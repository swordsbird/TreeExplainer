from sklearn.model_selection import train_test_split

import os
import sys
sys.path.append('.')
import pandas as pd
from scalable.model.base_model import BaseModel
from scalable.model.data_encoding import stock_encoding
import numpy as np
from scalable.config import data_path
from sklearn.metrics import confusion_matrix
from scalable.model.data_encoding import stock_encoding

random_state = 42

class Model(BaseModel):
    def __init__(self, model_name):
        super().__init__()
        self.data_name = 'stock'
        self.data_path = os.path.join(data_path, 'case2_stock/3year.csv')
        self.data_table = pd.read_csv(self.data_path)
        self.target = 'label'
        self.output_labels = ["decrease", "stable", "increase"]
        self.model_id = -1

        self.model_name = model_name
        if model_name == 'rf' or model_name == 'random forest':
            self.parameters = {
                'n_estimators': 150,
                'max_depth': 30,
                'random_state': random_state,
            }
        else:
            self.parameters = {
                'n_estimators': 430, 'learning_rate': 0.05599061280807796, 'max_depth': 6, 'feature_fraction': 0.7082156453527435, 'bagging_fraction': 0.8774051099368454, 'bagging_freq': 4, 'min_child_samples': 115,
                'class_weight': 'balanced',
                #'random_state': random_state,
                'verbosity': -1,
            }
    
    def init_data(self):
        data_table = self.data_table.drop('ticker', axis=1)
        data_table = data_table.drop('newPrice', axis = 1)
        data_table = data_table.drop('currentPrice', axis = 1)

        
        X = data_table.drop(self.target, axis=1)

        features = X.columns.tolist()
        for key in stock_encoding:
            index = 0
            for i in range(len(features)):
                if key in features[i]:
                    features[i] = key + '_' + stock_encoding[key][index]
                    index += 1
        X = X[features]

        y = data_table[self.target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.test_rating = X_test['rating'].values
        self.train_rating = X_train['rating'].values
        self.X_train = X_train.drop('rating', axis = 1).values
        self.y_train = y_train.values
        self.X_test = X_test.drop('rating', axis = 1).values
        self.y_test = y_test.values
        self.X = X.drop('rating', axis = 1).values
        self.y = y.values
        self.data_table = data_table.drop('rating', axis = 1)

        self.check_columns(self.data_table, self.target)

if __name__ == '__main__':
    model = Model('lightgbm')
    model.init_data()

    from tabnanny import verbose
    import optuna  # pip install optuna
    from sklearn.metrics import log_loss, precision_score, f1_score, accuracy_score
    from sklearn.model_selection import KFold
    from optuna.integration import LightGBMPruningCallback
    from lightgbm import LGBMClassifier
    random_state = 10

    X_train = model.X_train
    y_train = model.y_train
    X_test = model.X_test
    y_test = model.y_test

    def objective(trial, X_train, y_train, X_test, y_test):
        # 后面填充
        param_grid = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=20),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            #"lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0),
            #"lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.1, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 9),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
        }

        model = LGBMClassifier(**param_grid, verbose=-1, class_weight = 'balanced')
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        conf_mat = confusion_matrix(y_test, y_pred)
        accuracys = []
        for i in range(model.n_classes_):
            accuracy = conf_mat[i, i] / conf_mat[i].sum()
            accuracys.append(accuracy)
        train_accuracy = accuracy_score(y_test, y_pred)    
        
        return np.mean(accuracys)

    study = optuna.create_study(direction="maximize", study_name="LGBM Classifier")
    func = lambda trial: objective(trial, X_train, y_train, X_test, y_test)
    study.optimize(func, n_trials=100)
    
    print(study.best_params)
