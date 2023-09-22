from sklearn.model_selection import train_test_split

import os
import sys
sys.path.append('.')
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from scalable.model.stock_step2 import Model

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

    print('X_train', X_train.shape)
    print('y_train', y_train.shape)
    print('X_test', X_test.shape)
    print('y_test', y_test.shape)

    def objective(trial, X_train, y_train, X_test, y_test):
        # 后面填充
        param_grid = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=20),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            #"lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0),
            #"lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.1, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 9),
            "min_child_samples": trial.suggest_int("min_child_samples", 30, 300),
        }

        model = LGBMClassifier(**param_grid, verbose=-1, class_weight = 'balanced')
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        conf_mat = confusion_matrix(y_test, y_pred)
        accuracys = []

        tot = 0
        for i in range(model.n_classes_):
            accuracy = conf_mat[i, i] / conf_mat[i].sum()
            accuracys.append(accuracy)
            print(f'Accuracy on {model.classes_[i]}: {accuracy}')
            tot += conf_mat[i, i]
        tot /= len(y_pred)
        print(f'Accuracy: {tot}')
        
        return (accuracys[0] + accuracys[1] + accuracys[2] * 1.6) / 3.6

    study = optuna.create_study(direction="maximize", study_name="LGBM Classifier")
    func = lambda trial: objective(trial, X_train, y_train, X_test, y_test)
    study.optimize(func, n_trials=1500)
    
    print(study.best_params)
