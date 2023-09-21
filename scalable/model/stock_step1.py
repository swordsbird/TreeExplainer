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
        self.model_id = 101

        self.model_name = model_name
        if model_name == 'rf' or model_name == 'random forest':
            self.parameters = {
                'n_estimators': 150,
                'max_depth': 30,
                'random_state': random_state,
            }
        else:
            self.parameters = {
                'n_estimators': 370, 'learning_rate': 0.05316362156247843, 'max_depth': 6, 'feature_fraction': 0.8731725047139053, 'bagging_fraction': 0.8450537283888565, 'bagging_freq': 2, 'min_child_samples': 183,
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
    # print(model.X_train.mean())
    model.train()
    model.get_performance()

    y_pred_prob = model.clf.predict_proba(model.X_test)
    y_pred = model.clf.predict(model.X_test)
    conf_mat = confusion_matrix(model.y_test, y_pred)
    idx = np.zeros(len(model.X_test)) > 0
    ratios = []
    for i in np.argsort(y_pred_prob[:, 2])[::-1][:20]:
        ratios.append(model.test_rating[i])

    print(np.mean(ratios))
    print(ratios)
        
    accuracys = []
    num_classes = len(model.output_labels)
    for i in range(num_classes):
        accuracy = conf_mat[i, i] / conf_mat[i].sum()
        accuracys.append(accuracy)
        print(f'Accuracy on {model.output_labels[i]}: {accuracy}')

    model.generate_path()
    