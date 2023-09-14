from sklearn.model_selection import train_test_split

import os
import sys
sys.path.append('.')
import pandas as pd
import numpy as np
from scalable.config import data_path
from sklearn.metrics import confusion_matrix
from scalable.model.base_model import BaseModel
from scalable.model.data_encoding import stock_encoding

random_state = 42
yf_str_keys = ['industry', 'country', 'exchange', 'sector', 'previousConsensus']

class Model(BaseModel):
    def __init__(self, model_name):
        super().__init__()
        self.data_name = 'stock'
        self.data_path = os.path.join(data_path, 'case2_stock/stock2.csv')
        self.data_table = pd.read_csv(self.data_path)
        self.target = 'analystConsensus'
        self.output_labels = ["buy", "hold", "sell"]
        self.model_id = 13

        self.model_name = model_name
        if model_name == 'rf' or model_name == 'random forest':
            self.parameters = {
                'n_estimators': 150,
                'max_depth': 30,
                'random_state': random_state,
            }
        else:
            self.parameters = {
                'n_estimators': 370, 'learning_rate': 0.0962075640046511, 'max_depth': 4, 'lambda_l1': 4.70492739286485, 'lambda_l2': 8.937781634928339, 'feature_fraction': 0.6523773307210478, 'bagging_fraction': 0.7273406046748414, 'bagging_freq': 2, 'min_child_samples': 97,
                #'n_estimators': 390, 'learning_rate': 0.16943032398007013, 'max_depth': 7, 'lambda_l1': 7.728881674600999, 'lambda_l2': 6.525799696249104, 'feature_fraction': 0.28774241427852687, 'bagging_fraction': 0.7299353465088981, 'bagging_freq': 2, 'min_child_samples': 56,
                #'n_estimators': 310, 'learning_rate': 0.15796392940323642, 'max_depth': 7, 'lambda_l1': 8.842115501844644, 'lambda_l2': 3.288457547360635, 'feature_fraction': 0.4736821145868658, 'bagging_fraction': 0.6975866946873115, 'bagging_freq': 2, 'min_child_samples': 78,
                #'n_estimators': 330, 'learning_rate': 0.029233513856089702, 'max_depth': 12, 'lambda_l1': 2.662905300907257, 'lambda_l2': 0.6734919686040014, 'feature_fraction': 0.4500232685621936, 'bagging_fraction': 0.9731004417197868, 'bagging_freq': 5, 'min_child_samples': 98,
                'class_weight': 'balanced',
                'random_state': random_state,
            }
    
    def init_data(self):
        data_table = self.data_table.drop('ticker', axis=1)

        X = data_table.drop(self.target, axis=1)
        y = data_table[self.target]

        X_encoded = pd.get_dummies(X, columns = yf_str_keys)
        features = X_encoded.columns.tolist()
        for key in yf_str_keys:
            index = 0
            for i in range(len(features)):
                if key in features[i]:
                    features[i] = key + '_' + stock_encoding[key][index]
                    index += 1
        X_encoded = X_encoded[features]

        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
        
        self.test_rating = X_test['rating'].values
        self.train_rating = X_train['rating'].values
        self.X_train = X_train.drop('rating', axis = 1).values
        self.y_train = y_train.values
        self.X_test = X_test.drop('rating', axis = 1).values
        self.y_test = y_test.values
        self.X = X_encoded.drop('rating', axis = 1).values
        self.y = y.values
        self.data_table = X_encoded.drop('rating', axis = 1)

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
    for i in np.argsort(y_pred_prob[:, 0])[::-1][:20]:
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
    