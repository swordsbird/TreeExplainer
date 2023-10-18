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
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

random_state = 42

class Model(BaseModel):
    def __init__(self, model_name):
        super().__init__()
        self.data_name = 'stock'
        self.data_path = os.path.join(data_path, 'case2_stock/step/3year_3.csv')
        self.data_table = pd.read_csv(self.data_path)

        self.test_data_path = os.path.join(data_path, 'case2_stock/step/3month_3.csv')
        self.test_data_table = pd.read_csv(self.test_data_path)

        self.target = 'label'
        self.output_labels = ["increase", "decrease", "stable"]
        self.model_id = 105

        self.model_name = model_name
        if model_name == 'rf' or model_name == 'random forest':
            self.parameters = {
                'n_estimators': 150,
                'max_depth': 30,
                'random_state': random_state,
            }
        else:
            self.parameters = {
                'n_estimators': 450, 'learning_rate': 0.0642046991766106, 'max_depth': 10, 'feature_fraction': 0.8128422997766153, 'bagging_fraction': 0.8912579151709198, 'bagging_freq': 5, 'min_child_samples': 829,
                'class_weight': 'balanced',
                'verbosity': -1,
            }
            self.parameters2 = {
                'n_estimators': 330, 'learning_rate': 0.07211735911931144, 'max_depth': 6, 'feature_fraction': 0.4886664588097667, 'bagging_fraction': 0.8246631152639385, 'bagging_freq': 6, 'min_child_samples': 554,
                'class_weight': 'balanced',
                'verbosity': -1,
            }

    def train(self):
        if self.model_name == 'random forest' or self.model_name == 'rf':
            self.clf1 = RandomForestClassifier(**self.parameters)
            self.clf2 = RandomForestClassifier(**self.parameters)
        elif self.model_name == 'lightgbm' or self.model_name == 'lgbm':
            self.clf1 = LGBMClassifier(**self.parameters)
            self.clf2 = LGBMClassifier(**self.parameters2)
        self.clf1.fit(self.X_train, self.y_train_1)
        self.clf2.fit(self.X_train, self.y_train_2)

    def init_data(self):
        self.data_table = self.data_table.drop('date', axis=1)
        self.test_data_table = self.test_data_table.drop('date', axis=1)

        data_table = self.data_table.drop('ticker', axis=1)
        data_table = data_table.drop('newPrice', axis = 1)
        data_table = data_table.drop('currentPrice', axis = 1)

        features = data_table.columns.tolist()
        features = [k for k in features if k != 'rating' and k != 'label']

        for key in stock_encoding:
            index = 0
            for i in range(len(features)):
                if key in features[i]:
                    features[i] = key + '_' + stock_encoding[key][index]
                    index += 1

        X_train = data_table[features]
        y_train_1 = data_table[self.target] != 'stable'
        y_train_2 = data_table[self.target] == 'increase'
        X_test = self.test_data_table[features]
        y_test = self.test_data_table[self.target]

        self.train_rating = data_table['rating'].values
        self.test_rating = self.test_data_table['rating'].values
        self.X_train = X_train.values
        self.y_train_1 = y_train_1.values
        self.y_train_2 = y_train_2.values
        self.y_train = data_table[self.target].values
        self.X_test = X_test.values
        self.y_test = y_test.values
        self.X = X_train.values
        self.y = data_table[self.target].values
        self.data_table = data_table.drop('rating', axis = 1)

        self.check_columns(self.data_table, self.target)

if __name__ == '__main__':
    model = Model('lightgbm')
    model.init_data()
    # print(model.X_train.mean())
    model.train()

    y_pred1 = model.clf1.predict(model.X_train)
    y_pred2 = model.clf2.predict(model.X_train)
    y_pred = ['increase' if y_pred2[i] else ('decrease' if y_pred1[i] else 'stable') for i in range(len(y_pred1))]
    #y_pred = ['stable' if not y_pred1[i] else ('increase' if y_pred2[i] else 'decrease') for i in range(len(y_pred1))]
    conf_mat = confusion_matrix(model.y_train, y_pred)
    accuracys = []
    # model.output_labels = model.clf.classes_
    num_classes = len(model.output_labels)
    for i in range(num_classes):
        accuracy = conf_mat[i, i] / conf_mat[i].sum()
        accuracys.append(accuracy)
        print(f'Train Accuracy on {model.output_labels[i]}: {accuracy}')

    y_pred1 = model.clf1.predict(model.X_test)
    y_pred2 = model.clf2.predict(model.X_test)
    y_pred = ['increase' if y_pred2[i] else ('decrease' if y_pred1[i] else 'stable') for i in range(len(y_pred1))]
    #y_pred = ['stable' if not y_pred1[i] else ('increase' if y_pred2[i] else 'decrease') for i in range(len(y_pred1))]
    conf_mat = confusion_matrix(model.y_test, y_pred)
    accuracys = []
    # model.output_labels = model.clf.classes_
    num_classes = len(model.output_labels)
    for i in range(num_classes):
        accuracy = conf_mat[i, i] / conf_mat[i].sum()
        accuracys.append(accuracy)
        print(f'Test Accuracy on {model.output_labels[i]}: {accuracy}')
    print(conf_mat)
    model.generate_path()
