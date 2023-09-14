from sklearn.model_selection import train_test_split

import os
import sys
sys.path.append('.')
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scalable.model.base_model import BaseModel
from scalable.config import data_path

random_state = 10

class Model(BaseModel):
    def __init__(self, model_name):
        super().__init__()
        self.data_name = 'abalone'
        self.data_path = os.path.join(data_path, 'abalone.csv')
        self.data_table = pd.read_csv(self.data_path)
        self.target = 'Category size'
        self.output_labels = ['0', '1']

        self.model_name = model_name
        if model_name == 'rf' or model_name == 'random forest':
            self.parameters = {
                'n_estimators': 50,
                'max_depth': 7,
                'min_samples_split': 4,
                'min_samples_leaf': 1,
                'random_state': random_state,
            }
        else:
            self.parameters = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'num_leaves': 25,
                'max_depth': 7,
                'random_state': random_state,
            }
    def init_data(self):
        data_table = self.data_table

        category = np.repeat(0, data_table.shape[0])
        for i in range(0, data_table["Rings"].size):
            if data_table["Rings"][i] <= 11:
                category[i] = 0
            elif data_table["Rings"][i] > 11:
                category[i] = 1

        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(data_table['Sex'])
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

        data_table = data_table.drop(['Sex'], axis=1)
        data_table['Category size'] = category
        data_table = data_table.drop(['Rings'], axis=1)

        features = data_table.iloc[:, np.r_[0: 7]]
        labels = data_table.iloc[:, 7]

        X_train, X_test, y_train, y_test, X_gender_train, X_gender_test = \
            train_test_split(features, labels, onehot_encoded, random_state=random_state, test_size=0.25)

        X_train = np.concatenate((X_train, X_gender_train), axis=1)
        X_test = np.concatenate((X_test, X_gender_test), axis=1)

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X = np.concatenate((X_train, X_test), axis=0)
        self.y = np.concatenate((y_train, y_test), axis=0)
        self.data_table = data_table

        self.check_columns(data_table, self.target)

if __name__ == '__main__':
    model = Model('lightgbm')
    model.init_data()
    model.train()
    model.get_performance()
    model.generate_path()
