from sklearn.model_selection import train_test_split

import os
import sys
sys.path.append('.')
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC, SMOTE
from scalable.config import data_path
from scalable.model.base_model import BaseModel
from scalable.model.data_encoding import german_credit_encoding
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score

random_state = 42

class Model(BaseModel):
    def __init__(self, model_name):
        super().__init__()
        self.data_name = 'credit'
        self.data_path = os.path.join(data_path, 'case1_credit_card/credit_card_train1.csv')
        self.data_table = pd.read_csv(self.data_path)
        self.target = 'Approved'
        self.output_labels = ['0', '1']
        self.model_id = 2

        self.model_name = model_name
        if model_name == 'rf' or model_name == 'random forest':
            self.parameters = {
                'n_estimators': 200,
                'max_depth': 9,
                'min_samples_split': 7,
                'min_samples_leaf': 4,
                'max_features': 8,
                'random_state': random_state,
            }
        else:
            self.parameters = {
                'n_estimators': 100,
                'max_depth': 12,
                'random_state': random_state,
            }

    def init_data(self, sampling_rate = 1):
        qualitative_features = ['Gender', 'Married', 'BankCustomer', 'Job', 'Ethnicity', 'PriorDefault', 'Employed', 'DriversLicense', 'Citizen', 'ZipCode']
        binary_features = []
        data_table = self.data_table

        data_table['Age'] = [np.floor(x) for x in data_table['Age'].values]
        def income_transform(x):
            if x == 0:
                return 0
            elif x < 100:
                return 1
            elif x < 1000:
                return 2
            elif x < 5000:
                return 3
            else:
                return 5
        data_table['Income'] = [income_transform(x) for x in data_table['Income'].values]

        unique_values = np.unique(data_table['ZipCode'].values)
        unique_values = sorted(unique_values)
        unique_groups = []
        for val in unique_values:
            unique_groups.append((val, (data_table['ZipCode'].values == val).sum()))
        other_zipcodes = [zc for zc, zc_count in unique_groups if zc_count < 30]
        new_zipcode = [-1 if i in other_zipcodes else i for i in data_table['ZipCode'].values]
        data_table['ZipCode'] = new_zipcode
        unique_values = np.unique(new_zipcode)
        unique_values = sorted(unique_values)
        self.feature_values['ZipCode'] = unique_values
        self.transform['ZipCode'] = new_zipcode
        for feature in qualitative_features:
            unique_values = np.unique(data_table[feature].values)
            unique_values = sorted(unique_values)
            self.feature_values[feature] = unique_values
            if len(unique_values) == 2:
                unique_values = unique_values[1:]
                binary_features.append(feature)
            else:
                for i, val in enumerate(unique_values):
                    data_table[feature + ' - '+ str(val)] = data_table[feature].values == val
        for feature in qualitative_features:
            if feature not in binary_features:
                data_table = data_table.drop(feature, axis = 1)

        X = data_table.drop(self.target, axis=1).values
        y = data_table[self.target].values
        indices = np.arange(X.shape[0])
        indices_train, indices_test, y_train, y_test = train_test_split(indices, y, test_size=self.test_size, random_state=random_state)
        X_train = X[indices_train]
        X_test = X[indices_test]
        self.indices_train = indices_train
        self.indices_test = indices_test

        features = data_table.drop(self.target, axis=1).columns
        categorical_features = np.ones(len(features)) == 0
        for i in range(len(features)):
            if ' - ' in features[i]:
                categorical_features[i] = True
            else:
                print(features[i])
        sm = SMOTENC(random_state=42, categorical_features=categorical_features, sampling_strategy = sampling_rate)
        #sm = SMOTE(random_state=random_state)
        output = sm.fit_resample(X_train, y_train)
        X_train = output[0]
        y_train = output[1]

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        
        self.y_test = y_test
        self.X = X
        self.y = y
        self.data_table = data_table

        self.check_columns(data_table, self.target)

if __name__ == '__main__':

    model = Model('random forest')
    model.init_data()
    model.train()
    model.get_performance()
    model.generate_path()
    for p in model.paths:
        if 10 in p['range'] and p['range'][10][0] > 1000 and 5 in p['range'] and p['range'][5][0] > 1 and p['output'] == 1:
            print(p)
            for i in p['range']:
                print(model.features[i], p['range'][i])
