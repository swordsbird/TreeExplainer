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

random_state = 190

class Model(BaseModel):
    def __init__(self, model_name):
        super().__init__()
        self.data_name = 'german'
        self.data_path = os.path.join(data_path, 'discarded_case_german_credit/german_new.csv')
        self.data_table = pd.read_csv(self.data_path)
        self.target = 'credit_risk'
        self.output_labels = ['reject', 'accept']
        self.current_encoding = german_credit_encoding
        self.to_category_idx = [3]

        self.model_name = model_name
        if model_name == 'rf' or model_name == 'random forest':
            self.parameters = {
                'n_estimators': 104,
                'max_depth': 24,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 5,
                'random_state': random_state,
            }
        else:
            self.parameters = {
                'n_estimators': 100,
                'max_depth': 12,
                'random_state': random_state,
            }

    def init_data(self):
        qualitative_features = [
            'credit_history', 'purpose', 'other_debtors',
            'property', 'other_installment_plans',
            'housing', 'job', 'people_liable', 'telephone',
            'foreign_worker', 'number_credits',
        ]
        data_table = self.data_table
        for feature in qualitative_features:
            unique_values = np.unique(data_table[feature].values)
            sorted(unique_values)
            if int(unique_values[0]) == 0:
                for i in unique_values:
                    data_table[feature + ' - '+ str(i)] = data_table[feature].values == i
            else:
                for i in unique_values:
                    data_table[feature + ' - '+ str(int(i) - 1)] = data_table[feature].values == i
        data_table['personal_status_sex'] = 1 * (data_table['personal_status_sex'].values == 3)
        for feature in qualitative_features:
            data_table = data_table.drop(feature, axis = 1)
        #data_table = data_table.drop('Other installment plans', axis = 1)
        X = data_table.drop(self.target, axis=1).values
        y = data_table[self.target].values
        indices = np.arange(X.shape[0])
        indices_train, indices_test, y_train, y_test = train_test_split(indices, y, test_size=self.test_size, random_state=random_state)
        X_train = X[indices_train]
        X_test = X[indices_test]
        self.indices_train = indices_train
        self.indices_test = indices_test

        categorical_features = np.ones(51) > 0
        categorical_features[[0, 1, 2, 3, 4, 5, 7, 8]] = False
        sm = SMOTENC(random_state=190, categorical_features=categorical_features)
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
    