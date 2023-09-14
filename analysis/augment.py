from sklearn.model_selection import train_test_split

import os
import sys
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC, SMOTE
sys.path.append('.')
sys.path.append('..')
from scalable.config import data_path
from scalable.model.base_model import BaseModel
from sklearn.ensemble import RandomForestClassifier
from scalable.data_encoding import german_credit_encoding
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sdv.lite import SingleTablePreset
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

random_state = 42

new_label = {
    10: 0, 49: 0, 206: 0, 270: 0, 592: 0, 606: 0, 621: 0, 622: 0,
}

meta_json = {
    "columns": {
        "Gender": {
            "sdtype": "categorical"
        },
        "Age": {
            "sdtype": "numerical",
            "computer_representation": "Float"
        },
        "Debt": {
            "sdtype": "numerical",
            "computer_representation": "Float"
        },
        "YearsEmployed": {
            "sdtype": "numerical",
            "computer_representation": "Float"
        },
        "PriorDefault": {
            "sdtype": "categorical"
        },
        "Employed": {
            "sdtype": "categorical"
        },
        "DriversLicense": {
            "sdtype": "categorical"
        },
        "Citizen": {
            "sdtype": "categorical"
        },
        "ZipCode": {
            "sdtype": "categorical"
        },
        "Approved": {
            "sdtype": "categorical"
        },
        "Income": {
            "sdtype": "numerical",
            "computer_representation": "Float"
        },
        "CreditScore": {
            "sdtype": "numerical",
            "computer_representation": "Float"
        },
        "Married": {
            "sdtype": "categorical"
        },
        "BankCustomer": {
            "sdtype": "categorical"
        },
        "Job": {
            "sdtype": "categorical"
        },
        "Ethnicity": {
            "sdtype": "categorical"
        }
    },
    "METADATA_SPEC_VERSION": "SINGLE_TABLE_V1"
}


class Model(BaseModel):
    def __init__(self, model_name):
        super().__init__()
        self.data_name = 'credit'
        self.data_path = os.path.join(data_path, 'credit_card_train1.csv')
        self.data_table = pd.read_csv(self.data_path)
        self.data_table_original = self.data_table.copy()
        self.target = 'Approved'
        self.output_labels = ['0', '1']

        self.model_name = model_name
        if model_name == 'rf' or model_name == 'random forest':
            self.parameters = {
                'n_estimators': 200,
                'max_depth': 10,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'max_features': 8,
                'random_state': random_state,
            }
        else:
            self.parameters = {
                'n_estimators': 100,
                'max_depth': 12,
                'random_state': random_state,
            }

    def synthesize_data(self, label, num_rows):
        metadata = SingleTableMetadata.load_from_dict(meta_json)
        #synthesizer = SingleTablePreset(metadata, name='FAST_ML')
        synthesizer = CTGANSynthesizer(metadata)
        idx = self.data_table_original['Approved'] == label
        original_data = self.data_table_original[idx]
        synthesizer.fit(original_data)
        data = synthesizer.sample(num_rows=num_rows)
        data.loc['Approved', :] = label
        print(data.head())
        return data

    def transform_data(self, data_table):
        data_table = data_table.copy()
        data_table['ZipCode'] = [-1 if i in self.other_zipcodes else i for i in data_table['ZipCode'].values]
        for feature in self.qualitative_features:
            if feature not in self.binary_features:
                for val in self.feature_values[feature]:
                    data_table[feature + ' - '+ str(val)] = data_table[feature].values == val
        for feature in self.qualitative_features:
            if feature not in self.binary_features:
                data_table = data_table.drop(feature, axis = 1)

        X = data_table[self.data_table.columns].drop(self.target, axis=1).values
        y = data_table[self.target].values
        return X, y

    def init_data(self, sampling_rate = 1):
        qualitative_features = ['Gender', 'Married', 'BankCustomer', 'Job', 'Ethnicity', 'PriorDefault', 'Employed', 'DriversLicense', 'Citizen', 'ZipCode']
        self.qualitative_features = qualitative_features
        binary_features = []
        data_table = self.data_table

        data_table['Age'] = [np.floor(x) for x in data_table['Age'].values]

        unique_values = np.unique(data_table['ZipCode'].values)
        unique_values = sorted(unique_values)
        unique_groups = []
        for val in unique_values:
            unique_groups.append((val, (data_table['ZipCode'].values == val).sum()))
        other_zipcodes = [zc for zc, zc_count in unique_groups if zc_count < 30]
        new_zipcode = [-1 if i in other_zipcodes else i for i in data_table['ZipCode'].values]
        self.other_zipcodes = other_zipcodes
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
        self.binary_features = binary_features

        features = data_table.drop(self.target, axis=1).columns
        X = data_table.drop(self.target, axis=1).values
        y = data_table[self.target].values
        indices = np.arange(X.shape[0])
        indices_train, indices_test, y_train, y_test = train_test_split(indices, y, test_size=self.test_size, random_state=random_state)
        X_train = X[indices_train]
        X_test = X[indices_test]
        self.indices_train = indices_train
        self.indices_test = indices_test
        categorical_features = np.ones(len(features)) == 0

        for i in range(len(features)):
            if ' - ' in features[i]:
                categorical_features[i] = True
        sm = SMOTENC(random_state=42, categorical_features=categorical_features, sampling_strategy = sampling_rate)
        #sm = SMOTE(random_state=random_state)
        output = sm.fit_resample(X_train, y_train)
        X_train = output[0]
        y_train = output[1]
        for i, j in enumerate(indices_train):
            if j in new_label:
                y_train[i] = new_label[j]
                
        for i, j in enumerate(indices_test):
            if j in new_label:
                y_test[i] = new_label[j]


        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        
        self.y_test = y_test
        self.X = X
        self.y = y
        self.data_table = data_table

        self.check_columns(data_table, self.target)

model = Model('random forest')
model.init_data()
model.train()
model.get_performance()

data = pd.read_csv('synthetic.csv')
data = data[model.data_table_original.columns]

X, y = model.transform_data(data)
y_pred = model.clf.predict(X)
ret3 = (y == y_pred).sum()
print(ret3 / len(y))

model.model_id = -1
model.generate_path()

conds = {}

#conds['cond1'] = (data['Age'] <= 40.5) & (data['Income'] <= 246.5) & (data['Married'] == 0) & (data['Job'] != 6) & (data['Employed'] == 0) & (data['BankCustomer'] == 1)
conds['cond2'] = (data['Debt'] > 3.7) & (data['Married'] == 0) & (data['DriversLicense'] == 1) & (data['CreditScore'] <= 2.5) & (data['Job'] == 7)
conds['cond3'] = (data['Income'] <= 195.5) & (data['Married'] == 1) & (data['Job'] != 13) & (data['YearsEmployed'] <= 0.7)
conds['cond4'] = (data['Debt'] <= 1.8) & (data['YearsEmployed'] > 0.8) & (data['Employed'] == 0) & (data['DriversLicense'] == 0) & (data['Job'] != 1)
conds['cond5'] = (data['YearsEmployed'] <= 1.8) & (data['YearsEmployed'] > 0.6) & (data['CreditScore'] <= 2.5) & (data['DriversLicense'] == 1) & (data['Job'] != 5)
conds['cond6'] = (data['Married'] == 1) & (data['YearsEmployed'] > 4) & (data['Employed'] == 0) & (data['CreditScore'] <= 2.5) & (data['DriversLicense'] == 1) & (data['Ethnicity'] != 'Black') & (data['Citizen'] != 'ByOtherMeans')
conds['cond7'] = (data['Gender'] == 1) & (data['BankCustomer'] == 1) & (data['YearsEmployed'] > 2.8) & (data['CreditScore'] <= 0.5) & (data['DriversLicense'] == 0) & (data['Income'] <= 145) & (data['Ethnicity'] != 'Latino')
conds['cond8'] = (data['Debt'] > 1.1) & (data['Debt'] <= 4.2) & (data['Married'] == 0) & (data['Employed'] == 1) & (data['Income'] <= 148.5) & (data['Job'] != 5)
conds['cond9'] = (data['BankCustomer'] == 0) & (data['Employed'] == 1) & (data['Income'] <= 0.5) & (data['Ethnicity'] != 'Latino')
conds['cond10'] = (data['YearsEmployed'] <= 0.2) & (data['PriorDefault'] == 1) & (data['CreditScore'] <= 3.5) & (data['DriversLicense'] == 1) & (data['Income'] <= 0.5) & (data['ZipCode'] != 0)
# conds['cond11'] = (data['Gender'] == 1) & (data['Age'] > 40) & (data['Debt'] <= 1) & (data['YearsEmployed'] > 3.9) & (data['Employed'] == 0)
conds['cond12'] = (data['Age'] <= 23) & (data['Debt'] > 1.9) & (data['YearsEmployed'] > 2.5) & (data['Employed'] == 0) & (data['DriversLicense'] == 0) & (data['Job'] != 6)
conds['cond13'] = (data['Debt'] > 4.3) & (data['PriorDefault'] == 1) & (data['Employed'] == 0) & (data['Income'] <= 2019) & (data['Citizen'] != 'ByBirth')
conds['cond14'] = (data['Debt'] > 1.5) & (data['YearsEmployed'] <= 0.2) & (data['PriorDefault'] == 1) & (data['CreditScore'] <= 0.5) & (data['DriversLicense'] == 1) & (data['Income'] <= 475.5)
conds['cond15'] = (data['Married'] == 1) & (data['YearsEmployed'] > 3.9) & (data['Income'] <= 0.5) & (data['Ethnicity'] != 'Black') & (data['Citizen'] == 'ByBirth')
conds['cond16'] = (data['Gender'] == 0) & (data['Debt'] <= 10.8) & (data['Married'] == 1) & (data['YearsEmployed'] <= 4.6) & (data['PriorDefault'] == 1) & (data['CreditScore'] <= 0.5) & (data['Income'] <= 0.5)
conds['cond17'] = (data['Debt'] <= 0.6) & (data['PriorDefault'] == 1) & (data['Employed'] == 0) & (data['Income'] <= 437.5) & (data['Job'] != 9) & (data['Ethnicity'] != 'Latino') & (data['Citizen'] == 'ByBirth')
conds['cond18'] = (data['Debt'] > 2.4) & (data['Married'] == 1) & (data['YearsEmployed'] <= 0.1) & (data['PriorDefault'] == 1) & (data['CreditScore'] <= 0.5) & (data['Citizen'] == 'ByBirth')
conds['cond19'] = (data['Age'] <= 39) & (data['Debt'] > 1.8) & (data['Debt'] <= 9.4) & (data['BankCustomer'] == 1) & (data['PriorDefault'] == 1) & (data['Employed'] == 0) & (data['Income'] <= 385.5) & (data['Ethnicity'] == 'White')
# conds['cond20'] = (data['Gender'] == 1) & (data['YearsEmployed'] > 0.9) & (data['PriorDefault'] == 1) & (data['Employed'] == 0) & (data['Income'] <= 0.5) & (data['Job'] != 2) & (data['Citizen'] == 'ByBirth')

for k in conds:
    cond = conds[k]
    tdata = data[cond]
    pos_data = tdata[tdata['Approved'] == 1]
    neg_data = tdata[tdata['Approved'] == 0]
    pos = len(pos_data)
    neg = len(neg_data)
    n = pos + neg
    if pos < neg:
        neg_data = neg_data.sample(n = pos)
    else:
        pos_data = pos_data.sample(n = neg)

    #new_data = pd.concat((pos_data, neg_data), axis = 0)
    
    print(k, pos, neg)
    
    new_data = pos_data
    X, y = model.transform_data(new_data)
    y_pred = model.clf.predict(X)
    ret = (y == y_pred).sum()
    print(len(new_data), ret / len(new_data))
    
    new_data = neg_data
    X, y = model.transform_data(new_data)
    y_pred = model.clf.predict(X)
    ret = (y == y_pred).sum()
    print(len(new_data), ret / len(new_data))
    print()

from scalable.model_utils import ModelUtil
from scalable.anomaly_detection import LRAnomalyDetection

util = ModelUtil(data_name = 'credit', model_name = 'random forest')
old_data = model.data_table.copy()
current_data = []

for k in conds:
    cond = conds[k]
    min_score = 1
    for i in range(50):
        tdata = data[cond]
        pos_data = tdata[tdata['Approved'] == 1]
        neg_data = tdata[tdata['Approved'] == 0]
        pos = len(pos_data)
        neg = len(neg_data)
        n = pos + neg
        if pos < neg:
            neg_data = neg_data.sample(n = pos)
        else:
            pos_data = pos_data.sample(n = neg)

        new_data = pd.concat((pos_data, neg_data), axis = 0)
        new_data = new_data.sample(n = 20)
        
        new_data_table = pd.concat((new_data, old_data), axis = 0)

        if len(current_data) == 0:
            current = new_data
        else:
            current = pd.concat((current_data, new_data), axis = 0)
        X_train, y_train = model.transform_data(current)
        X_train = np.concatenate((X_train, model.X_train), axis = 0)
        y_train = np.concatenate((y_train, model.y_train), axis = 0)
        model.clf.fit(X_train, y_train)

        model.model_id = -1
        util.model = model
        model.generate_path()
        X, y = util.get_rule_matrix()
        y = y.astype(int)

        detect = LRAnomalyDetection(X, y)
        score = detect.score().mean()
        if score < min_score:
            min_score = score
            best_new_data = new_data
    if len(current_data) == 0:
        current_data = best_new_data
    else:
        current_data = pd.concat((current_data, best_new_data), axis = 0)
    best_new_data.to_csv(f'result/{k}_current.csv', index=False)
    print(f'{k}, {min_score}')