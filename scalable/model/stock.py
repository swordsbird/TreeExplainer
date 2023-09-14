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

random_state = 42
yf_str_keys = ['industry', 'country', 'exchange', 'sector', 'previousConsensus']

class Model(BaseModel):
    def __init__(self, model_name):
        super().__init__()
        self.data_name = 'stock'
        self.data_path = os.path.join(data_path, 'case2_stock/stock.csv')
        self.data_table = pd.read_csv(self.data_path)
        self.target = 'analystConsensus'
        self.output_labels = ["buy", "hold", "sell"]
        self.model_id = 10

        self.model_name = model_name
        if model_name == 'rf' or model_name == 'random forest':
            self.parameters = {
                'n_estimators': 150,
                'max_depth': 30,
                'random_state': random_state,
            }
        else:
            self.parameters = {
                'n_estimators': 290,
                'learning_rate': 0.03071937624385482, 'max_depth': 5, 'lambda_l1': 0.3172618592564328, 'lambda_l2': 3.1153779347793686, 'feature_fraction': 0.15592221901953243, 'bagging_fraction': 0.8564868111751227, 'bagging_freq': 6, 'min_child_samples': 56,
                #'n_estimators': 330, 'learning_rate': 0.029233513856089702, 'max_depth': 12, 'lambda_l1': 2.662905300907257, 'lambda_l2': 0.6734919686040014, 'feature_fraction': 0.4500232685621936, 'bagging_fraction': 0.9731004417197868, 'bagging_freq': 5, 'min_child_samples': 98,
                'class_weight': 'balanced',
                'random_state': random_state,
            }
    
    def init_data(self):
        data_table = self.data_table.drop('ticker', axis=1)
        for k in ['QTLD30', 'VMA20', 'BETA60', 'KSFT2', 'KSFT', 'IMIN30', 'IMXD20', 'RSV5', 'KMID2', 'RANK20', 'CORD5', 'VSUMP60', 'VSUMD60', 'VSUMN60', 'MA10', 'WVMA10', 'IMAX60', 'RANK30', 'SUMP20', 'SUMD20', 'RESI5', 'SUMN20', 'QTLD20', 'SUMP5', 'SUMD5', 'SUMN5', 'QTLU5', 'MA60', 'CNTP20', 'BETA30', 'IMXD10', 'ROC20', 'IMAX30', 'VSUMP30', 'VSUMD30', 'VSUMN30', 'VMA10', 'RSV20', 'BETA20', 'CNTP60', 'CNTN60', 'VMA5', 'RSQR20', 'RESI60', 'IMIN60', 'WVMA5', 'RSV30', 'QTLU60', 'VSUMP20', 'VSUMD20', 'VSUMN20', 'RANK60', 'CNTN30', 'CNTD20', 'QTLU30', 'ROC30', 'RSV60', 'CNTP30', 'QTLU10', 'QTLU20', 'BETA10', 'MA30', 'RSQR30', 'IMXD60', 'ROC10', 'KLOW2', 'RSV10', 'SUMP30', 'SUMD30', 'SUMN30', 'KUP2', 'CNTN20', 'RSQR60', 'RSQR5', 'RESI30', 'SUMP10', 'SUMN10', 'SUMD10', 'SUMN60', 'CNTD30', 'SUMD60', 'SUMP60', 'RSQR10', 'VSUMN10', 'VSUMD10', 'MA20', 'VSUMP10', 'CNTD60', 'CNTD10', 'VSUMN5', 'VSUMD5', 'VSUMP5', 'IMXD30', 'RANK5', 'RANK10', 'IMAX5', 'IMAX10', 'IMAX20', 'IMIN5', 'IMIN10', 'IMIN20', 'IMXD5', 'CNTP5', 'CNTP10', 'CNTN5', 'CNTN10', 'CNTD5']:
            data_table = data_table.drop(k, axis=1)
        # data_table = data_table.drop('KLEN', axis=1)
        # data_table.loc[data_table['ebitda'].isna(), 'ebitda'] = data_table['ebitda'].max()
        # data_table['ebitda'].fillna(data_table['ebitda'].min())
        # print('ebitda', data_table['ebitda'].isna().sum())
        # data_table = data_table.drop('rating', axis=1)
        # data_table = data_table.drop('rating', axis=1)
        '''
        for k in data_table.columns:
            if data_table[k].dtype == float or data_table[k].dtype == int:
                x = data_table[k].values
                x = x[~np.isnan(x)]
                q_hi  = np.quantile(x, 0.995)
                q_lo  = np.quantile(x, 0.005)
                #print(k, q_hi, q_lo)

                data_table.loc[data_table[k] > q_hi, k] = q_hi
                data_table.loc[data_table[k] < q_lo, k] = q_lo
                #if k == 'price':
                #    print(data_table[k])
        '''
        for category in yf_str_keys:
            category_counts = data_table[category].value_counts()
            if category == 'previousConsensus':
                continue
            low_count_categories = category_counts[category_counts < 50].index
            data_table.loc[data_table[category].isin(low_count_categories), category] = 'Others'

        X = data_table.drop(self.target, axis=1)
        y = data_table[self.target]
        print(y.value_counts())

        X_encoded = pd.get_dummies(X, columns = yf_str_keys)
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
        
        self.test_rating = X_test['rating'].values
        self.train_rating = X_train['rating'].values
        self.X_train = X_train.drop('rating', axis = 1).values
        self.y_train = y_train.values
        self.X_test = X_test.drop('rating', axis = 1).values
        self.y_test = y_test.values
        self.X = X_encoded.drop('rating', axis = 1).values
        self.y = y.values
        self.data_table = X_encoded

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
    