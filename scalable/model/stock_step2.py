from sklearn.model_selection import train_test_split

import os
import sys
sys.path.append('.')
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from scalable.model.base_model import BaseModel
from scalable.model.data_encoding import stock_encoding
from scalable.config import data_path
from scalable.model.data_encoding import stock_encoding

random_state = 42

selected_features = ['label', 'rating', 'payable', 'comprehensiveIncomeNetOfTax', 'profitLoss', 'dividendPayout', 'operatingCashflow', 'investments', 'changeInCash', 'marketCap', 'retainedEarningPerShareTTM', 'netInterestIncome', 'totalLiabilitiesTTM', 'totalAssetsTTM', 'revenuePerShareTTM', 'interestIncome', 'cashflowFromInvestment', 'estimatedEPS', 'reportedEPS', 'changeInOperatingLiabilities', 'totalCurrentLiabilities', 'pbRatioTTM', 'peRatioTTM', 'netIncomeParentTTM', 'operatingExpenses', 'interestExpense', 'changeInReceivables', 'currentAssetRatio', 'totalEquity', 'shortTermDebt', 'currentDebt', 'proceedsFromRepurchase', 'surprise', 'cashflowPerShareTTM', 'evToEbit', 'totalRevenueTTM', 'surprisePercentage', 'inventory', 'interestAndDebtExpense', 'grossProfit', 'operatingIncome', 'incomeTaxExpense', 'netIncomeTTM', 'depreciationAndAmortization', 'costOfRevenue', 'changeInInventory', 'totalCashflowTTM', 'currentRatio', 'psRatioTTM', 'pegRatioTTM', 'returnOnEquityTTM', 'returnOnAssetTTM', 'netIncomeToRevenueTTM', 'cashflowRatio', 'debtToAssetRatio', 'incRevenueTTM', 'KLEN', 'STD60', 'MIN5', 'LOW0', 'VSTD60', 'MIN60', 'WVMA30', 'MIN10', 'STD5', 'WVMA60', 'CORD60', 'MIN30', 'CORR60', 'CORR30', 'KLOW', 'QTLD5', 'WVMA20', 'MIN20', 'VSTD10', 'CORR20', 'CORD30', 'VSTD5', 'QTLD10', 'KUP', 'RESI20', 'MA5', 'CORR10', 'CORD20', 'HIGH0', 'RESI10', 'BETA5', 'BETA30', 'CORD10', 'KMID', 'OPEN0', 'ROC5', 'CORR5', 'industry_Others', 'industry_Biotechnology', 'industry_Software—Application', 'industry_Semiconductors', 'industry_Medical Devices', 'industry_Asset Management', 'industry_Software—Infrastructure', 'industry_Oil & Gas E&P', 'industry_Specialty Industrial Machinery', 'industry_Banks—Regional', 'country_United States', 'country_China', 'country_Others', 'country_Israel', 'country_Canada', 'country_United Kingdom', 'exchange_NMS', 'exchange_NCM', 'exchange_NGM', 'exchange_ASE', 'exchange_NYQ', 'exchange_Others', 'sector_Consumer Cyclical', 'sector_Healthcare', 'sector_Technology', 'sector_Consumer Defensive', 'sector_Industrials', 'sector_Communication Services', 'sector_Financial Services', 'sector_Basic Materials', 'sector_Energy', 'sector_Real Estate', 'sector_Utilities']

class Model(BaseModel):
    def __init__(self, model_name):
        super().__init__()
        self.data_name = 'stock'
        self.data_path = os.path.join(data_path, 'case2_stock/step/3year.csv')
        self.data_table = pd.read_csv(self.data_path)
        self.target = 'label'
        self.output_labels = ["decrease", "stable", "increase"]
        self.model_id = 110

        self.model_name = model_name
        if model_name == 'rf' or model_name == 'random forest':
            self.parameters = {
                'n_estimators': 150,
                'max_depth': 30,
                'random_state': random_state,
            }
        else:
            self.parameters = {
#                'n_estimators': 430, 'learning_rate': 0.05599061280807796, 'max_depth': 6, 'feature_fraction': 0.7082156453527435, 'bagging_fraction': 0.8774051099368454, 'bagging_freq': 4, 'min_child_samples': 115,
                'n_estimators': 490, 'learning_rate': 0.09639866565422027, 'max_depth': 8, 'feature_fraction': 0.49881480588365473, 'bagging_fraction': 0.9990501712986751, 'bagging_freq': 4, 'min_child_samples': 300,
                'class_weight': 'balanced',
                #'random_state': random_state,
                'verbosity': -1,
            }

    def init_data(self):
        data_table = self.data_table.drop('ticker', axis=1)
        data_table = data_table.drop('newPrice', axis = 1)
        data_table = data_table.drop('currentPrice', axis = 1)
        data_table = data_table[selected_features]

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
    for i in np.argsort(y_pred_prob[:, 1])[::-1][:20]:
        ratios.append(model.test_rating[i])

    print(np.mean(ratios))
    print(ratios)

    accuracys = []
    model.output_labels = model.clf.classes_
    num_classes = len(model.output_labels)
    for i in range(num_classes):
        accuracy = conf_mat[i, i] / conf_mat[i].sum()
        accuracys.append(accuracy)
        print(f'Accuracy on {model.output_labels[i]}: {accuracy}')

    model.generate_path()
    '''
    sorted_features = [(model.clf.feature_name_[i], model.clf.feature_importances_[i]) for i in range(len(model.clf.feature_importances_))]
    sorted_features = sorted(sorted_features, key = lambda x: -x[1])
    for k in sorted_features:
        i, j = k
        i = model.data_table.columns[int(i.split('_')[1])]
        print(i, j)
    '''
