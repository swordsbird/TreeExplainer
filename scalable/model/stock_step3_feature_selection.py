from sklearn.model_selection import train_test_split

import os
import sys
sys.path.append('.')
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from scalable.model.stock_step3 import Model
distribution = np.array([0.2970, 0.2260, 0.4770])

def trail(features):
    clf = LGBMClassifier(**model.parameters)
    X_train = model.X_train[features].values
    y_train = model.y_train.values
    X_test = model.X_test[features].values
    y_test = model.y_test.values

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    conf_mat = confusion_matrix(y_test, y_pred)
    accuracys = []

    tot = 0
    for i in range(clf.n_classes_):
        accuracy = conf_mat[i, i] / conf_mat[i].sum()
        accuracys.append(accuracy)
        print(f'Accuracy on {clf.classes_[i]}: {accuracy}')
        tot += conf_mat[i, i]
    tot /= len(y_pred)
    print(f'Accuracy: {tot}')
    accuracys = np.array(accuracys)
    
    sorted_features = [(clf.feature_name_[i], clf.feature_importances_[i]) for i in range(len(clf.feature_importances_))]
    sorted_features = sorted(sorted_features, key = lambda x: -x[1])

    s = []
    for k in sorted_features[:5]:
        i, j = k
        i = features[int(i.split('_')[1])]
        s.append(i)
    print(', '.join(s))

    gain = np.maximum(accuracys - distribution, -1)
    gain = (gain[0] * gain[1] * gain[2]) ** 0.33
    return gain

if __name__ == '__main__':
    model = Model('lightgbm')
    model.init_data()

    import optuna  # pip install optuna
    from sklearn.metrics import log_loss, precision_score, f1_score, accuracy_score
    from lightgbm import LGBMClassifier
    random_state = 10

    base_features = ['grossProfit', 'totalRevenue', 'costOfRevenue', 'operatingIncome', 'operatingExpenses', 'interestIncome', 'interestExpense', 'otherNonOperatingIncome', 'depreciationAndAmortization', 'incomeTaxExpense', 'interestAndDebtExpense', 'comprehensiveIncomeNetOfTax', 'ebit', 'ebitda', 'netIncome', 'totalAssets', 'totalCurrentAssets', 'inventory', 'totalNonCurrentAssets', 'goodwill', 'investments', 'totalLiabilities', 'totalCurrentLiabilities', 'payable', 'currentDebt', 'shortTermDebt', 'longTermDebt', 'totalEquity', 'retainedEarnings', 'commonStock', 'operatingCashflow', 'changeInOperatingLiabilities', 'changeInOperatingAssets', 'changeInReceivables', 'profitLoss', 'cashflowFromInvestment', 'proceedsFromRepurchase', 'changeInCash', 'reportedEPS', 'estimatedEPS', 'surprise', 'surprisePercentage', 'currentRatio', 'totalCashflowTTM', 'netIncomeTTM', 'totalRevenueTTM', 'totalLiabilitiesTTM', 'ebitTTM', 'ebitdaTTM', 'retainedEarningsTTM', 'totalEquityTTM', 'totalShares', 'netIncomeParent', 'netIncomeParentTTM', 'KMID', 'KLEN', 'KMID2', 'KUP2', 'KLOW', 'KSFT2', 'OPEN0', 'ROC5', 'ROC10', 'ROC20', 'ROC30', 'MA20', 'MA30', 'STD5', 'STD10', 'STD20', 'STD30', 'STD60', 'BETA5', 'BETA10', 'RESI10', 'RESI20', 'RESI30', 'RESI60', 'MIN5', 'MIN60', 'QTLU10', 'QTLD5', 'QTLD10', 'QTLD20', 'QTLD30', 'QTLD60', 'RSV5', 'RSV10', 'RSV20', 'RSV30', 'RSV60', 'CORR20', 'CORR30', 'CORR60', 'CORD5', 'CORD10', 'CORD20', 'CORD30', 'CORD60', 'CNTP60', 'CNTD60', 'SUMN5', 'SUMD5', 'VMA10', 'VMA20', 'VMA30', 'VMA60', 'VSTD20', 'VSTD60', 'WVMA10', 'WVMA20', 'WVMA30', 'WVMA60', 'marketCap', 'peRatioTTM', 'pcfRatioTTM', 'pbRatioTTM', 'psRatioTTM', 'evToEbit', 'earningsGrowth', 'revenuePerShareTTM', 'returnOnEquityTTM', 'returnOnAsset', 'returnOnAssetTTM', 'returnOnAssetNetTTM', 'netIncomeToRevenue', 'netIncomeToRevenueTTM', 'ebitToRevenue', 'ebitToRevenueTTM', 'cashflowPerShareTTM', 'cashflowRatio', 'retainedEarningPerShare', 'retainedEarningPerShareTTM', 'debtToAssetRatio', 'currentAssetRatio', 'revenueGrowth', 'country_United States', 'country_China', 'country_Others', 'country_Israel', 'country_Canada', 'country_United Kingdom', 'exchange_NMS', 'exchange_NCM', 'exchange_NGM', 'exchange_ASE', 'exchange_NYQ', 'exchange_Others', 'NATR']

    new_features = ['CDLBELTHOLD', 'CDLCLOSINGMARUBOZU', 'CDLDOJI', 'CDLHIGHWAVE', 'CDLHIKKAKE', 'CDLLONGLEGGEDDOJI', 'CDLLONGLINE', 'CDLSHORTLINE', 'CDLSPINNINGTOP', 'ADX', 'ADXR', 'APO', 'AROONOSC', 'BOP', 'CCI', 'CMO', 'DX', 'MFI', 'MINUS_DI', 'MINUS_DM', 'MOM', 'PLUS_DI', 'PLUS_DM', 'PPO', 'ROC', 'ROCP', 'ROCR', 'ROCR100', 'RSI', 'TRIX', 'ULTOSC', 'WILLR', 'ATR', 'TRANGE', 'AD', 'ADOSC', 'OBV', 'alpha001', 'alpha002', 'alpha003', 'alpha004', 'alpha005', 'alpha006', 'alpha007', 'alpha008', 'alpha009', 'alpha010', 'alpha011', 'alpha012', 'alpha013', 'alpha014', 'alpha015', 'alpha016', 'alpha017', 'alpha018', 'alpha019', 'alpha020', 'alpha021', 'alpha022', 'alpha023', 'alpha024', 'alpha025', 'alpha026', 'alpha027', 'alpha028', 'alpha029', 'alpha030', 'alpha031', 'alpha032', 'alpha033', 'alpha034', 'alpha035', 'alpha036', 'alpha037', 'alpha038', 'alpha039', 'alpha040', 'alpha041', 'alpha042', 'alpha043', 'alpha044', 'alpha045', 'alpha046', 'alpha047', 'alpha049', 'alpha050', 'alpha051', 'alpha052', 'alpha053', 'alpha054', 'alpha055', 'alpha056', 'alpha057', 'alpha060', 'alpha061', 'alpha062', 'alpha064', 'alpha065', 'alpha066', 'alpha068', 'alpha071', 'alpha072', 'alpha073', 'alpha074', 'alpha075', 'alpha077', 'alpha078', 'alpha081', 'alpha083', 'alpha085', 'alpha086', 'alpha088', 'alpha092', 'alpha094', 'alpha095', 'alpha096', 'alpha098', 'alpha099', 'alpha101']

    selected_features = []
       
    best_gain = trail(base_features)
    for k in new_features:
        current_features = base_features + selected_features + [k]
        print(f'------------{k}-------------')
        current_gain = trail(current_features)
        if current_gain > best_gain:
            best_gain = current_gain
            selected_features.append(k)
            print(f'new gain: {round(best_gain, 4)}, features: {selected_features}')

