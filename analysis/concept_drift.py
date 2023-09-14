import pandas as pd
import time
import requests
import json
import numpy as np
import os
import datetime
tickers = open('../data/tickers.json', 'r').read().split(',')
tickers = [k for k in tickers if os.path.exists(f'../data/new_price/{k}.csv')]

import qlib
from qlib.contrib.data.handler import Alpha158

data_handler_config = {
    "start_time": "2020-01-01",
    "end_time": "2023-08-21",
    "fit_start_time": "2020-01-01",
    "fit_end_time": "2023-07-17",
    "instruments": tickers,
    "freq": 'day',
}

qlib.init(provider_uri={'day': '~/qlib_data/us/' })
h = Alpha158(**data_handler_config)

alpha158_df = h.fetch(col_set="feature")
alpha158_df.index.names = ['date', 'asset']

closes = []
for ticker in tickers:
    df = pd.read_csv(f'../data/new_price/{ticker}.csv')
    df.rename(columns={'close': ticker}, inplace=True)
    df.set_index('date', inplace=True)
    closes.append(df[ticker])
close_prices = pd.concat(closes, axis=1)
opens = []
for ticker in tickers:
    df = pd.read_csv(f'../data/new_price/{ticker}.csv')
    df.rename(columns={'open': ticker}, inplace=True)
    df.set_index('date', inplace=True)
    opens.append(df[ticker])
open_prices = pd.concat(opens, axis=1)

new_df = []

for i, ticker in enumerate(tickers):
    if i % 100 == 0:
        print(i, ticker)
        
    json1 = json.load(open(f'../data/alphav/{ticker}_income.json', 'r'))
    json2 = json.load(open(f'../data/alphav/{ticker}_balance.json', 'r'))
    json3 = json.load(open(f'../data/alphav/{ticker}_cashflow.json', 'r'))
    json4 = json.load(open(f'../data/alphav/{ticker}_earnings.json', 'r'))
    jsons = [json1, json2, json3]

    error_flag = False
    for j in jsons:
        if 'quarterlyReports' not in j:
            error_flag = True
            break
    
    if error_flag:
        print(i, ticker, 'no valid data')
        continue

    dfs = [pd.DataFrame(x['quarterlyReports']) for x in jsons]
    dfs.append(pd.DataFrame(json4['quarterlyEarnings']))
    lens = [len(x) for x in dfs]
    min_len = min(lens)

    if min_len < 8:
        print(i, ticker, 'no enough length')
        continue
    
    dfs = [x[:min_len] for x in dfs]
    df = dfs[0]
    for another in dfs[1:]:
        for k in another.columns:
            if k not in df.columns:
                df[k] = another[k]   

    for k in ['costofGoodsAndServicesSold', 'currentNetReceivables']:
        df = df.drop(k, axis = 1)

    df.insert(1, 'symbol', ticker)
    df = df.rename({
        'fiscalDateEnding': 'date',
    }, axis=1)
    df = df.set_index(['date', 'symbol'])

    new_df.append(df)

    df = pd.concat(new_df, axis = 0)

for k in df.columns:
    if k == 'reportedDate':
        continue
    new_series = pd.to_numeric(df[k], errors='coerce')
    if new_series.isna().sum() > len(df) * 0.5:
        df = df.drop(k, axis = 1)
    else:
        df[k] = new_series

df1 = df.sort_index()
df2 = alpha158_df.sort_index()

selected_alpha = ['KLEN', 'STD60', 'MIN5', 'LOW0',
       'VSTD60', 'MIN60', 'WVMA30', 'MIN10', 'STD5', 'WVMA60', 'CORD60',
       'MIN30', 'CORR60', 'CORR30', 'KLOW', 'QTLD5', 'WVMA20', 'MIN20',
       'VSTD10', 'CORR20', 'CORD30', 'VSTD5', 'QTLD10', 'KUP', 'RESI20', 'MA5',
       'CORR10', 'CORD20', 'HIGH0', 'RESI10', 'BETA5', 'CORD10', 'KMID',
       'OPEN0', 'ROC5', 'CORR5']

s1 = df1.index.to_list()
ticker_date_index = {}

for ticker in tickers:
    s2 = [i for i in s1 if i[1] == ticker]
    if len(s2) == 0:
        continue
    reported_dates = [df1.loc[i]['reportedDate'].values[0] for i in s2]
    reported_dates = [pd.to_datetime(i) for i in reported_dates]
    ticker_date_index[ticker] = [reported_dates, s2]

valid_tickers = [k for k in tickers if k in ticker_date_index]
df1 = df1.drop('reportedDate', axis = 1)

def get_items_in_range(all_start_date, all_end_date):
    all_start_date = pd.to_datetime(all_start_date)
    all_end_date = pd.to_datetime(all_end_date)
    items = []
    while all_start_date < all_end_date:
        current_date = all_start_date
        while True:
            start_date = current_date.strftime('%Y-%m-%d')
            if not start_date in open_prices.index:
                current_date += datetime.timedelta(days=1)
                continue
            end_date = (current_date + datetime.timedelta(days=7)).strftime('%Y-%m-%d')
            if not end_date in open_prices.index:
                current_date += datetime.timedelta(days=1)
                continue
            break

        print(start_date, end_date)

        start_date_ = pd.to_datetime(start_date)
        for k in valid_tickers:
            try:
                i = np.searchsorted(ticker_date_index[k][0], start_date_) - 1
                idx = ticker_date_index[k][1][i]
                values = df1.loc[idx].values[0]
                item = {}
                for i, j in enumerate(df1.columns):
                    item[j] = values[i]
                alphas = df2.loc[start_date, k][selected_alpha].to_dict()
                for j in alphas:
                    item[j] = alphas[j]
                item['current_price'] = open_prices.loc[start_date][k]
                item['new_price'] = close_prices.loc[end_date][k]
                item['rating'] = item['new_price'] / item['current_price']
                if item['rating'] > 1.05:
                    item['label'] = 'increase'
                elif item['rating'] < 0.95:
                    item['label'] = 'decrease'
                else:
                    item['label'] = 'stable'
                item['ticker'] = k
                items.append(item)
            except:
                print('error on ', k)
                
        all_start_date += datetime.timedelta(days=30)
    return items

items = get_items_in_range('2021-08-24', '2023-07-20')
old_df = pd.DataFrame(items)
new_df = old_df.drop('rating', axis = 1)
new_df = new_df.drop('ticker', axis = 1)
new_df = new_df.drop('new_price', axis = 1)
new_df = new_df.drop('current_price', axis = 1)

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = new_df.drop('label', axis=1)
y = new_df['label']

for category in []:
    category_counts = X[category].value_counts()
    low_count_categories = category_counts[category_counts < 50].index
    X.loc[X[category].isin(low_count_categories), category] = 'Others'

X_encoded = pd.get_dummies(X, columns = [])
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

from sklearn.metrics import confusion_matrix

model = LGBMClassifier(**{'n_estimators': 330,
 'learning_rate': 0.14991634335549622,
 'max_depth': 7,
 'lambda_l1': 6.174980700757857,
 'lambda_l2': 5.849787573864054,
 'feature_fraction': 0.516828541523375,
 'bagging_fraction': 0.7252354560854817,
 'bagging_freq': 3,
 'min_child_samples': 48},
 class_weight = 'balanced',
verbose = -1)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)
print(conf_mat)

y_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_pred)
print("Accuracy(test / train): ", test_accuracy, train_accuracy)

for i in range(model.n_classes_):
    accuracy = conf_mat[i, i] / conf_mat[i].sum()
    print(f'Accuracy on {model.classes_[i]}: {accuracy}')

new_items = get_items_in_range('2023-08-14', '2023-08-30')
test_df0 = pd.DataFrame(new_items)
test_df = test_df0.drop('ticker', axis = 1)
test_df = test_df.drop('rating', axis = 1)
test_df = test_df.drop('new_price', axis = 1)
test_df = test_df.drop('current_price', axis = 1)

X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

y_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)
print(conf_mat)
print("Accuracy(test): ", test_accuracy)

for i in range(model.n_classes_):
    accuracy = conf_mat[i, i] / conf_mat[i].sum()
    print(f'Accuracy on {model.classes_[i]}: {accuracy}')

y_pred_prob = model.predict_proba(X_test)
idx = np.zeros(len(X_test)) > 0
for i in np.argsort(y_pred_prob[:, 1])[::-1][:20]:
    idx[i] = True

print(test_df0[idx]['rating'].mean())
