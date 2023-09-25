import pandas as pd
import time
import requests
import json
import numpy as np
import os
import datetime


tickers = open('../data/case2_stock/tickers.json', 'r').read().split(',')
tickers = [k for k in tickers if os.path.exists(f'../data/case2_stock/price/{k}.csv')]
tickers = [k for k in tickers if os.path.exists(f'../data/case2_stock/info/{k}.json')]
extended_df = pd.read_csv('../data/case2_stock/stock_info.csv')
extended_df = extended_df.set_index('ticker')
sp500_index = pd.read_csv('../data/case2_stock/s&p500.csv')
sp500_index['date'] = [x.strftime('%Y-%m-%d') for x in pd.to_datetime(sp500_index['date'])]
sp500_index = sp500_index.set_index('date')
extended_old_keys = ['changePercent', 'price',
       'totalValue', 'yearlyGain', 'payoutRatio', 'beta', 'marketCap',
       'priceToSalesTrailing12Months', 'trailingAnnualDividendRate',
       'profitMargins', 'sharesPercentSharesOut', 'heldPercentInsiders',
       'heldPercentInstitutions', 'shortRatio', 'shortPercentOfFloat',
       'bookValue', 'priceToBook', 'earningsQuarterlyGrowth',
       'netIncomeToCommon', 'trailingEps', 'forwardEps', 'pegRatio',
       '52WeekChange', 'lastDividendValue', 'lastDividendDate', 'totalCash',
       'totalDebt', 'currentRatio', 'totalRevenue', 'debtToEquity',
       'revenuePerShare', 'returnOnAssets', 'returnOnEquity', 'grossProfits',
       'freeCashflow', 'earningsGrowth', 'revenueGrowth', 'grossMargins'
       ]
info_df = []
for ticker in tickers:
       info = { 'ticker': ticker }
       obj = json.load(open(f'../data/case2_stock/info/{ticker}.json', 'r'))
       for k in obj:
              if type(obj[k]) != list:
                     info[k] = obj[k]
       info_df.append(info)
info_df = pd.DataFrame(info_df)
info_df = info_df.set_index('ticker')
info_df['totalShares'] = info_df['marketCap'] / info_df['currentPrice']

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
    df = pd.read_csv(f'../data/case2_stock/price/{ticker}.csv')
    df.rename(columns={'close': ticker}, inplace=True)
    df.set_index('date', inplace=True)
    closes.append(df[ticker])
close_prices = pd.concat(closes, axis=1)
opens = []
for ticker in tickers:
    df = pd.read_csv(f'../data/case2_stock/price/{ticker}.csv')
    df.rename(columns={'open': ticker}, inplace=True)
    df.set_index('date', inplace=True)
    opens.append(df[ticker])
open_prices = pd.concat(opens, axis=1)

new_df = []

for i, ticker in enumerate(tickers):
    if i % 100 == 0:
        print(i, ticker)
        
    json1 = json.load(open(f'../data/case2_stock/report/{ticker}_income.json', 'r'))
    json2 = json.load(open(f'../data/case2_stock/report/{ticker}_balance.json', 'r'))
    json3 = json.load(open(f'../data/case2_stock/report/{ticker}_cashflow.json', 'r'))
    json4 = json.load(open(f'../data/case2_stock/report/{ticker}_earnings.json', 'r'))
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

    for k in ['costofGoodsAndServicesSold', 'currentNetReceivables', 'propertyPlantEquipment']:
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
    
dropped_index = []
for i in df.index:
    values = [x for x in df.loc[i].values[0] if type(x) != str and not np.isnan(x)]
    if len(values) == 0 or np.max(values) > 1e17 or np.min(values) < -1e17:
        dropped_index.append(i)
df = df.drop(dropped_index)

df1 = df.sort_index()
df2 = alpha158_df.sort_index()

current_time = pd.to_datetime('2023-09-01')
to_be_removed = []
for ticker in tickers:
    index = df1.loc[df1.index.get_level_values('symbol') == ticker].index
    if len(index) < 4:
        to_be_removed.append(ticker)
        continue
    index = [pd.to_datetime(x[0]) for x in index]
    index.append(current_time)
    checkFlag = False
    index = index[-12:]
    for i in range(len(index) - 1):
        if index[i + 1] > index[i] + datetime.timedelta(days=200):
            to_be_removed.append(ticker)
            break
tickers = [x for x in tickers if x not in to_be_removed]

remove_features = [
    'cashAndCashEquivalentsAtCarryingValue',
    'cashAndShortTermInvestments',
    'intangibleAssets',
    'intangibleAssetsExcludingGoodwill',
    'incomeBeforeTax',
    'longTermInvestments',
    'shortTermInvestments',
    'otherCurrentAssets',
    'otherNonCurrentAssets',
    'totalNonCurrentLiabilities',
    'depreciationDepletionAndAmortization',
    'cashflowFromFinancing',
    'paymentsForRepurchaseOfCommonStock',
    'capitalExpenditures',
    'capitalLeaseObligations',
    'currentLongTermDebt',
    'shortLongTermDebtTotal',
    'otherCurrentLiabilities',
    'otherNonCurrentLiabilities',
    'dividendPayoutCommonStock',
    'paymentsForRepurchaseOfEquity',
    'netIncomeFromContinuingOperations',
    'sellingGeneralAndAdministrative',
    'nonInterestIncome',
    'commonStockSharesOutstanding',
    'paymentsForOperatingActivities',
    'proceedsFromIssuanceOfLongTermDebtAndCapitalSecuritiesNet',
    'proceedsFromRepaymentsOfShortTermDebt',
]

def calc_feature_ttm(name, last = False):
    name_ttm = name + 'TTM'
    df1[name].fillna(0)
    df1[name_ttm] = np.nan
    for ticker in tickers:
        v0 = df1.loc[df1.index.get_level_values('symbol') == ticker, name].values
        if len(v0) < 4:
            continue
        v1 = np.concatenate((np.zeros(1), v0[:-1]))
        v2 = np.concatenate((np.zeros(2), v0[:-2]))
        v3 = np.concatenate((np.zeros(3), v0[:-3]))
        v = v0 + v1 + v2 + v3
        v[:3] = np.nan
        df1.loc[df1.index.get_level_values('symbol') == ticker, name_ttm] = v
        if last:
            v_1 = np.concatenate((np.array([np.nan]), v[:-1]))
            df1.loc[df1.index.get_level_values('symbol') == ticker, name_ttm + '_1'] = v_1
            v_2 = np.concatenate((np.array([np.nan, np.nan, np.nan, np.nan]), v[:-4]))
            df1.loc[df1.index.get_level_values('symbol') == ticker, name_ttm + '_4'] = v_2


df1['currentRatio'] = df1['totalCurrentAssets'] / df1['totalCurrentLiabilities']
df1['operatingCashflow'].fillna(0)
df1['cashflowFromInvestment'].fillna(0)
df1['cashflowFromFinancing'].fillna(0)
df1['totalCashflow'] = df1['operatingCashflow'] + df1['cashflowFromInvestment'] + df1['cashflowFromFinancing']
calc_feature_ttm('totalCashflow', last = True)
df1['totalCashflowTTM'] /= 4
calc_feature_ttm('netIncome', last = True)
calc_feature_ttm('totalRevenue', last = True)
calc_feature_ttm('totalAssets', last = True)
df1['totalAssetsTTM'] /= 4
calc_feature_ttm('totalLiabilities', last = True)
df1['totalLiabilitiesTTM'] /= 4
calc_feature_ttm('ebit', last = True)
calc_feature_ttm('ebitda', last = True)
calc_feature_ttm('retainedEarnings', last = True)
calc_feature_ttm('totalShareholderEquity', last = True)

df1['totalShares'] = 0
for ticker in tickers:
    index = df1.loc[df1.index.get_level_values('symbol') == ticker].index
    if len(index) < 4:
        continue
    df1.loc[index, 'totalShares'] = info_df['totalShares'][ticker]

df1['factor'] = df1['commonStockSharesOutstanding'].copy()
for ticker in tickers:
    index = df1.loc[df1.index.get_level_values('symbol') == ticker].index
    if len(index) < 4:
        continue
    lastShares = df1.loc[index, 'commonStockSharesOutstanding'][-1]
    df1.loc[index, 'factor'] = lastShares / df1.loc[index, 'factor']

df1['netIncomeParent'] = df1['totalShares'] * df1['reportedEPS'] / df1['factor']
calc_feature_ttm('netIncomeParent', last = True)

selected_alpha = ['KLEN', 'STD60', 'MIN5', 'LOW0',
       'VSTD60', 'MIN60', 'WVMA30', 'MIN10', 'STD5', 'WVMA60', 'CORD60',
       'MIN30', 'CORR60', 'CORR30', 'KLOW', 'QTLD5', 'WVMA20', 'MIN20',
       'VSTD10', 'CORR20', 'CORD30', 'VSTD5', 'QTLD10', 'KUP', 'RESI20', 'MA5',
       'CORR10', 'CORD20', 'HIGH0', 'RESI10', 'BETA5', 'BETA30', 'CORD10', 'KMID',
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

extended_features = ['industry', 'country', 'exchange', 'sector']
extended_columns = []
for k in extended_features:
    extended_columns += [k + '_' + j for j in extended_df[k].unique()]

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
        factor = sp500_index.loc[end_date].value / sp500_index.loc[start_date].value
        for k in valid_tickers:
            try:
                i = np.searchsorted(ticker_date_index[k][0], start_date_)
                if i <= 0:
                    continue
                i -= 1
                idx = ticker_date_index[k][1][i]
                values = df1.loc[idx].values[0]
                item = {}
                for i, j in enumerate(df1.columns):
                    item[j] = values[i]
                alphas = df2.loc[start_date, k].to_dict()
                for j in df2.columns:
                    item[j] = alphas[j]
                item['currentPrice'] = open_prices.loc[start_date][k]
                item['newPrice'] = close_prices.loc[end_date][k]
                item['marketCap'] = item['currentPrice'] * item['totalShares']
                item['ev'] = item['marketCap'] + item['totalLiabilitiesTTM']
                if item['totalShareholderEquityTTM'] == 0 or np.isnan(item['totalShareholderEquityTTM']):
                    continue
                item['reportedEPS'] = item['reportedEPS'] / item['factor']
                item['estimatedEPS'] = item['estimatedEPS'] / item['factor']
                item['peRatioTTM'] = item['marketCap'] / item['netIncomeParentTTM']
                item['pcfRatioTTM'] = item['marketCap'] / item['totalCashflowTTM']
                item['pbRatioTTM'] = item['marketCap'] / item['totalShareholderEquityTTM']
                item['psRatioTTM'] = item['marketCap'] / item['totalRevenueTTM']
                item['evToEbit'] = item['ev'] / item['ebitdaTTM']
                # pe_ratio_ttm / (100*(net_profit_parent_company_ttm_0 - net_profit_parent_company_ttm_4) / net_profit_parent_company_ttm_4)
                item['earningsGrowthTTM'] = (100 * (item['netIncomeParentTTM'] - item['netIncomeParentTTM_4']) / item['netIncomeParentTTM_4'])
                item['pegRatioTTM'] = item['peRatioTTM'] / item['earningsGrowthTTM']
                item['rating'] = item['newPrice'] / item['currentPrice'] / factor
                item['revenuePerShareTTM'] = item['totalRevenueTTM'] / item['totalShares']
                item['returnOnEquityTTM'] = item['netIncomeParentTTM'] / item['totalShareholderEquity']
                item['returnOnAsset'] = item['ebit'] / item['totalAssets']
                item['returnOnAssetTTM'] = item['ebitTTM'] / item['totalAssetsTTM']
                item['returnOnAssetNetTTM'] = item['netIncomeParentTTM'] / item['totalAssetsTTM']
                item['netIncomeToRevenue'] = item['netIncome'] / item['totalRevenue']
                item['netIncomeToRevenueTTM'] = item['netIncomeTTM'] / item['totalRevenueTTM']
                item['ebitToRevenue'] = item['ebit'] / item['totalRevenue']
                item['ebitToRevenueTTM'] = item['ebitTTM'] / item['totalRevenueTTM']
                item['cashflowPerShare'] = item['totalCashflow'] / item['totalShares']
                item['cashflowPerShareTTM'] = item['totalCashflowTTM'] / item['totalShares']
                item['cashflowRatio'] = item['ebitda'] / item['interestExpense']
                item['retainedEarningPerShare'] = item['retainedEarnings'] / item['totalShares']
                item['retainedEarningPerShareTTM'] = item['retainedEarningsTTM'] / item['totalShares']
                item['debtToAssetRatio'] = item['totalLiabilities'] / item['totalAssets']
                item['currentAssetRatio'] = item['totalCurrentAssets'] / item['totalAssets']
                item['revenueGrowthTTM'] = item['totalRevenueTTM'] / item['totalRevenueTTM_4'] - 1
                if item['rating'] > 1.05:
                    item['label'] = 'increase'
                elif item['rating'] < 0.95:
                    item['label'] = 'decrease'
                else:
                    item['label'] = 'stable'
                item['ticker'] = k
                item['date'] = start_date
                for col in extended_columns:
                    i, j = col.split('_')
                    item[col] = extended_df[i][k] == j
                for i in remove_features:
                    del item[i]
                del item['factor']
                items.append(item)
            except:
                print('error on ', k)
                
        all_start_date += datetime.timedelta(days=15)
    return items

items = get_items_in_range('2018-07-01', '2023-08-30')
df = pd.DataFrame(items)
df.rename(columns={
    'changeInCashAndCashEquivalents': 'changeInCash',
    'currentAccountsPayable': 'payable',
    'proceedsFromRepurchaseOfEquity': 'proceedsFromRepurchase',
    'totalShareholderEquity': 'totalEquity',
    }, 
    inplace=True
)

for k in df.columns:
    if 'TTM_1' in k or 'TTM_4' in k:
        df = df.drop(k, axis = 1)

raw_df = df.copy()
extended_features = ['industry', 'country', 'exchange', 'sector']
for k in extended_features:
    raw_df[k] = ''
    for j in df.columns:
        if k in j:
            raw_df.loc[df[j], k] = j.split('_')[1]
            raw_df = raw_df.drop(j, axis = 1)

df.to_csv('stock.csv', index=False)