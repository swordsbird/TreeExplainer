import os
import sys
import pandas as pd

data = pd.read_csv('data/german_detailed.csv')
data2 = pd.read_csv('data/german.csv')

X = data2.values

c = [1, 1, -1, 1, 0, -1, 1, 1, -1, 0, 0, 1, -1, 1, 1, 0, -1, 0, -1, 1, 0]

'''
for i in range(len(X)):
    if X[i, 0] == 1:
        for j in range(len(X)):
            if X[j, 0] == 0:
                flag = 1
                for k in range(1, len(c)):
                    if c[k] == 0 and X[i, k] != X[j, k]:
                        flag = 0
                        continue
                    if c[k] == -1 and X[i, k] < X[j, k]:
                        flag = 0
                        continue
                    if c[k] == 1 and X[i, k] > X[j, k]:
                        flag = 0
                        continue
                if flag == 1:
                    print(i, data.values[i])
                    print(j, data.values[j])
                    data2['credit_risk'][j] = 1
cnt = []
for i in range(len(data)):
    if data['status'][i] != "no checking account":
        continue
    if data['savings'][i] != "unknown/no savings account":
        continue
    if data['credit_risk'][i] != 'Yes':
        continue
    if data['purpose'][i] != 'car (used)':
        continue
    if data['telephone'][i] != 'No':
        continue
    if data['number_credits'][i] != '1':
        continue
    if data['installment_rate'][i] != '>= 35':
        continue
    cnt.append(i)
    data2['credit_risk'][i] = 0
print('cnt', cnt)
'''
data2.to_csv('data/german_new.csv', index=None)