
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import sys
sys.path.append('..')
from backend.dataconfig import data_encoding
current_encoding = data_encoding['stock']
yf_str_keys = ['industry', 'country', 'exchange', 'sector', 'previousConsensus']

new_df = pd.read_csv('../data/stock2.csv')
new_df = new_df.drop('ticker', axis = 1)
new_df = new_df.drop('rating', axis = 1)


yf_str_keys = ['industry', 'country', 'exchange', 'sector']

X = new_df.drop('analystConsensus', axis=1)
y = new_df['analystConsensus']
        
X_encoded = pd.get_dummies(X, columns = yf_str_keys + ['previousConsensus'])

features = X_encoded.columns.tolist()
for key in yf_str_keys + ['previousConsensus']:
    index = 0
    for i in range(len(features)):
        if key in features[i]:
            features[i] = key + '_' + current_encoding[key][index]
            index += 1
X_encoded = X_encoded[features]

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

from sklearn.metrics import confusion_matrix

model = LGBMClassifier(**{
'n_estimators': 150, 'learning_rate': 0.05933959908326045, 'max_depth': 3, 'lambda_l1': 8.239690865892566, 'lambda_l2': 5.099588170103722, 'num_leaves': 180, 'feature_fraction': 0.2368029257738345, 'bagging_fraction': 0.8723237313172197, 'bagging_freq': 9, 'min_child_samples': 95},
                       class_weight = 'balanced',
                       random_state=42,
                       verbose = -1)

current_features = X_train.columns
drop_features = []
model.fit(X_train.values, y_train)

y_pred = model.predict(X_test.values)
test_accuracy = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)
print(conf_mat)

y_pred = model.predict(X_train.values)
train_accuracy = accuracy_score(y_train, y_pred)
print("Accuracy(test / train): ", test_accuracy, train_accuracy)

sorted_features = [(model.feature_name_[i], model.feature_importances_[i]) for i in range(len(model.feature_importances_))]
sorted_features = sorted(sorted_features, key = lambda x: -x[1])
for k in sorted_features[:20]:
    # pass
    print(k)

from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

classes = model.classes_
num_classes = len(classes)
y_pred = model.predict_proba(X_test.values)
y_test_bin = label_binarize(y_test, classes=classes)

accuracys = []
for i in range(num_classes):
    accuracy = conf_mat[i, i] / conf_mat[i].sum()
    accuracys.append(accuracy)
    print(f'Accuracy on {classes[i]}: {accuracy}')
print('Average accuracy: ', np.mean(accuracys))

auc_list = []

'''
for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred[:, i])
    roc_auc = auc(fpr, tpr)
    auc_list.append(roc_auc)
'''

for i in range(num_classes):
    for j in range(i+1, num_classes):
        y_i = y_pred[:, i]
        y_j = y_pred[:, j]
        true_i = y_test_bin[:, i]
        true_j = y_test_bin[:, j]
        
        y_ij = np.column_stack((y_i, y_j))
        true_ij = np.column_stack((true_i, true_j))
        
        # 计算AUC并添加到列表中
        auc = roc_auc_score(true_ij, y_ij)
        print(f'AUC between {classes[i]} and {classes[j]}: ', auc)
        auc_list.append(auc)
# 计算平均AUC

mean_auc = np.mean(auc_list)
print("Average AUC:", mean_auc)

from tabnanny import verbose
import optuna  # pip install optuna
from sklearn.metrics import log_loss, precision_score, f1_score, accuracy_score
from sklearn.model_selection import KFold
from optuna.integration import LightGBMPruningCallback
from lightgbm import LGBMClassifier
random_state = 10

def objective(trial, X_train, y_train, X_test, y_test):
    # 后面填充
    param_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=20),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.2),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.1, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 9),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
    }

    model = LGBMClassifier(**param_grid, verbose=-1, class_weight = 'balanced', random_state=42)
    model.fit(X_train.values, y_train)

    y_pred = model.predict(X_test.values)
    test_accuracy = accuracy_score(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred)
    classes = model.classes_
    num_classes = len(classes)

    y_pred = model.predict_proba(X_test.values)

    accuracys = []
    for i in range(num_classes):
        accuracy = conf_mat[i, i] / conf_mat[i].sum()
        accuracys.append(accuracy)

    mean_accuracy = (accuracys[0] + accuracys[2] + accuracys[1] * 2) / 4
    return mean_accuracy

study = optuna.create_study(direction="maximize", study_name="LGBM Classifier")
func = lambda trial: objective(trial, X_train, y_train, X_test, y_test)
study.optimize(func, n_trials=3000)

print(study.best_params)


model = LGBMClassifier(**study.best_params, verbose=-1, class_weight = 'balanced', random_state=42)
model.fit(X_train.values, y_train)

y_pred = model.predict(X_test.values)
test_accuracy = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)
print(conf_mat)
print("Accuracy(test / train): ", test_accuracy, train_accuracy)

accuracys = []
for i in range(num_classes):
    accuracy = conf_mat[i, i] / conf_mat[i].sum()
    accuracys.append(accuracy)

print(accuracys)