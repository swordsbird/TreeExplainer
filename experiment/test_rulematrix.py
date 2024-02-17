import sys
sys.path.append('.')
sys.path.append('..')

import os
import json
from scalable.utils import get_trained_model
from rulematrix.surrogate import rule_surrogate

import numpy as np
import random

random_state = 10

def train_surrogate(model, sampling_rate=2.0, n_rules=20, **kwargs):
    surrogate = rule_surrogate(model.predict,
                               X_train,
                               sampling_rate=sampling_rate,
                               is_continuous=None,
                               is_categorical=None,
                               is_integer=None,
                               number_of_rules=n_rules,
                               **kwargs)

    test_fidelity = surrogate.score(X_test)
    test_pred = surrogate.student.predict(X_test)
    test_accuracy = np.sum(test_pred == y_test) / len(y_test)
    return surrogate, test_accuracy, test_fidelity

config = json.loads(open("config.json", "r").read())
fname = "result/rulematrix.txt"
existing_result = set()
if os.path.exists(fname):
    f = open(fname, "r").read().split('\n')
    for line in f:
        if len(line) == 0:
            continue
        x = json.loads(line)
        k = f"{x['dataset']}-{x['model']}-{x['expected_rules']}"
        existing_result.add(k)

for n in config["number_of_rules"]:
    for data_name in config["dataset"]:
        for model_name in config["model"]:
            k = f'{data_name}-{model_name}-{n}'
            print(k)
            if k in existing_result:
                print('passed')
                continue
            model = get_trained_model(data_name, model_name)
            original_accuracy, prec, f1 = model.get_performance()
            original_accuracy = round(original_accuracy, 4)
            X_train = np.nan_to_num(model.X_train)
            y_train = model.y_train
            X_test = np.nan_to_num(model.X_test)
            y_test = model.y_test
            clf = model.clf
            sampling_rate = 4
            if len(X_train) > 5000 or data_name == 'abalone':
                sampling_rate = -1
            surrogate, accuracy, fidelity_test = train_surrogate(clf, sampling_rate, n, seed=random_state)
            accuracy = round(accuracy, 4)
            fidelity_test = round(fidelity_test, 4)

            ret = {
                'dataset': data_name,
                'model': model_name,
                'original_accuracy': original_accuracy,
                'surrogate_accuracy': accuracy,
                'fidelity': fidelity_test,
                'actual_rules': surrogate.number_of_rules,
                'expected_rules': n,
            }

            print(f'fidelity: {round(fidelity_test, 4)}')
            f = open(fname, 'a')
            f.write(json.dumps(ret) + '\n')
            f.close()
