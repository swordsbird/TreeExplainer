import sys
sys.path.append('.')
sys.path.append('..')

import os
import json
import pandas as pd
from scalable.utils import get_trained_model
from surrogate_rule import forest_info
from surrogate_rule import tree_node_info

import numpy as np

filter_threshold = {
    "support": 5,
    # "fidelity": .85,
    "fidelity": 0.8,
    # "num_feat": 4,
    "num_feat": 6,
    "num_bin": 3,
}
num_bin = filter_threshold['num_bin']
random_state = 10

def extract_rules_from_RF(model):
    X = model.X_train
    y_gt = model.y_train.tolist()
    y_pred = model.clf.predict(model.X_train).tolist()
    y_gt = np.array(y_gt)
    y_pred = np.array(y_pred)
    columns = model.features
    if len(columns) < X.shape[1]:
        for i in range(X.shape[1] - len(columns)):
            columns.append(f'new feature{i}')
    else:
        columns = columns[:X.shape[1]]
    df = pd.DataFrame(data=X, columns = columns)
    n_cls = len(model.output_labels)
    real_min = model.X_train.min(axis = 0)
    real_max = model.X_train.max(axis = 0)
    # train surrogate
    surrogate_obj = tree_node_info.tree_node_info()

    surrogate_obj.initialize(X=X, y=y_gt,
                             y_pred=y_pred, debug_class=-1,
                             attrs=columns, filter_threshold=filter_threshold,
                             n_cls=n_cls,
                             num_bin=num_bin, verbose=False
    ).train_surrogate_random_forest().tree_pruning()

    forest_obj = tree_node_info.forest()
    forest_obj.initialize(
        trees=surrogate_obj.tree_list, cate_X=surrogate_obj.cate_X,
        y=surrogate_obj.y, y_pred=surrogate_obj.y_pred, attrs=columns, num_bin=num_bin,
        real_percentiles=surrogate_obj.real_percentiles,
        real_min=surrogate_obj.real_min, real_max=surrogate_obj.real_max,
    ).construct_tree().extract_rules()

    forest = forest_info.Forest()

    forest.initialize(forest_obj.tree_node_dict, real_min, real_max, surrogate_obj.percentile_info,
        df, y_pred, y_gt,
        forest_obj.rule_lists,
        model.output_labels, 2)
    forest.initialize_rule_match_table()
    forest.initilized_rule_overlapping()
    try:
        res = forest.find_the_min_set()
    except:
        return False
    lattice = forest.get_lattice_structure(res['rules'])

    max_feat = 0
    min_feat = 111
    avg_feat = 0.0
    for rule in res['rules']:
        if (len(rule['rules']) > max_feat):
            max_feat = len(rule['rules'])
        if (len(rule['rules']) < min_feat):
            min_feat = len(rule['rules'])
        avg_feat += len(rule['rules'])
    return len(res['rules']), res['coverage'], res['correct_coverage']

config = json.loads(open("config.json", "r").read())
fname = "result/sure.txt"
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
            X_train = model.X_train
            y_train = model.y_train
            X_test = model.X_test
            y_test = model.y_test
            clf = model.clf
            n_selected, coverage, fidelity  = extract_rules_from_RF(model)
            fidelity = round(fidelity, 4)

            ret = {
                'dataset': model.data_name,
                'model': model.model_name,
                'original_accuracy': original_accuracy,
                'surrogate_accuracy': 0,
                'fidelity': fidelity,
                'actual_rules': n_selected,
                'expected_rules': n,
            }

            print(f'fidelity: {round(fidelity, 4)}')
            f = open(fname, 'a')
            f.write(json.dumps(ret) + '\n')
            f.close()
