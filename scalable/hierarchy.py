import numpy as np
import sys
sys.path.append('.')
sys.path.append('..')
from scalable.model_utils import ModelUtil
from scalable.algorithm.model_reduction import Extractor
from scalable.anomaly import LRAnomalyDetection
from backend.dataconfig import data_encoding
import pickle

def generate_model_paths(dataset, model_name):
    modelutil = ModelUtil(data_name = dataset, model_name = model_name)
    model = modelutil.model
    pickle.dump(model.paths, open('path.pkl', 'wb'))
    X, y = modelutil.get_rule_matrix()
    y = y.astype(int)
    #res = LRAnomalyDetection(X[:10000], y[:10000])
    res = LRAnomalyDetection(X, y)
    score = res.score(X, y)

    feature_importance = []

    for i in range(len(model.features)):
        j = modelutil.feature_pos[i][1]
        feature_importance.append((model.features[j], res.w[i]))
    feature_importance = sorted(feature_importance, key = lambda x: -x[1])

    for i, val in enumerate(score):
        model.paths[i]['score'] = val
        model.paths[i]['cost'] = val
        model.paths[i]['feature_vector'] = X[i] * np.abs(res.w)
        model.paths[i]['X'] = X[i]
        model.paths[i]['y'] = y[i]
    print('average score', np.mean(score))
    return modelutil

def param_xi_search(model, min_value, max_value, step, n = 80, class_weight=None):
    paths = model.paths
    alpha = model.parameters['n_estimators'] * n / len(paths)
    best_fidelity_test = 0
    best_xi = min_value
    xi = min_value
    pred_train = model.clf.predict(model.X_train)
    pred_test = model.clf.predict(model.X_test)
    ex = Extractor(paths, model.X_train, pred_train)
    it = 0
    last_update_it = 0
    while xi <= max_value:
        w, _, _, _ = ex.extract(n, xi * alpha, 0, class_weight=class_weight)
        fidelity_test = ex.evaluate(w, model.X_test, pred_test)
        print('xi', xi, fidelity_test)
        if fidelity_test > best_fidelity_test:
            best_fidelity_test = fidelity_test
            best_xi = xi
            last_update_it = it
        elif it - last_update_it > 4:
            break
        xi += step
        it += 1
    return best_xi

def param_lambda_search(model, min_value, max_value, step, xi = 0.5, n = 80, class_weight=None):
    paths = model.paths
    alpha = model.parameters['n_estimators'] * n / len(paths)
    lambda_ = min_value
    pred_train = model.clf.predict(model.X_train)
    pred_test = model.clf.predict(model.X_test)
    ex = Extractor(paths, model.X_train, pred_train)

    w, _, _, _ = ex.extract(n, xi * alpha, lambda_, class_weight=class_weight)
    best_fidelity = ex.evaluate(w, model.X_test, pred_test)
    best_lambda = min_value
    it = 0
    last_update_it = 0
    lambda_ += step

    while lambda_ <= max_value:
        w, _, _, _ = ex.extract(n, xi * alpha, lambda_)
        fidelity = ex.evaluate(w, model.X_test, pred_test)
        print('lambda', lambda_, fidelity)
        if fidelity > best_fidelity:
            best_fidelity = fidelity
            best_lambda = lambda_
            last_update_it = it
        elif fidelity >= best_fidelity * 0.99:
            best_lambda = lambda_
            last_update_it = it
        elif it - last_update_it > 4:
            break
        lambda_ += step
        it += 1
    return best_lambda

def generate_hierarchy(dataset, model_name, n = 80, xi = -1, lambda_ = -1, class_weight=None):
    modelutil = generate_model_paths(dataset, model_name)
    model = modelutil.model
    paths = model.paths
    n_fold = 4
    if xi == -1:
        xis = []
        for k_fold in range(1):#range(n_fold):
            xi = param_xi_search(model, 0.025, 1, 0.025, n)
            xis.append(xi)
        xi = np.mean(xis)

    if lambda_ == -1:
        lambda_ = param_lambda_search(model, 0, .5, .01, xi, n)
    print('The best parameter: xi', xi, 'lambda', lambda_)

    alpha = model.parameters['n_estimators'] * n / len(paths)
    ex = Extractor(paths, model.X_train, model.clf.predict(model.X_train))#, greedy = True)
    w, _, _, _ = ex.extract(n, xi * alpha, lambda_, class_weight=class_weight)
    [idx] = np.nonzero(w)
    accuracy_test = ex.evaluate(w, model.X_test, model.y_test)
    fidelity_test = ex.evaluate(w, model.X_test, model.clf.predict(model.X_test))

    model.accuracy = accuracy_test
    model.fidelity = fidelity_test
    level_info = {
        'fidelity_test': fidelity_test,
        'accuracy_test': accuracy_test,
        'xi': xi,
        'lambda_': lambda_,
    }
    avg_score = np.array([paths[i]['score'] for i in idx]).mean()
    print('avg_score', round(avg_score, 4), 'fidelity', round(fidelity_test, 4), 'accuracy', round(accuracy_test, 4))

    return model, paths, level_info, idx

str_keys = ['industry', 'country', 'exchange', 'sector', 'previousConsensus']

def post_process(dataset, model_name, model, paths, level_info, selected_idx):
    idx = selected_idx
    new_feature = {}
    features = [feature for feature in model.data_table.columns if feature != model.target]
    for index, feature in enumerate(model.features):
        is_cat = False
        for delimiter in [' - ', '_']:
            if delimiter not in feature:
                continue
            name, _ = feature.split(delimiter)
            if len([k for k in model.features if name in k]) == 1:
                continue
            if name not in new_feature:
                new_feature[name] = {}
            if feature not in new_feature[name]:
                new_feature[name][feature] = index
            is_cat = True
            break
        if not is_cat:
            new_feature[feature] = index

    output_data = {}
    current_encoding = data_encoding.get(dataset, {})

    if dataset == 'german':
        features = []
        feature_index = {}
        feature_type = {}
        for key in new_feature:
            if type(new_feature[key]) is int:
                i = new_feature[key][0]
                if key in current_encoding:
                    min_value = min(model.data_table[key].values)
                    max_value = max(model.data_table[key].values)
                    unique_values = np.unique(model.data_table[key].values) - min_value
                    sorted(unique_values)
                    features.append({
                        "name": key,
                        "range": [0, len(unique_values)],
                        "values": unique_values.tolist(),
                        "min": min_value,
                        "importance": model.clf.feature_importances_[i],
                        "dtype": "category",
                    })
                    feature_type[i] = "category"
                else:
                    values = model.data_table[key].values
                    values.sort()
                    n = len(values)
                    qmin, qmax = values[0], values[-1]
                    q5, q25, q50, q75, q95 = values[n * 5 // 100], values[n * 25 // 100], values[n * 50 // 100], values[n * 75 // 100], values[n * 95 // 100]
                    features.append({
                        "name": key,
                        "quantile": { "5": q5, "25": q25, "50": q50, "75": q75, "95": q95 },
                        "range": [qmin, qmax],
                        "importance": model.clf.feature_importances_[i],
                        "dtype": "number",
                    })
                feature_type[i] = "number"
                feature_index[i] = [len(features) - 1, 0]
            else:
                features.append({
                    "name": key,
                    "range": [0, len(new_feature[key])],
                    "importance": sum([model.clf.feature_importances_[i] for i in new_feature[key] if i != -1]),
                    "dtype": "category",
                })

                for index, i in enumerate(new_feature[key]):
                    if i != -1:
                        feature_index[i] = [len(features) - 1, index]
                        feature_type[i] = "category"

        #for i, feature in enumerate(features):
        #    print(feature, feature_type[i])

        for path in paths:
            new_range = {}
            for index in path['range']:
                i, j = feature_index[index]
                key = features[i]['name']
                if features[i]['dtype'] == 'number' or type(new_feature[key]) is int:
                    r = path['range'][index]
                    key = features[i]['name']
                    if model.data_table[key].dtype == np.int64:
                        if r[0] < 0:
                            r[0] = 0
                        if r[1] > features[i]['range'][1]:
                            r[1] = features[i]['range'][1]
                        if features[index]['range'][0] > 0:
                            if r[0] < int(r[0]) + 1e-7:
                                r[0] = int(r[0]) - 1
                            else:
                                r[0] = int(r[0])
                            if r[1] > int(r[1]) + 1e-7:
                                r[1] = int(r[1])
                        else:
                            if r[0] > int(r[0]) + 1e-7:
                                r[0] = int(r[0]) + 0.5
                            if r[1] > int(r[1]) + 1e-7:
                                r[1] = int(r[1]) + 0.5
                    new_range[i] = r
                else:
                    if 'min' in features[i]:
                        new_range[i] = [0] * features[i]['range'][1]
                        min_value = features[i]['min']
                        r = path['range'][index]
                        for j in range(features[i]['range'][1]):
                            if j + min_value >= r[0] and j + min_value <= r[1]:
                                new_range[i][j] = 1
                    else:
                        if i not in new_range:
                            new_range[i] = [0] * features[i]['range'][1]
                            if path['range'][index][0] <= 1 and 1 <= path['range'][index][1]:
                                new_range[i][j] = 1
                            else:
                                for k in range(len(new_range[i])):
                                    if k != j:
                                        new_range[i][k] = 1
                                    new_range[i][j] = 0
            path['range'] = new_range
            path['represent'] = False

        for i in idx:
            paths[i]['represent'] = True

        output_data = {
            'paths': paths,
            'features': features,
            'selected': [paths[i]['name'] for i in idx],
            'model_info': {
                'accuracy': model.accuracy,
                'info': level_info,
                'num_of_rules': len(paths),
                'dataset': dataset,
                'model': model_name,
            }
        }
    else:
        features = []
        feature_index = {}
        feature_type = {}
        for key in new_feature:
            if type(new_feature[key]) is int:
                i = new_feature[key]
                if key in current_encoding and len(current_encoding[key]) > 0:
                    min_value = min(model.data_table[key].values)
                    max_value = max(model.data_table[key].values)
                    features.append({
                        "name": key,
                        "range": [0, len(current_encoding[key])],
                        "values": current_encoding[key],
                        "min": min_value,
                        "importance": model.clf.feature_importances_[i],
                        "dtype": "category",
                    })
                    feature_type[i] = "category"
                    feature_index[i] = [len(features) - 1, 0]
                else:
                    x = model.data_table[key].values
                    x = x[~np.isnan(x)]
                    x.sort()
                    n = len(x)
                    if n == 0:
                        qmin = 0
                        qmax = 1
                        q5 = q25 = q50 = q75 = q95 = 0
                    else:
                        qmin, qmax = x[0], x[-1]
                        q5, q25, q50, q75, q95 = x[n * 5 // 100], x[n * 25 // 100], x[n * 50 // 100], x[n * 75 // 100], x[n * 95 // 100]
                    features.append({
                        "name": key,
                        "quantile": { "5": q5, "25": q25, "50": q50, "75": q75, "95": q95 },
                        "range": [qmin, qmax],
                        "importance": model.clf.feature_importances_[i],
                        "dtype": "number",
                    })
                    feature_type[i] = "number"
                    feature_index[i] = [len(features) - 1, 0]
            else:
                importance = 0
                for i in new_feature[key]:
                    if new_feature[key][i] != -1 and new_feature[key][i] < len(model.clf.feature_importances_):
                        importance += model.clf.feature_importances_[new_feature[key][i]]
                features.append({
                    "name": key,
                    "range": [0, max(2, len(new_feature[key]))],
                    "importance": importance,
                    "dtype": "category",
                })

                for index, j in enumerate(new_feature[key]):
                    i = new_feature[key][j]
                    if i != -1:
                        feature_index[i] = [len(features) - 1, index]
                        feature_type[i] = "category"

        for path in paths:
            new_range = {}
            for index in path['range']:
                i, j = feature_index[index]
                if features[i]['dtype'] == 'number':
                    r = path['range'][index]
                    key = features[i]['name']
                    if model.data_table[key].dtype == np.int64:
                        if r[0] < 0:
                            r[0] = 0
                        if r[1] > features[i]['range'][1]:
                            r[1] = features[i]['range'][1]
                        if features[index]['range'][0] > 0:
                            if r[0] < int(r[0]) + 1e-7:
                                r[0] = int(r[0]) - 1
                            else:
                                r[0] = int(r[0])
                            if r[1] > int(r[1]) + 1e-7:
                                r[1] = int(r[1])
                        else:
                            if r[0] > int(r[0]) + 1e-7:
                                r[0] = int(r[0]) + 0.5
                            if r[1] > int(r[1]) + 1e-7:
                                r[1] = int(r[1]) + 0.5
                    new_range[i] = r
                else:
                    key = features[i]['name']
                    if 'min' in features[i] and key in current_encoding:
                        new_range[i] = [0] * features[i]['range'][1]
                        min_value = features[i]['min']
                        r = path['range'][index]
                        for j in range(features[i]['range'][1]):
                            if j + min_value >= r[0] and j + min_value <= r[1]:
                                new_range[i][j] = 1
                    else:
                        if i not in new_range:
                            new_range[i] = [0] * features[i]['range'][1]
                            if path['range'][index][0] <= 1 and 1 <= path['range'][index][1]:
                                new_range[i][j] = 1
                            else:
                                for k in range(len(new_range[i])):
                                    if k != j:
                                        new_range[i][k] = 1
                                    new_range[i][j] = 0
            path['range'] = new_range
            path['represent'] = False

        for i in idx:
            paths[i]['represent'] = True

        output_data = {
            'paths': paths,
            'features': features,
            'selected': [paths[i]['name'] for i in idx],
            'model_info': {
                'accuracy': model.accuracy,
                'info': level_info,
                'num_of_rules': len(paths),
                'dataset': dataset,
                'model': model_name,
            }
        }
    return output_data

if __name__ == '__main__':
    '''
    dataset = 'credit4'
    model_name = 'random forest'
    model, paths, level_info, idx = generate_hierarchy(dataset, model_name, n = 80, xi=0.2, lambda_=0.2)
    data = post_process(dataset, model_name, model, paths, level_info, idx)

    dataset = 'bankruptcy'
    model_name = 'lightgbm'
    model, paths, level_info, idx = generate_hierarchy(dataset, model_name, n = 80, xi=0.1, lambda_=0.2)
    data = post_process(dataset, model_name, model, paths, level_info, idx)

    '''

    '''
    dataset = 'stock'
    model_name = 'lightgbm'
    model, paths, level_info, idx = generate_hierarchy(dataset, model_name, n = 80, xi=0.05, lambda_=0.1)
    data = post_process(dataset, model_name, model, paths, level_info, idx)
    '''
    dataset = 'stock_step3'
    model_name = 'lightgbm'
    model, paths, level_info, idx = generate_hierarchy(dataset, model_name, n = 80, xi=0.1, lambda_=0.2)
    data = post_process(dataset, model_name, model, paths, level_info, idx)

    import pickle
    pickle.dump(data, open('./output/dump/stock_step3.pkl', 'wb'))
