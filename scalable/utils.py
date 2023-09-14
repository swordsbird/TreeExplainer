import numpy as np
from scalable.model_utils import ModelUtil
from scalable import LRAnomalyDetection, Extractor
from backend.dataconfig import data_encoding
import pickle

def generate_model_paths(dataset, model_name):
    modelutil = ModelUtil(data_name = dataset, model_name = model_name)
    model = modelutil.model
    X, y = modelutil.get_rule_matrix()
    y = y.astype(int)
    res = LRAnomalyDetection(X[:1500], y[:1500])
    score = res.score(X, y)

    feature_importance = []

    for i in range(len(model.data_table.columns)):
        j = modelutil.feature_pos[i][1]
        feature_importance.append((model.data_table.columns[j], res.w[i]))
    feature_importance = sorted(feature_importance, key = lambda x: -x[1])
    #for it in feature_importance:
    #    print(it)

    print('score', len(score))
    for i, val in enumerate(score):
        model.paths[i]['score'] = val
        model.paths[i]['cost'] = val
        model.paths[i]['feature_vector'] = X[i] * np.abs(res.w)
        model.paths[i]['X'] = X[i]
        model.paths[i]['y'] = y[i]
    print('average score', np.mean(score))
    return modelutil