import sys
import numpy as np
sys.path.append('.')
sys.path.append('..')

from lib.tree_extractor import path_extractor
from lib.model_reduction import Extractor
from hierarchy_building.hierarchy import generate_hierarchy

ret = []
parameters = {
    'cancer_random forest': (0.1, 0.1),
    'cancer_lightgbm': (0.05, 0.3),
    'german_random forest': (0.25, 0.4),
    'german_lightgbm': (0.1, 0.3),
    'wine_random forest': (0.8, 0.8),
    'wine_lightgbm': (0.05, 0.4),
    'abalone_random forest': (0.1, 0.4),
    'abalone_lightgbm': (0.05, 0.5),
    'bankruptcy_random forest': (0.05, 1.2),
    'bankruptcy_lightgbm': (0.2, 1.8),
}

n = 80
#'cancer', 
for data_name in ['german', 'wine', 'abalone', 'bankruptcy']:
    for model_name in ['random forest', 'lightgbm']:
        xi, lambda_ = parameters.get(f'{data_name}_{model_name}', (-1, -1))
        model, paths, info, _ = generate_hierarchy(
            data_name, model_name, xi = xi, lambda_ = lambda_, n = n
        )
        print(f'DATA: {model.data_name}, MODEL: {model.model_name}')
        acc, prec, f1 = model.get_performance()
        print(f'Number of rules: {len(paths)}')

        alpha = model.parameters['n_estimators'] * n / len(paths)
        xi = info['xi']
        lambda_ = info['lambda_']
        ex = Extractor(paths, model.X_train, model.clf.predict(model.X_train))
        w, _, _, _ = ex.extract(n, xi * alpha, lambda_)
        idxes = np.flatnonzero(w)

        accuracy = ex.evaluate(w, model.X_test, model.y_test)
        accuracy = round(accuracy, 4)

        fidelity_test = ex.evaluate(w, model.X_test, model.clf.predict(model.X_test))
        fidelity_test = round(fidelity_test, 4)

        fidelity_train = ex.evaluate(w, model.X_train, model.clf.predict(model.X_train))
        fidelity_train = round(fidelity_train, 4)
        anomaly_score = np.mean([paths[i]['score'] for i in idxes])

        xi = round(xi, 4)
        lambda_ = round(lambda_, 4)
        anomaly_score = round(anomaly_score, 4)

        print(f'FIDELITY: {fidelity_test}')
        f = open('result/our_anomaly.txt', 'a')
        f.write('DATA: %s, MODEL: %s, xi: %s, lambda: %s, score: %s\n'%(model.data_name, model.model_name, xi, lambda_, anomaly_score))
        f.write('Accuracy: %s, Fidelity: %s\n'%(accuracy, fidelity_test))
        f.close()

        ex = Extractor(paths, model.X_train, model.clf.predict(model.X_train))
        w, _, _, _ = ex.extract(n, xi * alpha, 0)
        idxes = np.flatnonzero(w)

        accuracy = ex.evaluate(w, model.X_test, model.y_test)
        accuracy = round(accuracy, 4)

        fidelity_test = ex.evaluate(w, model.X_test, model.clf.predict(model.X_test))
        fidelity_test = round(fidelity_test, 4)

        fidelity_train = ex.evaluate(w, model.X_train, model.clf.predict(model.X_train))
        fidelity_train = round(fidelity_train, 4)
        anomaly_score = np.mean([paths[i]['score'] for i in idxes])
        anomaly_score = round(anomaly_score, 4)

        print(f'FIDELITY: {fidelity_test}')
        f = open('result/our_original.txt', 'a')
        f.write('DATA: %s, MODEL: %s, xi: %s, score: %s\n'%(model.data_name, model.model_name, xi, anomaly_score))
        f.write('Accuracy: %s, Fidelity: %s\n'%(accuracy, fidelity_test))
        f.close()
