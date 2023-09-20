import gurobipy as gp
from gurobipy import GRB
import numpy as np
import random
from copy import deepcopy
import time
from scalable.rule_query import Table

max_path_num = 10000
max_sample_num = 10000

def path_predict(X, paths):
    n_classes = paths[0].get('n_classes', 2)
    is_multiclass = n_classes > 2 
    classes = paths[0].get('classes', [])

    table = Table()
    for i in range(X.shape[1]):
        table.add(X[:, i])

    if is_multiclass:
        Y = np.zeros((X.shape[0], n_classes))
    else:
        Y = np.zeros(X.shape[0])

    for i, p in enumerate(paths):
        m = p.get('range')
        missing = p.get('missing', [])
        conds = [(key, m[key], key in missing) for key in m]
        samples = table.query(conds)
        d = p.get('weight') * p.get('value')
        if is_multiclass:
            for j in samples:
                Y[j, p['output_class']] += d
        else:
            for j in samples:
                Y[j] += d

    if is_multiclass:
        Y = np.array([classes[np.argmax(Y[i])] for i in range(Y.shape[0])])
    else:
        Y = np.where(Y > 0, 1, 0)
    return Y

class Extractor:
    def __init__(self, paths, X_train, y_train, cover = None, greedy = False):

        if len(X_train) > max_sample_num:
            idx = random.sample(range(len(X_train)), max_sample_num)
            self.X_raw = X_train[idx]
            self.y_raw = y_train[idx]
        else:
            self.X_raw = X_train
            self.y_raw = y_train
    
        self.table = Table()
        for i in range(self.X_raw.shape[1]):
            self.table.add(self.X_raw[:, i])

        for path in paths:
            path['skip'] = False

        self.paths = paths
        self.is_multiclass = paths[0].get('is_multiclass', False)
        self.n_classes = paths[0].get('n_classes', 2)
        self.n_paths = len(paths)
        self.classes = paths[0].get('classes', [])
        if self.n_classes == 2:
            self.classes = np.unique(y_train)
        for p in paths:
            if 'cost' not in p:
                p['cost'] = 1
        self.weight = np.array([p['cost'] for p in paths])
        self.cover = cover
        self.greedy = greedy

    def compute_accuracy_on_train(self, paths):
        y_pred = self.predict(self.X_raw, paths)
        return np.sum(np.where(y_pred == self.y_raw, 1, 0)) / len(self.X_raw)

    def evaluate(self, weights, X, y):
        paths = deepcopy(self.paths)
        for i in range(len(paths)):
            paths[i]['weight'] =  weights[i]
        y_pred = self.predict(X, paths)
        # y_pred = np.where(y_pred == 1, 1, 0)
        return np.sum(np.where(y_pred == y, 1, 0)) / len(X)

    def coverage(self, weights, X):
        paths = deepcopy(self.paths)
        for i in range(len(paths)):
            paths[i]['weight'] =  weights[i]# 1 if weights[i] > 0 else 0
        return self.coverage_raw(X, paths)

    def coverage_raw(self, X, paths):
        Y = np.zeros(X.shape[0])
        for i, p in enumerate(paths):
            if self.cover is None:
                ans = np.ones(X.shape[0])
                m = p.get('range') 
                missing = p.get('missing', [])
                for key in m:
                    if key in missing:
                        ans = ans * ((X[:, int(key)] >= m[key][0]) * (X[:, int(key)] < m[key][1]) + np.isnan(X[:, int(key)].astype(float)))
                    else:
                        ans = ans * (X[:, int(key)] >= m[key][0]) * (X[:, int(key)] < m[key][1])
            else:
                ans = self.cover[i]
            Y += ans * (p.get('weight') > 0)
        return Y

    def predict(self, X, paths):
        return path_predict(X, paths)
    
    '''
    def getConstraint(self, X, y, paths):
        table = Table()
        for i in range(X.shape[1]):
            table.add(X[:, i])

        n_dims = self.n_classes - 1
        n_samples = X.shape[0]
        self.n_constrs = n_samples * n_dims
        constrs = [[] for _ in range(self.n_constrs)]
        inv_constrs = [[] for _ in range(self.n_paths)]
        for p_i, path in enumerate(paths):
            if path['skip']:
                continue
            m = path.get('range')
            missing = path.get('missing', [])
            conds = [(key, m[key], key in missing) for key in m]
            samples = table.query(conds)
            path_output = self.classes[path['output_class']]
            value = float(path.get('value'))
            dim = 0
            for i in range(self.n_classes):
                i_class = self.classes[i]
                for j in range(i + 1, self.n_classes):
                    j_class = self.classes[j]
                    if i_class != path_output and j_class != path_output:
                        continue
                    if self.is_multiclass:
                        left_samples = [k for k in samples if y[k] == i_class or y[k] == j_class]
                    else:
                        left_samples = samples
                    for k in left_samples:
                        curr = value if y[k] == path_output else -value
                        constrs[k + dim * n_samples].append((p_i, curr))
                        inv_constrs[p_i].append((k + dim * n_samples, curr))
                    dim += 1
        return constrs, inv_constrs
    '''
    def getConstraint(self, X, y, paths, class_weight):
        table = Table()
        for i in range(X.shape[1]):
            table.add(X[:, i])

        n_dims = int(self.n_classes * (self.n_classes - 1) / 2)
        n_samples = X.shape[0]
        self.n_constrs = n_samples * n_dims
        constrs = [[] for _ in range(self.n_constrs)]
        inv_constrs = [[] for _ in range(self.n_paths)]
        constr_weight = np.ones(n_dims * n_samples)
        if class_weight == 'balanced':
            weight = np.array([(y == c).sum() for c in self.classes]).astype(np.float64)
            weight = 1.0 / (weight / weight.max())
            for ci in range(self.n_classes):
                class0 = self.classes[ci]
                for i in range(n_samples):
                    if y[i] == class0:
                        for dim in range(n_dims):
                            constr_weight[i + dim * n_samples] = weight[ci]

        for p_i, path in enumerate(paths):
            if path['skip']:
                continue
            m = path.get('range')
            missing = path.get('missing', [])
            conds = [(key, m[key], key in missing) for key in m]
            samples = table.query(conds)
            path_output = self.classes[path['output_class']]
            value = float(path.get('value'))
            dim = 0
            for i in range(self.n_classes):
                i_class = self.classes[i]
                for j in range(i + 1, self.n_classes):
                    j_class = self.classes[j]
                    if self.is_multiclass:
                        left_samples = [k for k in samples if y[k] == i_class or y[k] == j_class]
                    else:
                        left_samples = samples
                    for k in left_samples:
                        curr = value if y[k] == path_output else -value
                        constrs[k + dim * n_samples].append((p_i, curr))
                        inv_constrs[p_i].append((k + dim * n_samples, curr))
                    dim += 1
        return constrs, inv_constrs, constr_weight

    def extract(self, max_rules, tau, lambda_, method = 'maximize', class_weight=None):
        last = time.time()
        self.constrs, self.inv_constrs, self.constr_weight = self.getConstraint(self.X_raw, self.y_raw, self.paths, class_weight)
        paths_weight, obj = self.LP_extraction_maximize(self.weight, self.constrs, self.inv_constrs, self.constr_weight, max_rules, tau, lambda_)
        accuracy_origin1 = self.compute_accuracy_on_train(self.paths)
        for i in range(len(self.paths)):
            self.paths[i]['weight'] = 1 if paths_weight[i] > 0 else 0
        accuracy_new1 = self.compute_accuracy_on_train(self.paths)

        curr = time.time()
        print(f'TIME: {round(curr - last, 2)}s')
        return paths_weight, accuracy_origin1, accuracy_new1, obj

    def LP_extraction_maximize_round(self, score, z, inv_constrs, weight, max_rules, tau, lambda_):
        if self.greedy:
            z_candidates = range(len(z))
        else:
            z_candidates = np.flatnonzero(z)

        z0 = np.zeros(self.n_paths)
        v = np.zeros(self.n_constrs)
        loss_curr = 0
        for _ in range(max_rules):
            best_loss_gain = -1e10
            best_i = -1
            for i in z_candidates:
                v_curr = v.copy()
                for j, value in inv_constrs[i]:
                    v_curr[j] += z[i] * value
                loss_new = np.dot(np.minimum(v_curr, tau), weight).sum()
                loss_new += z[i] * score[i] * lambda_
                if loss_new - loss_curr > best_loss_gain:
                    best_loss_gain = loss_new - loss_curr
                    best_i = i
            for j, value in inv_constrs[best_i]:
                v[j] += z[best_i] * value
            loss_curr += best_loss_gain
            z0[best_i] = z[best_i]
            z_candidates = [i for i in z_candidates if i != best_i]
        
        z = z0
        z = z / np.sum(z)
        return z


    def LP_extraction_maximize(self, score, constrs, inv_constrs, weight, max_rules, tau, lambda_):
        if self.greedy:
            z = np.ones(self.n_paths)
            z = self.LP_extraction_maximize_round(score, z, inv_constrs, weight, max_rules, tau, lambda_)
            return z, 0

        zero = 1000
        model = gp.Model("MaximizeFidelity")
        z = model.addVars(range(self.n_paths), name="z", lb=0, ub=1)
        k = model.addVars(range(self.n_constrs), name="k", lb=0)
        objective = gp.quicksum(k[i] * weight[i] for i in range(self.n_constrs))
        objective += gp.quicksum(z[i] * score[i] * lambda_ for i in range(self.n_paths))
        model.addConstr(gp.quicksum(z[i] for i in range(self.n_paths)) <= max_rules, 'sum_z')
        for j in range(self.n_constrs):
            model.addConstr(k[j] <= zero + tau, f'k0_{j}')
            model.addConstr(k[j] <= zero + gp.quicksum(z[i] * value for i, value in constrs[j]), f'k1_{j}')

        model.setObjective(objective, sense=GRB.MAXIMIZE)
        model.optimize()

        z_values = model.getAttr('X', z)
        z = []
        for i in range(self.n_paths):
            z.append(z_values[i])
        z = np.array(z)
        z = self.LP_extraction_maximize_round(score, z, inv_constrs, weight, max_rules, tau, lambda_)

        return z, 0
