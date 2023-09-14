import pulp
import numpy as np
import random
from copy import deepcopy
import time

max_path_num = 10000
max_sample_num = 5000

class Extractor:
    def __init__(self, paths, X_train, y_train, cover = None):

        if len(X_train) > max_sample_num:
            idx = random.sample(range(len(X_train)), max_sample_num)
            self.X_raw = X_train[idx]
            self.y_raw = y_train[idx]
        else:
            self.X_raw = X_train
            self.y_raw = y_train

        if len(paths) > max_path_num:
            class_weight = np.array([(y_train == k).sum() for k in paths[0]['classes']])
            class_weight = class_weight / class_weight.max()
            for path in paths:
                output = path['output']
                if type(output) != int:
                    output = np.argmax(output)
                distribution = np.array(path['distribution']) * class_weight
                confidence = distribution[output] / distribution.sum()
                path['confidence'] = confidence
            conf_thres = np.quantile(np.array([p['confidence'] for p in paths]), 1 - max_path_num / len(paths))
            for path in paths:
                path['skip'] = path['confidence'] < conf_thres
        else:
            for path in paths:
                path['skip'] = False

        self.paths = paths
        self.is_multiclass = paths[0].get('is_multiclass', False)
        self.n_classes = paths[0].get('n_classes', 2)
        self.classes = paths[0].get('classes', [])
        self.mat = self.getMat(self.X_raw, self.y_raw, self.paths)
        for p in paths:
            if 'cost' not in p:
                p['cost'] = 1
        self.weight = np.array([p['cost'] for p in paths])
        #print('values', [p['value'] for p in paths])
        self.cover = cover

    def compute_accuracy_on_train(self, paths):
        y_pred = self.predict(self.X_raw, paths)
        # y_pred = np.where(y_pred == 1, 1, 0)
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
        if self.is_multiclass:
            Y = np.zeros((X.shape[0], self.n_classes))
        else:
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
            if self.is_multiclass:
                Y[:, p['output_class']] += ans * (p.get('weight') * p.get('value'))
            else:
                Y += ans * (p.get('weight') * p.get('value'))
        if self.is_multiclass:
            Y = np.array([self.classes[np.argmax(Y[i])] for i in range(Y.shape[0])])
        else:
            Y = np.where(Y > 0, 1, 0)
        return Y

    def getMat(self, X_raw, y_raw, paths):
        mat = np.array([self.path_score(p, X_raw, y_raw) for p in paths]).astype('float')
        return mat

    def path_score(self, path, X, y):
        value = float(path.get('value'))
        if self.is_multiclass:
            o = path['output_class']
            all_ans = []
            for i in range(self.n_classes):
                i_class = self.classes[i]
                for j in range(i + 1, self.n_classes):
                    j_class = self.classes[j]
                    if not path['skip'] and i == o:
                        ans = np.where((y == i_class) + (y == j_class), 1, 0) * np.where(y == i_class, value, -value)
                    elif not path['skip'] and j == o:
                        ans = np.where((y == i_class) + (y == j_class), 1, 0) * np.where(y == j_class, value, -value)
                    else:
                        ans = np.zeros(len(y))
                    all_ans.append(ans)
        else:
            if not path['skip']:
                y = y * 2 - 1
                ans = value * y
            else:
                ans = np.zeros(len(y))

        if not path['skip']:
            m = path.get('range')
            missing = path.get('missing', [])
            for key in m:
                if key in missing:
                    cond = ((X[:, int(key)] >= m[key][0]) * (X[:, int(key)] < m[key][1]) + np.isnan(X[:, int(key)].astype(float)))
                else:
                    cond = (X[:, int(key)] >= m[key][0]) * (X[:, int(key)] < m[key][1])
                if self.is_multiclass:
                    for i in range(len(all_ans)):
                        all_ans[i] = all_ans[i] * cond
                else:
                    ans = ans * cond
        
        if self.is_multiclass:
            ans = np.array(all_ans)

        return ans

    def extract(self, m, tau, lambda_, method = 'maximize'):
        mat = self.mat
        w = self.weight
        if method == 'maximize':
            paths_weight, obj = self.LP_extraction_maximize(w, mat, m, tau, lambda_)
        else:
            paths_weight, obj = self.LP_extraction_minimize(w, mat, m, tau, lambda_)
        accuracy_origin1 = self.compute_accuracy_on_train(self.paths)
        path_copy = deepcopy(self.paths)
        for i in range(len(path_copy)):
            path_copy[i]['weight'] = 1 if paths_weight[i] > 0 else 0
        accuracy_new1 = self.compute_accuracy_on_train(path_copy)
        return paths_weight, accuracy_origin1, accuracy_new1, obj

    def LP_extraction_minimize(self, score, y, m, tau, lambda_):
        m = pulp.LpProblem(sense=pulp.LpMaximize)
        var = []
        N = y.shape[1]
        M = y.shape[0]
        zero = 1000
        for i in range(M):
            var.append(pulp.LpVariable(f'z{i}', cat=pulp.LpContinuous, lowBound=0, upBound=1))
        for i in range(N):
            var.append(pulp.LpVariable(f'k{i}', cat=pulp.LpContinuous, lowBound=0))
        first_term = pulp.LpVariable('first', cat=pulp.LpContinuous, lowBound=0)
        second_term = pulp.LpVariable('second', cat=pulp.LpContinuous, lowBound=0)
        m.setObjective(first_term + second_term)
        m += (pulp.lpSum([var[j + M] for j in range(N)]) <= first_term)
        m += (pulp.lpSum([var[j] * score[j] * lambda_ for j in range(M)]) <= second_term)
        m += (pulp.lpSum([var[j] for j in range(M)]) <= m)
        m += (pulp.lpSum([var[j] for j in range(M)]) >= m)
        for j in range(N):
            m += (var[j + M] >= zero + tau - pulp.lpSum([var[k] * y[k][j] for k in range(M)]))
            m += (var[j + M] >= zero)

        m.solve(pulp.PULP_CBC_CMD(msg=False))  # solver = pulp.solver.CPLEX())#
        z = [var[i].value() for i in range(M)]
        for k in np.argsort(z)[:-m]:
            z[k] = 0
        z = z / np.sum(z)
        return z, (pulp.value(m.objective) - zero * N, first_term.value() - zero * N, second_term.value())

    def LP_extraction_maximize_round(self, score, z, y, m, tau, lambda_):
        
        if self.is_multiclass:
            K = y.shape[2]
            N = y.shape[1]
            M = y.shape[0]
            yt = y.copy().reshape((M, N * K))
            yt = yt.T
            z0 = np.zeros(len(z))
            z_candidates = np.flatnonzero(z)
            tau = tau * 1.0
            v = np.array([np.dot(z0, yt[j]) for j in range(N * K)])
            loss_curr = np.minimum(v, tau).sum()
            for _ in range(m):
                best_loss_gain = -1e10
                best_i = -1
                for i in z_candidates:
                    z1 = np.zeros(len(z))
                    z1[i] = z[i]
                    v = np.array([np.dot(z0 + z1, yt[j]) for j in range(N * K)])
                    loss_new = np.minimum(v, tau).sum()
                    loss_new += np.dot(z0 + z1, score) * lambda_
                    if loss_new - loss_curr > best_loss_gain:
                        best_loss_gain = loss_new - loss_curr
                        best_i = i
                v = np.array([np.dot(z0, yt[j]) for j in range(N * K)])
                loss_curr += best_loss_gain
                z0[best_i] = z[best_i]
                z_candidates = [i for i in z_candidates if i != best_i]
        else:
            N = y.shape[1]
            yt = y.T
            z0 = np.zeros(len(z))
            z_candidates = np.flatnonzero(z)
            tau = tau * 1.0
            v = np.array([np.dot(z0, yt[j]) for j in range(N)])
            loss_curr = np.minimum(v, tau).sum()
            for _ in range(m):
                best_loss_gain = -1e10
                best_i = -1
                for i in z_candidates:
                    z1 = np.zeros(len(z))
                    z1[i] = z[i]
                    v = np.array([np.dot(z0 + z1, yt[j]) for j in range(N)])
                    loss_new = np.minimum(v, tau).sum()
                    loss_new += np.dot(z0 + z1, score) * lambda_
                    if loss_new - loss_curr > best_loss_gain:
                        best_loss_gain = loss_new - loss_curr
                        best_i = i
                v = np.array([np.dot(z0, yt[j]) for j in range(N)])
                loss_curr += best_loss_gain
                z0[best_i] = z[best_i]
                z_candidates = [i for i in z_candidates if i != best_i]
        
        z = z0
        z = z / np.sum(z)
        return z


    def LP_extraction_maximize(self, score, y, m, tau, lambda_):
        p = pulp.LpProblem(sense=pulp.LpMaximize)
        var = []

        if self.is_multiclass:
            K = y.shape[2]
            N = y.shape[1]
            M = y.shape[0]
            zero = 1000
            for i in range(M):
                var.append(pulp.LpVariable(f'z{i}', cat=pulp.LpContinuous, lowBound=0, upBound=1))
            for i in range(N * K):
                var.append(pulp.LpVariable(f'k{i}', cat=pulp.LpContinuous, lowBound=0))
            p.setObjective(pulp.lpSum([var[j + M] for j in range(N * K)] + [var[j] * score[j] * lambda_ for j in range(M)]))
            p += (pulp.lpSum([var[j] for j in range(M)]) <= m)
            for l in range(K):
                for j in range(N):
                    p += (var[j + M + l * N] <= zero + tau)
                    if np.abs(y[:, j, l]).sum() == 0:
                        continue
                    p += (var[j + M + l * N] <= zero + pulp.lpSum([var[k] * y[k][j][l] for k in range(M) if y[k][j][l] != 0]))
        else:
            N = y.shape[1]
            M = y.shape[0]
            K = 1
            zero = 1000
            for i in range(M):
                var.append(pulp.LpVariable(f'z{i}', cat=pulp.LpContinuous, lowBound=0, upBound=1))
            for i in range(N):
                var.append(pulp.LpVariable(f'k{i}', cat=pulp.LpContinuous, lowBound=0))
            p.setObjective(pulp.lpSum([var[j + M] for j in range(N)] + [var[j] * score[j] * lambda_ for j in range(M)]))
            p += (pulp.lpSum([var[j] for j in range(M)]) <= m)
            for j in range(N):
                p += (var[j + M] <= zero + pulp.lpSum([var[k] * y[k][j] for k in range(M) if y[k][j] != 0]))
                p += (var[j + M] <= zero + tau)

        last = time.time()
        p.solve(pulp.GUROBI_CMD(msg=False))
        curr = time.time()
        print(f'TIME: {round(curr - last, 2)}s')

        z = np.array([var[i].value() for i in range(M)])
        #weighted_z = [var[i].value() * np.sum(self.paths[i]['distribution']) for i in range(M)]
        #for k in np.argsort(weighted_z)[:-m]:
        #    z[k] = 0
        #z = z / np.sum(z)
        z = self.LP_extraction_maximize_round(score, z, y, m, tau, lambda_)
        first_term = np.sum([var[j + M].value() for j in range(N * K)])
        second_term = np.sum([var[j].value() * score[j] * lambda_ for j in range(M)])
        print('first_term', first_term - zero * N * K, 'second_term', second_term)
        return z, pulp.value(p.objective) - zero * N
