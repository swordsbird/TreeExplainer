
from random import *
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import pairwise_distances
from annoy import AnnoyIndex
from dataconfig import cache_dir_path, data_encoding, data_setting
import sys
sys.path.append('..')
from scalable.anomaly import LRAnomalyDetection
from scalable.model import get_model
import os
import bisect

class DataLoader():
    def __init__(self, info, name, target):
        self.info = info
        self.name = name

        model = get_model(info['model_info']['dataset'], info['model_info']['model'].lower())
        model.init_data()
        model.train()
        self.model = model
        self.paths = self.info['paths']
        scores = np.array([p['score'] for p in self.paths])

        self.path_index = {}
        for index, path in enumerate(self.paths):
            self.path_index[path['name']] = index
            path['level'] = 2 if path['represent'] else 1
        max_level = 2
        self.selected_indexes = [path['name'] for path in self.paths if path['represent']]#self.model['selected']
        self.features = self.info['features']
        current_encoding = data_encoding.get(name, {})
        current_setting = data_setting.get(name, {})
        self.current_encoding = current_encoding
        self.current_setting = current_setting

        self.info['model_info']['target'] = target
        self.target = target
        if self.info['model_info']['model'] == 'LightGBM':
            self.info['model_info']['weighted'] = True
        else:
            self.info['model_info']['weighted'] = False

        if not os.path.exists(cache_dir_path):
            os.mkdir(cache_dir_path)

        vectors = []
        distribs = []
        for path in self.paths:
            distrib = np.array(path['distribution'])
            distrib = distrib / distrib.sum()# * 0.5
            #distrib.sum() #
            vectors.append(np.array(path['feature_vector']))
            distribs.append(distrib)
        max_sum = np.sqrt(np.max([(vector * vector).sum() for vector in vectors]))
        for i in range(len(vectors)):
            vectors[i] /= max_sum
        mats = []
        for i in range(len(vectors)):
            mats.append(np.concatenate((vectors[i], distribs[i]), axis = 0))
        path_mat = np.array(mats)
        X_mat = np.array([path['X'] for path in self.paths])
        y_mat = np.array([path['y'] for path in self.paths])
        np.seterr(divide='ignore', invalid='ignore')
        path_mat = path_mat.astype(np.float32)
        X_mat = X_mat.astype(np.float32)
        y_mat = y_mat.astype(np.float32)

        path_dist = pairwise_distances(X = path_mat, metric='euclidean')
        tree = AnnoyIndex(len(path_mat[0]), 'euclidean')
        for i in range(len(path_mat)):
            tree.add_item(i, path_mat[i])
        tree.build(10)
        self.tree = tree
        self.detector = LRAnomalyDetection(X_mat, y_mat)
        scores = self.detector.score()

        for i in range(len(self.paths)):
            self.paths[i]['anomaly'] = self.paths[i]['initial_anomaly'] = scores[i]
            self.paths[i]['represent'] = False
            self.paths[i]['children'] = []

        for level in range(max_level, 0, -1):
            ids = []
            for i in range(len(self.paths)):
                if self.paths[i]['level'] == level:
                    ids.append(i)
            for i in range(len(self.paths)):
                if self.paths[i]['level'] == level - 1:
                    self.paths[i]['children'] = []
                    nearest = -1
                    nearest_dist = 1e10
                    for j in ids:
                        if path_dist[i][j] < nearest_dist:# and self.paths[i]['output'] == self.paths[j]['output']:
                            nearest = j
                            nearest_dist = path_dist[i][j]
                    j = nearest
                    # self.paths[i]['father'] = j
                    self.paths[j]['children'].append(i)
        for i in range(len(self.paths)):
            self.paths[i]['father'] = i
        for i in range(len(self.paths)):
            children = [(path_dist[i][j], j) for j in self.paths[i]['children']]
            children = sorted(children)
            self.paths[i]['children'] = [j for _, j in children]
            for j in self.paths[i]['children']:
                self.paths[j]['father'] = i

        self.path_dict = {}
        for path in self.paths:
            self.path_dict[path['name']] = path

    def get_general_info(self, idxes = None):
        if idxes is None:
            positives = (self.data_table['Label'] == self.target_class).sum()
            total = len(self.data_table['Label'])
        else:
            positives = (self.data_table['Label'][idxes] == self.target_class).sum()
            total = len(idxes)
        return (positives, total, positives / total)

    def get_relevant_samples(self, idxes):
        samples = {}
        for i in idxes:
            for j in self.paths[i]['sample_id']:
                if j not in samples:
                    samples[j] = 1
                else:
                    samples[j] += 1
        thres = 1
        if len(samples) * 2 > len(self.model.data_table):
            thres += 1
        ret = [k for k in samples if samples[k] >= thres]
        ret = sorted(ret)
        return ret

    def get_encoded_path(self, idx):
        path = self.paths[idx]
        output = path['output']
        if type(output) != int:
            output = np.argmax(output)
        distribution = np.array(path['distribution']) * self.class_weight
        confidence = distribution[output] / distribution.sum()
        return {
            'labeled': self.path_index[path['name']] in self.detector.labeled_data,
            'name': path['name'],
            'idx': idx,
            'tree_index': path['tree_index'],
            'rule_index': path['rule_index'],
            'represent': path['represent'],
            'father': path['father'],
            'range': path['range'],
            'missing': path.get('missing', []),
            'level': path['level'],
            'weight': path['weight'],
            'anomaly': path['anomaly'],
            'initial_anomaly': path['initial_anomaly'],
            'num_children': len(path['children']),
            'distribution': path['distribution'],
            'confidence': confidence,
            'coverage': path['coverage'],
            #'feature_vector': path['feature_vector'],
            'output': output,
            'samples': path['sample_id'],
        }

    def model_info(self):
        return self.info['model_info']

    def set_data_table(self, data):
        # print('data length', len(self.model.y), len(data))
        if len(self.model.y) == len(data):
            pred_y = self.model.clf.predict(self.model.X)
        else:
            X, _ = self.model.transform_data(data)
            pred_y = self.model.clf.predict(X)

        try:
            data['Predict'] = [self.info['model_info']['targets'][i] for i in pred_y]
        except:
            data['Predict'] = pred_y
        col = data[self.target].values
        data = data.drop(self.target, axis = 1)
        data['Label'] = col

        setting = self.current_setting
        encoding = self.current_encoding

        for index, feature in enumerate(self.features):
            name = feature['name']
            if name in setting and 'scale' in setting[name]:
                feature['scale'] = setting[name]['scale']
            else:
                feature['scale'] = 'linear'

            if name in setting and 'display_name' in setting[name]:
                feature['display_name'] = setting[name]['display_name']

            if name in encoding:
                feature['values'] = encoding[name]
                if data[name].dtype == np.int64 and len(encoding[name]) > 0:
                    col = data[name].values
                    data = data.drop(name, axis = 1)
                    data[name] = [encoding[name][i] for i in col]
            else:
                r = feature['values'] = feature['range'] = data[name].astype(float).quantile([0.05, 0.95]).tolist()
                q = data[name].astype(float).quantile([0.05, 0.25, 0.5, 0.75, 0.95]).tolist()
                data.loc[data[name] < r[0], name] = r[0]
                data.loc[data[name] > r[1], name] = r[1]
                feature['avg'] = data[name].mean()
                feature['q'] = q

            if name not in encoding and data[name].dtype != 'object':
                r = feature['range']
                data.loc[data[name] < r[0], name] = r[0]
                data.loc[data[name] > r[1], name] = r[1]

        classes = self.paths[0]['classes']
        targets = [(i, j) for i, j in enumerate(classes)]
        self.info['model_info']['targets'] = [x[1] for x in targets]
        weight = np.array([(data['Label'] == c).sum() for c in classes]).astype(np.float64)
        weight = 1.0 / (weight / weight.max())
        self.class_weight = weight
        self.target_class = classes[1]

        data = data.fillna(-1)
        self.data_table = data

class DatasetLoader():
    def __init__(self):
        data_loader = {}

        '''
        data_table = pd.read_csv('../data/credit_card_train3.csv')
        data = pd.read_csv('../data/credit_card_train3.csv')
        #data_table = pd.read_csv('../data/credit_t.csv')
        #data = pd.read_csv('../data/credit_t.csv')
        target = 'Approved'
        targets = [(0, 'Rejected'), (1, 'Approved')]

        info = pickle.load(open('../output/dump/credit_step2.pkl', 'rb'))
        #info = pickle.load(open('../output/dump/credit_v7_0529.pkl', 'rb'))
        loader = DataLoader(data, info, 'credit', target, targets)
        loader.set_data_table(data_table)
        data_loader['credit_new'] = loader


        data_table = pd.read_csv('../data/case2_stock/1year_small_raw.csv')
        info = pickle.load(open('../output/dump/stock_step0_v1.pkl', 'rb'))
        loader = DataLoader(info, 'stock', 'label')
        loader.set_data_table(data_table)
        data_loader['stock'] = loader
        '''

        data_table = pd.read_csv('../data/case2_stock/step/3year_raw.csv')
        info = pickle.load(open('../output/dump/stock_step0.pkl', 'rb'))
        loader = DataLoader(info, 'stock', 'label')
        loader.set_data_table(data_table)
        data_loader['stock'] = loader


        '''
        data = pd.read_csv('../data/stock_ep.csv')
        target = 'analystConsensus'
        targets = [(0, 'Increase'), (1, 'Stable'), (2, 'Decrease')]

        info = pickle.load(open('../output/dump/stock_step1.1.pkl', 'rb'))
        loader = DataLoader(data, info, 'stock', target, targets)
        print('model', loader.model.X.shape, loader.model.y.shape, )
        loader.set_data_table(data_table)
        loader.class_weight = np.array([4, 1, 4])
        data_loader['stockep'] = loader

        data = pd.read_csv('../data/stock2.csv')
        target = 'analystConsensus'
        targets = [(0, 'Increase'), (1, 'Stable'), (2, 'Decrease')]

        info = pickle.load(open('../output/dump/stock_step2.pkl', 'rb'))
        loader = DataLoader(data, info, 'stock', target, targets)
        loader.set_data_table(data)
        loader.class_weight = np.array([4, 1, 4])
        data_loader['stock2'] = loader
        '''


        #loader.discretize()

        self.data_loader = data_loader

    def get(self, name):
        return self.data_loader[name]

