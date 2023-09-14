
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
    def __init__(self, data, info, name, target_feature, targets, target_value = 1):
        self.data_table = data
        self.info = info
        self.name = name
        self.target_map = dict(targets)
        self.class_weight = 1
        self.target_value = target_value
        model = get_model(info['model_info']['dataset'], info['model_info']['model'].lower())
        model.init_data()
        model.train()
        self.model = model
        self.paths = self.info['paths']
        scores = np.array([p['score'] for p in self.paths])
        # print('score', scores.mean(), np.quantile(scores, 0.1), np.quantile(scores, 0.2), np.quantile(scores, 0.8), np.quantile(scores, 0.9))
        self.path_index = {}
        for index, path in enumerate(self.paths):
            self.path_index[path['name']] = index
            path['level'] = 2 if path['represent'] else 1
        max_level = 2
        self.selected_indexes = [path['name'] for path in self.paths if path['represent']]#self.model['selected']
        self.features = self.info['features']
        # print(self.features)
        current_encoding = data_encoding.get(name, {})
        current_setting = data_setting.get(name, {})
        self.current_encoding = current_encoding
        self.current_setting = current_setting
        for index, feature in enumerate(self.features):
            if feature['name'] in current_setting and 'scale' in current_setting[feature['name']]:
                feature['scale'] = current_setting[feature['name']]['scale']
            else:
                feature['scale'] = 'linear'
            
            if feature['name'] in current_setting and 'display_name' in current_setting[feature['name']]:
                feature['display_name'] = current_setting[feature['name']]['display_name']

            if feature['name'] in current_encoding:
                #if len(current_encoding[feature['name']]) > 0:
                feature['values'] = current_encoding[feature['name']]
                #else:
                #    feature['values'] = [str(x) for x in model.feature_values[feature['name']]]
            else:
                # feature['values'] = feature['range']
                # q = data[feature['name']].astype(float).quantile([0.25, 0.5, 0.75]).tolist()
                feature['values'] = feature['range'] = data[feature['name']].astype(float).quantile([0.05, 0.95]).tolist()
                q = data[feature['name']].astype(float).quantile([0.05, 0.25, 0.5, 0.75, 0.95]).tolist()
                #q = [feature['range'][0]] + q + [feature['range'][1]]
                feature['avg'] = data[feature['name']].mean()
                feature['q'] = q
        self.info['model_info']['target'] = target_feature
        self.info['model_info']['targets'] = [x[1] for x in targets]
        self.target = target_feature
        if self.info['model_info']['model'] == 'LightGBM':
            self.info['model_info']['weighted'] = True
        else:
            self.info['model_info']['weighted'] = False

        if not os.path.exists(cache_dir_path):
            os.mkdir(cache_dir_path)
        
        # cache_path = os.path.join(cache_dir_path, name + '.pkl')
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
        # pickle.dump({ 'paths': self.paths, 'scores': self.detector.scores }, open(cache_path, 'wb'))
        self.path_dict = {}
        for path in self.paths:
            self.path_dict[path['name']] = path
            path['sample'] = np.array(path['sample'])
        
        #self.init_corr()
    

    def init_corr(self):
        corr = self.data_table.corr()
        self.corr = corr
        n_cols = len(self.data_table.columns)
        corr_values = []
        for i in range(n_cols):
            k1 = self.data_table.columns[i]
            for k2 in self.data_table.columns[i + 1:]:
                if k1 != k2 and abs(corr[k1][k2]) > 0:
                    corr_values.append(abs(corr[k1][k2]))
        self.has_high_corr_thres = np.quantile(corr_values, 0.975)
        self.has_low_corr_thres = np.quantile(corr_values, 0.85)
        features = [[i, x['importance'], x['name']] for i, x in enumerate(self.features)]
        features = sorted(features, key = lambda x: -x[1])
        for i, x in enumerate(features):
            k1 = x[2]
            for j in range(i + 1, len(features)):
                k2 = features[j][2]
                if abs(corr[k1][k2]) > self.has_high_corr_thres:
                    features[i][1] += features[j][1]
                    features[j][1] = 0
        features = [x for x in features if x[1] > 0]
        features = sorted(features, key = lambda x: -x[1])
        self.independent_features = [x[2] for x in features]

    def discretize(self):
        for feature in self.features:
            values = np.unique(self.data_table[feature['name']])
            #print(values)
            if len(values) > 15:
                feature['discretize'] = True
            else:
                feature['discretize'] = False
                continue
            #values = self.data_table[feature['name']].values.copy()
            values.sort()
            feature['values'] = values
            n = len(values)
            frac_n = 1.0 / n
            feature['uniques'] = n
            new_values = self.data_table[feature['name']]
            new_values = [bisect.bisect_left(values, x) * frac_n for x in new_values]
            self.data_table[feature['name']] = new_values
            new_values = self.original_data[feature['name']]
            #if feature['name'] == 'Total debt/Total net worth':
            #    print(new_values[:10])
            new_values = [bisect.bisect_left(values, x) * frac_n for x in new_values]
            #if feature['name'] == 'Total debt/Total net worth':
            #    print(new_values)
            self.original_data[feature['name']] = new_values
            q = self.data_table[feature['name']].quantile([0.25, 0.5, 0.75]).tolist()
            q = [feature['range'][0]] + q + [feature['range'][1]]
            feature['avg'] = self.data_table[feature['name']].mean()
            feature['q'] = q
            #if feature['name'] == 'Total debt/Total net worth':
            #    print(q)
            #    print(values)
        for path in self.paths:
            for i in path['range']:
                if self.features[i]['discretize']:
                    [left, right] = path['range'][i]
                    path['range'][i] = [
                        bisect.bisect_left(self.features[i]['values'], left) / self.features[i]['uniques'],
                        bisect.bisect_right(self.features[i]['values'], right) / self.features[i]['uniques'],
                    ]
    
    def get_relevant_features(self, idxes, k = 6):
        feature_count = {}
        for i in idxes:
            for j in self.paths[i]['range']:
                if j not in feature_count:
                    feature_count[j] = 0
                feature_count[j] += 1
        path_relevant_features = [(j, feature_count[j]) for j in feature_count]
        path_relevant_features = sorted(path_relevant_features, key = lambda x: -x[1])
        #path_top_relevant_features = path_relevant_features[:k]
        path_top_relevant_features = [x for x in path_relevant_features if x[1] >= len(idxes) // 2]
        if len(path_top_relevant_features) < k:
            path_top_relevant_features = path_relevant_features[:k]
        path_top_relevant_features = [self.features[x[0]]['name'] for x in path_top_relevant_features]
        return path_top_relevant_features

    def get_feature_hint(self, path_idxes, sample_idxes, target, n = 8):
        top_relevant_features = self.get_relevant_features(path_idxes)
        feature_candidates = []
        for x in self.independent_features:
            flag = False
            for y in top_relevant_features:
                if abs(self.corr[x][y]) > self.has_low_corr_thres:
                    flag = True
                    break
            if flag:
                continue
            feature_candidates.append(x)
        pattern_candidates = []
        prob = (self.data_table[self.target][sample_idxes] == target).sum() / len(sample_idxes)
        prob_general = (self.data_table[self.target] == target).sum() / len(self.data_table[self.target])
        deltas = []
        for x in feature_candidates:
            sorted_idxes = sorted(sample_idxes, key = lambda i: self.data_table[x][i])
            feature_median = self.data_table[x][sorted_idxes].median()
            if feature_median == 0:
                continue
            first_half = sorted_idxes[:len(sorted_idxes) // 2]
            second_half = sorted_idxes[len(sorted_idxes) // 2:]
            prob_greater = (self.data_table[self.target][second_half] == target).sum() / len(second_half)
            prob_smaller = (self.data_table[self.target][first_half] == target).sum() / len(first_half)
            second_half = np.flatnonzero(self.data_table[x] >= feature_median).tolist()
            first_half = np.flatnonzero(self.data_table[x] < feature_median).tolist()
            # unbalanced feature
            ratio = len(first_half) / len(self.data_table[x])
            if ratio < 0.25 or ratio > 0.75:
                continue
            prob_greater_general = (self.data_table[self.target][second_half] == target).sum() / len(second_half)
            prob_smaller_general = (self.data_table[self.target][first_half] == target).sum() / len(first_half)
            delta = abs(prob_greater_general - prob_smaller_general)
            deltas.append(delta)
            if prob_greater > prob:
                if prob_greater_general - prob_general < prob_greater - prob:
                    pattern_candidates.append((prob_greater, delta, (x, '>=', feature_median), (len(first_half), len(second_half))))
            elif prob_smaller > prob:
                if prob_smaller_general - prob_general < prob_smaller - prob:
                    pattern_candidates.append((prob_smaller, delta, (x, '<', feature_median), (len(first_half), len(second_half))))
        if len(pattern_candidates) == 0:
            return []
        pattern_candidates = sorted(pattern_candidates, key = lambda x: -x[0])
        prob_max = pattern_candidates[0][0]
        #for k in pattern_candidates:
        #    print(k)
        mid_delta = np.quantile(deltas, 0.75)
        pattern_candidates = [x for x in pattern_candidates if x[0] > (prob + prob_max) / 2 and (x[1] < mid_delta or x[0] == prob_max)]
        return [(x[0], x[2]) for x in pattern_candidates[:n]]

    def get_general_info(self, idxes = None):
        if idxes is None:
            positives = (self.data_table[self.target] == self.target_value).sum()
            total = len(self.data_table[self.target])
        else:
            positives = (self.data_table[self.target][idxes] == self.target_value).sum()
            total = len(idxes)
        return (positives, total, positives / total)
            
    def get_relevant_samples(self, idxes):
        sample_array = self.paths[idxes[0]]['sample'].copy()
        for i in idxes[1:]:
            sample_array = sample_array + self.paths[i]['sample']
        thres = 1
        if (sample_array >= thres).sum() * 2 > len(sample_array):
            thres += 1
        return np.flatnonzero(sample_array >= thres).tolist()

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
            'missing': path.get('missing', {}) ,
            'level': path['level'],
            'weight': path['weight'],
            'anomaly': path['anomaly'],
            'initial_anomaly': path['initial_anomaly'],
            'num_children': len(path['children']),
            'distribution': path['distribution'],
            'confidence': confidence,
            'coverage': path['coverage'],
            'feature_vector': path['feature_vector'],
            'output': output,
            'samples': np.flatnonzero(path['sample']).tolist(),
        }

    def model_info(self):
        return self.info['model_info']
    
    def set_original_data(self, original_data):
        self.original_data = original_data
        print('len', len(self.model.y), len(original_data))
        if len(self.model.y) == len(original_data):
            pred_y = self.model.clf.predict(self.model.X)
        else:
            X, _ = self.model.transform_data(original_data)
            # print('shape', X.shape)
            pred_y = self.model.clf.predict(X)
        try:
            self.original_data['Predict'] = [self.info['model_info']['targets'][i] for i in pred_y]
        except:
            self.original_data['Predict'] = pred_y
        for index, feature in enumerate(self.features):
            name = feature['name']
            if name in self.current_encoding and self.original_data[name].dtype == np.int64 and len(self.current_encoding[name]) > 0:
                col = self.original_data[name].values
                self.original_data = self.original_data.drop(name, axis = 1)
                self.original_data[name] = [self.current_encoding[name][i] for i in col]
            elif name == 'ZipCode':
                self.original_data[name] = self.model.transform[name]
            elif name == 'Age':
                self.original_data[name] = [np.round(x) for x in self.original_data[name].values]
        target_feature = self.info['model_info']['target']
        try:
            self.original_data['Label'] = [self.target_map[i] for i in self.original_data[target_feature]]
        except:
            self.original_data['Label'] = self.original_data[target_feature]
        self.original_data = self.original_data.drop(target_feature, axis = 1)
        self.original_data = self.original_data.fillna(-1)
        self.sample_feature_importance = [np.zeros(len(self.features)) for i in range(len(pred_y))]
        for path in self.paths:
            d = np.zeros(len(self.features))
            for k in path['range']:
                d[k] += 1
            for i in np.flatnonzero(path['sample']).tolist():
                self.sample_feature_importance[i] += d
        max_value = 0
        for i in range(len(self.sample_feature_importance)):
            value = self.sample_feature_importance[i].max()
            if value > max_value:
                max_value = value
        for i in range(len(self.sample_feature_importance)):
            self.sample_feature_importance[i] /= max_value

class DatasetLoader():
    def __init__(self):
        data_loader = {}

        '''
        original_data = pd.read_csv('../data/german_detailed.csv')
        data = pd.read_csv('../data/german.csv')
        target = 'credit_risk'
        targets = [('No', 'Rejected'), ('Yes', 'Approved')]
        
        info = pickle.load(open('../output/dump/german_0425_new_v2.pkl', 'rb'))
        #        info = pickle.load(open('../model/output/german0120v2.pkl', 'rb'))
        loader = DataLoader(data, info, 'german', target, targets)
        loader.set_original_data(original_data)
        data_loader['german'] = loader

        
        original_data = pd.read_csv('../data/credit_card_train3.csv')
        data = pd.read_csv('../data/credit_card_train3.csv')
        #original_data = pd.read_csv('../data/credit_t.csv')
        #data = pd.read_csv('../data/credit_t.csv')
        target = 'Approved'
        targets = [(0, 'Rejected'), (1, 'Approved')]
        
        info = pickle.load(open('../output/dump/credit_step2.pkl', 'rb'))
        #info = pickle.load(open('../output/dump/credit_v7_0529.pkl', 'rb'))
        loader = DataLoader(data, info, 'credit', target, targets)
        loader.set_original_data(original_data)
        data_loader['credit_new'] = loader

        
        original_data = pd.read_csv('../data/credit_card.csv')
        data = pd.read_csv('../data/credit_card.csv')
        target = 'Approved'
        targets = [(0, 'Rejected'), (1, 'Approved')]
        
        info = pickle.load(open('../output/dump/credit_v62.pkl', 'rb'))
        loader = DataLoader(data, info, 'credit', target, targets)
        loader.set_original_data(original_data)
        data_loader['credit'] = loader

        original_data = pd.read_csv('../data/bank.csv')
        data = pd.read_csv('../data/bank.csv')
        target = 'Bankrupt?'
        targets = [(0, 'Non-bankrupt'), (1, 'Bankrupt')]
        
        info = pickle.load(open('../output/dump/bankruptcy_0624.pkl', 'rb'))
        loader = DataLoader(data, info, 'bankruptcy', target, targets)
        loader.set_original_data(original_data)
        data_loader['bankruptcy'] = loader
        '''

        data_table = pd.read_csv('../data/stock.csv')
        missing_value = {
            'Price-to-200dmA': 0.7246266539050062,
            'high52Weeks-low52Weeks': 24.578989990670866,
            'forwardEps': 11.285804913255696,
            'priceToBook': 22.490465087691305,
            'lastSplitDate': 1120528384.642443, 'shortRatio': 2.5391538461538463, 'price': 38.94054225271952, 'revenueGrowth': 4.460847289285703, 'ebitda': 3950703735.5001807, 'overallRisk': 4.144977168949771, 'fiveYearAvgDividendYield': 1.5671215978102193, 'threeMonthsGain': -0.07503341729577415, 'peRatio': 269.68785962999976, 'sharesOutstanding': 28331990.546841864, 'Price-to-50emA': 0.9000800663393471, 'earningsGrowth': 10.29059612893402, 'debtToEquity': 888.9829046684775, 'heldPercentInsiders': 0.2606314371301533, 'returnOnEquity': -2.6668378490915616, 'shortPercentOfFloat': 0.12313902412266173, 'quickRatio': 7.935160461005822, 'pegRatio': -13.97196532432434, 'floatShares': 1909764937.4617662, 'profitMargins': -1.1923611471229094, 'enterpriseToEbitda': 49.05842818905651, 'numberOfAnalystOpinions': 2.3942065491183877, 'grossProfits': 3507929393.2665787, 'dividendRate': 2.453668008298757, 'returnOnAssets': -0.13015627193973134, 'sharesPercentSharesOut': 0.03689692369863095, 'marketCap': 1547011258.389735, 'yearlyGain': -0.24682226771331056, 'lastDividendDate': 1463982839.3452349, 'earningsQuarterlyGrowth': 11.911862279999953, 'freeCashflow': 1843690520.9939518, 'volumeAvg30d': 720281.6558583333, 'sixMonthsGain': 0.6363147479419944, 'lastDividendValue': 4.2558167682651575, 'Price-to-21emA': 0.9118564917592868, 'netIncomeToCommon': 1240708435.1191192, 'operatingMargins': -153.98933996649353, 'currentRatio': 8.89738094734251, 'oneMonthGain': -0.11131898414619693, 'forwardPE': 16.878388628845983, '52WeekChange': -28.06212372516145, 'operatingCashflow': 2173679121.397712, 'payoutRatio': 2.0866253953038227, 'ebitdaMargins': -1.1505340020911479, 'totalCash': 2698311809.2497363, 'totalCashPerShare': 141.4650005664051, 'volumeAvg10d': 273867.66662610293, 'revenuePerShare': 295.5754175918238, 'enterpriseToRevenue': 321.54042317331164, 'changeAmount': -1.4752920191885883, 'totalDebt': 2103419894.492228, 'beta': 1.3215074924694283, 'sharesShort': 1433814.463485473, 'bookValue': 11.173071299728756, 'shareHolderRightsRisk': 5.657894736842105, 'heldPercentInstitutions': 0.6289776691151036, 'trailingEps': -5.689057973577236, 'volumeAvg10d-volumeAvg90d': -163183.94176706823, 'changePercent': -3.9497649877862573, 'trailingAnnualDividendYield': 0.010717866217803032, 'volume-volumeAvg30d': -90429.78827361562, 'priceToSalesTrailing12Months': 397.99374886936147, 'boardRisk': 5.852564102564102, 'trailingPegRatio': 13.695802119836967, 'trailingPE': 506.79647751713486, 'volume': 260449.6291513819, 'volumeAvg90d': 642797.4090141129, 'sharesShortPriorMonth': 4685047.976606057, 'trailingAnnualDividendRate': 0.729409090909091, 'volume-volumeAvg10d': -92420.82775919727, 'volume-volumeAvg90d': -89930.7713815789, 'grossMargins': -0.7259286120786217, 'compensationRisk': 4.903846153846154, 'totalRevenue': 2778511550.117754, 'dividendYield': 0.058159833696226446, 'totalValue': 97783625.40658924, 'auditRisk': 5.442028985507246, 'compensationAsOfEpochDate': 1617213600.0, 'impliedSharesOutstanding': 413640524.76}
        for k in data_table.columns:
            if data_table[k].dtype == float or data_table[k].dtype == int:
                x = data_table[k].values
                x = x[~np.isnan(x)]
                q_lo  = np.quantile(x, 0.005)

                #data_table.loc[data_table[k].isna(), k] = missing_value.get(k, q_lo)
            elif data_table[k].dtype == object and k != 'ticker':
                category_counts = data_table[k].value_counts()
                # print(k, category_counts)
                low_count_categories = category_counts[category_counts < 50].index
                data_table.loc[data_table[k].isin(low_count_categories), k] = 'Others'

        data = pd.read_csv('../data/stock.csv')
        target = 'analystConsensus'
        targets = [(0, 'Increase'), (1, 'Stable'), (2, 'Decrease')]
        
        info = pickle.load(open('../output/dump/stock_v3.pkl', 'rb'))
        loader = DataLoader(data, info, 'stock', target, targets)
        loader.set_original_data(data_table)
        loader.class_weight = np.array([4, 1, 4])
        data_loader['stock'] = loader


        data = pd.read_csv('../data/stock_ep.csv')
        target = 'analystConsensus'
        targets = [(0, 'Increase'), (1, 'Stable'), (2, 'Decrease')]
        
        info = pickle.load(open('../output/dump/stock_step1.1.pkl', 'rb'))
        loader = DataLoader(data, info, 'stock', target, targets)
        print('model', loader.model.X.shape, loader.model.y.shape, )
        loader.set_original_data(data_table)
        loader.class_weight = np.array([4, 1, 4])
        data_loader['stockep'] = loader
        
        data = pd.read_csv('../data/stock2.csv')
        target = 'analystConsensus'
        targets = [(0, 'Increase'), (1, 'Stable'), (2, 'Decrease')]
        
        info = pickle.load(open('../output/dump/stock_step2.pkl', 'rb'))
        loader = DataLoader(data, info, 'stock', target, targets)
        loader.set_original_data(data)
        loader.class_weight = np.array([4, 1, 4])
        data_loader['stock2'] = loader
        
        
        
        #loader.discretize()

        self.data_loader = data_loader
        
    def get(self, name):
        return self.data_loader[name]
        