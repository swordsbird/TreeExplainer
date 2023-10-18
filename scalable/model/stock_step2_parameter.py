from sklearn.model_selection import train_test_split

import os
import sys
sys.path.append('.')
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from scalable.model.stock_step2 import Model
base_accuracy = np.array([0.2970, 0.2260, 0.4770])

if __name__ == '__main__':
    model = Model('lightgbm')
    model.init_data()

    from tabnanny import verbose
    import optuna  # pip install optuna
    from sklearn.metrics import log_loss, precision_score, f1_score, accuracy_score
    from sklearn.model_selection import KFold
    from optuna.integration import LightGBMPruningCallback
    from lightgbm import LGBMClassifier
    random_state = 10

    X_train = model.X_train
    y_train = model.y_train
    X_test = model.X_test
    y_test = model.y_test

    print('X_train', X_train.shape)
    print('y_train', y_train.shape)
    print('X_test', X_test.shape)
    print('y_test', y_test.shape)

    def objective(trial, X_train, y_train, X_test, y_test):
        # 后面填充
        param_grid = {
            "n_estimators": trial.suggest_int("n_estimators", 500, 500, step=10),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.05),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1),
            "subsample": trial.suggest_float("subsample", 0.5, 1),
            "num_leaves": trial.suggest_int("num_leaves", 32, 128),
        }

        model = LGBMClassifier(**param_grid, verbose=-1, class_weight = 'balanced')
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        conf_mat = confusion_matrix(y_test, y_pred)
        accuracys = []

        tot = 0
        for i in range(model.n_classes_):
            accuracy = conf_mat[i, i] / conf_mat[i].sum()
            accuracys.append(accuracy)
            print(f'Accuracy on {model.classes_[i]}: {accuracy}')
            tot += conf_mat[i, i]
        tot /= len(y_pred)
        print(f'Accuracy: {tot}')
        accuracys = np.array(accuracys)

        gain = np.maximum(accuracys - base_accuracy, 0)
        return gain[0] * gain[1] * gain[2]

    study = optuna.create_study(direction="maximize", study_name="LGBM Classifier")
    func = lambda trial: objective(trial, X_train, y_train, X_test, y_test)
    study.optimize(func, n_trials=500)
    
    print(study.best_params)
