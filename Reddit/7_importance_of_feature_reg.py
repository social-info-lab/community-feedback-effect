from __future__ import division, print_function

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.extmath import safe_sparse_dot, log_logistic
# from sklearn.utils.fixes import expit
import numpy as np
import pandas as pd
from pandas import HDFStore
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.sparse import csr_matrix
from scipy.sparse import vstack, hstack, save_npz, load_npz
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
from sklearn.model_selection import GridSearchCV
from scipy.optimize import fmin_l_bfgs_b
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.extmath import safe_sparse_dot, log_logistic
from scipy.special import expit
import copy
import time
from sklearn.model_selection import train_test_split
import numpy.random as npr
from sklearn.model_selection import ShuffleSplit

"""l-bfgs-b L1-Logistic Regression solver"""


# Author: Vlad Niculae <vlad@vene.ro>
# Suggested by Mathieu Blondel

def _l1_logistic_loss_grad(w_extended, X, y, C, D, k, idx, ignore2w):
    # print(k)
    _, n_features = X.shape
    w = w_extended[:n_features] - w_extended[n_features:]

    yz = y * safe_sparse_dot(X, w)

    # Logistic loss is the negative of the log of the logistic function.
    out = -np.sum(log_logistic(yz))
    # out += .5 * alpha * np.dot(w, w)  # L2

    w_extended_ = copy.copy(w_extended)
    if ignore2w == 0:
        reg_idx = list(range(0, idx-1))
        reg_idx2 = list(range(n_features, n_features+idx-1))
        w_extended_[reg_idx] = 0.
        w_extended_[reg_idx2] = 0.
    elif ignore2w == 1:
        reg_idx = list(range(idx, n_features))
        reg_idx2 = list(range(n_features+idx, 2*n_features))
        w_extended_[reg_idx] = 0.
        w_extended_[reg_idx2] = 0.
    else:
        n_user = int((idx-1)/2)
        reg_idx = list(range(0, n_user)) + list(range(idx, n_features))
        reg_idx2 = list(range(n_features, n_features + n_user)) + list(range(n_features + idx, 2 * n_features))
        w_extended_[reg_idx] = 0.
        w_extended_[reg_idx2] = 0.


    if ignore2w > 0:
        w_ = w[idx:]
        w_ = np.transpose(w_.flatten().reshape(k, -1))
        # print(w_.shape)
        Dsmooth = w_[1:, :] - w_[:-1, :]
        zero = np.zeros((1, k))

        Dsmooth = np.concatenate((Dsmooth, zero), axis=0)
        Dsmooth = np.transpose(Dsmooth)
        Dsmooth = Dsmooth.flatten()
        Dsmooth_squared = Dsmooth * Dsmooth

        # out += alpha * w_extended.sum()
        out += C * w_extended_.sum() + D * Dsmooth_squared.sum()  # L1, w_extended is non-negative

        z = expit(yz)
        z0 = (z - 1) * y

        grad = safe_sparse_dot(X.T, z0)
        grad = np.concatenate([grad, -grad])

        # grad += alpha * w  # L2
        # grad += alpha +  # L1
        D_grad = np.zeros((n_features,))
        D_grad[idx:] = Dsmooth
        D_grad = np.concatenate([D_grad, -D_grad])
        grad += C - 2 * D * D_grad
    else:
        out += C + w_extended.sum()
        z = expit(yz)
        z0 = (z-1)*y
        grad = safe_sparse_dot(X.T, z0)
        grad = np.concatenate([grad, -grad])
        grad += C

    return out, grad


class LbfgsL1Logistic(BaseEstimator, ClassifierMixin):
    def __init__(self, tol=1e-3, C=1.0, D=1.0, no_topics=100, smooth_start_index=2, ignore2w=0):
        """Logistic Regression Lasso solved by L-BFGS-B
        Solves the same objective as sklearn.linear_model.LogisticRegression
        Parameters
        ----------
        alpha: float, default: 1.0
            The amount of regularization to use.
        tol: float, default: 1e-3
            Convergence tolerance for L-BFGS-B.
        """
        self.tol = tol
        self.C = C
        self.D = D
        self.no_topics = no_topics
        self.smooth_start_index = smooth_start_index
        self.ignore2w = ignore2w

    def fit(self, X, y):
        n_samples, n_features = X.shape

        coef0 = np.zeros(2 * n_features)
        w, f, d = fmin_l_bfgs_b(_l1_logistic_loss_grad, x0=coef0, fprime=None,
                                pgtol=self.tol,
                                bounds=[(0, None)] * n_features * 2,
                                args=(X, y, self.C, self.D, self.no_topics, self.smooth_start_index, self.ignore2w))
        self.coef_ = w[:n_features] - w[n_features:]

        return self

    def predict(self, X):
        return np.sign(safe_sparse_dot(X, self.coef_))

def getTrainTestIndexes():
    with open('test_indexes.txt') as f1, open('train_indexes.txt') as f2:
        test_lines = f1.readlines()
        train_lines = f2.readlines()

    test_indexes = [int(indx.strip()) for indx in test_lines]
    train_indexes = [int(indx.strip()) for indx in train_lines]

    return train_indexes, test_indexes

def classify_data(X, Y, data_name=''):
    no_users = 7072 #10915 #10921
    no_topics = 100

    if data_name == 'model_user':
        clf = LogisticRegression(penalty='l1')
        #parameters = {'C': [10000000]}
        parameters = {'C':[0.01, 0.1, 1.0, 10, 100]}
    elif data_name == 'model_user_event':
        clf = LbfgsL1Logistic(tol=1e-8, no_topics=100, smooth_start_index=no_users+1, ignore2w=1)
        parameters = {'C': [0.01, 0.1, 1.0], 'D': [1.0, 10]}
    elif data_name.startswith('model_user_fb'):
        clf = LbfgsL1Logistic(tol=1e-8, no_topics=100, smooth_start_index=no_users+1, ignore2w=0)
        parameters = {'C': [0.01,0.1,1.0,10, 100], 'D':[0.0]}
    elif data_name.startswith('model_user_event_fb'):
        clf = LbfgsL1Logistic(tol=1e-8, no_topics=100, smooth_start_index=2*no_users+1, ignore2w=2)
        parameters = {'C': [0.01, 0.1, 1.0], 'D': [1.0, 10.0]}
    else:
        clf = LogisticRegression(penalty='l1')
        parameters = {'C':[0.01, 0.1, 1.0, 10, 100]}
        #parameters = {'C': [0.01,0.1, 1.0, 10.0, 100.0]}
        #clf = LogisticRegression(penalty='l2', C = 100000000)
        #parameters = {'C': [100000000]}

    Y[Y == 0] = -1
    print(Y)

    #### Get Train vs Test Split ######
    train_indexes, test_indexes = getTrainTestIndexes()

    npr.seed(23)
    # shuf_list = npr.permutation(len(Y))
    X_train, X_test = X[train_indexes], X[test_indexes]
    y_train, y_test = Y[train_indexes], Y[test_indexes]

    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    print(X_train.shape)
    print('-----begin cross-validation-----')
    clf = GridSearchCV(clf, parameters, cv=10)
    clf.fit(X_train, y_train)

    best_C = clf.best_params_['C']

    if data_name.startswith('model_user_event'):
        best_D = clf.best_params_['D']
    else:
        best_D = 0

    chosen_clf = clf.best_estimator_

    best_idxs = np.where(clf.cv_results_['mean_test_score'] == clf.best_score_)[0]
    std_best_idxs = clf.cv_results_['std_test_score'][best_idxs]
    best_idx = np.where(clf.cv_results_['std_test_score'] == np.min(std_best_idxs))[0]
    best_cv_score, best_cv_std = clf.best_score_, clf.cv_results_['std_test_score'][best_idx]

    print(chosen_clf)
    print('CV score and std @95%', best_cv_score)

    pred = chosen_clf.predict(X_train)
    acc = accuracy_score(pred, y_train)
    print("Training Accuracy Score is: ", acc)
    pred_test = chosen_clf.predict(X_test)
    acc = accuracy_score(pred_test, y_test)
    print("Test Accuracy Score is: ", acc)
    print('Non-zero features: ', len(np.where(chosen_clf.coef_ != 0)[0]))

    print('-----begin bootstrapping -----')
    no_bootstraps = 200
    alpha = 0.95

    n_train = len(y_train)
    boot_samples = np.random.choice(train_indexes, (no_bootstraps, n_train))

    model_parameters = []
    accuracies = []
    mccs = []
    f1_scores = []
    npr.seed(23)
    for k in range(no_bootstraps):
        if k % 25 == 0:
            print('bootstrap, ', k)
        train_index = list(boot_samples[k])
        Xtrain, ytrain = X[train_index], Y[train_index]
        chosen_clf.fit(Xtrain, ytrain)

        pred = chosen_clf.predict(X_test)
        acc = accuracy_score(pred, y_test)
        f1 = f1_score(y_test, pred, average='macro')
        mcc = matthews_corrcoef(y_test, pred)
        accuracies.append(acc)
        f1_scores.append(f1)
        mccs.append(mcc)
        model_parameters.append(chosen_clf.coef_)

    model_parameters = np.array(model_parameters)
    model_parameters_mean = np.median(model_parameters, axis=0)
    model_parameters_mean = model_parameters_mean.flatten()

    p = ((1.0 - alpha) / 2.0) * 100
    lower_params = np.percentile(model_parameters, p, axis=0)
    p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    upper_params = np.percentile(model_parameters, p, axis=0)

    model_parameters_std = (upper_params - lower_params) / 2

    model_parameters_ci = model_parameters_std.flatten()
    model_parameters_std = (2.58 / np.sqrt(no_bootstraps)) * np.std(model_parameters, axis=0)
    model_parameters_std = model_parameters_std.flatten()
    print('CI error, 2*SE ', model_parameters_ci[0], model_parameters_std[0])

    accuracies = np.array(accuracies)
    mean_acc = np.mean(accuracies)
    mean_mcc = np.mean(np.array(mccs))
    mean_f1_score = np.mean(np.array(f1_scores))

    '''
    p = ((1.0 - alpha) / 2.0) * 100
    lower_acc = max(0.0, np.percentile(accuracies, p))
    p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    upper_acc = min(1.0, np.percentile(accuracies, p)) 

    acc_ci = (upper_acc - lower_acc) / 2
    '''

    std_acc = (2.58/np.sqrt(no_bootstraps)) * np.std(accuracies)
    std_mcc = (2.58 / np.sqrt(no_bootstraps)) * np.std(mccs)
    std_f1 = (2.58 / np.sqrt(no_bootstraps)) * np.std(f1_scores)
    print("Mean Test acc and std: ", mean_acc, std_acc)
    print("Mean Test mcc and std: ", mean_mcc, std_mcc)
    print("Mean Test f1-score and std: ", mean_f1_score, std_f1)

    print("Mean Feedback estimate & std: ", model_parameters_mean[0], model_parameters_std[0])

    model_parameters = np.array([model_parameters_mean, model_parameters_ci, model_parameters_std])
    np.savetxt('model_parameters_' + data_name + 'median.txt', model_parameters)
    np.savetxt('accuracies_'+data_name+'.txt', accuracies)
    np.savetxt('matthew_corr_' + data_name + '.txt', mccs)
    np.savetxt('f1_scores_' + data_name + '.txt', f1_scores)
    print(best_C, best_D, mean_acc, std_acc)
    return best_C, best_D, mean_acc, std_acc, mean_mcc, std_mcc, mean_f1_score, std_f1


if __name__ == '__main__':
    no_time_bins_per_day = 1
    print('Number of bins per day ', no_time_bins_per_day)

    data_name_suffix = ['model_c0', 'model_user', 'model_probTopic', 'model_user_probTopic', 'model_user_event', 'model_user_event_fb_pudt', 'model_user_event_fb_favdt']
    #data_name_suffix = ['model_user_event_fb_pudt', 'model_user_event_fb_favdt', 'model_user_event_fb_finaldt']

    Yfilename = 'outputY100.csv'
    Y = np.loadtxt(Yfilename)
    Y = Y.ravel()
    print('Y', Y.shape)
    no_res_cols = 8

    model_Res = np.zeros((len(data_name_suffix), no_res_cols))
    for i, suffix_name in enumerate(data_name_suffix):
        print(suffix_name)
        #if suffix_name == 'model_fb':
        #    Xfile_name = str(no_time_bins_per_day) + 'bins_per_day_featuresX_' + suffix_name + '.csv'
        #    X = np.loadtxt(Xfile_name)
        #    X = X.reshape(-1, 1)
        #else:
        Xfile_name = 'featuresX_' + suffix_name + '.npz'
        X = load_npz(Xfile_name)
        print('X.shape', X.shape)

        t0 = time.time()
        res = classify_data(X, Y, suffix_name)
        model_Res[i, :] = res
        t1 = time.time()
        print('time taken: ', t1 - t0)

    res_df = pd.DataFrame(model_Res, index=data_name_suffix,
                          columns=['best param C', 'best param D', 'test_accuracy', 'test_accuracy_error',
                                   'test_mcc', 'test_mcc_error', 'test_f1', 'test_f1_error'])

    res_df.to_csv('feature_importance_user.csv', sep=',')
