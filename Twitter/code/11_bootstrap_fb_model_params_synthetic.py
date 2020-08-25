from __future__ import division, print_function

import numpy as np
import pandas as pd
import argparse
import sys
from pandas import HDFStore
from sklearn.metrics import accuracy_score
from scipy.sparse import csr_matrix
from scipy.sparse import vstack, hstack, save_npz, load_npz
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
from sklearn.model_selection import GridSearchCV
from scipy.optimize import fmin_l_bfgs_b
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.extmath import safe_sparse_dot, log_logistic
#from sklearn.utils.fixes import expit
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
    # w[regularized_alphas] = 0.
    yz = y * safe_sparse_dot(X, w)

    # Logistic loss is the negative of the log of the logistic function.
    out = -np.sum(log_logistic(yz))
    # out += .5 * alpha * np.dot(w, w)  # L2

    w_extended_ = copy.copy(w_extended)
    # don't regularize \alphas
    if ignore2w == 0:
        reg_idx = list(range(1, idx))
        reg_idx2 = list(range(n_features+1, n_features + idx))
        w_extended_[reg_idx] = 0.
        w_extended_[reg_idx2] = 0.
    # model_user_event
    elif ignore2w == 1:
        reg_idx = list(range(idx, n_features))
        reg_idx2 = list(range(n_features + idx, 2 * n_features))
        w_extended_[reg_idx] = 0.
        w_extended_[reg_idx2] = 0.
    # model_user_event_fb
    else:
        reg_idx = list(range(1, idx)) + list(range(idx, n_features))
        reg_idx2 = list(range(n_features, n_features + idx)) + list(range(n_features + idx, 2 * n_features))
        # if len(regularized_alphas) > 0:
        #    unpenalized_idx = list(set(list(range(0, idx - 1))) - set(regularized_alphas)) + list(
        #        set(list(range(idx, n_features))) - set(idx + np.array(regularized_alphas)))
        #    penalized_idx = list(regularized_alphas) + list(idx + np.array(regularized_alphas))
        #    w_extended_[penalized_idx] = w_extended_[penalized_idx]*1000000
        #    w_extended_[unpenalized_idx] = 0.
        # else:
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
        out += C * w_extended_.sum() + 0.5 * D * Dsmooth_squared.sum()  # L1, w_extended is non-negative

        z = expit(yz)
        z0 = (z - 1) * y

        grad = safe_sparse_dot(X.T, z0)
        grad = np.concatenate([grad, -grad])

        # grad += alpha * w  # L2
        # grad += alpha +  # L1
        D_grad = np.zeros((n_features,))
        D_grad[idx:] = Dsmooth
        D_grad = np.concatenate([D_grad, -D_grad])
        grad += C - D * D_grad
    else:
        out += C + w_extended.sum()
        z = expit(yz)
        z0 = (z - 1) * y
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

        intercept = np.ones(X.shape[0]).reshape(X.shape[0], 1)
        X = hstack([intercept, X])
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


def classify_data(X_user, Y, data_name, no_topics):
    no_users = 3248# 1145 #2312 #1145
    no_topics = no_topics

    Y[Y == 0] = -1

    print(X_user.shape)
    print(Y.shape)

    if data_name == 'model_user_fb':
        clf_uef = LbfgsL1Logistic(C=1.0, D=0, tol=1e-8, no_topics=no_topics, smooth_start_index=no_users, ignore2w=0)
        #clf_uef = LogisticRegression(penalty='l1', fit_intercept=True)

    else:
        clf_uef = LbfgsL1Logistic(C=0.1, D = 10, tol=1e-8, no_topics=no_topics, smooth_start_index=no_users +1,
                              ignore2w=2)
        #clf_uef = LogisticRegression(C = 1.0, penalty='l1', fit_intercept=True)

    parameters = {'C': [0.01, 0.1, 1], 'D': [0.1, 1.0, 10]}
    clf = GridSearchCV(clf_uef, parameters, n_jobs=10, cv=10)
    clf.fit(X_user, Y)

    with open('columnnames2.txt') as f1:
        col_lines = f1.readlines()

    user_cols = {}
    u = 0
    for col in col_lines:
        if col.startswith('user_'):
            user_n = col.strip()
            user_cols[user_n[5:]] = u
            u += 1

    data_df = pd.read_csv('../data/regression_data/synthetic_prob_u_k_t.csv')
    users_of_posts = data_df[['user']].values.flatten()

    users_of_posts = np.array([user_cols[user] for user in users_of_posts], dtype='int')

    print('-----begin bootstrapping -----')
    no_bootstraps = 200
    alpha = 0.95

    n_samples = len(Y)
    print('# samples ', n_samples)
    samples = np.arange(n_samples)
    ss = ShuffleSplit(n_splits=no_bootstraps, test_size=0.25, random_state=42)
    npr.seed(23)
    n_percentile = 2
    bootstraps_alpha_perUser = np.zeros((no_bootstraps, no_users))
    # percentile_values = np.zeros((no_bootstraps, n_percentile+1))
    model_params = []
    k = -1
    for train_index, test_index in ss.split(samples):
        k += 1
        if k % 25 == 0:
            print('bootstrap, ', k)

        Xtrain_uef, ytrain_uef = X_user[train_index], Y[train_index]
        Xtest_uef, ytest_uef, users_uef = X_user[test_index], Y[test_index], users_of_posts[test_index]

        print('train model_user_event_fb')
        clf_uef.fit(Xtrain_uef, ytrain_uef)
        clf_coefs = clf_uef.coef_
        if data_name == 'model_user_fb':
            clf_coefs = clf_uef.coef_[0]
        print(clf_coefs[:10])
        alpha_coefs = clf_coefs[:no_users]  # coefficient for per-uer feedback estimate

        bootstraps_alpha_perUser[k] = alpha_coefs
        coefs = csr_matrix(clf_coefs)
        model_params.append(coefs)

    #np.savetxt('syn_bootstraps_alpha_perUser_' + data_name + '2.txt', bootstraps_alpha_perUser)
    model_params = vstack(model_params)
    save_npz('syn_bootstraps_coefs_' + data_name  + '2.npz', model_params)

    print(bootstraps_alpha_perUser)
    return bootstraps_alpha_perUser


if __name__ == '__main__':
    num_topics = 2

    data_name_suffix = ['model_user_fb', 'model_user_event_fb']
    #data_name_suffix = ['model_user_event_fb']

    Yfilename = 'outputY_LDA' + str(num_topics) + '.csv'
    Y = np.loadtxt(Yfilename)
    Y = Y.ravel()
    print(Y.shape)
    no_res_cols = 5

    for data_name in data_name_suffix:
        Xfile_name = 'featuresX_syn_' + data_name + '_LDA' + str(num_topics) + '.npz'
        X = load_npz(Xfile_name)

        t0 = time.time()
        classify_data(X, Y, data_name, no_topics=num_topics)
        t1 = time.time()
        print('time taken: ', t1 - t0)

