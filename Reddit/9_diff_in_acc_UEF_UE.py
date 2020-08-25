from __future__ import division, print_function

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
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

def _l1_logistic_loss_grad(w_extended, X, y, C, D, k, idx, ignore2w, regularized_alphas):
    # print(k)
    _, n_features = X.shape
    w = w_extended[:n_features] - w_extended[n_features:]
    #w[regularized_alphas] = 0.
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
        #reg_idx = list(range(0, idx-1)) + list(range(idx, n_features))
        #reg_idx2 = list(range(n_features, n_features + idx-1)) + list(range(n_features + idx, 2 * n_features))
        
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
    def __init__(self, tol=1e-3, C=1.0, D=1.0, no_topics=100, smooth_start_index=2, ignore2w=0, regularized_alphas = []):
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
        self.regularized_alphas = regularized_alphas

    def fit(self, X, y):
        n_samples, n_features = X.shape

        coef0 = np.zeros(2 * n_features)
        w, f, d = fmin_l_bfgs_b(_l1_logistic_loss_grad, x0=coef0, fprime=None,
                                pgtol=self.tol,
                                bounds=[(0, None)] * n_features * 2,
                                args=(X, y, self.C, self.D, self.no_topics, self.smooth_start_index, self.ignore2w, self.regularized_alphas))
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

def classify_data(X_base, X_user, Y, data_name = ''):
    no_users = 10915
    no_topics = 100

    clf_uef = LbfgsL1Logistic(C=0.1, D=10, tol=1e-8, no_topics=no_topics, smooth_start_index=2*no_users+1, ignore2w=2)
    #clf_base = LbfgsL1Logistic(C=0.1, D=10, tol=1e-8, no_topics=no_topics, smooth_start_index=no_users+1, ignore2w=1)

    with open('columnnames.txt') as f1:
        col_lines = f1.readlines()
                
    user_cols = {}
    u=0
    for col in col_lines:
        if col.startswith('user_'):
            user_n = col.strip()
            user_cols[user_n[5:]]=u
            u+=1

    data_df = pd.read_csv('regression_data/logit_reg_data.csv')
    users_of_posts = data_df[['user']].values.flatten()

    users_of_posts = np.array([user_cols[user] for user in users_of_posts], dtype='int')
    
    Y[Y == 0] = -1

    #### Get Train vs Test Split ######
    train_indexes, test_indexes = getTrainTestIndexes()
    X_train, X_test = X_user[train_indexes], X_user[test_indexes]
    y_train, y_test = Y[train_indexes], Y[test_indexes]

    print('-----begin bootstrapping -----')
    no_bootstraps = 200
    alpha = 0.95
    npr.seed(23)

    n_train = len(y_train)
    boot_samples = np.random.choice(train_indexes, (no_bootstraps, n_train))

    n_groups = 2
    diff_in_accuracies = np.zeros((no_bootstraps, n_groups))
    
    data_fb = np.loadtxt('bootstraps_alpha_perUser_pudt.txt')
    median_fb = np.median(data_fb, axis=0)
    err_ci = (np.percentile(data_fb, 99.5, axis=0)- np.percentile(data_fb, 0.5, axis=0))/2.0
    median_fb_pos = median_fb - err_ci
    median_fb_neg = median_fb + err_ci

    group_idxs = []
    pos_fb_idx = np.where(median_fb_pos > 0.0)[0]
    neg_fb_idx = np.where(median_fb_pos <= 0.0)[0]
    other_idx = list((set(list(range(no_users))) - set(pos_fb_idx)) - set(neg_fb_idx))
    #print(len(neg_fb_idx), len(pos_fb_idx))
    
    group_idxs.append(neg_fb_idx)
    #group_idxs.append(other_idx)
    group_idxs.append(pos_fb_idx)


    for k in range(no_bootstraps):
        if k % 25 == 0:
            print('bootstrap, ', k)
        train_index = list(boot_samples[k])
        Xtrain_uef, ytrain_uef = X_user[train_index], Y[train_index]
        Xtest_uef, ytest_uef, users_uef = X_user[test_indexes], Y[test_indexes], users_of_posts[test_indexes]
        print('train model_user_event_fb')
        clf_uef.fit(Xtrain_uef, ytrain_uef)
        coefs = copy.deepcopy(clf_uef.coef_)
        rel_coefs = copy.copy(coefs[:no_users])   #coefficient for per-uer feedback estimate
        
        for i in range(n_groups):
            sel_idxs = group_idxs[i]
            imp_user_idxs = [u for u, idx in enumerate(users_uef) if idx in sel_idxs]

            test_uef, testY = Xtest_uef[imp_user_idxs], ytest_uef[imp_user_idxs]

            clf_base = LbfgsL1Logistic(C=0.1, D=10, tol=1e-8, no_topics=no_topics, smooth_start_index=2 * no_users + 1,
                                      ignore2w=2, regularized_alphas=sel_idxs)
           
            c_Xtrain_uef = Xtrain_uef.tolil()
            c_Xtrain_uef = copy.deepcopy(c_Xtrain_uef)
            c_Xtrain_uef[:,sel_idxs] = 0.
            c_Xtrain_uef = c_Xtrain_uef.tocsr()
            clf_base.fit(c_Xtrain_uef, ytrain_uef)
            
            c_test_uef = copy.deepcopy(test_uef.tolil())
            c_test_uef[:,sel_idxs] = 0.
            c_test_uef = c_test_uef.tocsr()
            

            pred_base = clf_base.predict(c_test_uef)
            pred_uef = clf_uef.predict(test_uef)

            acc_base = accuracy_score(pred_base, testY)
            acc_uef  = accuracy_score(pred_uef, testY)

            diff_in_acc = acc_uef - acc_base
            
            diff_in_accuracies[k][i] = diff_in_acc
            
        print('diff in accuracy', diff_in_accuracies[k])
   
    np.savetxt('diff_in_accuracies_2G_pudt.txt', diff_in_accuracies)

    
    diff_in_accuracies_mean = np.nanmean(diff_in_accuracies, axis=0)
    diff_in_accuracies_mean = diff_in_accuracies_mean.flatten()

    p = ((1.0 - alpha) / 2.0) * 100
    lower_params = np.percentile(diff_in_accuracies, p, axis=0)
    p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    upper_params = np.percentile(diff_in_accuracies, p, axis=0)

    diff_in_accuracies_std = (upper_params - lower_params) / 2

    diff_in_accuracies_ci = diff_in_accuracies_std.flatten()
    diff_in_accuracies_std = 1.96 * np.nanstd(diff_in_accuracies, axis=0)
    diff_in_accuracies_std = diff_in_accuracies_std.flatten()

    print('mean diff in acc ', diff_in_accuracies_mean)
    print('ci diff in acc ', diff_in_accuracies_ci)
    print('std diff in acc ', diff_in_accuracies_std)
    #print('#CV', sp)

    data_to_save = np.array([diff_in_accuracies_mean, diff_in_accuracies_ci, diff_in_accuracies_std])
    np.savetxt('diff_in_accuracies_2groups_pudt.txt', data_to_save)

    return diff_in_accuracies_mean





if __name__ == '__main__':
    no_time_bins_per_day = 1
    print('Number of bins per day ', no_time_bins_per_day)

    data_name_suffix = ['model_user_event', 'model_user_event_fb_pudt']

    Yfilename = 'outputY100.csv'
    Y = np.loadtxt(Yfilename)
    Y = Y.ravel()
    print(Y.shape)
    no_res_cols = 6

    Xfile_name = 'featuresX_'+data_name_suffix[0]+'.npz'
    X_base  =  load_npz(Xfile_name)

    Xfile_name = 'featuresX_' + data_name_suffix[1] + '.npz'
    X_user = load_npz(Xfile_name)

    t0 = time.time()
    classify_data(X_base, X_user, Y)
    t1 = time.time()
    print('time taken: ', t1 - t0)

