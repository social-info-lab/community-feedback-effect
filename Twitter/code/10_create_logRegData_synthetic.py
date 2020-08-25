"""
    Write to file the IV, normal regression and EXO inference data for each tweet.
    @author: 2018 David Adelani, <dadelani@mpi-sws.org> and contributors.
"""
from __future__ import division, print_function

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.extmath import safe_sparse_dot, log_logistic
import copy

import os
import re
import csv
import sys
import datetime
import argparse
from scipy import stats
import pandas as pd
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix
from scipy.sparse import vstack, hstack, save_npz
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from numpy.random import shuffle


def build_logit_data(source_dir):
    from sklearn import preprocessing
    df = pd.read_csv(source_dir + "/regression_data/synthetic_prob_u_k_t.csv")
    # df = df.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    #timestamp, user, topic, prob_topic, event_time_bin, g_k_t, prob_k_t, feedback, prob_u_k_t
    #5.78703703704e-05, user_4597, 0, -0.140088793171, 0, 0.0, 0, -5.232336522945216, 0

    df_data = df.drop(['timestamp'], axis=1)
    df_data['prob_topic'] = preprocessing.scale(df_data['prob_topic'].values)

    all_users = Counter(list(df_data[['user']].values.flatten()))
    no_user = len(all_users)
    no_samples = df_data.shape[0]
    print('Number of posts considered ', no_samples)
    print('Number of users considered ', no_user)
    all_topics = Counter(list(df_data[['topic']].values.flatten()))
    no_topic = len(all_topics)
    all_time_bins = Counter(list(df_data[['event_time_bin']].values.flatten()))
    no_time = len(all_time_bins)
    print('(user, topic, time): ', no_user, no_topic, no_time)

    df_data[['topic']] = df_data[['topic']].astype(object)
    df_data[['event_time_bin']] = df_data[['event_time_bin']].astype(object)
    df_data = pd.get_dummies(df_data)

    print(df_data.head())

    columnnames = list(df_data.columns.values)
    print(columnnames[:10])

    no_nonCat_features = 0  # i.e number of non-categorical features
    for col in columnnames:
        if col.startswith('user_'):
            break
        no_nonCat_features += 1

    print('# of non categorical features: ', no_nonCat_features)
    np.savetxt('columnnames2.txt', columnnames, fmt="%s")

    return df_data, no_user, no_samples, no_nonCat_features


def expound_features(all_df_data, chunk_start, chunk_stop, no_user, num_topics, no_nonCat_features, data_name=''):
    no_users = no_user
    no_topics = num_topics

    df_data = all_df_data.iloc[chunk_start:chunk_stop]

    no_fixed_features = no_nonCat_features
    no_time_bins = len(list(df_data.columns.values)[(no_fixed_features + no_users + no_topics):])

    df_data = df_data.values


    fb = df_data[:, 3]
    prob_topic = df_data[:, [0]].ravel()

    y = df_data[:, [4]].ravel()

    start_index = no_fixed_features
    end_index = no_fixed_features + no_users
    user_fields = df_data[:, list(range(start_index, end_index))]


    ncol_user = user_fields.shape[1]
    per_user_feedback_put = user_fields * np.transpose(np.array([fb, ] * ncol_user))

    if data_name.startswith('syn_model_user_event'):
        start_index = end_index
        end_index = end_index + no_topics
        topic_fields = df_data[:, list(range(start_index, end_index))]

        start_index = end_index
        end_index = end_index + no_time_bins
        time_fields = df_data[:, list(range(start_index, end_index))]

        ncol = time_fields.shape[1]

        time_by_topic = []

        for k in range(topic_fields.shape[1]):
            # element-wise multiplying a matrix by a vector
            res = time_fields * np.transpose(np.array([topic_fields[:, k], ] * ncol))
            time_by_topic.append(res)

        time_by_topic = np.column_stack(time_by_topic)

    if data_name == 'syn_model_user_fb':
        X = per_user_feedback_put
    elif data_name == 'syn_model_user_event_fb':
        X = np.column_stack((per_user_feedback_put, time_by_topic))

    return X, y


# export_data(df_data, no_user, num_topics, no_samples, data_name='model_c0')
def export_data(df_data, no_user, num_topics, no_train_samples, no_nonCat_features, data_name=''):
    no_samples = no_train_samples
    batch_size = 5000

    chunk_indexes = [i for i in range(0, no_samples, batch_size)] + [no_samples]

    all_X = []
    all_Y = []
    for k in range(len(chunk_indexes) - 1):
        if k % 100 == 0:
            print(k)
        X, y = expound_features(df_data, chunk_indexes[k], chunk_indexes[k + 1], no_user, num_topics,
                                no_nonCat_features, data_name)

        X = csr_matrix(X)

        all_X.append(X)
        all_Y.append(y)

    all_X = vstack(all_X)

    all_Y = np.concatenate(all_Y)
    all_Y = all_Y.flatten().ravel()

    feature_filename = 'featuresX_' + data_name + '_LDA' + str(num_topics) + '.npz'

    save_npz(feature_filename, all_X)

    np.savetxt('outputY_LDA' + str(num_topics) + '.csv', all_Y)

    print('Dataset ' + feature_filename + ' exported')


if __name__ == '__main__':
    feedback_data_path = 'ranked_feedback_data'

    suffix_name = feedback_data_path.split('_')[-1]

    pos_path = os.getcwd().rfind('/')
    dir = os.getcwd()
    source_dir = dir[:pos_path + 1] + "data"

    df_data, no_user, no_train_samples, no_nonCat_features = build_logit_data(source_dir)

    num_topics = 2

    export_data(df_data, no_user, num_topics, no_train_samples, no_nonCat_features, data_name='syn_model_user_fb')
    export_data(df_data, no_user, num_topics, no_train_samples, no_nonCat_features, data_name='syn_model_user_event_fb')