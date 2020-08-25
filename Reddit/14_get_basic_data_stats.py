"""
    Write to file the IV, normal regression and EXO inference data for each tweet.
    @author: 2018 David Adelani, <dadelani@mpi-sws.org> and contributors.
"""
from __future__ import division, print_function

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.extmath import safe_sparse_dot, log_logistic
# from sklearn.utils.fixes import expit
import copy

import os
import csv
import datetime
from scipy import stats
import pandas as pd
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix
from scipy.sparse import vstack, hstack, save_npz
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from numpy.random import shuffle
import json

start_date = datetime.datetime.fromtimestamp(float('1451606400'))


def preprocessTweets(path):
    user_data = {}
    for user_file in sorted(
            os.listdir(path)):
        if user_file == "user_tweet_size.csv": continue
        if not user_file.endswith('.csv'): continue
        with open(path + "/" + user_file) as text:
            tweets = text.readlines()

        times = []
        post_ids = []
        n_feedback = []
        n_feedback_dt = []
        per_fb_dt = []
        per_fb = []
        ranked_fbs = []
        scores = []
        final_no_comments = []
        topic_seq = []

        # timestamp, post_id, numFb, numFb_dt, percent_numFb_dt, median_numFb_dt, feedback_rank, subreddit_id

        for i, evt in enumerate(tweets[1:]):
            evt_data = evt.strip().split(',')
            if len(evt_data) < 3: continue
            t_n = float(evt_data[0])
            date_time = datetime.datetime.fromtimestamp(t_n)
            tweet_time = float((date_time - start_date).total_seconds()) / 86400.0
            # print(tweet_time)

            if tweet_time > 0 and tweet_time < 182:
                post_id = evt_data[1]
                fb = float(evt_data[2])
                fb_dt = float(evt_data[3])
                p_fb_dt = float(evt_data[4])
                m_fb_dt = float(evt_data[5])
                ranked_fb = int(float(evt_data[6]))
                score = int(float(evt_data[7]))
                final_comment = float(evt_data[8])
                topic = evt_data[9]

                times.append(tweet_time)
                post_ids.append(post_id)
                n_feedback.append(fb)
                n_feedback_dt.append(fb_dt)
                per_fb_dt.append(p_fb_dt)
                per_fb.append(m_fb_dt)
                ranked_fbs.append(ranked_fb)
                topic_seq.append(topic)
                scores.append(score)
                final_no_comments.append(final_comment)

        user_data[user_file[:-4]] = (times, post_ids, n_feedback, n_feedback_dt, per_fb_dt, per_fb, ranked_fbs,
                                     scores, final_no_comments, topic_seq)

    return user_data


##################################################################################################################
# check if the current and next action (topic, action) are the same
##################################################################################################################
def get_for_same_consecutive_action(curr, next, action):
    if action == 'topic':
        same_action = (curr == next)
    elif action == 'hashtag':
        same_action = (len(curr) >= 1 and len(list(set(curr) & set(next))) > 0)
    else:  # licw category
        same_action = (curr >= 1 and next >= 1)

    return same_action


##################################################################################################################
# -------------------bin sequence of data based on a criterion------------------------------------------
##################################################################################################################
def bin_data(seq, bins):
    # input: seqs
    # bins : criteria to bin the data e.g bin ranked data into quartiles where bins = [-0.1, 0.25, 0.5, 0.75, 1.0]
    # replace values between (-0.1, 0.25) by 0, (0.25, 0.5) by 1,...
    # seq_new = np.copy(seq)
    seq = np.array(seq)
    seq_new = -np.ones(seq.shape, dtype=int)
    for j, bin in enumerate(bins[:-1]):
        seq_new[(seq > bins[j]) & (seq <= bins[j + 1])] = j

    seq_new = seq_new.astype(int)
    return seq_new


def logit(x):
    return np.log(x / (1 - x))


def write_microAverageData(source_dir, user_data, no_time_bins_per_day):
    num_users = len([user for user in user_data if len(user_data[user][0]) >= 50])


    f_cnt = 0
    r_cnt = 0
    user_con = 0
    no_posts_with_fb = 0
    avg_fb = []
    avg_nposts = []

    perUser_probTopic = {}
    username2id = {}
    median_post_time = {}
    num_of_posts_by_timebin_topic = defaultdict(lambda: defaultdict(int))

    macro_avg = []
    n_rep_all = 0

    for u, user in enumerate(user_data.keys()):

        times, post_ids, n_feedback, n_feedback_dt, per_fb_dt, per_fb, ranked_fbs, \
        scores, final_no_comments, topic_seq = user_data[user]

        if len(n_feedback) < 50: continue
        # ----user attributes ----
        per_fb = np.array(per_fb)
        idxs_with_fb = np.where(np.array(per_fb) >= 0)[0]
        avg_nFb = np.mean(per_fb[idxs_with_fb])
        sum_nFb = np.sum(per_fb[idxs_with_fb])
        n_topics_used = len(np.unique(topic_seq))
        no_posts = len(topic_seq)
        #avg_fb.append(np.mean(n_feedback))
        #avg_nposts.append(no_posts)

        # ---forward pre-process ---
        dt = np.array(list(np.diff(times)) + [1.0]) * 60 * 60 + 0.000001
        # print(np.where(dt < 0)[0])
        dt_bins = dt  # model.bin_data(dt, int_arr_times)
        ranked_fbs = np.array(ranked_fbs) / 100.0  # rescale from percentage to [0,1]
        idx_not_0 = np.array(np.where(np.array(per_fb) >= 0)[0])
        per_fb = np.array(per_fb)
        # print(per_fb[:10])
        some_nos = np.copy(range(len(idx_not_0)))
        np.random.shuffle(some_nos)
        # some_nos = some_nos.astype('int')
        # s_per_fb = per_fb[idx_not_0][some_nos]

        final_no_comments = np.array(final_no_comments)
        idxs_no_fb = np.where(final_no_comments < 0)[0]
        final_no_comments[idxs_no_fb] = np.nan
        non_null_len = len(np.where(~np.isnan(scores))[0])
        ranked_final_fb = (pd.Series(final_no_comments / dt).rank().values) / non_null_len

        scores = np.array(scores)
        ranked_Scores = pd.Series(scores / dt).rank(pct=True).values

        num_tweets = len(post_ids)
        user_con += 1
        username2id[user] = u
        median_post_time[u] = np.median(dt)

        # probability of topic for each user
        count_topic = Counter(topic_seq)
        N = sum(count_topic.values())
        K = 100  # number of topics
        # Lidstone smoothing
        prob_topic = dict([(z, float(cnt + 1) / (N + K)) for z, cnt in count_topic.items()])
        log_prob_topic = dict([(z, logit(float(cnt + 1) / (N + K))) for z, cnt in count_topic.items()])
        log_prob_topic['oov'] = logit(1.0 / (N + K))
        perUser_probTopic['user_' + str(u)] = log_prob_topic

        no_of_topicRep = defaultdict(int)
        k = 0
        per_user_events = []
        for i in range(1, len(post_ids) - 1):

            # forward
            if per_fb[i] < 0:
                continue

            istopicRepeated = get_for_same_consecutive_action(topic_seq[i], topic_seq[i + 1], 'topic')

            no_of_topicRep[topic_seq[i]] += 1
            fb_rank = round(ranked_fbs[i], 1)

            V = len(Counter(no_of_topicRep))
            prob_t_R = float(no_of_topicRep[topic_seq[i]] + 1) / (i + 1 + V)

            event_time_bin = int(times[i] * no_time_bins_per_day)
            event_f = [post_ids[i], n_feedback[i], n_feedback_dt[i], per_fb_dt[i], per_fb[k], fb_rank,
                       dt_bins[i - 1], dt_bins[i], logit(prob_topic[topic_seq[i]]), logit(prob_t_R),
                       'user_' + str(u),
                       int(istopicRepeated), 1.0 / float(num_tweets), ranked_Scores[i], ranked_final_fb[i],
                       topic_seq[i], event_time_bin
                       ]
            k += 1
            per_user_events.append(event_f)


        if u % 1000 == 0: print(u)

        if k >= 50:
            fbs = []
            user_con += 1

            n_rep = 0
            for event_f in per_user_events:
                f_cnt += 1
                if event_f[1] > 0: no_posts_with_fb += 1
                fbs.append(event_f[1])

                topic = event_f[-2]
                evt_tb = event_f[-1]
                num_of_posts_by_timebin_topic[topic][evt_tb]+=1

                if event_f[11]==1:
                    n_rep+=1
                    n_rep_all+=1

            macro_avg.append(n_rep / float(k))

            avg_nposts.append(k)
            avg_fb.append(np.mean(fbs))

    print('No of tweets (Forward direction): ', f_cnt)
    print('No of posts with fb: ', no_posts_with_fb)
    print('average number of posts:', np.mean(avg_nposts))
    print('average number of fb:', np.nanmean(avg_fb))
    print('Macro Prob topic Rep:', np.mean(macro_avg))
    print('Micro Prob topic Rep:', n_rep_all / f_cnt)

    for p in [10,20,30,40,50,60,70,80,90,100]:
        print('percentile: ', p, np.percentile(avg_nposts, p))


    with open('num_of_posts_by_timebin_topic.json', 'w') as fp:
        json.dump(num_of_posts_by_timebin_topic, fp)

    #np.savetxt('num_of_posts_by_timebin_topic.csv', num_of_posts_by_timebin_topic)

def save_train_indexes():
    df = pd.read_csv("regression_data/logit_reg_data.csv")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(how="all")

    ### Train/Test Split ###
    df['index'] = list(range(df.shape[0]))
    users_set = set(list(df['user'].values))
    print('# of users: ', len(users_set))

    print('Before filtering: ', df.shape)

    train_idxs =  {}
    avail_users = 0
    for u, user in enumerate(list(users_set)):
        if u%1000==0:
            print(u, user)
        user_posts_idxs = list(df[df['user'] == user]['index'].values)
        test_start = len(user_posts_idxs) - 3
        if len(user_posts_idxs) >= 50:
            per_user_idxs = sorted(user_posts_idxs[:test_start])
            per_user_idxs = [str(i) for i in per_user_idxs]
            train_idxs[user] = per_user_idxs
            avail_users+=1
        else:
            #train_idxs +=user_posts_idxs
            df = df.drop(df[df['user']==user].index)

    print(train_idxs[user])

    print('# users in test set ', avail_users)
    print('After filtering: ', df.shape)

    with open('train_indexes.json', 'w') as fp:
        json.dump(train_idxs, fp)


def getTrainTestIndexes():
    with open('test_indexes.txt') as f1, open('train_indexes.txt') as f2:
        test_lines = f1.readlines()
        train_lines = f2.readlines()

    test_indexes = [int(indx.strip()) for indx in test_lines]
    train_indexes = [int(indx.strip()) for indx in train_lines]

    return train_indexes, test_indexes

def get_frac_users_post_in_sampled_data():
    no_users = 7072 #10915 #10921
    no_topics = 100

    with open('train_indexes.json', 'r') as fp:
        user_indexes_dict = json.load(fp)

    train_indexes, test_indexes = getTrainTestIndexes()

    users_dict = dict([(user, u) for u, user in enumerate(user_indexes_dict.keys())])

    print('-----begin bootstrapping -----')
    no_bootstraps = 200

    n_train = len(train_indexes)
    boot_samples = np.random.choice(train_indexes, (no_bootstraps, n_train))

    num_posts_boot_user = np.zeros((no_bootstraps, len(user_indexes_dict)))

    for k in range(no_bootstraps):
        #if k % 25 == 0:
        print('bootstrap, ', k)
        train_index = list(boot_samples[k])

        for user in user_indexes_dict:
            user_indexes = np.array(user_indexes_dict[user])
            user_indexes = user_indexes.astype(int)
            min_index, max_index = user_indexes[0], user_indexes[-1]
            subset_indexes = user_indexes[(user_indexes>=min_index) & (user_indexes<=max_index)]

            num_sample_posts = len(set(subset_indexes).intersection(train_index)) #len(set(subset_indexes) & set(train_index))

            num_posts_boot_user[k][users_dict[user]] = num_sample_posts

    np.savetxt('num_posts_sample_user.txt', num_posts_boot_user)

if __name__ == '__main__':
    source_dir = "user_data"
    num_topics = 100
    user_data = preprocessTweets(source_dir)

    print('preprocessing done! ')

    # print out micro average data
    print('write logistic regression data')
    no_time_bins_per_day = 1
    write_microAverageData(source_dir, user_data, no_time_bins_per_day)

    #save_train_indexes()

    #get_frac_users_post_in_sampled_data()

