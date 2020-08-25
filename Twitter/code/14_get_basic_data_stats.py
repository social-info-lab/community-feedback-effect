"""
    Write to file the IV, normal regression and EXO inference data for each tweet.
    @author: 2018 David Adelani, <dadelani@mpi-sws.org> and contributors.
"""
from __future__ import division, print_function

import numpy as np
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
from numpy.random import shuffle
import json

start_date = datetime.datetime.fromtimestamp(float('1451606400'))


def preprocessTweets(path, feedback_data_path, topic_model_dir):
    tweet_path = path + "/" + feedback_data_path

    with open(tweet_path + "/user_tweet_size.csv") as f:
        users_lines = f.readlines()
    list_of_users = [line.split(',')[0] + '.csv' for line in users_lines]
    sorted_users = sorted(list_of_users)
    anony_userIds = dict([(user, 'user_' + str(u)) for u, user in enumerate(sorted_users)])

    user_data = {}

    for user_file in anony_userIds:
        # first path is where the topics of tweets are located
        # second path is where the feedback data are located
        # tweets_feats is a list of (tweet_arr_time, retweet_cnt, favorite_cnt, tweet_stem)
        # tweet_arr_time needs to be converted to float i.e difference in seconds between the start_date timestamp and the tweet_arr_time.

        times = []
        tweet_ids = []
        rt_data_avail_seq = []
        n_feedback = []
        n_feedback_dt = []
        per_fb_dt = []
        per_fb = []
        ranked_fbs = []
        topic_seq = []
        favorites = []
        final_fbs = []

        with open(
                "../../../Twitter-LDA/" + topic_model_dir + "/src/data/ModelRes/test/TextWithLabel/" + user_file) as f1_top, \
                open(tweet_path + "/" + user_file) as f2:
            tweets_topic = f1_top.readlines()
            tweets_feats = f2.readlines()

        for line in tweets_topic:
            tokens = re.split(r'[:=]+', line)
            topic = int(tokens[2])
            topic_seq.append(topic)

        if (len(topic_seq) != len(tweets_feats[1:])): continue

        # timestamp,tweet_id,isRetweetersDataAvail?,numFb,numFb_dt,percent_numFb_dt,percentile_numFb,feedback_rank, favorites

        for i, evt in enumerate(tweets_feats[1:]):
            evt_data = evt.strip().split(',')
            t_n = float(evt_data[0])
            date_time = datetime.datetime.fromtimestamp(t_n)
            tweet_time = ((date_time - start_date).total_seconds()) / 86400.0

            if tweet_time >= 182: continue
            tweet_id = evt_data[1]
            isRT_available = int(evt_data[2])

            fb = float(evt_data[3])
            fb_dt = float(evt_data[4])
            p_fb_dt = float(evt_data[5])
            m_fb_dt = float(evt_data[6])
            ranked_fb = int(float(evt_data[7]))
            # topic = evt_data[8]
            favorite_cnt = int(evt_data[8])
            final_cnt = float(evt_data[9])

            times.append(tweet_time)
            tweet_ids.append(tweet_id)
            rt_data_avail_seq.append(isRT_available)
            n_feedback.append(fb)
            n_feedback_dt.append(fb_dt)
            per_fb_dt.append(p_fb_dt)
            per_fb.append(m_fb_dt)
            ranked_fbs.append(ranked_fb)
            favorites.append(favorite_cnt)
            final_fbs.append(final_cnt)

        user_id = anony_userIds[user_file]
        user_data[user_id] = (times, tweet_ids, rt_data_avail_seq, n_feedback, n_feedback_dt, per_fb_dt,
                              per_fb, ranked_fbs, topic_seq, favorites, final_fbs)

    print('# of users in the Twitter data: ', len(user_data))

    return user_data


##################################################################################################################
# check if the current and next action (topic, action) are the same
##################################################################################################################
def get_for_same_consecutive_action(curr, next, action, rt_avail):
    if action == 'topic':
        same_action = (curr == next)
        is_rt_avail = rt_avail == 1
    elif action == 'hashtag':
        same_action = (len(curr) >= 1 and len(list(set(curr) & set(next))) > 0)
        is_rt_avail = (rt_avail == 1 and len(curr) >= 1)
    else:  # licw category
        same_action = (curr >= 1 and next >= 1)
        is_rt_avail = (rt_avail == 1 and curr >= 1)

    return is_rt_avail, same_action


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


##################################################################################################################
# get rank data for a user.
##################################################################################################################
def getRankData(retweets, bins, recomputeStatus):
    rt_bin = -np.ones((len(retweets),))
    inds = np.where(np.array(recomputeStatus) == 1)[0]
    retweets_bin_t = np.array(retweets)[inds]
    rt_cdf = stats.rankdata(retweets_bin_t, "average") / len(retweets_bin_t)
    rt_bin[inds] = bin_data(rt_cdf, bins)

    return rt_bin


def logit(x):
    return np.log(x / (1 - x))


def write_microAverageData(source_dir, user_data, num_topics):
    num_users = len([user for user in user_data if len(user_data[user][0]) >= 50])

    f_cnt = 0
    r_cnt = 0
    user_con = 0
    no_posts_with_fb = 0
    avg_fb = []
    avg_nposts = []
    perUser_probTopic = {}
    median_post_time = {}

    num_of_posts_by_timebin_topic = defaultdict(lambda: defaultdict(int))

    macro_avg = []
    n_rep_all = 0
    for u, user in enumerate(user_data.keys()):

        times, tweet_ids, rt_data_avail_seq, n_feedback, n_feedback_dt, per_fb_dt, per_fb, \
        ranked_fbs, topic_seq, favorites, final_fbs = user_data[user]

        if len(n_feedback) < 50: continue
        # ----user attributes ----
        per_fb = np.array(per_fb)
        idxs_with_fb = np.where(np.array(per_fb) >= 0)[0]
        avg_nFb = np.mean(per_fb[idxs_with_fb])
        sum_nFb = np.sum(per_fb[idxs_with_fb])
        n_topics_used = len(np.unique(topic_seq))
        no_posts = len(topic_seq)
        avg_fb.append(np.mean(n_feedback))
        avg_nposts.append(no_posts)

        # ---forward pre-process ---
        dt = np.array(list(np.diff(times)) + [1.0]) * 60 * 60 + 0.000001
        final_fbs = np.array(final_fbs)
        idxs_no_fb = np.where(final_fbs < 0)[0]
        final_fbs[idxs_no_fb] = np.nan
        non_null_len = len(np.where(~np.isnan(final_fbs))[0])
        ranked_final_fb = (pd.Series(final_fbs / dt).rank().values) / non_null_len

        fav_seq = pd.Series(np.array(favorites) / dt).rank(pct=True).values
        # dt = np.diff(times)
        dt_bins = dt
        ranked_fbs = np.array(ranked_fbs) / 100.0  # rescale from percentage to [0,1]
        per_fb = np.array(per_fb)
        some_nos = np.copy(range(len(ranked_fbs)))
        np.random.shuffle(some_nos)
        some_nos = some_nos.astype('int')
        # per_fb = per_fb[some_nos]

        median_post_time[u] = np.median(dt)
        num_tweets = len(tweet_ids)


        per_fb_dt = np.array(per_fb_dt)

        # probability of topic for each user
        count_topic = Counter(topic_seq)
        N = sum(count_topic.values())
        K = num_topics  # number of topics
        # Lidstone smoothing
        prob_topic = dict([(z, float(cnt + 1) / (N + K)) for z, cnt in count_topic.items()])
        log_prob_topic = dict([(z, logit(float(cnt + 1) / (N + K))) for z, cnt in count_topic.items()])
        log_prob_topic['oov'] = logit(1.0 / (N + K))

        perUser_probTopic['user_' + str(u)] = log_prob_topic

        no_of_topicRep = defaultdict(int)
        # data are considered for the shuffled data
        per_user_events = []
        k = 0
        for i in range(1, len(tweet_ids) - 1):
            # forward

            if n_feedback_dt[i] < 0: continue
            #if n_feedback[i] > 0: no_posts_with_fb += 1
            no_of_topicRep[topic_seq[i]] += 1
            _, istopicRepeated = get_for_same_consecutive_action(topic_seq[i], topic_seq[i + 1],
                                                                 'topic', rt_data_avail_seq[i])

            V = len(Counter(no_of_topicRep))
            prob_t_R = float(no_of_topicRep[topic_seq[i]] + 1) / (i + 1 + V)

            fb_rank = round(ranked_fbs[i], 1)

            event_time_bin = int(times[i])  # int(times[i] * no_time_bins_per_day)

            event_f = [times[i], n_feedback[i], n_feedback_dt[i], per_fb_dt[i], per_fb[i], fb_rank, dt_bins[i - 1],
                       dt_bins[i], logit(prob_topic[topic_seq[i]]), logit(prob_t_R), user, int(istopicRepeated),
                       1.0 / float(num_tweets), fav_seq[i], ranked_final_fb[i], topic_seq[i], event_time_bin
                       ]

            if rt_data_avail_seq[i] == 1:
                per_user_events.append(event_f)
                k += 1

        if u % 1000 == 0: print(u)

        if k >= 50:
            fbs = []
            user_con += 1

            n_rep = 0
            for event_f in per_user_events:
                f_cnt+=1
                if event_f[1] > 0: no_posts_with_fb += 1
                fbs.append(event_f[1])

                topic = event_f[-2]
                evt_tb = event_f[-1]
                num_of_posts_by_timebin_topic[topic][evt_tb] += 1

                if event_f[11]==1:
                    n_rep+=1
                    n_rep_all+=1

            macro_avg.append(n_rep/float(k))

            avg_nposts.append(k)
            avg_fb.append(np.mean(fbs))

    print('No of tweets (Forward direction): ', f_cnt)
    print('Number of users considered ', user_con)
    print('No of posts with fb: ', no_posts_with_fb)
    print('Average amt of fb:', np.mean(avg_fb))
    print('Average no of posts:', np.mean(avg_nposts))
    print('Macro Prob topic Rep:', np.mean(macro_avg))
    print('Micro Prob topic Rep:', n_rep_all/f_cnt)

    import json

    with open('perUser_probTopic.json', 'w') as fp:
        json.dump(perUser_probTopic, fp)

    with open('median_post_time.json', 'w') as fp:
        json.dump(median_post_time, fp)

    for p in [10,20,30,40,50,60,70,80,90, 95, 99, 100]:
        print('percentile: ', p, np.percentile(avg_nposts, p))

    with open('num_of_posts_by_timebin_topic.json', 'w') as fp:
        json.dump(num_of_posts_by_timebin_topic, fp)

def save_train_indexes():
    df = pd.read_csv(source_dir + "/regression_data/logit_reg_data_" + str(num_topics) + ".csv")
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
    feedback_data_path = 'ranked_feedback_data'

    suffix_name = feedback_data_path.split('_')[-1]

    pos_path = os.getcwd().rfind('/')
    dir = os.getcwd()
    source_dir = dir[:pos_path + 1] + "data"

    parser = argparse.ArgumentParser(description='create logistic regression data with LDA inferred topics')
    parser.add_argument('---num_lda_topics', type=int,
                        help='number of topics required for topic modeling: 50, 100 or 200')

    args = parser.parse_args()

    if len(sys.argv) < 2:
        print('please, specify the number of topics used for LDA topic modeling: 50, 100 or 200, for example'
              ' \n python 4_create_logReg_dataset.py ---num_lda_topics 100')
        sys.exit(1)  # abort because of error

    if args.num_lda_topics == 50 or args.num_lda_topics == 200:
        num_topics = args.num_lda_topics
        topic_model_dir = 'Twitter-LDA' + str(num_topics)
    else:
        num_topics = 100
        topic_model_dir = 'Twitter-LDA'

    user_data = preprocessTweets(source_dir, feedback_data_path, topic_model_dir)

    print('preprocessing done! ')

    # print out micro average data
    print('write logistic regression data')
    write_microAverageData(source_dir, user_data, num_topics)

    #save_train_indexes()

    #get_frac_users_post_in_sampled_data()