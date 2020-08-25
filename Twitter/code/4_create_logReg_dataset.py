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

start_date = datetime.datetime.fromtimestamp(float('1451606400'))


def preprocessTweets(path, feedback_data_path, topic_model_dir):

    tweet_path = path + "/" + feedback_data_path

    with open(tweet_path + "/user_tweet_size.csv") as f:
        users_lines = f.readlines()
    list_of_users = [line.split(',')[0] + '.csv' for line in users_lines]
    sorted_users = sorted(list_of_users)
    anony_userIds = dict([(user, 'user_'+str(u)) for u, user in enumerate(sorted_users)])

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

        with open("../../../Twitter-LDA/"+topic_model_dir+"/src/data/ModelRes/test/TextWithLabel/" + user_file) as f1_top, \
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
            #topic = evt_data[8]
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
    return np.log(x/(1-x))

def write_microAverageData(source_dir, user_data, num_topics):
    num_users = len([user for user in user_data if len(user_data[user][0]) >= 50])

    ff = open(source_dir + "/regression_data/logit_reg_data_"+str(num_topics)+".csv", 'w')
    fg = open("user_attributes.csv", 'w')


    try:
        writer_f = csv.writer(ff, dialect='excel')

        column_names = ['timestamp', 'numFb', 'numFb_dt', 'percent_numFb_dt', 'percent_numFb', 'log_numFb', 'prevDelay',
                        'delay_btw_tweets',  'prob_topic', 'prob_topic_rep', 'user', 'isTopicRepeated',
                        'weights', 'favorite', 'log_numFbdt', 'topic', 'event_time_bin'
                        ]
    
        writer_s = csv.writer(fg, dialect='excel')
        writer_s.writerow(['user', 'AvgPerFb', 'SumFb' 'noPosts', 'noTopics'])
        writer_f.writerow(column_names)
        f_cnt = 0
        r_cnt = 0
        user_con= 0
        no_posts_with_fb = 0
        avg_fb = []
        avg_nposts = []
        perUser_probTopic = {}
        median_post_time = {}

        for u, user in enumerate(user_data.keys()):

            times, tweet_ids, rt_data_avail_seq, n_feedback, n_feedback_dt, per_fb_dt, per_fb, \
            ranked_fbs, topic_seq, favorites, final_fbs = user_data[user]

            if len(n_feedback) < 50: continue
            # ----user attributes ----
            per_fb = np.array(per_fb)
            idxs_with_fb = np.where(np.array(per_fb)>=0)[0]
            avg_nFb = np.mean(per_fb[idxs_with_fb])
            sum_nFb = np.sum(per_fb[idxs_with_fb])
            n_topics_used = len(np.unique(topic_seq))
            no_posts = len(topic_seq)
            avg_fb.append(np.mean(n_feedback))
            avg_nposts.append(no_posts)


            writer_s.writerow([user, avg_nFb, sum_nFb, no_posts, n_topics_used])


            # ---forward pre-process ---
            dt = np.array(list(np.diff(times)) + [1.0]) * 60 * 60 +0.000001
            final_fbs = np.array(final_fbs)
            idxs_no_fb = np.where(final_fbs < 0)[0]
            final_fbs[idxs_no_fb] = np.nan
            non_null_len = len(np.where(~np.isnan(final_fbs))[0])
            ranked_final_fb = (pd.Series(final_fbs/dt).rank().values) / non_null_len

            fav_seq = pd.Series(np.array(favorites)/dt).rank(pct=True).values
            #dt = np.diff(times)
            dt_bins = np.array(list(np.diff(times)) + [86400.0]) * 60 * 60 * 24
            ranked_fbs = np.array(ranked_fbs) / 100.0  # rescale from percentage to [0,1]
            per_fb = np.array(per_fb)
            some_nos = np.copy(range(len(ranked_fbs)))
            np.random.shuffle(some_nos)
            some_nos = some_nos.astype('int')
            #per_fb = per_fb[some_nos]

            median_post_time[u] = np.median(dt)
            num_tweets = len(tweet_ids)
            user_con += 1


            per_fb_dt = np.array(per_fb_dt)

            # probability of topic for each user
            count_topic = Counter(topic_seq)
            N = sum(count_topic.values())
            K = num_topics  # number of topics
            # Lidstone smoothing
            prob_topic = dict([(z, float(cnt + 1) / (N + K)) for z, cnt in count_topic.items()])
            log_prob_topic = dict([(z, logit(float(cnt + 1)/(N+K))) for z, cnt in count_topic.items()])
            log_prob_topic['oov'] = logit(1.0/(N+K))
            
            perUser_probTopic['user_'+str(u)] = log_prob_topic

            no_of_topicRep = defaultdict(int)
            # data are considered for the shuffled data
            per_user_events = []
            k = 0
            for i in range(1, len(tweet_ids) - 1):
                # forward

                if n_feedback_dt[i] < 0: continue
                if dt_bins[i] < 30: continue
                if n_feedback[i] > 0: no_posts_with_fb +=1
                no_of_topicRep[topic_seq[i]] += 1
                _, istopicRepeated = get_for_same_consecutive_action(topic_seq[i], topic_seq[i + 1],
                                                                     'topic', rt_data_avail_seq[i])

                V = len(Counter(no_of_topicRep))
                prob_t_R = float(no_of_topicRep[topic_seq[i]] + 1) / (i + 1 + V)


                fb_rank = round(ranked_fbs[i], 1)

                event_time_bin = int(times[i]) #int(times[i] * no_time_bins_per_day)
                logndt = np.log(n_feedback_dt[i] * 24 * 60 + 1)
                if np.isnan(logndt):
                    print(n_feedback_dt[i])
                    logndt=0
                event_f = [times[i], n_feedback[i], n_feedback_dt[i], per_fb_dt[i], per_fb[i], np.log(n_feedback[i] + 1),
                           dt[i-1], dt[i], logit(prob_topic[topic_seq[i]]), logit(prob_t_R), user, int(istopicRepeated),
                           1.0 / float(num_tweets), fav_seq[i], logndt , topic_seq[i + 1],
                           event_time_bin
                           ]

                if rt_data_avail_seq[i] == 1:
                    per_user_events.append(event_f)
                    k+=1

                    f_cnt += 1

            if u % 1000 == 0: print(u)

            if k>=50:
                for event_f in per_user_events:
                    writer_f.writerow(event_f)


    finally:
        ff.close()
        fg.close()
    print('No of tweets (Forward direction): ', f_cnt)
    print('Number of users considered ', user_con)
    print('No of posts with fb: ',  no_posts_with_fb)
    print('Average amt of fb:', np.mean(avg_fb))
    print('Average no of posts:', np.mean(avg_nposts))
    
    import json

    with open('perUser_probTopic.json', 'w') as fp:
        json.dump(perUser_probTopic, fp)

    with open('median_post_time.json', 'w') as fp:
        json.dump(median_post_time, fp)


def build_logit_data(source_dir, num_topics):
    from sklearn import preprocessing
    df = pd.read_csv(source_dir + "/regression_data/logit_reg_data_"+str(num_topics)+".csv")
    #df = df.replace([np.inf, -np.inf], np.nan).dropna(how="all")

    ### Train/Test Split ###
    df['index'] = list(range(df.shape[0]))
    users_set = set(list(df['user'].values))
    print('# of users: ', len(users_set))

    print('Before filtering: ', df.shape)

    train_idxs, test_idxs = [], []
    avail_users = 0
    for u, user in enumerate(list(users_set)):
        if u % 1000 == 0:
            print(u, user)
        user_posts_idxs = list(df[df['user'] == user]['index'].values)
        test_start = len(user_posts_idxs) - 3
        if len(user_posts_idxs) >= 50:
            test_idxs += user_posts_idxs[test_start:]
            train_idxs += user_posts_idxs[:test_start]
            avail_users += 1
        else:
            # train_idxs +=user_posts_idxs
            df = df.drop(df[df['user'] == user].index)

    print('# users in test set ', avail_users)
    print('After filtering: ', df.shape)

    print('# users in test set ', avail_users)

    with open('train_indexes.txt', 'w') as f:
        for t_idx in train_idxs:
            f.write(str(t_idx) + '\n')

    with open('test_indexes.txt', 'w') as f:
        for t_idx in test_idxs:
            f.write(str(t_idx) + '\n')

    df = df.drop(['index'], axis=1)
    df_data = df.drop(['timestamp'], axis=1)
    df_data['numFb'] = preprocessing.scale(df_data['numFb'].values) #np.log(df_data['numFb'].values+1)
    df_data['numFb_dt'] = preprocessing.scale(df_data['numFb_dt'].values) #np.log(df_data['numFb_dt'].values * 24 *60 +1)
    df_data['prevDelay'] = preprocessing.scale(df_data['prevDelay'].values)
    df_data['delay_btw_tweets'] = preprocessing.scale(df_data['delay_btw_tweets'].values)
    df_data['prob_topic'] = preprocessing.scale(df_data['prob_topic'].values)
    df_data['prob_topic_rep'] = preprocessing.scale(df_data['prob_topic_rep'].values)
    df_data['log_numFbdt'].fillna(0, inplace=True)

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

    no_nonCat_features = 0 # i.e number of non-categorical features
    for col in columnnames:
        if col.startswith('user_'):
            break
        no_nonCat_features+=1

    print('# of non categorical features: ', no_nonCat_features)
    np.savetxt('columnnames.txt', columnnames, fmt="%s")

    return df_data, no_user, no_samples, no_nonCat_features


def expound_features(all_df_data, chunk_start, chunk_stop, no_user, num_topics, no_nonCat_features, data_name=''):
    no_users = no_user
    no_topics = num_topics

    df_data = all_df_data.iloc[chunk_start:chunk_stop]

    no_fixed_features = no_nonCat_features
    no_time_bins = len(list(df_data.columns.values)[(no_fixed_features + no_users + no_topics):])

    df_data = df_data.values

    n_fb = df_data[:, 0]
    fb_dt = df_data[:, 1]
    per_fb_dt = df_data[:, 2]
    per_fb = df_data[:, 3]
    log_nfb = df_data[:, 4]
    prevDelays = df_data[:, 5]
    delays = df_data[:, 6]
    prob_topic = df_data[:, [7]].ravel()
    prob_topic_rep = df_data[:, [8]].ravel()
    y = df_data[:, [9]].ravel()
    fav = df_data[:, [11]].ravel()
    log_nfbdt = df_data[:, [12]].ravel()
    c0 = np.ones(df_data.shape[0]).ravel()

    start_index = no_fixed_features
    end_index = no_fixed_features + no_users
    user_fields = df_data[:, list(range(start_index, end_index))]

    ncol_user = user_fields.shape[1]

    if data_name.startswith('model_user_event'):
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

    if data_name == 'model_c0':
        X = c0.reshape(-1, 1)
    elif data_name == 'model_user':
        X = user_fields
    elif data_name == 'model_probTopic':
        X = prob_topic.reshape(-1,1)
    elif data_name == 'model_user_probTopic':
        X = np.column_stack((prob_topic, user_fields))
    elif data_name == 'model_user_fb':
        per_user_feedback_pudt = user_fields * np.transpose(np.array([per_fb_dt, ] * ncol_user))
        X = np.column_stack((per_user_feedback_pudt, prob_topic, user_fields))
    elif data_name == 'model_user_event':
        X = np.column_stack((prob_topic, user_fields, time_by_topic))
    elif data_name == 'model_user_event_fb_pu':
        per_user_feedback_pu = user_fields * np.transpose(np.array([per_fb, ] * ncol_user))
        X = np.column_stack((per_user_feedback_pu, prob_topic, user_fields, time_by_topic))
    elif data_name == 'model_user_event_fb_favdt':
        per_user_feedback_fav = user_fields * np.transpose(np.array([fav, ] * ncol_user))
        X = np.column_stack((per_user_feedback_fav, prob_topic, user_fields, time_by_topic))
    elif data_name == 'model_user_event_fb_pudt':
        per_user_feedback_pudt = user_fields * np.transpose(np.array([per_fb_dt, ] * ncol_user))
        X = np.column_stack((per_user_feedback_pudt, prob_topic, user_fields, time_by_topic))
    elif data_name == 'model_user_event_fb_logn':
        per_user_feedback_logn = user_fields * np.transpose(np.array([log_nfb, ] * ncol_user))
        X = np.column_stack((per_user_feedback_logn, prob_topic, user_fields, time_by_topic))
    elif data_name == 'model_user_event_fb_logndt':
        per_user_feedback_logndt = user_fields * np.transpose(np.array([log_nfbdt, ] * ncol_user))
        X = np.column_stack((per_user_feedback_logndt, prob_topic, user_fields, time_by_topic))
    elif data_name == 'model_user_event_fb_fbdt':
        per_user_feedback_fbdt = user_fields * np.transpose(np.array([fb_dt, ] * ncol_user))
        X = np.column_stack((per_user_feedback_fbdt, prob_topic, user_fields, time_by_topic))
    elif data_name == 'model_user_event_fb_99':
        per_user_feedback99 = user_fields * np.transpose(np.array([np.ones(user_fields.shape[0]) * 0.99, ] * ncol_user))
        X = np.column_stack((per_user_feedback99, prob_topic, user_fields, time_by_topic))
    elif data_name == 'model_user_event_fb_50':
        per_user_feedback50 = user_fields * np.transpose(np.array([np.ones(user_fields.shape[0]) * 0.50, ] * ncol_user))
        X = np.column_stack((per_user_feedback50, prob_topic, user_fields, time_by_topic))
    else:
        per_user_feedback = user_fields * np.transpose(np.array([n_fb, ] * ncol_user))
        X = np.column_stack((per_user_feedback, prob_topic, user_fields, time_by_topic))

    return X, y


# export_data(df_data, no_user, num_topics, no_samples, data_name='model_c0')
def export_data(df_data, no_user, num_topics, no_train_samples, no_nonCat_features, data_name=''):
    no_samples = no_train_samples
    batch_size = 5000

    chunk_indexes = [i for i in range(0, no_samples, batch_size)] + [no_samples]

    all_X = []
    all_Y = []
    for k in range(len(chunk_indexes) - 1):
        if k%100==0:
            print(k)
        X, y = expound_features(df_data, chunk_indexes[k], chunk_indexes[k + 1], no_user, num_topics,
                                no_nonCat_features, data_name)

        X = csr_matrix(X)

        all_X.append(X)
        all_Y.append(y)

    all_X = vstack(all_X)

    all_Y = np.concatenate(all_Y)
    all_Y = all_Y.flatten().ravel()

    feature_filename = 'featuresX_' + data_name + '_LDA'+str(num_topics)+'.npz'

    save_npz(feature_filename, all_X)

    np.savetxt('outputY_LDA'+str(num_topics)+'.csv', all_Y)

    print('Dataset ' + feature_filename + ' exported')



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
        topic_model_dir = 'Twitter-LDA'+str(num_topics)
    else:
        num_topics = 100
        topic_model_dir = 'Twitter-LDA'


    user_data = preprocessTweets(source_dir, feedback_data_path, topic_model_dir)

    print('preprocessing done! ')

    # print out micro average data
    print('write logistic regression data')
    write_microAverageData(source_dir, user_data, num_topics)
    df_data, no_user, no_train_samples, no_nonCat_features = build_logit_data(source_dir, num_topics)

    export_data(df_data, no_user, num_topics, no_train_samples, no_nonCat_features, data_name='model_user_event_fb_fb')
    export_data(df_data, no_user, num_topics, no_train_samples, no_nonCat_features, data_name='model_user_event_fb_logn')
    export_data(df_data, no_user, num_topics, no_train_samples, no_nonCat_features, data_name='model_user_event_fb_pu')

    export_data(df_data, no_user, num_topics, no_train_samples, no_nonCat_features, data_name='model_user_event_fb_fbdt')
    export_data(df_data, no_user, num_topics, no_train_samples, no_nonCat_features, data_name='model_user_event_fb_logndt')
    export_data(df_data, no_user, num_topics, no_train_samples, no_nonCat_features, data_name='model_user_event_fb_pudt')


    #export_data(df_data, no_user, num_topics, no_train_samples, no_nonCat_features, data_name='model_c0')
    #export_data(df_data, no_user, num_topics, no_train_samples, no_nonCat_features, data_name='model_user_fb')
    #export_data(df_data, no_user, num_topics, no_train_samples, no_nonCat_features, data_name='model_user')
    #export_data(df_data, no_user, num_topics, no_train_samples, no_nonCat_features, data_name='model_probTopic')
    #export_data(df_data, no_user, num_topics, no_train_samples, no_nonCat_features, data_name='model_user_probTopic')
    #export_data(df_data, no_user, num_topics, no_train_samples, no_nonCat_features, data_name='model_user_event_fb_99')
    #export_data(df_data, no_user, num_topics, no_train_samples, no_nonCat_features, data_name='model_user_event_fb_50')
