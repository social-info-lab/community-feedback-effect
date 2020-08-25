"""
    This is to rank the feedback using probability integral transformation.
    RTs are ranked for different users separately, with time delay until the next tweet incorporated in the ranking.
    @author: 2018 David Adelani, <dadelani@mpi-sws.org> and contributors.
"""


import os
import csv
import sys
import datetime
import time
import numpy as np
import pandas as pd

start_date = datetime.datetime.fromtimestamp(float('1451606400'))

#########################################################################################################
# For each user, store the sequence of tweets' attributes e.g timestamp, topic_seq, retweets
#########################################################################################################
def preprocessTweets(path, unranked_data_folder):
    tweet_path = path +"/"+unranked_data_folder
    user_data = {}
    u = 0
    for user_file in sorted(os.listdir(tweet_path)):
        with open(tweet_path+"/"+user_file) as text:
            tweets = text.readlines()

        times = []
        timestamps = []
        retweets = []
        favorites = []
        tweet_ids = []

        user = user_file[:-4]
        #print(u)
        u+=1
        if user == 'user_tweet_size':
            continue
        # first row is the column names e.g 'Timestamp', 'retweet_cnt', 'favorite_cnt', 'tweet_id', 'tweet_stem', 'hashtags'
        for i, evt in enumerate(tweets[1:]):
            evt_data = evt.strip().split(',')
            t_n = float(evt_data[0])
            retweet_cnt = int(evt_data[1])
            tweet_id = evt_data[3]
            favorite_cnt = int(evt_data[2])

            timestamps.append(evt_data[0])
            date_time = datetime.datetime.fromtimestamp(t_n)
            tweet_time = ((date_time - start_date).total_seconds()) / 86400.0
            # tweet_time = int(t_n)
            times.append(tweet_time)
            retweets.append(retweet_cnt)
            tweet_ids.append(tweet_id)
            favorites.append(favorite_cnt)

        user_data[user] = (timestamps, tweet_ids, np.zeros(len(retweets), dtype='int'), np.array(retweets),
                                     np.zeros(len(retweets)), np.zeros(len(retweets)), np.zeros(len(retweets)),
                           np.zeros(len(retweets)), times, favorites, np.array(retweets))

    return user_data


#########################################################################################################
# For each user, build feedback ranking matrix. The dimension is (#time_delays, 10_percentile_scores)
#########################################################################################################
def build_feedback_ranking_matrix(times, tweet_ids, tweet_rtTime_dict, num_tweets, num_bins=100):
    delays = list(np.logspace(np.log10(1.0 / 86400.0), np.log10(30), num=num_bins - 1)) + [np.inf]
    delay_len = len(delays)
    tweet_delay_matrix = np.nan * np.ones(shape=(num_tweets, delay_len))

    # build tweet delay matrix
    for t, tweet_id in enumerate(tweet_ids):
        if tweet_id not in tweet_rtTime_dict: continue
        rt_times = tweet_rtTime_dict[tweet_id]

        for d, delay in enumerate(delays):
            idx_before_next_tweet = np.where((rt_times <= times[t] + delay))[0]  # #RT obtained after each time delay
            tweet_delay_matrix[t][d] = len(idx_before_next_tweet)  # total number of RT within time range

    # rank the #RT in each delay bins
    # percentile score : 10%, 20%, 30%, ..., 100%
    percentile_score = np.array(range(1, 11)) * 10
    delay_percentileScore_matrix = np.nan * np.ones(shape=(delay_len, len(percentile_score)))
    for p, percent in enumerate(percentile_score):
        delay_percentileScore_matrix[:, p] = np.nanpercentile(tweet_delay_matrix, percent, axis=0)

    return delay_percentileScore_matrix


def explore_feedback_representations(post_times, post_ids, post_comments_times):
    # build tweet delay matrix
    post_feedback = np.nan * np.ones(shape=(len(post_ids)-1,))
    post_feedback_div_delay = np.nan * np.ones(shape=(len(post_ids) - 1,))
    for t, post_id in enumerate(post_ids[:-1]):
        if post_id not in post_comments_times: continue
        fb_times = np.array(post_comments_times[post_id])

        idx_before_next_post = np.where((fb_times <= post_times[t+1]))[0]
        post_feedback[t] = len(idx_before_next_post)  # total number of fb within time range
        delay = post_times[t+1] - post_times[t]
        if delay>0.0:
            post_feedback_div_delay[t] = len(idx_before_next_post) / (delay*24.0*60.0)  # divide by delay in minutes
        else:
            #print(delay, post_times[t+1], post_times[t], len(idx_before_next_post))
            post_feedback_div_delay[t] = np.nan

    non_null_len = len(np.where(~np.isnan(post_feedback_div_delay))[0])
    fb_div_delay_per = pd.Series(post_feedback_div_delay).rank().values
    fb_div_delay_per = fb_div_delay_per / non_null_len
    #fb_div_delay_per = rankdata(post_feedback_div_delay, "average") / len(post_feedback_div_delay)

    non_null_len = len(np.where(~np.isnan(post_feedback))[0])
    post_feedback_per = pd.Series(post_feedback).rank().values
    post_feedback_per = post_feedback_per / non_null_len
    #fb_div_delay_median = post_feedback_div_delay / np.nanmedian(post_feedback_div_delay)
    #print(fb_div_delay_median)

    post_feedback[np.isnan(post_feedback)] = -1
    post_feedback_div_delay[np.isnan(post_feedback_div_delay)] = -1
    fb_div_delay_per[np.isnan(fb_div_delay_per)] = -1
    post_feedback_per[np.isnan(post_feedback_per)] = -1
    #fb_div_delay_median[np.isinf(fb_div_delay_median)] = -1


    post_feedback = np.array(list(post_feedback)+[-1])
    post_feedback_div_delay = np.array(list(post_feedback_div_delay) + [-1])
    fb_div_delay_per = np.array(list(fb_div_delay_per) + [-1])
    post_feedback_per = np.array(list(post_feedback_per) + [-1])

    return post_feedback, post_feedback_div_delay, fb_div_delay_per, post_feedback_per


#########################################################################################################
# assign percentile score to a tweet by selecting the appropriate time delay vector from the rankMatrix
#########################################################################################################

def choose_timeBin_and_percentileScore(delay, num_rts, rankMatrix, num_bins=100):
    # choose time bin

    possible_delays = np.array(
        list(np.logspace(np.log10(1.0 / 86400.0), np.log10(30), num=num_bins - 1)) + [np.inf])

    some_inds = np.where(delay < possible_delays)[0]
    time_bin = sorted(some_inds)[0]
    time_bin_vector = rankMatrix[time_bin]
    some_inds = np.where(num_rts <= time_bin_vector)[0]
    if len(some_inds) == 0:
        rank = 100
        print ('surprise')
    else:
        rank = (sorted(some_inds)[0] + 1) * 10
    return rank


#########################################################################################################
# update feedback data, write the feedback data into file
#########################################################################################################
def updateFeedbackData(source_dir, retweet_dir, user_data, unranked_data_folder):

    # retrieve tweet and times of retweets from tweet_retweeters.csv
    tweet_rtTime_dict = {}
    tt = 0

    print("load retweeters' data, takes some time .... (~ 5 Million lines) ")
    with open(retweet_dir + "/tweet_retweetTimes.csv") as f:
        for line in f:
            tweet_retweets = line.strip().split(',')
            tweet_id = tweet_retweets[0]
            retweeters_times = tweet_retweets[1:]
            rt_times = [( (datetime.datetime.fromtimestamp(float(rt_time)) - start_date).total_seconds()) / 86400.0
            for rt_time in retweeters_times
            ]
            tweet_rtTime_dict[tweet_id] = np.array(rt_times)
            tt += 1
            if tt % 1000000 == 0: print(tt)

    print('retweeters data loaded')


    folder_ext = "ranked_feedback_data"

    user_tweet_path = source_dir + "/"+folder_ext

    if not os.path.exists(user_tweet_path):
        os.makedirs(user_tweet_path)

    num_bins = 100
    user_rankingMatrix = dict()
    user_numTweets = {}
    for u, user in enumerate(user_data):
        if u%1000==0: print('#users processed:', u)
        tweet_ids = user_data[user][1]
        times = user_data[user][8]
        num_tweets = len(tweet_ids)
        if num_tweets < 50: continue

        rankMatrix_rt = build_feedback_ranking_matrix(times, tweet_ids, tweet_rtTime_dict, num_tweets)

        user_rankingMatrix[user] = rankMatrix_rt

        post_feedback, post_feedback_div_delay, fb_div_delay_per, fb_div_delay_median = \
            explore_feedback_representations(times, tweet_ids, tweet_rtTime_dict)

        for i, time in enumerate(times):

            if tweet_ids[i] in tweet_rtTime_dict:
                rt_times = tweet_rtTime_dict[tweet_ids[i]]

                if i + 1 == num_tweets: continue
                time_next_tweet = times[i + 1]
                time_curr_tweet = times[i]
                time_prev_tweet = 0
                if i > 0:  time_prev_tweet = times[i - 1]
                delay_curr_next = time_next_tweet - time_curr_tweet

                #idx_curr_next = np.where(rt_times < (time_curr_tweet + delay_curr_next))[0]
                idx_curr_next = np.where(rt_times < time_next_tweet)[0]

                num_rt_curr_next = len(idx_curr_next)
                #if len(rt_times) < 50:
                #    print(time_curr_tweet, time_next_tweet, rt_times)
                #    print(num_rt_curr_next, post_feedback[i], delay_curr_next, len(rt_times))
                
                #usr_rts.append((user_data[user][3][i], num_rt_curr_next, len(rt_times)))
                user_data[user][2][i] = 1
                user_data[user][3][i] = num_rt_curr_next

                user_data[user][4][i] = post_feedback_div_delay[i]
                user_data[user][5][i] = fb_div_delay_per[i]
                user_data[user][6][i] = fb_div_delay_median[i]
                # update retweet_cnt
                user_data[user][7][i] = choose_timeBin_and_percentileScore(delay_curr_next, num_rt_curr_next,
                                                                           rankMatrix_rt)
                user_data[user][10][i] = len(rt_times)
                
                #usr_rts.append((user_data[user][3][i], user_data[user][10][i], len(rt_times)))
                tweet_rtTime_dict[tweet_ids[i]] = []
            else:
                user_data[user][2][i] = -1
                user_data[user][3][i] = -1
                user_data[user][4][i] = -1
                user_data[user][5][i] = -1
                user_data[user][6][i] = -1
                user_data[user][7][i] = -1

                user_data[user][10][i] = -1
            

        # stores feedback data and tweets for each user (sorted by time)
        #print(usr_rts[:20])
        # for user in list_of_users:
        f = open(user_tweet_path + "/" + str(user) + ".csv", 'w')
        user_numTweets[user] = len(user_data[user][0])
        try:
            writer = csv.writer(f, dialect='excel')
            writer.writerow(['timestamp', 'tweet_id', 'isRetweetersDataAvail?','numFb', 'numFb_dt',
                             'percent_numFb_dt', 'percentile_numFb', 'feedback_rank', 'favorites', 'final_numFb' ])

            for i in range(len(user_data[user][0])):
                writer.writerow([user_data[user][0][i], user_data[user][1][i], user_data[user][2][i],
                                 user_data[user][3][i], user_data[user][4][i], user_data[user][5][i],
                                 user_data[user][6][i], user_data[user][7][i], user_data[user][9][i],
                                 user_data[user][10][i]]
                                )
        finally:
            f.close()
        user_data[user] = []


    user_numTweets_sorted = sorted(user_numTweets.items(),
                                   key=lambda x: (-x[1], x[0]))  # sort users by number of tweets

    f = open(user_tweet_path + "/user_tweet_size.csv", 'w')
    try:
        writer = csv.writer(f, dialect='excel')
        for user_len in user_numTweets_sorted:
            writer.writerow(list(user_len))
            # print user_len
    finally:
        f.close()
    return user_data

if __name__ == '__main__':
    pos_path = os.getcwd().rfind('/')
    dir = os.getcwd()
    source_dir = dir[:pos_path + 1] + "data"
    retweet_dir = "../../../retweet_data"

    if len(sys.argv) < 2:
        print('please, specify the unranked data folder or the default folder will be used ')
        unranked_data_folder = 'tweet_feedback_data'
    else:
        unranked_data_folder = sys.argv[1]

    user_data = preprocessTweets(source_dir, unranked_data_folder)

    print ('pre-processing done, extract user_tweets')

    start = time.time()
    print('start ranking of feedback ... (takes around 30 minutes)')
    user_data = updateFeedbackData(source_dir, retweet_dir, user_data, unranked_data_folder)
    end = time.time()
    print('time duration ', end - start)
