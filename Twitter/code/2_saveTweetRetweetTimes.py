'''
in  dictionary  (key=tweet_id, value = (user_id, #rt, #fav, tweet_time, tweet_pos)) from tweet_feedback_data folder
 2) read in the large dataset of retweets and write out a subset of it with the same tweet_ids as that of the tweets_10k dataset / tweet_feedback_data folder.

'''

import os
import re
import csv
from collections import defaultdict, Counter
import datetime
import matplotlib.pyplot as plt
import numpy as np
import bisect
import string
from scipy import stats
import argparse
import sys

def preprocessTweets(path):
    with open(path + "/tweet_feedback_data/user_tweet_size.csv") as f:
        users_lines = f.readlines()
    list_of_users = [line.split(',')[0] + '.csv' for line in users_lines]

    start_date = datetime.datetime.fromtimestamp(float('1451606400'))

    tweets = {}  # tweets in tweets_10k dataset with their attributes
    user_data = {}
    for ur, user_file in enumerate(list_of_users):
        print('user'+str(ur))
        #  path is where the feedback data are located
        # tweets_feats is a list of (tweet_arr_time, retweet_cnt, favorite_cnt, tweet_stem)
        # tweet_arr_time needs to be converted to float i.e difference in seconds between the start_date timestamp and the tweet_arr_time.

        with open(path + "/tweet_feedback_data/" + user_file) as f2:
            tweets_feats = f2.readlines()

        times = []
        timestamps = []
        retweets = []
        favorites = []
        tweet_stems = []
        tweet_ids = []
        tweets_hashtags = []

        for i, evt in enumerate(tweets_feats[
                                1:]):  # first row is the column names e.g 'Timestamp','retweet_cnt', 'favorite_cnt', 'tweet_id', 'tweet_stem', hashtags
            evt_data = evt.strip().split(',')
            tweet_stem = evt_data[4]
            hashtags = []
            if len(evt_data) > 5: hashtags = evt_data[5:]
            t_n = float(evt_data[0])
            retweet_cnt = int(evt_data[1])
            favorite_cnt = int(evt_data[2])
            tweet_pos = i
            user_id = user_file[:-4]
            tweet_id = evt_data[3]

            timestamps.append(evt_data[0])
            date_time = datetime.datetime.fromtimestamp(t_n)
            tweet_time = ((date_time - start_date).total_seconds()) / 86400.0
            # tweet_time = int(t_n)
            tweets[tweet_id] = (user_id, retweet_cnt, favorite_cnt, tweet_time, tweet_pos, t_n)
            times.append(tweet_time)
            retweets.append(retweet_cnt)
            favorites.append(favorite_cnt)
            tweet_stems.append(str(tweet_stem))
            tweet_ids.append(tweet_id)
            tweets_hashtags.append(hashtags)
            user_data[user_file[:-4]] = (retweets, favorites, times, tweet_stems, np.zeros(len(retweets)),
                tweet_ids, np.copy(np.array(retweets)), np.zeros(len(retweets), dtype='int'), tweets_hashtags,
                timestamps,
                np.zeros(len(retweets)), np.zeros(len(retweets)), np.zeros(len(retweets)), np.zeros(len(retweets)))

    return user_data, tweets


def getRetweeters(retweet_dir, tweets):

    retweeters_dir = retweet_dir
    jan_mar = retweeters_dir+"/Jan_March"    #retweets for tweets from January-March
    apr_june = retweeters_dir + "/Apr_June"   # retweets for tweets from April-June

    retweet_dict = defaultdict(list)  #retweet_dict : dict(tweet, list(user_id, retweet_time))
    for months_dir in [jan_mar, apr_june]:
        print(months_dir)
        for rt_out_file in sorted(os.listdir(months_dir)):
            with open(months_dir + "/" + rt_out_file) as text:
                retweets = text.readlines()
            # retweet is  tweet: list(retweeter, time_retweet)
            for retweet in retweets:
                retweet_data =  retweet.strip().split()
                if len(retweet_data) <= 1 or retweet_data[0] not in tweets: continue   #no retweet yet for the tweet
                tweet = retweet_data[0]  #format is user_id;time e.g  2928882141;1448552433 3064357018;1448365533
                rt_time_list = retweet_data[1:]
                # turn it into a list of tuples i.e (user_id, timestamp)
                rt_time_list = [(rt_time[0:rt_time.index(';')], rt_time[rt_time.index(';') + 1:]) for rt_time in
                                rt_time_list if ';' in rt_time]
                # convert time to float, difference between start_date (Jan01 2016) time stamp and the current timestamp
                rt_time_list = [
                    str(rt_time[1])
                    for rt_time in rt_time_list
                    ]
                retweet_dict[tweet].extend(rt_time_list)

    print('total # tweets with retweets: ', len(retweet_dict))

    f = open(retweeters_dir + "/tweet_retweetTimes.csv", 'w')
    try:
        writer = csv.writer(f, dialect='excel')
        for tweet, retweets in retweet_dict.items():
            writer.writerow([tweet]+retweets)
    finally:
        f.close()
    return  retweet_dict

if __name__ == '__main__':
    source_dir = "../data"
    retweet_dir = "../../retweet_data"

    parser = argparse.ArgumentParser(description='Save Re-tweet times for all tweets')
    parser.add_argument('---user_data_path', type=str, help='the directory where per user tweets are located')
    parser.add_argument('---retweets_dir', type=str, help='the directory where the retweets data is located')

    args = parser.parse_args()

    if len(sys.argv) < 2:
        print('please, specify the directory where retweets data and preprocessed user tweets are stored, for example'
                                                  ' \n python 2_saveTweetRetweetTimes.py ---retweets_dir "../../../retweet_data" ---user_data_path "../data" ')
        sys.exit(1)  # abort because of error

    user_data, tweets = preprocessTweets(args.user_data_path)
    print('# of tweets in the dataset', len(tweets))
    retweet_dict = getRetweeters(args.retweets_dir, tweets)  # slow, hence, the result is saved into a file
