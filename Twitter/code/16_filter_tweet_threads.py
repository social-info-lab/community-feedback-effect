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

        tweet_texts = []

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
            tweet_text = evt_data[4]

            timestamps.append(evt_data[0])
            date_time = datetime.datetime.fromtimestamp(t_n)
            tweet_time = ((date_time - start_date).total_seconds()) #/ 86400.0
            # tweet_time = int(t_n)
            times.append(tweet_time)
            retweets.append(retweet_cnt)
            tweet_ids.append(tweet_id)
            favorites.append(favorite_cnt)
            tweet_texts.append(tweet_text)

        user_data[user] = (timestamps, tweet_ids, retweets, times, favorites, tweet_texts)

    return user_data


def filter_tweets(source_dir, user_data):
    num_users = len([user for user in user_data if len(user_data[user][0]) >= 50])

    ff = open(source_dir + "/regression_data/tweet_threads.csv", 'w')
    try:
        writer_f = csv.writer(ff, dialect='excel')

        column_names = ['timestamp1', 'timestamp2', 'tweet1', 'tweet2', 'delay']
        writer_f.writerow(column_names)
        f_cnt = 0
        user_con = 0
        no_posts_with_fb = 0

        for u, user in enumerate(user_data.keys()):
            timestamps, tweet_ids, retweets, times, favorites, tweet_texts = user_data[user]

            if len(timestamps) < 50: continue

            # ---forward pre-process ---
            dt = np.array(list(np.diff(times)) + [86400.0])
            user_con += 1
            for i in range(1, len(tweet_ids) - 1):
                if dt[i] < 60:
                    writer_f.writerow([timestamps[i], timestamps[i+1], tweet_texts[i], tweet_texts[i+1], dt[i]])


    finally:
        ff.close()
    print('No of tweets (Forward direction): ', f_cnt)
    print('Number of users considered ', user_con)


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
    print('filter tweets')

    filter_tweets(source_dir, user_data)

    end = time.time()
    print('time duration ', end - start)
