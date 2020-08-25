'''
preprocess data in tweet_10k (tweets posted each day); 
the code writes
1) users tweets/stems in "/Twitter-LDA/src/data/Data4Model/test" folder - this is used by Twitter-Lda to cluster tweets of users,
2) the feedback data (tweet_arr_time, tweet_id, #RT gotten after 1h, #favorites,...,hashtags)  for all tweets of a user into tweet_feedback_data folder

'''
import os
import pandas as pd
import re
import csv
import sys
from collections import defaultdict
import datetime
import numpy as np
import string
import argparse


def extract_tweet_attrs_and_stems(source_dir, tweet_path, no_lda_topics, twitter_lda_path):
    # the data to be processed is a 6 months (01-01-2016 to 30-06-2016) twitter data. 
    # The tweets of each day are stored in a file with date as it's file name
    # we need to process the each day's tweets and group tweets by user. 
    # E.g From "day -> [(tweet1, user1,..), (tweet2, user2,..), (tweet3, user2,..),... ]"  ===> " user2 -> [(tweet2, ..), (tweet3,..),...] "
    # "/Twitter-LDA/src/data/Data4Model/test" folder stores " user2 -> [(tweet2), (tweet3),...] ". A file for each user
    # tweet_feedback_data/ folder stores  " user2 -> [(time, #rt, #fav, tweet),...] "
    j=0
    evt_no=0
    users_tweets =  defaultdict(list)
    users_tweets_mentions = defaultdict(list)
    tweets_ids = {}
    # '#' excluded to retain hashtags in tweet stems
    punctuations = ''.join(list(set(list(string.punctuation)) - set(['#'])))
    for filename in sorted(os.listdir(tweet_path)):  #sorting is very important, to preserve the order of tweet arrival time
        with open(tweet_path+"/"+filename) as text:
            tweets = text.readlines()
        for i, tweet in enumerate(tweets):
            if i>0:
                data = re.split(r'\t+', tweet)
                tweet_stem = data[11].strip()
                tweet_stem = ' '.join(word.lower().strip(punctuations) for word in tweet_stem.split())
                tweet = data[10].strip().lower()
                hashtags =  list(set([word  for word in tweet_stem.split() if word.startswith("#")]))
                mentions = list(set([word for word in tweet.split() if word.startswith("@")]))
                retweet_id = int(data[6])   #tweet id that is being retweeted
                tweet_time = data[7]
                tweet_id = data[0]
                tweet_user = data[1]    #user id
                retweet_count = data[2]
                favorite_count = data[3]
                reply_user_id = int(data[4])
                reply_tweet_id = int(data[5])
                event = [int(tweet_time), retweet_count, favorite_count, tweet_id, tweet_stem]+hashtags  #modified 01-09-2017  # added tweet_id on July 18 2017
                #if tweet_id in tweets_ids: print tweet_id, tweet_user
            
                if(retweet_id == 0 and reply_user_id == 0 and reply_tweet_id == 0):
                    if len(users_tweets[tweet_user]) > 0 and tweet_id == users_tweets[tweet_user][-1][3]:
                        del users_tweets[tweet_user][-1]
                        del users_tweets_mentions[tweet_user][-1]

                    users_tweets[tweet_user].append(event)
                    users_tweets_mentions[tweet_user].append([tweet_id]+mentions)
                    tweets_ids[tweet_id] = retweet_count
                    evt_no+=1
        j+=1
        print("File: "+filename)
    print("total events: "+str(evt_no))

    if no_lda_topics == 100:
        tw_lda_dir = 'Twitter-LDA'
    else:
        tw_lda_dir = 'Twitter-LDA'+str(no_lda_topics)

    user_tweet_path = source_dir+"/tweet_feedback_data"   #stores feedback data and tweets for each user (sorted by time)
    
    tweet_stem_path = twitter_lda_path+"/"+tw_lda_dir+"/src/data/Data4Model/test"   #stores tweets/stems used for lda topic clustering.

    if not os.path.exists(user_tweet_path):
        os.makedirs(user_tweet_path)

    user_numTweets = {}
    now_evts = 0

    for user in sorted(users_tweets):
        #if user not in user_followers: continue
        f = open(user_tweet_path+"/"+str(user) + ".csv", 'w')
        f2 = open(tweet_stem_path+"/"+str(user) + ".csv", 'w')
        user_numTweets[user] = len(users_tweets[user])
        now_evts += len(users_tweets[user])
        try:
            writer = csv.writer(f, dialect='excel')
            writer2 = csv.writer(f2, dialect='excel')
            writer.writerow(['Timestamp','retweet_cnt', 'favorite_cnt', 'tweet_id', 'tweet_stem', 'hashtags'])
            for k, event in enumerate(users_tweets[user]):
                writer.writerow(event)
                writer2.writerow([event[4]])
                
        finally:
            f.close()
            f2.close()
          
    
    print('No of events after filtering same timestamp: ', now_evts)
    user_numTweets_sorted = sorted(user_numTweets.items(), key=lambda x: (-x[1], x[0]))  #sort users by number of tweets

    f = open(user_tweet_path+"/user_tweet_size.csv", 'w')
    try:
        writer = csv.writer(f, dialect='excel')
        for user_len in user_numTweets_sorted:
            writer.writerow(list(user_len))
    finally:
        f.close()

    path = twitter_lda_path+"/"+tw_lda_dir+"/src/data/Data4Model/test"
    num_users = 0
    new_path = twitter_lda_path+"/"+tw_lda_dir+"/src/data"
    f = open(new_path + "/filelist_test.txt", 'w')
    for user_file in os.listdir(path):
        if (user_file == 'user_tweet_size.csv'): continue
        f.write(user_file + "\n")
        num_users += 1

    f.close()

    print('Number of users for Twitter-LDA to process ', num_users)


if __name__ == '__main__':
    pos_path =  os.getcwd().rfind('/')
    dir = os.getcwd()
    print(dir)
    source_dir = dir[:pos_path + 1] + "data"
    tweet_path = "../../tweets_10k";

    parser = argparse.ArgumentParser(description='Preprocess tweets -- extract feedback data & data for topic modeling')
    parser.add_argument('---num_lda_topics', type=int, help='number of topics required for topic modeling: 50, 100 or 200')
    parser.add_argument('---tweets_path', type=str, default='../../../tweets_10k', help='the directory of the 8k/10K expert tweets')
    parser.add_argument('---twitter_lda_path', type=str, default='../../../Twitter-LDA', help='the directory where the Twitter-LDA you clone from github is located')
    parser.add_argument('---user_data_path', type=str, default=source_dir, help='the directory where you want to store the per user tweets data')

    args = parser.parse_args()

    if len(sys.argv) < 2:
        print('please, specify the number of topics required for LDA topic modeling: 50, 100 or 200, for example'
              ' \n python 1_preprocessTweets.py ---num_lda_topics 100 ---tweets_path "../../../tweets_10k" ---twitter_lda_path "../../../Twitter-LDA/" ---user_data_path "../data"')
        sys.exit(1)  # abort because of error

    extract_tweet_attrs_and_stems(args.user_data_path, args.tweets_path, args.num_lda_topics, args.twitter_lda_path)

