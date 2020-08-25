from bz2 import BZ2File as bzopen
import json
import codecs
import csv
from collections import defaultdict
import numpy as np
import datetime
import os

start_date = datetime.datetime.fromtimestamp(float('1451606400'))


###############################################################################################
# Extract posts attributes AND feedback data
###############################################################################################
def extract_active_users_posts():

    user_noPosts = defaultdict(int)
    user_posts = defaultdict(list)

    with open('reddit_FirstHalf2016.csv', 'r') as f:
        lines = f.readlines()

    for line in lines[1:]:
        line_data = line.strip().split(',')
        post_id, subbreddit_id, user, post_timestamp, num_comments = line_data

        date_time = datetime.datetime.fromtimestamp(float(post_timestamp))
        post_time = ((date_time - start_date).total_seconds()) / 86400.0

        user_posts[user].append((post_id, subbreddit_id, post_timestamp, post_time, num_comments))
        user_noPosts[user] += 1

    sorted_user_noPosts = sorted(user_noPosts.items(), key=lambda kv: kv[1], reverse=True)
    print('# of users ', len(sorted_user_noPosts))

    avg_noPost = np.mean(list(user_noPosts.values()))
    print('Average number of posts per user ', avg_noPost)

    per_subreddit_posts = defaultdict(list)
    per_user_posts = defaultdict(list)
    noUser_bigAve = 0
    min_noPosts = 50
    total_no_posts = 0
    for user, noPost in user_noPosts.items():
        if noPost >= min_noPosts:
            noUser_bigAve += 1
            total_no_posts += noPost

            per_user_posts[user] = user_posts[user]
            for post_data in user_posts[user]:
                post_id, subbreddit_id, post_timestamp, post_time, num_comments = post_data


                per_subreddit_posts[subbreddit_id].append((post_id, post_timestamp, post_time, user, num_comments))


    print('# users with posts bigger than 50 ', noUser_bigAve)
    print('# posts submitted by the users that posts at least than 50 times ', total_no_posts)

    print('\n Active user posts extracted ')

    return per_subreddit_posts, per_user_posts


def extract_post_comments():

    post_comments_times = {}

    with open("reddit_comments.txt") as f:
        for line in f:
            post_id_feedback = line.strip().split(',')
            post_id = post_id_feedback[0]
            feedback_times = post_id_feedback[1:]

            fb_time_list = [
                ((datetime.datetime.fromtimestamp(float(fb_time)) - start_date).total_seconds()) / 86400.0
                for fb_time in feedback_times]

            post_comments_times[post_id] = fb_time_list

    print(post_comments_times[post_id])

    return post_comments_times



###############################################################################################
# Rank number of comments (feedback)
###############################################################################################
def build_feedback_ranking_matrix(post_times, post_ids, post_comments_times, num_posts, num_bins=100):
    delays = list(np.logspace(np.log10(1.0 / 86400.0), np.log10(30), num=num_bins - 1)) + [np.inf]
    delay_len = len(delays)
    post_delay_matrix = np.nan * np.ones(shape=(num_posts, delay_len))

    # build tweet delay matrix
    for t, post_id in enumerate(post_ids):
        if post_id not in post_comments_times: continue
        fb_times = np.array(post_comments_times[post_id])

        for d, delay in enumerate(delays):
            idx_before_next_post = np.where((fb_times <= post_times[t] + delay))[0]
            post_delay_matrix[t][d] = len(idx_before_next_post)  # total number of fb within time range

    avg_post_delay_vector = np.nanmean(post_delay_matrix, axis=0)

    return avg_post_delay_vector


def choose_timeBin_and_percentileScore(delay, num_fbs, rankMatrix, num_bins=100):
    # choose time bin

    possible_delays = np.array(
        list(np.logspace(np.log10(1.0 / 86400.0), np.log10(30), num=num_bins - 1)) + [np.inf])

    some_inds = np.where(delay < possible_delays)[0]
    time_bin = sorted(some_inds)[0]
    time_bin_vector = rankMatrix[time_bin]

    some_inds = np.where(num_fbs <= time_bin_vector)[0]
    if len(some_inds) == 0:
        rank = 100
        print('surprise')
    else:
        rank = (sorted(some_inds)[0] + 1) * 10

    return rank


def countComment_before_nextPost_and_rank(per_user_posts, per_subreddit_posts, post_comments_times):

    # get the next post time of a user
    postID_postTimes = {}
    all_cur_posts = []
    all_next_times = []
    for i, user in enumerate(per_user_posts):
        if i % 1000 == 0: print(i, user)
        per_user_data = list(zip(*per_user_posts[user]))
        # [post_id, subbreddit_id, post_date, num_comments]
        post_ids = np.array(per_user_data[0], dtype=str)
        post_dates = np.array(per_user_data[3])
        sort_idxs = np.argsort(post_dates)


        cur_post_ids = post_ids[sort_idxs]
        next_post_times = post_dates[sort_idxs]
        next_post_times = list(next_post_times[1:]) + [180.0]


        all_cur_posts += list(cur_post_ids)
        all_next_times+= next_post_times

    postID_postTimes = dict(list(zip(all_cur_posts, all_next_times)))

    num_bins = 100
    user_rankingMatrix = []
    user_numTweets = []
    for i, user in enumerate(per_user_posts):
        if i % 1000 == 0: print("reddit_user", i, user)
        per_user_data = list(zip(*per_user_posts[user]))
        # [post_id, subbreddit_id, post_timestamp, post_time, num_comments]
        post_ids, subreddit_ids, post_timestamps, post_times, num_comments_list = per_user_data
        sort_idxs = np.argsort(post_times)

        post_ids = np.array(post_ids, dtype=str)[sort_idxs]
        subreddit_ids = np.array(subreddit_ids, dtype=str)[sort_idxs]
        post_timestamps = np.array(post_timestamps, dtype=str)[sort_idxs]
        post_times = np.array(post_times)[sort_idxs]
        num_comments_list = np.array(num_comments_list)[sort_idxs]

        N_posts = len(post_ids)
        if N_posts < 50: continue
        user_numTweets.append(N_posts)

        rankMatrix_rt = build_feedback_ranking_matrix(post_times, post_ids, post_comments_times, N_posts)

        user_rankingMatrix.append(rankMatrix_rt)


    delays = list(np.logspace(np.log10(1.0 / 86400.0), np.log10(30), num=num_bins - 1)) + [np.inf]
    user_rankingMatrix = np.array(user_rankingMatrix)
    cleaned_data = np.ma.masked_array(user_rankingMatrix, np.isnan(user_rankingMatrix))

    user_rankingMatrix_ave = np.ma.average(cleaned_data, axis=0, weights=user_numTweets)
    user_rankingMatrix_ave.filled(np.nan)
    data_to_export = np.array([delays, user_rankingMatrix_ave])
    np.savetxt('avg_nComments_delay_weighted.csv', data_to_export, delimiter=',', fmt='%5s')
    print(user_rankingMatrix.shape)
    print(data_to_export.shape)


if __name__ == '__main__':
    per_subreddit_posts, per_user_posts = extract_active_users_posts()
    post_comments_times = extract_post_comments()
    user_posts = countComment_before_nextPost_and_rank(per_user_posts, per_subreddit_posts, post_comments_times)




