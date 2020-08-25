from bz2 import BZ2File as bzopen
import json
import codecs
import csv
from collections import defaultdict
import numpy as np
import datetime
import os
from scipy.stats import rankdata
import pandas as pd

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
        post_id, subbreddit_id, user, post_timestamp, num_comments, scores = line_data

        date_time = datetime.datetime.fromtimestamp(float(post_timestamp))
        post_time = ((date_time - start_date).total_seconds()) / 86400.0

        user_posts[user].append((post_id, subbreddit_id, post_timestamp, post_time, num_comments, scores))
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
                post_id, subbreddit_id, post_timestamp, post_time, num_comments, scores = post_data


                per_subreddit_posts[subbreddit_id].append((post_id, post_timestamp, post_time, user, num_comments, scores))


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

    # rank the #FB in each delay bins
    # percentile score : 0.1, 0.2, ...,0.9, 1.0
    percentile_score = np.array(range(1, 11)) * 10
    delay_percentileScore_matrix = np.nan * np.ones(shape=(delay_len, len(percentile_score)))
    for p, percent in enumerate(percentile_score):
        delay_percentileScore_matrix[:, p] = np.nanpercentile(post_delay_matrix, percent, axis=0)

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
        delay = post_times[t+1] -post_times[t]
        if delay>0:
            post_feedback_div_delay[t] = len(idx_before_next_post) / (delay*24.0*60.0)  # divide by delay in minutes
        else:
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


    '''
    # build rank matrix of number of comments per subreddit
    subredditID_rankingMatrix = dict()
    for subreddit_id in per_subreddit_posts:
        subreddit_data = list(zip(*per_subreddit_posts[subreddit_id]))
        post_ids, post_timestamps, post_times, users, num_comments_list = subreddit_data

        rankMatrix = build_feedback_ranking_matrix(post_times, post_ids, post_comments_times, len(post_ids))
        subredditID_rankingMatrix[subreddit_id] = rankMatrix
    '''

    user_post_path = "user_data"

    if not os.path.exists(user_post_path):
        os.makedirs(user_post_path)

    # rank number of comments using rank matrix
    ranked_peruser_feedback = defaultdict(list)
    number_of_comments_dist = []
    delays_dist = []
    user_rankingMatrix = {}
    for i, user in enumerate(per_user_posts):
        if i%1000==0: print("reddit_user", i, user)
        per_user_data = list(zip(*per_user_posts[user]))
        # [post_id, subbreddit_id, post_timestamp, post_time, num_comments]
        post_ids, subreddit_ids, post_timestamps, post_times, num_comments_list, scores_list = per_user_data
        sort_idxs = np.argsort(post_times)

        post_ids = np.array(post_ids, dtype=str)[sort_idxs]
        subreddit_ids = np.array(subreddit_ids, dtype=str)[sort_idxs]
        post_timestamps = np.array(post_timestamps, dtype=str)[sort_idxs]
        post_times = np.array(post_times)[sort_idxs]
        num_comments_list = np.array(num_comments_list)[sort_idxs]
        scores_list = np.array(scores_list)[sort_idxs]


        rankMatrix = build_feedback_ranking_matrix(post_times, post_ids, post_comments_times, len(post_ids))

        user_rankingMatrix[user] = rankMatrix


        post_feedback, post_feedback_div_delay, fb_div_delay_per, fb_div_delay_median = \
            explore_feedback_representations(post_times, post_ids, post_comments_times)

        f = open(user_post_path + "/" + str(user) + ".csv", 'w')
        try:
            writer = csv.writer(f, dialect='excel')
            writer.writerow(['timestamp', 'post_id', 'numFb', 'numFb_dt', 'percent_numFb_dt', 'per_numFb', 'feedback_rank', 'final_numComments', 'scores','subreddit_id'])

            for i, post_id in enumerate(post_ids):

                if post_id in post_comments_times:
                    next_time = postID_postTimes[post_id]
                    fb_times = np.array(post_comments_times[post_id])
                    idx_curr_next = np.where(fb_times < next_time)[0]
                    num_comments = len(idx_curr_next)
                    number_of_comments_dist.append(num_comments)
                    delay = next_time - post_times[i]
                    delays_dist.append(delay)
                    fb_rank = choose_timeBin_and_percentileScore(delay, num_comments, user_rankingMatrix[user])
                    final_numComments = len(fb_times)
                    # timestamp, post_id, total_comments,  number_feedback, feedback_rank, subreddit_id
                    writer.writerow([post_timestamps[i], post_id, post_feedback[i], post_feedback_div_delay[i],
                                     fb_div_delay_per[i], fb_div_delay_median[i], fb_rank, final_numComments, scores_list[i], subreddit_ids[i]])
                else:
                    writer.writerow([post_timestamps[i], post_id, -1, -1,
                                     -1, -1, -1, -1, scores_list[i], subreddit_ids[i]])

        finally:
            f.close()

    np.savetxt('number_of_comments_dist.csv', number_of_comments_dist, delimiter=',', fmt='%5s')
    np.savetxt('delays_dist.csv', delays_dist, delimiter=',', fmt='%5s')

if __name__ == '__main__':
    per_subreddit_posts, per_user_posts = extract_active_users_posts()
    post_comments_times = extract_post_comments()
    user_posts = countComment_before_nextPost_and_rank(per_user_posts, per_subreddit_posts, post_comments_times)




