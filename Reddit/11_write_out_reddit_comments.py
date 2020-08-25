from bz2 import BZ2File as bzopen
import json
import codecs
import csv
from collections import defaultdict
import pandas as pd

df = pd.read_csv("regression_data/logit_reg_data.csv")
user_post_ids = set(list(df['tweet_id'].values))

def extract_comments(filename, top_n_subreddits):
    post_commentTimes = defaultdict(list)
    # reading a bz2 archive

    with bzopen(filename, "r") as bzfin:
        """ Handle lines here """
        cbz = codecs.iterdecode(bzfin, "utf-8")
        # print(len(list(cbz)))
        for i, line in enumerate(cbz):
            if i%10000000 == 0:
                print(i)

            try:
                post_dict = json.loads(line.rstrip())

                #
                # {
                #    "created_utc":1506816000,
                #    "link_id":"t3_73ieyz"
                # }

                post_id, comment_date, subreddit_id, comment_body = post_dict['link_id'], post_dict['created_utc'], post_dict[
                    'subreddit_id'], post_dict['body']

                if subreddit_id in top_n_subreddits and post_id in user_post_ids:
                    #print(comment_date, comment_body)
                    post_commentTimes[post_id].append([comment_date, comment_body])


            except:
                continue

    return post_commentTimes


if __name__ == '__main__':
    with open('top_100_byNoComments_outof_top1000_byNoSubscribers.csv', 'r') as f:
        lines = f.readlines()

    top_n_subreddits = {}
    for subreddit in lines:
        subreddit_id, subreddit_name, no_subs, no_comments = subreddit.strip().split(',')
        top_n_subreddits[subreddit_id] = subreddit_name

    comment_files = ['RC_2016-01.bz2', 'RC_2016-02.bz2', 'RC_2016-03.bz2', 'RC_2016-04.bz2', 'RC_2016-05.bz2',
                     'RC_2016-06.bz2']

    #comment_files = ['RC_2016-01.bz2']

    f = open("reddit_timestamp_comments.txt", 'w')
    try:
        writer = csv.writer(f, dialect='excel')

        for filename in comment_files:
            print(filename)
            post_commentTimes = extract_comments(filename, top_n_subreddits)
            print('# comments of ', filename, len(post_commentTimes))
            for post_id in post_commentTimes:
                for comment_date, comment_body in post_commentTimes[post_id]:
                    #print(comment_date, comment_body)
                    writer.writerow([post_id] + [comment_date] + [comment_body])
    finally:
        f.close()

    print('# of reddit posts with comments ', len(post_commentTimes))






