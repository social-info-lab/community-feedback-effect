from bz2 import BZ2File as bzopen
import json
import codecs
import csv
from collections import defaultdict


subreddits_noComments = defaultdict(int)

def extract_comments(filename, top_n_subreddits):
    # reading a bz2 archive

    with bzopen(filename, "r") as bzfin:
        """ Handle lines here """
        cbz = codecs.iterdecode(bzfin, "utf-8")

        reddit_submissions = []
        for i, line in enumerate(cbz):

            try:
                post_dict = json.loads(line.rstrip())

                '''
                {
                    "created_utc":1506816000,
                    "link_id":"t3_73ieyz"
                }
                '''
                post_id, comment_date, subreddit_id = post_dict['link_id'], post_dict['created_utc'], post_dict['subreddit_id']

                if i % 100000 == 0:
                    print(i, post_id, comment_date, subreddit_id)

                if subreddit_id in top_n_subreddits:
                    subreddits_noComments[subreddit_id]+=1


            except:
                continue



    return reddit_submissions


if __name__ == '__main__':
    with open('top_1000_subreddits.csv', 'r') as f:
        lines = f.readlines()

    top_n_subreddits = {}
    for subreddit in lines:
        subreddit_id, subreddit_name, no_subs = subreddit.strip().split(',')
        top_n_subreddits[subreddit_id] = (subreddit_name, no_subs)


    comment_files = ['RC_2016-01.bz2', 'RC_2016-02.bz2', 'RC_2016-03.bz2', 'RC_2016-04.bz2', 'RC_2016-05.bz2',
                         'RC_2016-06.bz2']

    #comment_files = ['RC_2011-01.bz2']


    for filename in comment_files:
        print(filename)
        extract_comments(filename, top_n_subreddits)

    sorted_dict_subreddits = sorted(subreddits_noComments.items(), key=lambda kv: kv[1], reverse=True)

    f = open("top_100_byNoComments_outof_top1000_byNoSubscribers.csv", 'w')
    try:
        writer = csv.writer(f, dialect='excel')
        writer.writerow(['subreddit_id', 'subbreddit_name', 'number_of_subscribes', 'number_of_comments'])
        for id, noComments in sorted_dict_subreddits[:100]:
            writer.writerow([id, top_n_subreddits[id][0], top_n_subreddits[id][1], noComments])
    finally:
        f.close()

    #print('# of reddit posts with comments ', len(post_commentTimes))






