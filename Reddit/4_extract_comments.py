from bz2 import BZ2File as bzopen
import json
import codecs
import csv
from collections import defaultdict


post_commentTimes = defaultdict(list)

def extract_comments(filename, top_n_subreddits):
    # reading a bz2 archive

    with bzopen(filename, "r") as bzfin:
        """ Handle lines here """
        cbz = codecs.iterdecode(bzfin, "utf-8")
        #print(len(list(cbz)))

        reddit_submissions = []
        
        for i, line in enumerate(cbz):

            try:
                post_dict = json.loads(line.rstrip())

                #
                #{
                #    "created_utc":1506816000,
                #    "link_id":"t3_73ieyz"
                #}
                
                post_id, comment_date, subreddit_id = post_dict['link_id'], post_dict['created_utc'], post_dict['subreddit_id']

                #if i % 100000 == 0:
                #    print(i, post_id, comment_date, subreddit_id)

                if subreddit_id in top_n_subreddits:
                    post_commentTimes[post_id].append(comment_date)

        
            except:
                continue

        

    return reddit_submissions


if __name__ == '__main__':
    with open('top_100_byNoComments_outof_top1000_byNoSubscribers.csv', 'r') as f:
        lines = f.readlines()

    top_n_subreddits = {}
    for subreddit in lines:
        subreddit_id, subreddit_name, no_subs, no_comments  = subreddit.strip().split(',')
        top_n_subreddits[subreddit_id] = subreddit_name


    comment_files = ['RC_2016-01.bz2', 'RC_2016-02.bz2', 'RC_2016-03.bz2', 'RC_2016-04.bz2', 'RC_2016-05.bz2',
                         'RC_2016-06.bz2']

    #comment_files = ['RC_2011-01.bz2']


    for filename in comment_files:
        print(filename)
        extract_comments(filename, top_n_subreddits)
    
    f = open("reddit_comments.txt", 'w')
    try:
        writer = csv.writer(f, dialect='excel')
        for post_id in post_commentTimes:
            writer.writerow([post_id]+post_commentTimes[post_id])
    finally:
        f.close()


    print('# of reddit posts with comments ', len(post_commentTimes))

    




