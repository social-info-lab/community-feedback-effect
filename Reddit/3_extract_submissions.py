from bz2 import BZ2File as bzopen
import json
import codecs
import csv
from collections import defaultdict


num_posts_per_subreddit = defaultdict(int)

def extract_submissions(filename, top_n_subreddits):
    # reading a bz2 archive

    with bzopen(filename, "r") as bzfin:
        """ Handle lines here """
        cbz = codecs.iterdecode(bzfin, "utf-8")

        reddit_submissions = []
        for i, line in enumerate(cbz):
            try:
                post_dict = json.loads(line.rstrip())
                #print(post_dict)

                '''
                {
                'num_comments': 0,
                'author': 'magicks',
                'name': 't3_eut41',
                'subreddit_id': 't5_2qh0u',
                'created_utc': 1293944394,
                }

                n_topics, n_posts
                100, 220304
                200, 255879
                500, 306173
                1000, 338408

                '''
                # score = upvotes - downvotes
                post_id, subreddit_id, user, post_date, num_comments, score  = post_dict['name'], post_dict['subreddit_id'],\
                                                                       post_dict['author'], post_dict['created_utc'],\
                                                                       post_dict['num_comments'], post_dict['score']

                #print(post_id, num_comments, score)
                if subreddit_id in top_n_subreddits and user !='[deleted]':
                    reddit_submissions.append([post_id, subreddit_id, user, post_date, num_comments, score])
                    num_posts_per_subreddit[subreddit_id]+=1

            except:
                continue



    return reddit_submissions


if __name__ == '__main__':
    with open('top_100_byNoComments_outof_top1000_byNoSubscribers.csv', 'r') as f:
        lines = f.readlines()

    top_n_subreddits = {}
    for subreddit in lines:
        subreddit_id, subreddit_name, no_subs, noComments = subreddit.strip().split(',')
        top_n_subreddits[subreddit_id] = subreddit_name

    submission_files = ['RS_2016-01.bz2', 'RS_2016-02.bz2', 'RS_2016-03.bz2', 'RS_2016-04.bz2', 'RS_2016-05.bz2', 'RS_2016-06.bz2']

    #submission_files = ['RS_2011-01.bz2']

    total_no_posts = 0
    f = open("reddit_FirstHalf2016.csv", 'w')
    try:
        writer = csv.writer(f, dialect='excel')
        writer.writerow(['post_id', 'subbreddit_id', 'user', 'pos t_date', 'num_comments', 'score'])
        
        for filename in submission_files:
            print(filename)
            posts_attributes = extract_submissions(filename, top_n_subreddits)

            total_no_posts += len(posts_attributes)
            for post_attributes in posts_attributes:
                writer.writerow(post_attributes)

    finally:
        f.close()

    subreddits_with_most_posts = sorted(num_posts_per_subreddit.items(), key=lambda kv: kv[1], reverse=True)

    with open('subreddits_noPosts.json', 'w') as fp:
        json.dump(subreddits_with_most_posts, fp)


    print('# of subreddits with at least one post', len(subreddits_with_most_posts))
    print('# posts in the top_n subreddits', total_no_posts)







