import wget

comment_files = ['RC_2016-01.bz2','RC_2016-02.bz2','RC_2016-03.bz2','RC_2016-04.bz2','RC_2016-05.bz2','RC_2016-06.bz2']
submission_files = ['RS_2016-01.bz2','RS_2016-02.bz2','RS_2016-03.bz2','RS_2016-04.bz2','RS_2016-05.bz2','RS_2016-06.bz2']

url_comment_prefix = "https://files.pushshift.io/reddit/comments/"
url_post_prefix = "https://files.pushshift.io/reddit/submissions/"

filename = wget.download("https://files.pushshift.io/reddit/subreddits/subreddits_basic.csv")

for i, file in enumerate(comment_files):
    filename = wget.download(url_comment_prefix+comment_files[i])
    filename = wget.download(url_post_prefix + submission_files[i])




