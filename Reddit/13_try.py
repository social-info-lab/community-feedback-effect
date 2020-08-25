import pandas as pd
import numpy as np

df = pd.read_csv("regression_data/logit_reg_data.csv")
df = df.replace([np.inf, -np.inf], np.nan).dropna(how="all")

### Train/Test Split ###
df['index'] = list(range(df.shape[0]))
users_set = set(list(df['user'].values))
print('# of users: ', len(users_set))

train_idxs, test_idxs = [], []
avail_users = 0
for u, user in enumerate(list(users_set)):
    if u%1000==0:
        print(u, user)
    user_posts_idxs = list(df[df['user'] == user]['index'].values)
    test_start = len(user_posts_idxs) - 3
    if len(user_posts_idxs) >= 25:
        test_idxs += user_posts_idxs[test_start:]
        train_idxs += user_posts_idxs[:test_start]
        avail_users+=1
    else:
        #train_idxs +=user_posts_idxs
        df = df.drop(df[df['user']==user].index)
        #print(df.shape)

print('# users in test set ', avail_users)
print(df.shape)