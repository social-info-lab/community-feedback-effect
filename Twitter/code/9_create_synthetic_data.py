import pandas as pd
import numpy as np
import json

np.random.seed(2)

#import pandas as pd
#data = pd.read_csv('logit_reg_data_100.csv')
#new_data = data[['timestamp', 'user', 'topic', 'event_time_bin']]
#new_data.head()
#new_data.to_csv('subset_twitter_data.csv', index=False)
# >>> new_data.to_csv('subset_twitter_data.csv', index=False)

def logit(x):
    return np.log(x/(1-x))

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def event_function(t, t0, pi = 4, g0 = 4):
    if t < t0:
        return 0
    else:
        return g0*np.exp(-1*(t-t0)/pi)


def chooseT0(Evt, n):
    if n <= Evt[0]:
        return Evt[0]
    elif n > Evt[-1]:
        return Evt[-1]
    else:
        for j, v in enumerate(Evt):
            if n <= v:
                return Evt[j-1]

def create_synthetic_data():
    data = pd.read_csv('../data/regression_data/subset_twitter_data.csv') #, nrows=100000)
    data = data.sort_values(by=['timestamp'])
    data = data[data['timestamp'] <= 31]

    user_noPosts = data[['user', 'timestamp']].groupby('user').size().to_frame('count').reset_index()
    users_selected = user_noPosts[user_noPosts['count'] > 50]

    data = data[data['user'].isin(users_selected['user'].values)]
    print(data.head())
    print(data.shape)

    number_of_events = 5
    number_of_topics = 2
    number_of_days = 20
    events_per_topic = np.random.randint(number_of_days, size=(number_of_topics, number_of_events))
    events_per_topic = np.sort(events_per_topic, axis = 1)
    print(events_per_topic)
    event_g_k_t = np.zeros(data.shape[0])
    prob_k_t = np.zeros(data.shape[0])
    prob_u_k_t = np.zeros(data.shape[0])
    fb = np.zeros(data.shape[0])

    users = data['user'].unique()
    user_idx = dict([(user, u) for u, user in enumerate(users)])
    alpha_u = np.random.randint(2, size=len(users)) #np.random.choice(2, size=len(users), p=[0.5, 0.5])# #np.random.normal(0, 1, len(users))
    alpha_pos = 1
    alpha_u[alpha_u>0] = alpha_pos
    print('# positive: ', len(np.where(alpha_u == alpha_pos )[0]))
    print('# zero: ', len(np.where(alpha_u == 0)[0]))
    alphas_users = dict([(user, int(alpha_u[u])) for user, u in user_idx.items()])
    print('# users: ', len(alphas_users))
    with open('users_alphas_synthetic2.json', 'w') as f:
        json.dump(alphas_users, f)

    sigma = 0.5

    # 100 topics and 5 events
    '''
    sel_idxs = []
    topics = []
    for i in range(data.shape[0]):
        if i%500000 == 0: print(i)
        t = data['timestamp'].iloc[i]
        if t <= 20:
            sel_idxs.append(i)
        else:
            continue
        topic = data['topic'].iloc[i]
        topics.append(topic)
        n = data['event_time_bin'].iloc[i]   # day index of the year
        prob_topic = data['prob_topic'].iloc[i]
        t0 = chooseT0(events_per_topic[topic], n)
        event_g_k_t[i] = event_function(t, t0)
        prob_k_t[i] = sigmoid(prob_topic+event_g_k_t[i])
        fb[i] = np.random.normal(event_g_k_t[i], sigma, 1)[0]
        user =  data['user'].iloc[i]

        prob_u_k_t[i] = sigmoid(prob_topic + event_g_k_t[i] + alphas_users[user] * fb[i])

    '''
    # binary topics
    topics = []
    prob_topics = np.zeros(data.shape[0])
    alphas_posts = np.zeros(data.shape[0])

    events_per_topic = {0: [0, 15], 1:[5, 20]}
    for i in range(data.shape[0]):
        if i % 500000 == 0: print(i)
        t = data['timestamp'].iloc[i]

        topic = np.random.randint(2, size=1)[0] #data['topic'].iloc[i]
        topics.append(topic)
        n = data['event_time_bin'].iloc[i]  # day index of the year
        prob_topic = -0.5 #-1.0 #logit(0.5)
        prob_topics[i] = prob_topic
        t0 = chooseT0(events_per_topic[topic], n)
        g_k_t = event_function(t, t0)
        #if topic == 0: g_k_t = 0

        user = data['user'].iloc[i]
        alphas_posts[i] = alphas_users[user]
        event_g_k_t[i] = g_k_t
        A = 0
        fb[i] = np.random.normal(A*event_g_k_t[i], 2*sigma, 1)[0] #np.random.poisson(5+event_g_k_t[i], 1)


        #val2 = prob_topic + alphas_users[user] * fb[i] #+ event_g_k_t[i]
        #if val2 > 0:
        #    print(val2)


    fb_posts =  pd.Series(fb).rank(pct=True).values
    print(len(np.where(fb_posts > 0.5)[0]))
    val = prob_topics + alphas_posts * fb_posts
    print(val[:50])
    prob_k_t = sigmoid(prob_topics + alphas_posts * fb_posts)
    prob_u_k_t = sigmoid(prob_topics + event_g_k_t + alphas_posts * fb_posts)
    #prob_u_k_t = sigmoid(prob_topics + 0 + alphas_posts * fb_posts)
    data['g_k_t'] = event_g_k_t
    data['prob_k_t'] = np.random.binomial(1, prob_k_t, len(prob_k_t))

    print(len(np.where(prob_k_t > 0.5)[0]))
    data['topic'] = topics
    data.to_csv('../data/regression_data/synthetic_prob_k_t.csv', index=False)

    data['feedback'] = fb
    data['prob_topic']= prob_topics
    data['prob_u_k_t'] = np.random.binomial(1, prob_u_k_t, len(prob_u_k_t))
    print(len(np.where(prob_u_k_t > 0.5)[0]))
    data['topic'] = topics

    data.to_csv('../data/regression_data/synthetic_prob_u_k_t.csv', index=False)



if __name__ == '__main__':
    new_data = create_synthetic_data()
