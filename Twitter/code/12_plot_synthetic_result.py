import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from scipy.sparse import vstack, hstack, save_npz, load_npz
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, recall_score, precision_score, classification_report


def get_Stats_positiveUsers(result_file):

    with open('users_alphas_synthetic2.json') as fh:
        users_alphas = json.load(fh)

    no_user = len(users_alphas)
    print('#users: ', no_user)

    pos_alpha = []
    for user, alpha in users_alphas.items():
        if alpha > 0:
            pos_alpha.append(alpha)

    print('# positive users', len(pos_alpha))
    print('# negative users', no_user - len(pos_alpha))


    with open('columnnames2.txt') as f1:
        col_lines = f1.readlines()

    user_cols = {}
    u = 0
    alphas_array = np.zeros(no_user)
    for col in col_lines:
        if col.startswith('user_'):
            user_n = col.strip()
            user_cols[user_n[5:]] = u
            alphas_array[u] = users_alphas[user_n[5:]]
            u += 1

    print('original: ', alphas_array[:10])




    data_coef = load_npz(result_file)
    data_coef = data_coef.toarray()
    data = data_coef[:, 0:no_user]
    if 'model_user_event_fb' in result_file:
        data = data_coef[:, 1:no_user+1]

    print('inferred: ', data[0,:10])

    median_fb = np.median(data, axis=0)

    print('MAE :', np.mean(np.absolute(alphas_array - median_fb)))




    err_ci = (np.percentile(data, 99.5, axis=0) - np.percentile(data, 0.5, axis=0)) / 2.0
    #err_ci = 1.96*np.std(data, axis=0)
    #print(err_ci)
    top_users_score = np.percentile(median_fb, 99)
    #print('95 percentile score ', np.percentile(median_fb, 99))
    median_fb_pos = median_fb - err_ci
    median_fb_neg = median_fb + err_ci

    idxs_ones = np.where(median_fb_pos > 0)[0]
    idxs_minus = np.where(median_fb_neg < 0)[0]


    print('Accuracy, positive users')
    inf_alpha = np.zeros(no_user)
    inf_alpha[idxs_ones] = 1

    acc_pos = accuracy_score(alphas_array, inf_alpha)

    print('Accuracy zero users')
    inf_alpha = np.zeros(no_user)
    inf_alpha[idxs_ones] = 1
    inf_alpha[idxs_minus] = 1
    acc_zero = accuracy_score(alphas_array, inf_alpha)
    print("Accuracy: ", (acc_pos+acc_zero)/2)
    #print('Confusion matrix', confusion_matrix(alphas_array, inf_alpha))
    #print(classification_report(alphas_array, inf_alpha))
    #print('recall', recall_score(alphas_array, inf_alpha))
    #print('precision', precision_score(alphas_array, inf_alpha))

    pos_fb_idx = np.where(median_fb_pos > 0.0)[0]
    no_with_pos_fb = len(pos_fb_idx)
    pos_fb = median_fb[pos_fb_idx]
    print('Number of users with positive FB', no_with_pos_fb, no_with_pos_fb / no_user)
    print(np.min(pos_fb))

    neg_fb_idx = np.where(median_fb_neg < 0.0)[0]
    no_with_neg_fb = len(neg_fb_idx)
    neg_fb = median_fb[neg_fb_idx]
    print('Number of users with Non-Positive FB', no_with_neg_fb, no_with_neg_fb / no_user)

    other_idx = list(set(list(range(no_user))) - (set(pos_fb_idx) | set(neg_fb_idx)))
    print(len(other_idx))
    no_others = len(median_fb) - no_with_pos_fb - no_with_neg_fb
    print('Number of users that have neither pos or neg FB', no_others, no_others / no_user)

    print('Top pos users', no_with_pos_fb / len(median_fb))

    if 'model_user_event_fb' in result_file:
        topic_time_evt_matrix = np.median(data_coef[:, no_user + 1:], axis=0).flatten().reshape(2, -1)
        topic_time_evt_ci = 2.58*(np.std(data_coef[:, no_user + 1:], axis=0).flatten().reshape(2, -1)) / np.sqrt(data_coef.shape[0])

        #print(topic_time_evt_matrix[0])
        #print(topic_time_evt_matrix[1])

        np.savetxt('syn_inferred_TopicTrends2.txt', topic_time_evt_matrix)
        np.savetxt('syn_inferred_TopicTrends2_err.txt', topic_time_evt_ci)

def plot_topic_trends():
    data = pd.read_csv('../data/regression_data/synthetic_prob_u_k_t.csv')

    inf_data = np.loadtxt('syn_inferred_TopicTrends2.txt').reshape(2, -1)
    inf_data_err = np.loadtxt('syn_inferred_TopicTrends2_err.txt').reshape(2, -1)

    new_data_topic0 = data[data['topic'] == 0]
    new_data_topic1 = data[data['topic'] == 1]


    fig = plt.figure()
    plt.title('Timestamp vs Event')
    plt.ylabel('g_k(t)')
    plt.xlabel('Timestamp (by days)')
    plt.margins(y=.1, x=.1)
    plt.plot(new_data_topic0['timestamp'], new_data_topic0['g_k_t'], linestyle='-', color='b',  label = 'Topic 0')
    plt.plot(new_data_topic1['timestamp'], new_data_topic1['g_k_t'], linestyle='-', color='r',  label = 'Topic 1')

    plt.plot(inf_data[0],  linestyle='None', color='b', marker = '>', label = 'Inferred Topic 0')
    plt.plot(inf_data[1],  linestyle='None', color='r',  marker = 'o', label = 'Inferred Topic 1')

    plt.legend(loc='best')
    plt.savefig("timestamp_event1.png", bbox_inches='tight')



if __name__ == '__main__':
    print('Baseline model: model_user_fb \n ========================')
    get_Stats_positiveUsers("syn_bootstraps_coefs_model_user_fb2.npz")
    print('Full model: model_user_fb \n ========================')
    get_Stats_positiveUsers("syn_bootstraps_coefs_model_user_event_fb2.npz")

    plot_topic_trends()