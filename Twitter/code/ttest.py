from scipy import stats
import numpy as np
res_user_event_fb = np.loadtxt('accuracies_model_user_event_fb_pudt_100.txt')
res_user_event = np.loadtxt('accuracies_model_user_event_100.txt')

print(np.mean(res_user_event_fb))
print(np.mean(res_user_event))
print(stats.ttest_ind(res_user_event_fb,res_user_event))

