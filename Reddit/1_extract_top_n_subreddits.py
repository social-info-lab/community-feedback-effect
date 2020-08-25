import csv


top_n=1000

with open("subreddits_basic.csv", 'r') as f:
    lines = f.readlines()

dict_subreddit_name_subscribers = {}
dict_subreddit_subscribers = {}
dict_subreddit_name = {}
for line in lines:
    per_line = line.strip().split(',')
    if len(per_line) < 5: continue

    try:
        subreddit_id, subreddit_name, subreddit_subscribers = per_line[1], per_line[3], int(per_line[4])
    except:
        continue

    if subreddit_subscribers > 0:
        dict_subreddit_subscribers[subreddit_id] = subreddit_subscribers
        dict_subreddit_name[subreddit_id] = subreddit_name


sorted_dict_subreddits = sorted(dict_subreddit_subscribers.items(), key=lambda kv:kv[1], reverse=True)

f = open("top_1000_subreddits.csv", 'w')
try:
    writer = csv.writer(f, dialect='excel')
    writer.writerow(['subreddit_id', 'subbreddit_name', 'number_of_subscribes'])
    for id, num_subscribers in sorted_dict_subreddits[:top_n]:
        writer.writerow([id, dict_subreddit_name[id], num_subscribers])
        dict_subreddit_name_subscribers[id] = (dict_subreddit_name[id], num_subscribers)
finally:
    f.close()

print('# of topics: ', len(dict_subreddit_name_subscribers))
