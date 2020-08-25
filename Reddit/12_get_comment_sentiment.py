from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from collections import Counter

df = pd.read_csv('reddit_timestamp_comments.txt')
df.columns = ['post_id', 'timestamp', 'comment']
sentences = list(df['comment'].values)
print('# of sentences', len(sentences))

analyzer = SentimentIntensityAnalyzer()
sentiments = []
for s, sentence in enumerate(sentences):
    sentiment = 2
    try:
        vs = analyzer.polarity_scores(sentence)
        if vs['compound'] >= 0.05:
            sentiment = 1
        elif vs['compound'] <= -0.05:
            sentiment = -1
        else:
            sentiment = 0
    except:
        print(sentence)
        pass

    if s%100000 == 0:
        print(s)
        print("{:-<65} {}".format(sentence, str(vs)), sentiment)
    sentiments.append(sentiment)

print(Counter(sentiments))
df['sentiment'] = sentiments

print(df.head())

df.to_csv('reddit_timestamp_comment_sentiment.tsv', index=False, sep='\t')


