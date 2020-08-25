import os
import re
import csv
from collections import defaultdict, Counter
import datetime
import matplotlib.pyplot as plt
import numpy as np
import bisect
from scipy import stats
from wordcloud import WordCloud
import matplotlib.image as mpimg

def getTopicScoresForWordClouds(topic_file_name):

    with open(topic_file_name) as f:
        lines =  f.readlines()

    topics_scores = defaultdict(list)
    topic_line = ''
    for line in lines:
        if 'Topic' in line:
            topic_line_vals = re.split('[ :\t\n]',line)
            topic_line_vals = [val for val in topic_line_vals if val!='']
            topic_line = topic_line_vals[1]
            topics_scores[topic_line].append((topic_line_vals[2], float(topic_line_vals[3])))
        else:
            line_vals = re.split('[\t\n]',line)[1:-1]
            topics_scores[topic_line].append((line_vals[0], float(line_vals[1])))

    return topics_scores

def generateWordClouds(topic_scores, topic):
    # Initialize the word cloud
    wc = WordCloud(
        background_color="white",
        max_words=20,
        width=800,
        height=600
    )

    # Generate the cloud
    wc.generate_from_frequencies(dict(topic_scores))

    # Save the could to a file
    wc.to_file("wordClouds/wordword_cloud"+topic+".png")

if __name__ == '__main__':
    topic_file_name = 'WordsInTopics.txt'
    topics_scores = getTopicScoresForWordClouds(topic_file_name)
    for topic in topics_scores:
        generateWordClouds(topics_scores[topic], topic)
