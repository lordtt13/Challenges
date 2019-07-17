# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 21:46:00 2019

@author: tanma
"""

import re
import pickle 
import pandas as pd 
from collections import Counter


train = pd.read_csv('train.csv')
train = train[:1000000]
X = train['comment_text']

med = []
for i in X:
    review = re.sub('[^a-zA-Z\']', ' ', str(i))
    review = review.lower()
    review = review.split(' ')
    med.append(review)

cleaned = []
for i in med:
    cleaned.append(' '.join(i))

actual_words = []
for i in cleaned:
    review = i.split()
    for j in review:
        actual_words.append(j)

counts = Counter(actual_words)

"""
actual_stopwords = []
for i,j in zip(counts,counts.values()):
    if(j < 6 or j > 10000):
        actual_stopwords.append(i)
    elif(len(list(i)) < 3):
        actual_stopwords.append(i)

     
actual_split = []
for i in cleaned:
    review = i.split()
    review = [word for word in review if not word in actual_stopwords]
    actual_split.append(' '.join(review))
"""

train['comment_text'] = cleaned

with open ("data_array.pkl","wb") as file:
    arr = pickle.dump(train[['comment_text','id','target']],file)