# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 08:44:54 2020

@author: vaibhav_bhanawat
"""
def split_train_test_data(dataset):
    train_positive_tweets = dataset[(dataset.SentimentCategory == 4)]
    train_negative_tweets = dataset[(dataset.SentimentCategory == 0)]
    

    test_negative_tweets = train_negative_tweets.head(100000)
    test_positive_tweets = train_positive_tweets.head(100000)
    
     # taking only 8 million records from 1.6 million data for fast computation.
    train_negative_tweets.drop(train_negative_tweets.head(100000).index, inplace=True)
    train_positive_tweets.drop(train_positive_tweets.head(100000).index, inplace=True)
    
    # merge training dataset
    train_tweets =  train_negative_tweets.append(train_positive_tweets)
    
    # merge test dataset
    test_tweets = test_negative_tweets.append(test_positive_tweets)
    return train_tweets, test_tweets;

import pandas as pd

# import dataset
dataset = pd.read_csv('sentiments_training_data.csv', encoding = 'latin_1', 
                      error_bad_lines = False, names = ['SentimentCategory', 'Id', 'Time', 'Flag', 'User', 'Comment'])

import nltk
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# downloading Stop words to remove all the unnecessary words 
#Total 1600000 records we are spliting 1400000(train) + 2000000(test) 
nltk.download('stopwords')

# spliting test and training dataset. Data is not random
train_tweets, test_tweets = split_train_test_data(dataset)
train_tweets = train_tweets.reset_index(drop = True)
test_tweets = test_tweets.reset_index(drop = True)

train_corpus = []
for i in range(0, len(train_tweets)):
    comment = re.sub('[^a-zA-Z]', ' ', train_tweets['Comment'][i])
    comment = comment.lower()
    comment = comment.split()
    porter = PorterStemmer()
    comment = [porter.stem(word) for word in comment if not word in set(stopwords.words('english'))]
    comment = ' '.join(comment)
    train_corpus.append(comment)

test_corpus = []
for i in range(0, len(test_tweets)):
    comment = re.sub('[^a-zA-Z]', ' ', test_tweets['Comment'][i])
    comment = comment.lower()
    comment = comment.split()
    porter = PorterStemmer()
    comment = [porter.stem(word) for word in comment if not word in set(stopwords.words('english'))]
    comment = ' '.join(comment)
    test_corpus.append(comment)

# create bag of words for training and test data
from sklearn.feature_extraction.text import CountVectorizer
countVector = CountVectorizer(max_features = 1000)
train_indep_matrix = countVector.fit_transform(train_corpus).toarray()
train_dep_matrix = train_tweets.iloc[:, 0]

test_indep_matrix = countVector.fit_transform(test_corpus).toarray()
test_dep_matrix = test_tweets.iloc[:, 0]


# use any classification model
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state = 0)
classifier.fit(train_indep_matrix, train_dep_matrix)
test_pred = classifier.predict(test_indep_matrix)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_dep_matrix, test_pred)
