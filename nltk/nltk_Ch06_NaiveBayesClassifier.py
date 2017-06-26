# coding: utf-8
import nltk
import random
from nltk.corpus import movie_reviews

# Extract all the words in the movie_reviews corpus
all_words = nltk.FreqDist(word for word in movie_reviews.words())

# extract the most common out of it
test=all_words.most_common(2000)

word_vector, freq_vector = zip(*test)

# Function to extract the feature set from each file
def extract_features(vector, filename):
    temp={}
    word_in_file=set(movie_reviews.words(filename))
    word_in_file = [word.lower() for word in word_in_file]
    for word in vector:
        temp[word]=word in word_in_file
    return temp

# test: extract_features(word_vector,'neg/cv999_14636.txt')

total_set = [(extract_features(word_vector,fileid),category) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]

# the data set will be in serial order , need to shuffle it
random.shuffle(total_set)

# which we will use to verify
test_set = total_set[501:]

# the training set
trial_set=total_set[0:500]

# The naive bayes classifier - train it
classifier = nltk.NaiveBayesClassifier.train(trial_set)

# classifier.classify(test_set[0])
# test_set[0]
# classifier.classify(test_set[0][0])
# classifier.classify(test_set[1][0])
# test_set[1][1]

print nltk.classify.accuracy(classifier, test_set)

