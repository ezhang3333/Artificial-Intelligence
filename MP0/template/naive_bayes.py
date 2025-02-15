# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Last Modified 8/23/2023


"""
This is the main code for this MP.
You only need (and should) modify code within this file.
Original staff versions of all other files will be used by the autograder
so be careful to not modify anything else.
"""


import reader
import math
from tqdm import tqdm
from collections import Counter

stop_words = {"the", "a", "an", "is", "are", "was", "were", "to", "of","for", "and", "but", "or", "on", "in"}

'''
util for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    
    
    def remove_stop_words(dataset):
        filtered_dataset = []

        for review in dataset:
            filtered_review = []

            for word in review:
                if word not in stop_words:
                    filtered_review.append(word)

            filtered_dataset.append(filtered_review)

        return filtered_dataset    

    train_set = remove_stop_words(train_set)
    dev_set = remove_stop_words(dev_set)
    
    
    return train_set, train_labels, dev_set, dev_labels


"""
Main function for training and predicting with naive bayes.
    You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def naive_bayes(train_labels, train_data, dev_data, laplace=1.4, pos_prior=0.8, silently=False):
    print_values(laplace,pos_prior)

    pos_reviews = []
    neg_reviews = []

    for i in range(len(train_labels)):
        if(train_labels[i] == 1):
            pos_reviews.append(train_data[i])
        else:
            neg_reviews.append(train_data[i])


    pos_counter = Counter()
    neg_counter = Counter()

    for review in pos_reviews:
        for word in review:
                pos_counter[word] += 1

    for review in neg_reviews:
        for word in review:
                neg_counter[word] += 1
        

    total_pos_words = sum(pos_counter.values())
    total_neg_words = sum(neg_counter.values())

    vocab = set(pos_counter.keys()).union(set(neg_counter.keys()))
    vocab_size = len(vocab)
    if vocab_size == 0:
        vocab_size = 1

    pos_word_probs = {}
    neg_word_probs = {}

    for word in vocab:
        pos_word_probs[word] = math.log((pos_counter[word] + laplace) / (total_pos_words + (laplace * vocab_size)))
        neg_word_probs[word] = math.log((neg_counter[word] + laplace) / (total_neg_words + (laplace * vocab_size)))

    log_default_pos = math.log((laplace) / (total_pos_words + (laplace * vocab_size)))
    log_default_neg = math.log((laplace) / (total_neg_words + (laplace * vocab_size)))


    log_pos_prior = math.log(pos_prior)
    log_neg_prior = math.log(1 - pos_prior)

    yhats = []
    for doc in tqdm(dev_data, disable=silently):
        log_prob_pos = log_pos_prior
        log_prob_neg = log_neg_prior        

        for word in doc:
            log_prob_pos += pos_word_probs.get(word, log_default_pos)
            log_prob_neg += neg_word_probs.get(word, log_default_neg) 

        #print(f"Review: {doc}")  # Print words in the review
        #print(f"Log P(Positive): {log_prob_pos}, Log P(Negative): {log_prob_neg}")
        #print(f"Prediction: {1 if log_prob_pos > log_prob_neg else 0}")

        if log_prob_pos > log_prob_neg:
            yhats.append(1)
        else:
            yhats.append(0)

    return yhats
