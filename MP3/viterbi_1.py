"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""

import math
from collections import defaultdict, Counter
from math import log

# Note: remember to use these two elements when you find a probability is 0 in the training data.
epsilon_for_pt = 1e-5
emit_epsilon = 1e-5   # exact setting seems to have little or no effect


def training(sentences):    
    """
    Computes initial tags, emission words and transition tag-to-tag probabilities
    :param sentences:
    :return: intitial tag probs, emission words given tag probs, transition of tags to tags probs
    """
    init_prob = defaultdict(lambda: 0)
    trans_prob = defaultdict(lambda: defaultdict(lambda: 0))
    emit_prob = defaultdict(lambda: defaultdict(lambda: 0))
    tag_count = defaultdict(lambda: 0)
    vocab = set()
    
    # Count occurrences
    for sentence in sentences:
        prev_tag = None
        for i, (word, tag) in enumerate(sentence):
            vocab.add(word)
            tag_count[tag] += 1
            emit_prob[tag][word] += 1
            
            if i == 0:
                init_prob[tag] += 1  # Initial tag count
            if prev_tag is not None:
                trans_prob[prev_tag][tag] += 1
            prev_tag = tag
    
    # Compute smoothed probabilities
    total_sentences = sum(init_prob.values())
    total_tags = len(tag_count)
    vocab_size = len(vocab)
    
    # Initial probabilities
    for tag in tag_count:
        init_prob[tag] = (init_prob[tag] + epsilon_for_pt) / (total_sentences + epsilon_for_pt * (total_tags + 1))
    
    # Transition probabilities
    for tag_a in tag_count:
        total_trans = sum(trans_prob[tag_a].values())
        for tag_b in tag_count:
            trans_prob[tag_a][tag_b] = (trans_prob[tag_a][tag_b] + epsilon_for_pt) / (total_trans + epsilon_for_pt * (total_tags + 1))
    
    # Emission probabilities
    for tag in tag_count:
        total_emissions = sum(emit_prob[tag].values())
        for word in emit_prob[tag]:
            emit_prob[tag][word] = (emit_prob[tag][word] + emit_epsilon) / (total_emissions + emit_epsilon * (vocab_size + 1))
        # Assign probability for unknown words
        emit_prob[tag]['UNKNOWN'] = emit_epsilon / (total_emissions + emit_epsilon * (vocab_size + 1))
    
    return init_prob, emit_prob, trans_prob

def viterbi_stepforward(i, word, prev_prob, prev_predict_tag_seq, emit_prob, trans_prob):
    """
    Does one step of the viterbi function
    :param i: The i'th column of the lattice/MDP (0-indexing)
    :param word: The i'th observed word
    :param prev_prob: A dictionary of tags to probs representing the max probability of getting to each tag at in the
    previous column of the lattice
    :param prev_predict_tag_seq: A dictionary representing the predicted tag sequences leading up to the previous column
    of the lattice for each tag in the previous column
    :param emit_prob: Emission probabilities
    :param trans_prob: Transition probabilities
    :return: Current best log probs leading to the i'th column for each tag, and the respective predicted tag sequences
    """
    log_prob = {}
    predict_tag_seq = {}
    
    for curr_tag in emit_prob:
        max_prob = float('-inf')
        best_prev_tag = None
        best_sequence = None
        
        for prev_tag in prev_prob:
            transition = math.log(trans_prob[prev_tag].get(curr_tag, math.log(epsilon_for_pt)))
            emission = math.log(emit_prob[curr_tag].get(word, emit_prob[curr_tag].get('UNKNOWN', math.log(emit_epsilon))))
            prob = prev_prob[prev_tag] + transition + emission
            
            if prob > max_prob:
                max_prob = prob
                best_prev_tag = prev_tag
                best_sequence = prev_predict_tag_seq[best_prev_tag] + [curr_tag]
        
        log_prob[curr_tag] = max_prob
        predict_tag_seq[curr_tag] = best_sequence
    
    return log_prob, predict_tag_seq

def viterbi_1(train, test, get_probs=training):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    init_prob, emit_prob, trans_prob = get_probs(train)
    
    predicts = []
    
    for sen in range(len(test)):
        sentence=test[sen]
        length = len(sentence)
        log_prob = {}
        predict_tag_seq = {}
        # init log prob
        for t in emit_prob:
            if t in init_prob:
                log_prob[t] = log(init_prob[t])
            else:
                log_prob[t] = log(epsilon_for_pt)
            predict_tag_seq[t] = []

        # forward steps to calculate log probs for sentence
        for i in range(length):
            log_prob, predict_tag_seq = viterbi_stepforward(i, sentence[i], log_prob, predict_tag_seq, emit_prob,trans_prob)
            
        
        bestFinalTag = max(log_prob, key = log_prob.get)
        bestSequence = predict_tag_seq[bestFinalTag]

        predicts.append([(sentence[i], bestSequence[i]) for i in range(length)])

    return predicts