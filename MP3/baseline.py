"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    
    wordTags = {}
    testDataTags = []

    for sentence in train:
        for pair in sentence:
            word = pair[0]
            tag = pair[1]

            if word not in wordTags:
                wordTags[word] = {}
        
            if tag not in wordTags[word]:
                wordTags[word][tag] = 0

            wordTags[word][tag] += 1

    mostPopularTag = ""
    tagCounts = {}

    for word,tagDicts in wordTags.items():
        for tags,num in tagDicts.items():
            if tags not in tagCounts:
                tagCounts[tags] = 0

            tagCounts[tags] += num

    mostPopularTag = max(tagCounts, key = tagCounts.get)


    for sentence in test:
        sentenceList = []

        for word in sentence:
            if word not in wordTags:
                sentenceList.append((word, mostPopularTag))
            else:
                bestTag = max(wordTags[word], key = wordTags[word].get)
                sentenceList.append((word, bestTag))

        testDataTags.append(sentenceList)
        
    return testDataTags