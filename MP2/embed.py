import numpy as np

def initialize(data, dim):
    '''
    Initialize embeddings for all distinct words in the input data.
    Most of the dimensions will be zero-mean unit-variance Gaussian random variables.
    In order to make debugging easier, however, we will assign special geometric values
    to the first two dimensions of the embedding:

    (1) Find out how many distinct words there are.
    (2) Choose that many locations uniformly spaced on a unit circle in the first two dimensions.
    (3) Put the words into those spots in the same order that they occur in the data.

    Thus if data[0] and data[1] are different words, you should have

    embedding[data[0]] = np.array([np.cos(0), np.sin(0), random, random, random, ...])
    embedding[data[1]] = np.array([np.cos(2*np.pi/N), np.sin(2*np.pi/N), random, random, random, ...])

    ... and so on, where N is the number of distinct words, and each random element is
    a Gaussian random variable with mean=0 and standard deviation=1.

    @param:
    data (list) - list of words in the input text, split on whitespace
    dim (int) - dimension of the learned embeddings

    @return:
    embedding - dict mapping from words (strings) to numpy arrays of dimension=dim.
    '''
    embedding = {}

    for word in data:
        if word not in embedding:              
            embedding[word] = np.random.randn(dim)

    
    distinct = len(embedding)
    count = 0
    for key, value in embedding.items():
        angle = (2*np.pi*count) / distinct
        value[0] = np.cos(angle)
        value[1] = np.sin(angle)
        count += 1

    return embedding

def gradient(embedding, data, t, d, k):
    '''
    Calculate gradient of the skipgram NCE loss with respect to the embedding of data[t]

    @param:
    embedding - dict mapping from words (strings) to numpy arrays.
    data (list) - list of words in the input text, split on whitespace
    t (int) - data index of word with respect to which you want the gradient
    d (int) - choose context words from t-d through t+d, not including t
    k (int) - compare each context word to k words chosen uniformly at random from the data

    @return:
    g (numpy array) - loss gradients with respect to embedding of data[t]
    '''
    vectorT = embedding[data[t]]
    gradient = np.zeros_like(vectorT)   
    
    for i in range(t-d, t+d+1):
        if i != t and 0 <= i < len(data):
            vectorTC = embedding[data[i]]

            sigmoid = 1 / (1 + np.exp(-np.dot(vectorT, vectorTC)))
            similarity = (sigmoid - 1) * vectorTC
            
            noise = 0
            for m in range(k):
                randomInt = np.random.randint(0,len(data))
                vectorI = embedding[data[randomInt]]
              
                sigmoid = 1 / (1 + np.exp(-np.dot(vectorT,vectorI)))

                noise += (sigmoid * vectorI)

            noise /= k   

            gradient += (similarity + noise)

    return gradient
           
def sgd(embedding, data, learning_rate, num_iters, d, k):
    '''
    Perform num_iters steps of stochastic gradient descent.

    @param:
    embedding - dict mapping from words (strings) to numpy arrays.
    data (list) - list of words in the input text, split on whitespace
    learning_rate (scalar) - scale the negative gradient by this amount at each step
    num_iters (int) - the number of iterations to perform
    d (int) - context width hyperparameter for gradient computation
    k (int) - noise sample size hyperparameter for gradient computation

    @return:
    embedding - the updated embeddings
    '''

    for i in range(num_iters):
        randomInt = np.random.randint(0, len(data))

        gradientVector = gradient(embedding, data, randomInt, d, k)
        
        embedding[data[randomInt]] = embedding[data[randomInt]] - (learning_rate * gradientVector)

    return embedding
    

