import sys, random
import numpy as np
import reader

'''
Perform one layer of transformer inference on a dataset
using embeddings, positional_embeddings, and weight matrices 
specified in the file model.json
'''

def softmax(logits):
    '''
    Return the row-wise softmax of a matrix.  
    @param:
    logits - any numpy array
    @return:
    probs - probs[i,j] = exp(logits[i,j])/sum(exp(logits[i,:])), but 
      be careful to normalize so that you avoid overflow errors!
    '''

    logits = np.array(logits)
    if logits.ndim == 1:
        logits = logits.reshape(1, -1)
    
    max_in_rows = np.max(logits, axis=1, keepdims=True)
    stable_logits = logits - max_in_rows
    
    exps = np.exp(stable_logits)
    softmax_vals = exps / np.sum(exps, axis=1, keepdims=True)
    
    if softmax_vals.shape[0] == 1:
        return softmax_vals.flatten()
    
    return softmax_vals

    

def forward(XK, XQ, WK, WO, WQ, WV):
    '''
    Perform one layer of transformer inference, using trained model, on given data.

    @param:
    XK - (T-2)-by-V array containing embeddings of words to be used for keys and values
    XQ - 2-by-V array containing embeddings of words to be used for queries
    WK - V-by-d array mapping X to K
    WO - d-by-V array mapping C to O
    WQ - V-by-d array mapping X to Q
    WV - V-by-d array mapping X to V

    @return:
    A - 2-by-(T-2) array, A[i,j] is attention the i'th query pays to the j'th key
    C - 2-by-d array, context vectors from which P is computed
    K - (T-2)-by-d array, key vectors computed from XK
    O - 2-by-V array, O[i,j] is probability that i'th output word should be j
    Q - 2-by-d array, query vectors computed from XQ
    V - (T-2)-by-d array, value vectors computed from XK
    '''

    K = np.dot(XK, WK)  
    V = np.dot(XK, WV) 
    Q = np.dot(XQ, WQ) 
    
    scores = np.dot(Q, K.T)  
    
    A = softmax(scores)
    C = np.dot(A, V)  
    
    logits = np.dot(C, WO)  
    O = softmax(logits)
    
    return A, C, K, O, Q, V


def generate(embeddings, vocabulary, WK, WO, WQ, WV):
    '''
    Perform inference on the provided embeddings, and report the generated sentences.
    
    @param:
    embeddings - a list of one-hot embedding matrices, one per sentence
    vocabulary - a list of words in the vocabulary
    WK - V-by-d array mapping X to K
    WO - d-by-V array mapping C to O
    WQ - V-by-d array mapping X to Q
    WV - V-by-d array mapping X to V

    @return:
    generated - a list of generated sentences, each as a list of space-separated words.
      The first T-2 words of each sentence should be vocabulary items indexed by the
      argmax of the first T-2 embeddings.  The last 2 words of each sentence should be
      vocabulary items indexed by the argmax of the two outputs computed by running
      the transformer with the provided WK, WO, WQ, and WV.
    '''
    
    generated = []
    
    for embedding in embeddings:
        XK, XQ, Y = reader.define_task(embedding)
        
        prompt_indices = np.argmax(XK, axis=1)
        prompt_words = [vocabulary[idx] for idx in prompt_indices]
        
        A, C, K, O, Q, V_matrix = forward(XK, XQ, WK, WO, WQ, WV)
        
        output_indices = np.argmax(O, axis=1)
        output_words = [vocabulary[idx] for idx in output_indices]
        
        sentence = prompt_words + output_words
        generated.append(sentence)
    
    return generated
    
    '''
    generated = []
    for X in embeddings:
        XK, XQ, _ = reader.define_task(X)
        _, _, _, O, _, _ = forward(XK, XQ, WK, WO, WQ, WV)
        sentence = [vocabulary[np.argmax(word)] for word in X]
        sentence.extend(vocabulary[np.argmax(O[i])] for i in range(O.shape[0]))
        generated.append(" ".join(sentence))
    return generated
    '''


def cross_entropy_loss(O, Y):
    '''
    Calculate losses from network outputs O if target one-hot vectors are Y.

    @param:
    O - NQ-by-V array.  O[n,v]=probability that n'th output is v.
    Y - NQ-by-V array. Y[n,v]=1 if n'th target is v, else Y[n,v]=0.
    
    @return:
    L - cross-entropy loss, summed over all rows
    dO - NQ-by-V array.  Derivatives of the loss with respect to the elements of O.
    '''
    O = np.array(O)
    Y = np.array(Y)

    O = np.maximum(O, sys.float_info.min)

    L = -np.sum(Y * np.log(O))       

    dO = - Y / O

    return L, dO

def gradient(XK, XQ, Y, WK, WO, WQ, WV, A, C, K, O, Q, V):
    '''
    Compute gradient of cross-entropy loss with respect to WK, WO, WQ, and WV
    given the input data in K, Q, and V, and the target outputs in Y.
    
    @param:
    XK - one embedding per row, first n-2 words in the sentence
    XQ - one embedding per row, 3rd-from-last and 2nd-from-last words in the sentence
    Y - one embedding per row, last two words in the sentence
    O - 2-by-V array, O[i,j] is probability that i'th output word should be j
    C - 2-by-d array, context vectors from which O is computed
    V - (T-2)-by-d array, value vectors of which each row of C is a weighted average
    A - 2-by-(T-2) array, A[i,j] is attention the i'th query pays to the j'th key
    K - (T-2)-by-d array, key vectors computed from XK
    Q - 2-by-d array, query vectors computed from XQ

    @return:
    dWK - gradient of cross-entropy with respect to WK
    dWO - gradient of cross-entropy with respect to WO
    dWQ - gradient of cross-entropy with respect to WQ
    dWV - gradient of cross-entropy with respect to WV
    '''
    
    d_logits = O - Y
    dWO = np.dot(C.T, d_logits)
    dC = np.dot(d_logits, WO.T)
    dA = np.dot(dC, V.T)
    dV_from_C = np.dot(A.T, dC)
    sum_A_dA = np.sum(A * dA, axis=1, keepdims=True)
    d_scores = A * (dA - sum_A_dA)
    dQ = np.dot(d_scores, K)
    dK_from_scores = np.dot(d_scores.T, Q)
    dWQ = np.dot(XQ.T, dQ)
    dWK = np.dot(XK.T, dK_from_scores)
    dWV = np.dot(XK.T, dV_from_C)

    return dWK, dWO, dWQ, dWV
    

def train(embeddings, WK, WO, WQ, WV, learningrate, num_iters):
    '''
    Train a transformer using stochastic gradient descent (SGD).
    Each iteration of SGD should choose one training sentence, uniformly at random,
    compute the loss and loss gradient for that one sentence,
    then adjust the parameters WK, WO, WQ and WV in the direction of the negative
    gradient scaled by the learningrate.

    @param:
    embeddings - embeddings[i][j,:] is one-hot vector of the j'th word in the i'th training sentence
    WK - the matrix that multiplies each embedding to produce a key
    WO - the matrix that multiplies the context vector to produce an output logit vector
    WQ - the matrix that multiplies each embedding to produce a query
    WV - the matrix that multiplies each embedding to produce a value
    learningrate - scalar learning rate
    num_iters - number of iterations of SGD to perform

    @return:
    losses - losses[t]=cross-entropy loss of t'th iteration
    WK - what WK has become after num_iters of training
    WO - what WO has become after num_iters of training
    WQ - what WQ has become after num_iters of training
    WV - what WV has become after num_iters of training
    '''
    
    losses = []

    for i in range(num_iters):
        index = random.randint(0, len(embeddings) - 1)
        X = embeddings[index]

        XK, XQ, Y = reader.define_task(X)
        A, C, K, O, Q, V = forward(XK, XQ, WK, WO, WQ, WV)

        L, dO = cross_entropy_loss(O, Y)
        losses.append(L)
        dWK, dWO, dWQ, dWV = gradient(XK, XQ, Y, WK, WO, WQ, WV, A, C, K, O, Q, V)
        WK -= learningrate * dWK
        WO -= learningrate * dWO
        WQ -= learningrate * dWQ
        WV -= learningrate * dWV


    return losses, WK, WO, WQ, WV

