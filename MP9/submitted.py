'''
Replace each RuntimeError with code that does what's
specified in the docstring, then submit to autograder.
'''
import numpy as np

def utility_gradients(logit, reward):
    '''
    Calculate partial derivatives of expected rewards with respect to logits.

    @param:
    logit - player i plays move 1 with probability 1/(1+exp(-logit[i]))
    reward - reward[i,a,b] is reward to player i if player 0 plays a, and player 1 plays b

    @return:
    gradients - gradients[i]= dE[reward[i,:,:]]/dlogit[i]
    utilities - utilities[i] = E[reward[i,:,:]]
      where the expectation is computed over the distribution of possible moves by both players.
    '''

    z = np.asarray(logit)               
    R = np.asarray(reward)               

    p = 1 / (1 + np.exp(-z))            
    dp = p * (1 - p)                    

    P = np.stack([1 - p, p], axis=1)     

    uA = P[0] @ R[0] @ P[1]
    uB = P[0] @ R[1] @ P[1]
    utilities = np.array([uA, uB])

    vA = R[0] @ P[1]      
    vB = P[0] @ R[1]      

    gradA = dp[0] * (vA[1] - vA[0])
    gradB = dp[1] * (vB[1] - vB[0])
    gradients = np.array([gradA, gradB])

    return gradients, utilities
    


    

def strategy_gradient_ascent(logit, reward, nsteps, learningrate):
    path      = np.zeros((nsteps+1, 2), dtype=float)
    utilities = np.zeros((nsteps+1, 2), dtype=float)

    z = np.asarray(logit, dtype=float)
    path[0] = z
    _, utils = utility_gradients(z, reward)
    utilities[0] = utils

    for t in range(1, nsteps+1):
        grads, utils = utility_gradients(z, reward)
        z = z + learningrate * np.array(grads)
        path[t]      = z
        utilities[t] = utils

    # **drop the initial (t=0) row so shapes become (nsteps,2)**
    return path[1:], utilities[1:]

    

def mechanism_gradient(logit, reward):
    '''
    Calculate partial derivative of mechanism loss with respect to rewards.

    @param:
    logit - The goal is to make this pair of strategies a Nash equlibrium:
        player i plays move 1 with probability 1/(1+exp(-logit[i])), else move 0
    reward - reward[i,a,b] is reward to player i if player 0 plays a, and player 1 plays b

    @return:
    gradient - gradient[i,a,b]= derivative of loss w.r.t. reward[i,a,b]
    loss - half of the mean-squared strategy mismatch.
        Mean = average across both players.
        Strategy mismatch = difference between the expected reward that
        the player earns by cooperating (move 1) minus the expected reward that
        they earn by defecting (move 0).
    '''
    R = np.asarray(reward, dtype=float)  # (2,2,2)
    z = np.asarray(logit, dtype=float)   # (2,)

    # 1) compute scalar probs
    p = 1/(1+np.exp(-z))           # [pA, pB]
    pA, pB = p

    # 2) build full strategy vectors
    pA_vec = np.array([1-pA, pA])  # shape (2,)
    pB_vec = np.array([1-pB, pB])

    R_A = R[0]  # (2,2)
    R_B = R[1]

    # 3) mismatches
    #    mA = [−1,1]·(R_A @ pB_vec)
    #    mB = (pA_vec @ R_B)·[−1,1]
    mA = (R_A[1,:] - R_A[0,:]) @ pB_vec
    mB = pA_vec @ (R_B[:,1] - R_B[:,0])

    # 4) loss
    loss = 0.5 * (mA*mA + mB*mB)

    # 5) gradient w.r.t. reward
    grad = np.zeros_like(R)

    # ∂ℒ/∂R_A[a,b] = mA * (∂mA/∂R_A[a,b])
    #             = mA * pB_vec[b] * (+1 if a==1 else -1)
    signA = np.array([-1.0, +1.0])           # for a=0,1
    grad[0] = mA * (signA[:,None] * pB_vec[None,:])

    # ∂ℒ/∂R_B[a,b] = mB * (∂mB/∂R_B[a,b])
    #             = mB * pA_vec[a] * (+1 if b==1 else -1)
    signB = np.array([-1.0, +1.0])           # for b=0,1
    grad[1] = mB * (pA_vec[:,None] * signB[None,:])

    return grad, loss

def mechanism_gradient_descent(logit, reward, nsteps, learningrate):
    '''
    nsteps of gradient descent on the mean-squared strategy mismatch
    using simultaneous gradient ascent.

    @param:
    logit - The goal is to make this pair of strategies a Nash equlibrium:
        player i plays move 1 with probability 1/(1+exp(-logit[i])), else move 0.
    reward - Initial setting of the rewards.
        reward[i,a,b] is reward to player i if player 0 plays a, and player 1 plays b
    nsteps - number of steps of gradient descent to perform
    learningrate - learning rate

    @return:
    path - path[t,i,a,b] is the reward to player i of the moves (a,b) after t steps 
      of gradient descent (path[0,:,:,:] = initial reward).
    loss - loss[t] is half of the mean-squared strategy mismatch at iteration [t].
        Mean = average across both players.
        Strategy mismatch = difference between the expected reward that
        the player earns by cooperating (move 1) minus the expected reward that
        they earn by defecting (move 0).
    '''
    reward = np.asarray(reward, dtype=float)
    path   = np.zeros((nsteps+1, *reward.shape), dtype=float)
    loss   = np.zeros(nsteps+1, dtype=float)

    r = reward.copy()
    path[0] = r
    grad, L = mechanism_gradient(logit, r)
    loss[0] = L

    for t in range(1, nsteps+1):
        r = r - learningrate * grad

        path[t] = r
        grad, L = mechanism_gradient(logit, r)
        loss[t] = L

    return path, loss

    
