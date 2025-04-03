'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
import numpy as np

epsilon = 1e-3

def compute_transition(model):
    '''
    Parameters:
    model - the MDP model returned by load_MDP()

    Output:
    P - An M x N x 4 x M x N numpy array. P[r, c, a, r', c'] is the probability that the agent will move from cell (r, c) to (r', c') if it takes action a, where a is 0 (left), 1 (up), 2 (right), or 3 (down).
    '''

    M, N = model.M, model.N
    # Initialize the transition probability array with zeros.
    P = np.zeros((M, N, 4, M, N))
    
    # Define the intended movement for each action.
    # 0: left, 1: up, 2: right, 3: down.
    actions = {
        0: (0, -1),  # left
        1: (-1, 0),  # up
        2: (0, 1),   # right
        3: (1, 0)    # down
    }
    
    # Loop over all cells.
    for r in range(M):
        for c in range(N):
            # For terminal states, leave P[r, c, :, :, :] as zeros.
            if model.TS[r, c]:
                continue
            # For non-terminal states, calculate the transitions for each action.
            for a in range(4):
                # Get the intended movement vector.
                intended_move = actions[a]
                # Compute the moves for the two perpendicular directions:
                # Left-turn (counter-clockwise) and right-turn (clockwise)
                left_move = (-intended_move[1], intended_move[0])   # 90° counter-clockwise
                right_move = (intended_move[1], -intended_move[0])    # 90° clockwise
                
                # Retrieve the probabilities for the three outcomes from model.D.
                p_intended = model.D[r, c, 0]
                p_left = model.D[r, c, 1]
                p_right = model.D[r, c, 2]
                
                # Helper: determine next cell given a move.
                def next_state(curr_r, curr_c, move):
                    new_r, new_c = curr_r + move[0], curr_c + move[1]
                    # Check for boundary violations or hitting a wall.
                    if new_r < 0 or new_r >= M or new_c < 0 or new_c >= N or model.W[new_r, new_c]:
                        return curr_r, curr_c
                    return new_r, new_c

                # Intended move.
                nr, nc = next_state(r, c, intended_move)
                P[r, c, a, nr, nc] += p_intended

                # Left (counter-clockwise) move.
                nr, nc = next_state(r, c, left_move)
                P[r, c, a, nr, nc] += p_left

                # Right (clockwise) move.
                nr, nc = next_state(r, c, right_move)
                P[r, c, a, nr, nc] += p_right

    return P
       

def compute_utility(model, U_current, P):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()
    U_current - The current utility function, which is an M x N array
    P - The precomputed transition matrix returned by compute_transition()

    Output:
    U_next - The updated utility function, which is an M x N array
    '''

    # Using numpy's einsum to compute the expected utility for each action.
    # Q[r,c,a] = sum_{r',c'} P[r,c,a,r',c'] * U_current[r',c']
    Q = np.einsum('rcaij,ij->rca', P, U_current)
    # Update each state's utility:
    # U_next(s) = R(s) + gamma * max_a Q(s, a)
    U_next = model.R + model.gamma * np.max(Q, axis=2)
    return U_next


def value_iterate(model):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()

    Output:
    U - The utility function, which is an M x N array
    '''
    
    # Predefined epsilon for convergence.
    epsilon = 1e-3
    # Precompute the transition matrix.
    P = compute_transition(model)
    # Initialize the utility function to all zeros.
    U_current = np.zeros((model.M, model.N))
    
    # Perform value iteration, with a maximum of 100 iterations.
    for _ in range(100):
        U_next = compute_utility(model, U_current, P)
        # Check for convergence.
        if np.all(np.abs(U_next - U_current) < epsilon):
            U_current = U_next.copy()
            break
        U_current = U_next.copy()
    return U_current
