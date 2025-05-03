import numpy as np
import utils


class Agent:
    def __init__(self, actions, Ne=40, C=40, gamma=0.7, display_width=18, display_height=10):
        # HINT: You should be utilizing all of these
        self.actions = actions
        self.Ne = Ne  # used in exploration function
        self.C = C
        self.gamma = gamma
        self.display_width = display_width
        self.display_height = display_height
        self.reset()
        # Create the Q Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        
    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self, model_path):
        utils.save(model_path, self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self, model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        # HINT: These variables should be used for bookkeeping to store information across time-steps
        # For example, how do we know when a food pellet has been eaten if all we get from the environment
        # is the current number of points? In addition, Q-updates requires knowledge of the previously taken
        # state and action, in addition to the current state from the environment. Use these variables
        # to store this kind of information.
        self.points = 0
        self.s = None
        self.a = None
    
    def update_n(self, state, action):
        # increment visit count for (state, action)
        self.N[state][action] += 1
        return 

    def update_q(self, s, a, r, s_prime):
        # learning rate alpha = C / (C + N(s,a))
        alpha = self.C / (self.C + self.N[s][a])
        # estimate of optimal future value
        max_q_next = max(self.Q[s_prime][a_prime] for a_prime in self.actions)
        # TD update
        self.Q[s][a] += alpha * (r + self.gamma * max_q_next - self.Q[s][a])
        return 
        
    def act(self, environment, points, dead):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y]
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: chosen action between utils.UP, utils.DOWN, utils.LEFT, utils.RIGHT
        '''
        s_prime = self.generate_state(environment)

        # If training and we have a previous state-action, do updates
        if self._train and self.s is not None and self.a is not None:
            # compute reward
            if dead:
                reward = -1
            elif points > self.points:
                reward = 1
            else:
                reward = -0.1

            # update N then Q
            self.update_n(self.s, self.a)
            self.update_q(self.s, self.a, reward, s_prime)

        # Choose action: exploration or greedy
        if self._train:
            # exploration function f(s,a)
            def f(a):
                return 1 if self.N[s_prime][a] < self.Ne else self.Q[s_prime][a]
            # tie-break by higher action code: RIGHT>LEFT>DOWN>UP
            action = max(self.actions, key=lambda a: (f(a), a))
        else:
            # evaluation: greedy
            action = max(self.actions, key=lambda a: (self.Q[s_prime][a], a))

        # update internal trackers
        self.s = s_prime
        self.a = action
        self.points = points
        
        return action

    def generate_state(self, environment):
        '''
        Discretize raw environment into a state-tuple for Q-table indexing.
        '''
        FOOD_DIR_X, FOOD_DIR_Y, ADJOINING_WALL_X, ADJOINING_WALL_Y, ADJOINING_BODY_TOP, ADJOINING_BODY_BOTTOM, ADJOINING_BODY_LEFT,ADJOINING_BODY_RIGHT = 0,1,2,3,4,5,6,7
        state = [0,0,0,0,0,0,0,0]
        snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y = environment
        # food dirs
        if(snake_head_x > food_x):
            state[FOOD_DIR_X] = 1
        elif(snake_head_x < food_x):
            state[FOOD_DIR_X] = 2
        if(snake_head_y > food_y):
            state[FOOD_DIR_Y] = 1
        elif(snake_head_y < food_y):
            state[FOOD_DIR_Y] = 2

        # adjoining walls
        rock_x_list = [rock_x, rock_x+1]
        # THE X PART
        if snake_head_y == rock_y: 
            for cur_rock_x in rock_x_list:
                if snake_head_x == cur_rock_x+1:
                    state[ADJOINING_WALL_X] = 1
            if state[ADJOINING_WALL_X] == 0:
                if snake_head_x == 1:
                    state[ADJOINING_WALL_X] = 1
                elif snake_head_x == self.display_width - 2:
                    state[ADJOINING_WALL_X] = 2
                else:
                    for cur_rock_x in rock_x_list:
                        if snake_head_x == cur_rock_x-1:
                            state[ADJOINING_WALL_X] = 2
        else:
            if snake_head_x == 1:
                state[ADJOINING_WALL_X] = 1
            elif snake_head_x == self.display_width - 2:
                state[ADJOINING_WALL_X] = 2  
        # THE Y PART
        for cur_rock_x in rock_x_list: 
            if snake_head_x == cur_rock_x:
                if snake_head_y == rock_y+1:
                    state[ADJOINING_WALL_Y] = 1
                elif snake_head_y == 1:
                    state[ADJOINING_WALL_Y] = 1
                elif snake_head_y == self.display_height - 2:
                    state[ADJOINING_WALL_Y] = 2
                elif snake_head_y == rock_y-1:
                    state[ADJOINING_WALL_Y] = 2
        else:
            if snake_head_y == 1:
                state[ADJOINING_WALL_Y] = 1
            elif snake_head_y == self.display_height - 2:
                state[ADJOINING_WALL_Y] = 2  

        # adjoining body
        up, down, left, right = (snake_head_x, snake_head_y-1), (snake_head_x, snake_head_y+1), (snake_head_x-1, snake_head_y), (snake_head_x+1, snake_head_y)
        for position in snake_body:
            if position == up:
                state[ADJOINING_BODY_TOP] = 1
            if position == down:
                state[ADJOINING_BODY_BOTTOM] = 1
            if position == left:
                state[ADJOINING_BODY_LEFT] = 1
            if position == right:
                state[ADJOINING_BODY_RIGHT] = 1

        return tuple(state)