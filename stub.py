# Imports.
import numpy as np
import numpy.random as npr

import random

from collections import deque

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras import regularizers


import time

import pygame as pg
from SwingyMonkey import SwingyMonkey

class Learner(object):

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        
        self.state = None
        self.action = None
        
        self.action_history = []
        
        self.state_size = 10  # monkey_top/bottom/velocity, feature1~4, tree_dist/top/bottom
        self.action_size = 2  # either jump (1) or swing (0)
 
        self.memory = []
        
        self.gamma = 0.9999
        self.epsilon = 0  # exploration rate
        self.epsilon_min = 0.01  # explore at least at this rate
        self.epsilon_decay = 0.99  # how fast exploration rate decays
        self.learning_rate = 0.01
        self.lamda = 0.001
        
        self.tick = 0
        self.vel1 = 0.0
        self.vel2 = 0.0
        self.gravity = None
        
        self.Q_model = self._build_Q_model()
        
    def _build_Q_model(self):
        # Neural Net for Deep-Q learning Model
        
        model = Sequential()
        #model.add(Dense(24, input_dim=self.state_size, activation='relu', kernel_regularizer=regularizers.l2(self.lamda)))
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(24, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(24, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(24, activation='relu')) # reached 8 tree
        model.add(Dropout(0.5))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=self.learning_rate))
        return model 

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.state = None
        self.action = None
        self.tick = 0
        self.memory = []
        self.action_history = []
        return
    
    def _remember(self, state, action, last_state, last_action, last_reward):
        self.memory.append( (state, action, last_state, last_action, last_reward) )
    
    def _state2input(self, state):
        
        tree_width = 115
        
        delta_t1 = state["tree"]["dist"]/25.0
        delta_t2 = (state["tree"]["dist"]+tree_width)/25.0
        delta_y1 = 0.5*self.gravity*(delta_t1**2) + state["monkey"]["vel"]*delta_t1
        delta_y2 = 0.5*self.gravity*(delta_t2**2) + state["monkey"]["vel"]*delta_t2
        
        
        Q_model_input = [state["monkey"]["top"],
                         state["monkey"]["bot"],
                         state["monkey"]["vel"],
                         self.gravity*(delta_t1**2),
                         self.gravity*(delta_t2**2),
                         delta_y1,
                         delta_y2,
                         state["tree"]["dist"],
                         state["tree"]["top"],
                         state["tree"]["bot"]]
        #Q_model_input = np.reshape(Q_model_input, [1, self.state_size])
        return Q_model_input
    
    def train(self, score):
        if len(self.memory) == 0:
            print(".",end="")
            return
        
        print("Lasted: %3d / Epsilon: %4f/ memory size: %4d / " % (self.tick, self.epsilon, len(self.memory) ), end="")
        #print(''.join([str(c) for c in self.action_history]))
        
        if score>30:
            training_epoch = 30
        else:
            training_epoch = score
        
        X_train = []
        y_train = []
        for state, action, last_state, last_action, last_reward in self.memory:    
                
                s = self._state2input(state)
                ls = self._state2input(last_state)
                
                Q_sa = ( last_reward + self.gamma*np.amax(self.Q_model.predict( np.reshape(s, [1, self.state_size]) )[0]) )
                target_f = self.Q_model.predict( np.reshape(ls, [1, self.state_size]) )[0]
                target_f[action] = Q_sa
                
                x = ls
                X_train.append(x)
                y_train.append(target_f)
                
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        self.Q_model.fit(X_train, y_train, epochs=training_epoch, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return
    
    def action_callback(self, state):

        self.tick += 1

        # Act0: just swing
        if self.tick == 1:
            self.vel1 = state["monkey"]["vel"]

            self.state = state
            self.action = 0
            self.action_history.append(0)
            return False # just swing in tick1

        # Act1: just swing
        elif self.tick == 2:
            self.vel2 = state["monkey"]["vel"]
            self.gravity = self.vel2 - self.vel1

            self.last_state  = self.state
            self.last_action = self.action
            self.state = state
            self.action = 0
            self.action_history.append(0)
            return False # also just swing in tick2

        # Act randomly to explore
        elif np.random.rand() <= self.epsilon:
            # random action biased toward swinging since it seems that jumping applies large momentum and often overshoots
            # biasing toward swinging might increase survival rate before hitting the first tree
            # hence allowing for the model to learn to pass the first tree
            n = random.randint(1,101)
            if n>70:
                random_action = 1
            else:
                random_action = 0

            self.last_state  = self.state
            self.last_action = self.action
            self.state = state
            self.action = random_action
            self.action_history.append( random_action )
            return (random_action==1) # randomly explore
        
        # Act to maximize predicted Q-value
        if (state["tree"]["top"]+state["tree"]["bot"])/2.0 < state["monkey"]["bot"]:
            action = 0
        elif (state["tree"]["top"]+state["tree"]["bot"])/2.0 > state["monkey"]["top"]:
            action = 1
        else:  
            #Q_prediction = self.Q_model.predict( self._state2input(state) )
            Q_prediction = self.Q_model.predict( np.reshape(self._state2input(state), [1, self.state_size]) )
            
            action = np.argmax(Q_prediction[0])
        
        self.last_state  = self.state
        self.last_action = self.action
        self.state = state
        self.action = action
        if self.last_reward == None:
            self.last_reward = 0.0
        self._remember(self.state, self.action, self.last_state, self.last_action, self.last_reward)
        self.action_history.append( self.action )
        return (self.action==1)

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        
        state = self.state
        last_state = self.last_state
        
        if reward == 1: # just passed a tree
            reward = 5.0 # further incentivize passing trees => seems surprisingly effective for learning!

        if reward == -5:
            reward = -10
        if state["tree"]["top"] < state["monkey"]["top"]:
            reward -= 1.0
            if last_state is not None:
                if last_state["tree"]["top"] < last_state["monkey"]["top"]:
                    reward -= 1.0
        elif state["tree"]["bot"] > state["monkey"]["bot"]:
            reward -= 1.0
            if last_state is not None:
                if last_state["tree"]["bot"] > last_state["monkey"]["bot"]:
                    reward -= 1.0
        
        #if len(self.action_history) > 5:
        #    if np.sum(self.action_history[-5:]) > 4:
        #        reward -= 1.0
        #        print("X",end="")
        
        self.last_reward = reward


def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)
        
        # Loop until you hit something.
        while swing.game_loop():
            pass
        
        # Save score history.
        hist.append(swing.score)
        print("[Game #%d / Score: %d / " % (ii,swing.score), end="")
        # Train learner on the last game
        toc = time.time()
        learner.train(swing.score)
        tic = time.time()
        print("training time: %3.3f]" % float(tic-toc))
        
        # Reset last_state, last_action, last_reward, and game memory of the learner (learned parameters are retained).
        learner.reset()
        
    return


if __name__ == '__main__':

    # Select agent.
    agent = Learner()

    #print("Initial \n", agent.Q_model.layers[0].get_weights()[0])
    agent.Q_model.load_weights('my_weights.h5', by_name=True)
    #agent.epsilon = 0.015
    #print("loading \n", agent.Q_model.layers[0].get_weights()[0])
    
    # Empty list to save history.
    hist = []

    # Run games. 
    run_games(agent, hist, 10, 10)
    
    #agent.Q_model.save_weights('my_weights.h5')
    #print("Saved \n", agent.Q_model.layers[0].get_weights()[0])

    # Save history. 
    #np.save('hist',np.array(hist))


