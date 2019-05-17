"""
author: Maxime Darrin

Model for the mini decision game which only learns to detect squares
"""

from collections import deque

from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, Activation, MaxPooling2D
from keras.initializers import glorot_uniform
from keras.optimizers import Adam

from agent import DeepQLearner

def preprocess(observation):
    return observation.reshape((1,100,100,1)) / 255

class DQLDecision(DeepQLearner):

    def build_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, (8, 8), padding='same', strides=(4, 4), kernel_initializer=glorot_uniform(seed=None),
                              input_shape=self.input_shape))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=glorot_uniform(seed=None)))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer=glorot_uniform(seed=None)))
        self.model.add(Activation('relu'))
        self.model.add(Flatten())
        self.model.add(Dense(512, kernel_initializer=glorot_uniform(seed=None)))
        self.model.add(Activation('relu'))
        self.model.add(Dense(self.action_space))
        adam = Adam(lr=self.learning_rate)
        self.model.compile(loss='mse', optimizer=adam)
