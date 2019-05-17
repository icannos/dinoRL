import cv2
import numpy as np

from agent import DeepQLearnerStep, DeepQLearner

from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, Activation, MaxPooling2D
from keras.initializers import glorot_uniform
from keras.optimizers import Adam


def preprocess(observation):
    """
    Reshape and resize the observation vector before processing it.
    :param observation: observations from the gym env
    :return: A smaller and reshaped vector (keras friendly)
    """
    observation = observation.reshape((150, 600, 3))

    observation = cv2.Canny(observation, 200, 300)

    observation = observation[35:130, 50:]
    observation = cv2.resize(observation, (200, 30))

    # Image.fromarray(observation).save("test.jpg")

    return observation / 255


class DinoKStepQLearner(DeepQLearnerStep):

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

class DinoQLearner(DeepQLearner):

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
