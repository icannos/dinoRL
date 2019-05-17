"""

Used to visualize the behavior of a trained dino in Chrome browser.

author: Maxime Darrin
"""
from keras.models import load_model
from keras import backend as K
import cv2
import numpy as np
from train_dino import preprocess


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the dino controlled by the specified model.")
    parser.add_argument("model", type=str, default="file", help="Path to the model")

    args = parser.parse_args()

    # Load the model
    model = load_model(args.model)

    import gym
    import gym_chrome_dino

    env = gym.make('ChromeDino-v0')

    while True:
        observation = env.reset()
        done = False

        observation = preprocess(observation)
        state = np.stack([observation, observation, observation, observation], axis=2).reshape([1, 30, 200, 4])

        while not done:

            Q = model.predict(state)[0]
            action = np.argmax(Q)
            observation, reward, done, info = env.step(action)

            observation = preprocess(observation)

            state = np.append(observation.reshape((1, 30, 200, 1)), state[:, :, :, :3], axis=3)

        print(env.unwrapped.game.get_score())
