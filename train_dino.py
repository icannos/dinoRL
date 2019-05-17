"""
author: Maxime Darrin

Train the dino using the single step Deep Q Learning Algorithm
"""

import os

import cv2

import numpy as np
from dinoAgent import DinoQLearner, preprocess


def train(path, episodes=100, init_model=None, exploration_max=1,
          exploration_min=0.01, exploration_decay=0.99, batch_size=32, gamma=0.99):
    """
    Train the model.
    :param episodes: number of episodes (ie of games)
    :return: the dino
    """

    import gym
    import gym_chrome_dino

    env = gym.make('ChromeDinoNoBrowser-v0')

    dino = DinoQLearner(3, input_shape=(30, 200, 4),
                        mem_size=50000,
                        learning_rate=0.001,
                        exploration_max=exploration_max,
                        exploration_min=exploration_min,
                        exploration_decay=exploration_decay,
                        batch_size=batch_size,
                        gamma=gamma)

    if init_model:
        dino.load(init_model)

    for i in range(episodes):
        observation = env.reset()
        done = False

        steps = 0

        observation = preprocess(observation)

        state = np.stack([observation, observation, observation, observation], axis=2).reshape([1, 30, 200, 4])

        print("episode: ", i)

        while not done:
            action = dino.act(state)

            observation, reward, done, info = env.step(action)

            observation = preprocess(observation)

            next_state = np.append(observation.reshape((1, 30, 200, 1)), state[:, :, :, :3], axis=3)

            reward = reward - 0.05 if action == 1 else reward

            dino.store_exp(state, action, reward, next_state, done)

            state = next_state

            steps += 1

            dino.exp_review()

        if not i % 10:
            dino.large_review(50)

        if not i % 100:
            dino.dump(path + str(i) + ".chkpt")

        print("score", env.unwrapped.game.get_score())

    return dino


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DeepQLearner for chrome dino")
    parser.add_argument("-e", "--episodes", type=int, default=1000, help="Number of episodes to play")
    parser.add_argument("-b", "--batch", type=int, default=64, help="Batch size")
    parser.add_argument("-i", "--init", type=str, help="Model used to begin the training")
    parser.add_argument("-d", "--exploration-decay", type=float, default=0.99, help="Decay of the exploration rate")
    parser.add_argument("-xmax", "--exploration-max", type=float, default=1, help="Max exploration rate")
    parser.add_argument("-xmin", "--exploration-min", type=float, default=0.01, help="Min exploration rate")
    parser.add_argument("-g", "--gamma", type=float, default=0.99, help="Discount Rate")
    parser.add_argument("savefile", type=str, default="file", help="Where to save the model")

    args = parser.parse_args()

    if args.init is not None:
        print("Loading model")
        dino = train(args.savefile, args.episodes, args.init, exploration_max=args.exploration_max,
                     exploration_min=args.exploration_min, exploration_decay=args.exploration_decay,
                     batch_size=args.batch,
                     gamma=args.gamma)
    else:
        dino = train(args.savefile, args.episodes, exploration_max=args.exploration_max,
                     exploration_min=args.exploration_min, exploration_decay=args.exploration_decay,
                     batch_size=args.batch,
                     gamma=args.gamma)

    dino.dump(args.savefile)
