"""
author: Maxime Darrin

Train the dino using the implementation of the K-step DeepQ Learning Algorithm
"""

import numpy as np

from dinoAgent import preprocess, DinoKStepQLearner


def train(path, episodes=100, init_model=None, exploration_max=1,
          exploration_min=0.001, exploration_decay=0.99, batch_size=32, gamma=0.99, steps=10):
    """
    Train the model.
    :param episodes: number of episodes (ie of games)
    :return: the dino
    """

    import gym
    import gym_chrome_dino


    env = gym.make('ChromeDino-v0')

    dino = DinoKStepQLearner(2, input_shape=(30, 200, 4),
                             mem_size=500,
                             learning_rate=0.001,
                             exploration_max=exploration_max,
                             exploration_min=exploration_min,
                             exploration_decay=exploration_decay,
                             batch_size=batch_size,
                             gamma=gamma)

    if init_model:
        dino.load(init_model)

    env.unwrapped.game.set_acceleration(0.001)

    best_score = 0

    for i in range(episodes):
        observation = env.reset()
        done = False

        observation = preprocess(observation)

        state = np.stack([observation, observation, observation, observation], axis=2).reshape([1, 30, 200, 4])

        print("episode: ", i)

        seq = []

        while not done:
            action = dino.act(state)

            observation, reward, done, info = env.step(action)

            observation = preprocess(observation)
            next_state = np.append(observation.reshape((1, 30, 200, 1)), state[:, :, :, :3], axis=3)

            if done:
                seq.append((state, action, reward, next_state, done))
                dino.store_exp(seq)
                dino.exp_review()
                seq = []
            elif len(seq) < steps:
                seq.append((state, action, reward, next_state, done))
            else:
                dino.store_exp(seq)
                seq = [(state, action, reward, next_state, done)]
                dino.exp_review()

            state = next_state

        if not i % 50:
            dino.dump(path + str(i) + ".chkpt")

        print("score", env.unwrapped.game.get_score())

        if env.unwrapped.game.get_score() > best_score:
            best_score = env.unwrapped.game.get_score()
            dino.dump(path + "BEST.chkpt")

    return dino


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DeepQLearner for chrome dino")
    parser.add_argument("-e", "--episodes", type=int, default=1000, help="Number of episodes to play")
    parser.add_argument("-b", "--batch", type=int, default=3, help="Batch size")
    parser.add_argument("-s", "--steps", type=int, default=8, help="Number of steps to proceed during exploration")
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
                     gamma=args.gamma, steps=args.steps)
    else:
        dino = train(args.savefile, args.episodes, exploration_max=args.exploration_max,
                     exploration_min=args.exploration_min, exploration_decay=args.exploration_decay,
                     batch_size=args.batch,
                     gamma=args.gamma, steps=args.steps)

    dino.dump(args.savefile)
