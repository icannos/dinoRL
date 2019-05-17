import numpy as np


class miniDecisionGame:

    def __init__(self):

        self.action_space = 3
        self.observation_space = (100, 100, 1)

        self.state = None
        self.expected_action = None

        self.done = True

        self.reward_good = 0.01
        self.reward_bad = -1

    def step(self, action):
        if action == self.expected_action:
            self.expected_action, self.state = self.draw_state()
            return self.state, self.reward_good, self.done, 0

        else:
            self.done = True
            return self.state, self.reward_bad, self.done, 0

    def reset(self):
        self.done = False

        self.expected_action, self.state = self.draw_state()

        return self.state

    def draw_state(self):
        s = np.random.randint(0, 3)

        if s == 0:
            return s, np.zeros(self.observation_space)

        elif s == 1:
            state = np.zeros(self.observation_space, dtype=np.float)

            state[1, 4, 0] = 255
            state[1, 5, 0] = 255
            state[2, 4, 0] = 255
            state[2, 5, 0] = 255

            return s, state

        elif s == 2:
            state = np.zeros(self.observation_space, dtype=np.float)

            state[7, 4, 0] = 255
            state[7, 5, 0] = 255
            state[8, 4, 0] = 255
            state[8, 5, 0] = 255

            return s, state
