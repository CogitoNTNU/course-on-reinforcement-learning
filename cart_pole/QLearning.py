import pickle

import numpy as np
from collections import defaultdict
from Config import Config as Confg
from Agent import Agent


class QLearningAgent(Agent):
    class Config(Confg):
        # Env variables
        env = "CartPole-v1"
        ac_dim = 2
        ob_dim = 4

        # hyper-parameters
        lr =  # TODO
        epsilon =  # TODO
        gamma =  # TODO

        # experiment
        wandb_name = env
        episodes =  # TODO
        eval_freq =  # TODO
        print_freq = # TODO

    def __init__(self, cfg, lr_decay=lambda lr, i: lr, epsilon_decay=lambda epsilon, i: epsilon):
        super().__init__(cfg)

        self.lr = cfg.lr
        self.lr_decay = lr_decay

        self.epsilon = cfg.epsilon
        self.epsilon_decay = epsilon_decay

        # Using a dictionary as the q-function
        self.q_values = defaultdict(lambda: [0] * self.cfg.ac_dim)

        self.num_updates = 0

    def _greedy_action(self, state):
        # Get the best action
        return  # TODO

    def _exploration_action(self):
        # Get a random action
        return  # TODO

    def act(self, state):
        if np.random.rand() < self.cfg.epsilon:
            return  # TODO
        else:
            return  # TODO

    def save(self, path):
        values = dict(self.q_values)
        with open(path, 'wb') as f:
            pickle.dump(values, f)

    def load(self, path):
        with open(path, 'rb') as f:
            values = pickle.load(f)
        self.q_values = defaultdict(lambda: [0] * self.cfg.ac_dim)
        self.q_values.update(values)

    def update_q_values(self, state, reward, action, next_state):
        self.q_values[state][action] +=  # TODO
        self.num_updates += 1

    def decay_lr(self, i):
        self.lr = self.lr_decay(self.lr, i)

    def decay_epsilon(self, i):
        self.epsilon = self.epsilon_decay(self.epsilon, i)
