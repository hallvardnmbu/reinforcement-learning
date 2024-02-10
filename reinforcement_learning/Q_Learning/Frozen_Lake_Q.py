import gymnasium as gym
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class QLearningAgent:
    def __init__(self, state_space, action_space, lr,
                 gamma, epsilon, epsilon_decay, epsilon_minimum):
        self.state_space = state_space
        self.action_space = action_space
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_minimum = epsilon_minimum
        self.Q_table = pd.DataFrame(data=np.zeros(shape=(action_space, state_space)),
                                    columns=[f"State {i}" for i in range(state_space)],
                                    index=[f"Action {i}" for i in range(action_space)])

    def greedy_epsilon_action_choice(self, state):
        if np.random.random() > self.epsilon:
            action = self.greedy_action_choice(state)
        else:
            action = np.random.choice(range(self.action_space))
        self.decrement_epsilon()
        return action

    def decrement_epsilon(self):
        self.epsilon = max(self.epsilon_minimum, self.epsilon - self.epsilon_decay)

    def greedy_action_choice(self, state):
        action = self.Q_table.iloc[:, state].idxmax()
        return int(action.split(" ")[-1])

    def learn(self, old_state, action, reward, new_state):
        Q_observed = reward + self.gamma * self.Q_table.iloc[:, new_state].max()
        Q_expected = self.Q_table.at[f"Action {action}", f"State {old_state}"]
        self.Q_table.at[f"Action {action}", f"State {old_state}"] = Q_expected + self.lr * (Q_observed - Q_expected)


