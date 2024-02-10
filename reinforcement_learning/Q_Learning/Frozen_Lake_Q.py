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


class FrozenLakeEnvironment:
    def __init__(self, lr=0.8, gamma=0.9, epsilon=0.99, 
                 epsilon_decay=0.0001, epsilon_minimum=0.01, 
                 map_name="4x4", is_slippery=False):
        self.map_name = map_name
        self.is_slippery = is_slippery
        self.env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=is_slippery)
        self.agent = QLearningAgent(self.env.observation_space.n, self.env.action_space.n, lr, gamma, epsilon, epsilon_decay, epsilon_minimum)

    def run(self, episodes, render=False):
        reward_list = []
        for episode in range(episodes):
            state = self.env.reset()[0]
            terminated = False
            truncated = False
            acc_reward = 0

            while not terminated and not truncated:
                if render:
                    self.env.render()

                action = self.agent.greedy_epsilon_action_choice(state)
                new_state, reward, terminated, truncated, _ = self.env.step(action)

                self.agent.learn(state, action, reward, new_state)

                acc_reward += reward
                state = new_state

            reward_list.append(acc_reward)

        self.env.close()
        return reward_list
    

    def train(self, episodes):
        self.env = gym.make("FrozenLake-v1", map_name = self.map_name, is_slippery = self.is_slippery)
        rewards = self.run(episodes, render=False)
        plt.plot(rewards)
        plt.show()

    def test(self, episodes):
        self.env = gym.make("FrozenLake-v1", map_name = self.map_name, 
                            is_slippery = self.is_slippery, render_mode = "human")
        self.run(episodes, render=True)
        


# Example usage:
environment = FrozenLakeEnvironment()
environment.train(1000)  # Train without rendering
environment.test(10)  # Test with rendering
