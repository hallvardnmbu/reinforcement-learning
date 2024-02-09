import gymnasium as gym
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run(episodes, render=False):
	if render:
		render_mode = "human"
	else:
		render_mode = None

	env = gym.make("FrozenLake-v1", map_name = "4x4", is_slippery = False, render_mode = render_mode)

	state_space = env.observation_space.n
	action_space = env.action_space.n

	reward_list = []

	agent = SimpleQAgent(state_space, action_space, lr = 0.8, gamma = 0.9, epsilon = 0.99, epsilon_decay=0.0001, epsilon_minimum=0.01)
	agent.Q_table.head()
	for i in range(episodes):
		state = env.reset()[0]
		terminated = False
		truncated = False

		acc_reward = 0
		while(not terminated and not truncated):
			action = agent.greedy_epsilon_action_choice(state)
			new_state, reward, terminated, truncated, _ = env.step(action)

			agent.learn(state, action, reward, new_state)

			acc_reward += reward
			state = new_state
		
		reward_list.append(acc_reward)

	env.close()

	plt.plot(reward_list)
	plt.show()

	return agent

class SimpleQAgent:
	def __init__(self, state_space, action_space, lr, gamma, epsilon, epsilon_decay, epsilon_minimum):
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
		print(self.Q_table)
			
	def greedy_epsilon_action_choice(self, state):
		random_val = np.random.random()
		if random_val > self.epsilon:
			action = self.greedy_action_choice(state)
		else:
			action = np.random.choice(range(self.action_space))
		self.decrement_epsilon()
		return action
	
	def decrement_epsilon(self):
		if self.epsilon > self.epsilon_minimum:
			self.epsilon -= self.epsilon_decay
		else:
			self.epsilon = self.epsilon_minimum
	
	def greedy_action_choice(self, state):
		action = self.Q_table[f"State {state}"].idxmax()
		action = int(action.split(" ")[-1])
		return action
	
	def learn(self, old_state, action, reward, resulting_state):
		Q_observed = reward + self.gamma * self.Q_table[f"State {resulting_state}"].max()
		Q_expected = self.Q_table[f"State {old_state}"][f"Action {action}"]
		temporal_diff_err = Q_observed - Q_expected
		self.Q_table[f"State {old_state}"][f"Action {action}"] = Q_expected + self.lr * temporal_diff_err

run(1000)