import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from simple_deep_Q_network import SimpleDeepQNetwork

# Implementation of the network shown in "Modern Reinforcement Learning: Deep Q Agents (PyTorch & TF2)"

class Agent():
	def __init__(self, network, input_dim, n_actions, lr, gamma, epsilon, epsilon_decay, epsilon_minimum):
		self.input_dim = input_dim
		self.n_actions = n_actions
		self.lr = lr
		self.gamma = gamma
		self.epsilon = epsilon
		self.epsilon_decay = epsilon_decay
		self.epsilon_minimum = epsilon_minimum
		self.possible_actions = [i for i in range(self.n_actions)]
		self.network = network
	
	def choose_action(self, state):
		# Go with highest Q random number greater than epsilon, else, random
		random_val = np.random.random()
		if random_val > self.epsilon:
			actions = self.network.forward(state)
			action = torch.argmax(actions).item()
		else:
			action = np.random.choice(self.possible_actions)
		
		return action
	
	def decremet_epsilon(self):
		if self.epsilon > self.epsilon_minimum:
			self.epsilon -= self.epsilon_decay
		else:
			self.epsilon = self.epsilon_minimum

		def learn(self, state, action, reward, state_):
			self.network.optimizer.zero_grad()
			states = torch.tensor(state, dtype=torch.float)
			actions = torch.tensor(action)
			rewards = torch.tensor(reward)
			states_ = torch.tensor(state_, dtype = torch.float)

			q_pred = self.Q.forward(states)[actions]
			q_next = self.Q.forward(states_).max()
			q_target = reward + self.gamma * q_next

			loss = self.network.loss(q_target, q_pred)
			loss.backward()
			self.network.optimizer.step()
			self.decremet_epsilon()