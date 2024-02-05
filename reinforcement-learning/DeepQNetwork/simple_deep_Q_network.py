import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Implementation of the network shown in "Modern Reinforcement Learning: Deep Q Agents (PyTorch & TF2)"

class SimpleDeepQNetwork(nn.module):
	def __init__(self, n_actions, input_dim, lr):
		super(SimpleDeepQNetwork, self).__init__()

		self.fc1 = nn.Linear(*input_dim, 64)
		self.fc2 = nn.Linear(64, n_actions)

		self.optimizer = torch.optim.Adam(self.parameters(), lr = lr)
		self.loss = nn.MSELoss()

	def forward(self, state):
		a1 = F.relu(self.fc1(state))
		actions = self.fc2(a1)

		return actions

