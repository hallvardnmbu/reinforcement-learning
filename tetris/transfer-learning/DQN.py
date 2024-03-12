"""
Value-based vision agent for reinforcement learning using transfer learning.

This agent has to transform the input state tensors to images, thus leading to a highly
inefficient agent. This example is therefore only meant to demonstrate the use of transfer
learning in a reinforcement learning setting.
"""

import random
from collections import deque, namedtuple

import torch
import numpy as np
from PIL import Image


class TransferDeepQ:
    """Value-based vision agent for reinforcement learning."""
    Memory = namedtuple("Memory",
                        ["state", "action", "reward", "new_state", "steps"])

    def __init__(self,
                 transfer,
                 actions,
                 optimizer,
                 batch_size=32,
                 **other):
        """
        Value-based vision agent for reinforcement learning.

        Parameters
        ----------
        transfer : dict
            Contains the pre-trained model and its pre-processing function. The dictionary must
            contain the following keys:

            network : torchvision.models.X
                Pre-trained model for transfer learning.
            preprocess : torchvision.transforms.X
                Pre-processing function for the model.

        optimizer : dict
            Contains the optimizer for the model and its hyperparameters. The dictionary must
            contain the following keys:

            optimizer : torch.optim.X
                The optimizer for the model.
            lr : float
                Learning rate for the optimizer.
            **hyperparameters : dict, optional
                Additional hyperparameters for the optimizer.
        batch_size : int, optional
            Number of samples to train on.
        shape : tuple of int, optional
            Shape of the input state space.
        other
            Additional parameters.

            discount : float, optional
                Discount factor for future rewards.
                --> 0: only consider immediate rewards
                --> 1: consider all future rewards equally
            gamma : float, optional
                Discount factor for Q-learning.
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ARCHITECTURE
        # ------------------------------------------------------------------------------------------
        # Freezing the layers of the pre-trained model and adding a new fully connected layer for
        # the fine-tuning.

        self.network = transfer["network"]

        for param in self.network.parameters():
            param.requires_grad = False

        self.network.fc = torch.nn.Linear(self.network.fc.in_features, actions)

        self.preprocess = transfer["preprocess"]

        # LEARNING
        # ------------------------------------------------------------------------------------------
        # Default discount factor is 0.99, as suggested by the Google DeepMind paper "Human-level
        # control through deep reinforcement learning" (2015).

        eps_rate = other.get("exploration_rate", 0.9)
        eps_steps = other.get("exploration_steps", 1500)
        eps_min = other.get("exploration_min", 0.01)
        self.parameter = {
            "rate": eps_rate,
            "decay": (eps_rate - eps_min) / eps_steps,
            "min": eps_min,

            "discount": other.get("discount", 0.99),
            "gamma": other.get("gamma", 0.95),
        }

        self.optimizer = optimizer["optimizer"](self.network.parameters(), lr=optimizer["lr"],
                                                **optimizer.get("hyperparameters", {}))

        self.memory = {
            "batch_size": batch_size,
            "memory": deque(maxlen=other.get("memory", 2500)),
            "game": deque([])
        }

        self.network.to(self.device)

    def forward(self, state):
        """
        Forward pass with nonmodified output.

        Parameters
        ----------
        state : torch.Tensor
            Observed state.

        Returns
        -------
        output : torch.Tensor
        """
        state = self.preprocess(state).to(self.device)

        return self.network(state)

    def action(self, state):
        """
        Greedy action selection with stochastic exploration.

        Parameters
        ----------
        state : torch.Tensor
            Observed state.

        Returns
        -------
        action : torch.Tensor
            Selected action.
        """
        if np.random.rand() < self.parameter["rate"]:
            action = torch.tensor([np.random.choice(self.network.fc.out_features)],
                                  dtype=torch.long)
        else:
            action = self.forward(state).argmax(1)

        return action.to(self.device)

    def learn(self, network):
        """
        Q-learning algorithm; a value-based method.

        Parameters
        ----------
        network : torch.nn.Module
            Reference network for Q-learning.

        Returns
        -------
        loss : float
            Relative loss.

        Raises
        ------
        ValueError
            If no reference network is passed.

        Notes
        -----
        In order for the agent to best learn the optimal actions, it is common to evaluate the
        expected future rewards. Then, the agent can adjust its predicted action values so that
        this expected reward is maximized.
        """
        memory = random.sample(self.memory["memory"],
                               min(self.memory["batch_size"], len(self.memory["memory"])))

        _steps = [game.steps for game in memory]
        steps = [sum(_steps[:i + 1]) - 1 for i in range(len(_steps))]

        with torch.cuda.amp.autocast():
            # Using autocast to reduce memory usage and speed up training.
            states = torch.cat([torch.stack(game.state).squeeze() for game in memory]).unsqueeze(1)
            actions = torch.cat([torch.stack(game.action) for game in memory]).to(self.device)
            _states = torch.cat([game.new_state for game in memory])
            rewards = torch.cat([torch.stack(game.reward) for game in memory]).to(self.device)

            state_images = self.preprocess(Image.fromarray(states)).to(self.device)

            # EXPECTED FUTURE REWARDS
            # --------------------------------------------------------------------------------------
            # The expected reward given an action is the sum of all future (discounted) rewards.
            # This is achieved by reversely adding the observed reward and the discounted
            # cumulative future rewards. The rewards are then standardized.

            _reward = 0
            for i in reversed(range(len(rewards))):
                _reward = 0 if i in steps else _reward
                _reward = _reward * self.parameter["discount"] + rewards[i]
                rewards[i] = _reward
            rewards = ((rewards - rewards.mean()) / (rewards.std() + 1e-7)).view(-1, 1)

            # Q-LEARNING
            # --------------------------------------------------------------------------------------
            # The Q-learning implementation is based on the letter by Google DeepMind ("Human-level
            # control through deep reinforcement learning", 2015). Which is based on the Bellman
            # equation for the optimal action-value function. The optimal action-value function is
            #
            #  Q*(s, a) = Q(s, a) + alpha * (r + gamma * max_a' Q'(s', a') - Q(s, a))
            #
            # where Q*(s, a) is the optimal action-value function, Q(s, a) is the current
            # action-value, Q'(s', a') is the approximated action-value, alpha is the learning rate,
            # r is the expected reward and gamma is the discount factor.
            #
            # DeepMind further simplifies this equation to:
            #
            #  Q*(s, a) = r + gamma * max_a' Q'(s', a')
            #
            # where Q' is a copy of the agent, which is updated every C steps.

            actual = self.network(state_images).gather(1, actions.view(-1, 1))

            with torch.no_grad():
                new_states = torch.roll(states, -1, 0)
                new_states[torch.tensor(steps)] = _states

                new_images = self.preprocess(Image.fromarray(new_states)).to(self.device)

                optimal = (rewards +
                           self.parameter["gamma"] * network(new_images).max(1).values.view(-1, 1))

            # As Google DeepMind suggests, the optimal Q-value is set to r if the game is over.
            for step in steps:
                optimal[step] = rewards[step]

            # BACKPROPAGATION
            # --------------------------------------------------------------------------------------

            loss = torch.nn.functional.mse_loss(actual, optimal)

        self.optimizer.zero_grad()
        loss.backward()

        # # Clamping gradients as per the Google DeepMind paper.
        for param in self.network.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        # EXPLORATION RATE DECAY
        # ------------------------------------------------------------------------------------------

        self.parameter["rate"] = max(self.parameter["rate"] - self.parameter["decay"],
                                     self.parameter["min"])

        del states, actions, new_states, rewards, _reward, actual, optimal
        torch.cuda.empty_cache()

        return (loss.item() / steps[-1]) * 10000

    def remember(self, state, action, reward):
        """
        Append state, action and reward to agents memory of the current game.

        Parameters
        ----------
        state : torch.Tensor
        action : torch.Tensor
        reward : torch.Tensor
        """
        self.memory["game"].append((state, action, reward))

    def memorize(self, new_state, steps):
        """
        Append game to agent memory for mini-batch training.

        Parameters
        ----------
        new_state : torch.Tensor
            Last observed state in the game.
        steps : int
            Number of steps in the game (i.e., game length).
        """
        self.memory["memory"].append(self.Memory(*zip(*self.memory["game"]), new_state, steps))
        self.memory["game"].clear()
