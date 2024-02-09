"""Reinforcement learning Agents."""

from collections import namedtuple, deque
from abc import ABC, abstractmethod
import random

import numpy as np
import torch


class Agent(ABC, torch.nn.Module):
    """Base Agent for reinforcement learning."""
    def __init__(self,
                 network,
                 optimizer,
                 **other):
        """
        Base Agent for reinforcement learning.

        Parameters
        ----------
        network : dict
            Contains the architecture for the model.
            The dictionary must contain the following keys:

            inputs : int
                Number of input nodes (observations).
            outputs : int
                Number of output nodes (actions).
            nodes : list, optional
                Number of nodes for each hidden layer.
        optimizer : dict
            Contains the optimizer for the model and its hyperparameters. The dictionary must
            contain the following keys:

            optim : torch.optim.X
                The optimizer for the model.
            lr : float
                Learning rate for the optimizer.
            **hyperparameters : dict, optional
                Additional hyperparameters for the optimizer.
        other : dict
            Additional parameters.

            discount : float, optional
                Discount factor for future rewards.
                --> 0: only consider immediate rewards
                --> 1: consider all future rewards equally
        """
        super().__init__()

        # ARCHITECTURE
        # ------------------------------------------------------------------------------------------

        if "nodes" not in network:
            network["nodes"] = [25]

        for i, (_in, _out) in enumerate(zip([network["inputs"]] + network["nodes"],
                                            network["nodes"] + [network["outputs"]])):
            setattr(self, f"layer_{i}", torch.nn.Linear(_in, _out, dtype=torch.float32))

        # LEARNING
        # ------------------------------------------------------------------------------------------
        # Default discount factor is 0.99, as suggested by the Google DeepMind paper "Human-level
        # control through deep reinforcement learning" (2015).

        self.discount = other.get("discount", 0.99)
        self.optimizer = optimizer["optim"](self.parameters(), lr=optimizer["lr"],
                                            **optimizer.get("hyperparameters", {}))

        self.memory = {}

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

        _output = torch.relu(self.layer_0(state))

        for i in range(1, len(self._modules) - 1):
            _output = torch.relu(getattr(self, f"layer_{i}")(_output))

        output = getattr(self, f"layer_{len(self._modules)-1}")(_output)

        return output

    @abstractmethod
    def action(self, state):
        """
        Abstract method for action selection.

        Parameters
        ----------
        state : torch.Tensor
            Observed state.

        Returns
        -------
        action : int
            Selected action.
        """

    @abstractmethod
    def learn(self):
        """
        Abstract method for learning.

        Returns
        -------
        float
            Either the gradient, loss, Q-value, etc.
        """

    @abstractmethod
    def memorize(self, *args):
        """
        Abstract method for memorizing.

        Parameters
        ----------
        *args : list
            Positional arguments to memorize.
        """


class PolicyGradientAgent(Agent):
    """Policy-based Agent for reinforcement learning."""
    def __init__(self,
                 network,
                 optimizer,
                 **other):
        """
        Policy-based Agent for reinforcement learning.

        Parameters
        ----------
        network : dict
            Contains the architecture for the model.
            The dictionary must contain the following keys:

            inputs : int
                Number of input nodes (observations).
            outputs : int
                Number of output nodes (actions).
            nodes : list, optional
                Number of nodes for each hidden layer.
        optimizer : dict
            Contains the optimizer for the model and its hyperparameters. The dictionary must
            contain the following keys:

            optim : torch.optim.X
                The optimizer for the model.
            lr : float
                Learning rate for the optimizer.
            **hyperparameters : dict, optional
                Additional hyperparameters for the optimizer.
        other : dict
            Additional parameters.

            discount : float, optional
                Discount factor for future rewards.
                --> 0: only consider immediate rewards
                --> 1: consider all future rewards equally
        """
        super().__init__(network, optimizer, **other)

        self.memory["logarithm"] = []
        self.memory["reward"] = []

    def action(self, state):
        """
        Stochastic action selection.

        Parameters
        ----------
        state : torch.Tensor
            Observed state.

        Returns
        -------
        action : int
            Selected action.
        logarithm : torch.Tensor
            Logarithm of the selected action probability.
        """
        actions = torch.softmax(self(state), dim=-1)

        action = np.random.choice(range(actions.shape[0]), 1,
                                  p=actions.detach().numpy())[0]
        logarithm = torch.log(actions[action])

        return action, logarithm

    def learn(self, network=None):
        """
        REINFORCE algorithm; a policy-based gradient method, with respect to the last game played.

        Returns
        -------
        gradient : float

        Notes
        -----
        In order for the Agent to best learn the optimal actions, it is common to evaluate the
        expected future rewards. Then, the Agent can adjust its predicted action probabilities
        (policy) so that this expected reward is maximized. This is done through the REINFORCE
        algorithm, which computes the policy gradient. Algorithm modified from:

         https://medium.com/@thechrisyoon/deriving-policy-gradients-and-implementing-reinforce
         -f887949bd63
        """
        if network:
            print("Reference network passed. This is not used in PolicyGradientAgent.")

        rewards = torch.tensor(self.memory["reward"], dtype=torch.float32)

        # EXPECTED FUTURE REWARDS
        # ------------------------------------------------------------------------------------------
        # The expected reward given an action is the sum of all future (discounted) rewards. This is
        # achieved by reversely adding the observed reward and the discounted cumulative future
        # rewards. The rewards are then standardized.

        _reward = 0
        for i in reversed(range(len(rewards))):
            _reward = _reward * self.discount + rewards[i]
            rewards[i] = _reward
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-9)

        # POLICY GRADIENT
        # ------------------------------------------------------------------------------------------
        # The policy gradient is the gradient of the expected reward with respect to the action
        # taken (policy). This is computed by multiplying the logarithm of the selected action
        # probability (see `action` method) with the standardized expected reward — previously
        # calculated. The overall gradient is then the sum of all these products.

        gradient = torch.zeros_like(rewards)
        for i, (logarithm, reward) in enumerate(zip(self.memory["logarithm"], rewards)):
            gradient[i] = -logarithm * reward
        gradient = gradient.sum()

        # BACKPROPAGATION
        # ------------------------------------------------------------------------------------------
        # The gradient is then used to update the Agent's policy. This is done by backpropagating
        # with the optimizer using the gradient.

        self.optimizer.zero_grad()
        gradient.backward()
        self.optimizer.step()

        self.memory = {key: [] for key in self.memory.keys()}

        return gradient.item()

    def memorize(self, *args):
        """
        Append state, action and reward to Agent memory.

        Parameters
        ----------
        *args : list
            Positional arguments to memorize.
        """
        logarithm, reward = args

        self.memory["logarithm"].append(logarithm)
        self.memory["reward"].append(reward)


class ValueDeepQAgent(Agent):
    """Value-based Agent for reinforcement learning."""
    Memory = namedtuple("Memory",
                        ["state", "action", "new_state", "reward"])

    def __init__(self,
                 network,
                 optimizer,
                 **other):
        """
        Value-based Agent for reinforcement learning.

        Parameters
        ----------
        network : dict
            Contains the architecture for the model.
            The dictionary must contain the following keys:

            inputs : int
                Number of input nodes (observations).
            outputs : int
                Number of output nodes (actions).
            nodes : list, optional
                Number of nodes for each hidden layer.
        optimizer : dict
            Contains the optimizer for the model and its hyperparameters.
            The dictionary must contain the following keys:

            optim : torch.optim.X
                The optimizer for the model.
            lr : float
                Learning rate for the optimizer.
            **hyperparameters : dict, optional
                Additional hyperparameters for the optimizer.
        other : dict
            Additional parameters.

            discount : float, optional
                Discount factor for future rewards.
                --> 0: only consider immediate rewards
                --> 1: consider all future rewards equally
            alpha : float, optional
                Learning rate for the Q-learning algorithm.
            exploration_rate : float, optional
                Initial exploration rate.
            exploration_decay : float, optional
                Exploration rate decay.
            exploration_min : float, optional
                Minimum exploration rate.
            batch_size : int, optional
                Number of samples to use for learning.
            memory_size : int, optional
                Maximum number of samples to store in memory.
        """
        super().__init__(network, optimizer, **other)

        self.gamma = other.get("gamma", 0.95)

        self.explore = {
            "rate": other.get("exploration_rate", 0.9),
            "decay": other.get("exploration_decay", 0.999),
            "min": other.get("exploration_min", 0.01),
        }

        self.batch_size = other.get("batch_size", 32)
        self.memory = deque(maxlen=other.get("memory_size", 10000))

    def action(self, state):
        """
        Greedy action selection with stochastic exploration.

        Parameters
        ----------
        state : torch.Tensor
            Observed state.

        Returns
        -------
        action : int
            Selected action.
        actions : torch.Tensor
            Q-values for each action.
        """
        if np.random.rand() < self.explore["rate"]:
            action = torch.tensor([np.random.choice(
                range(next(reversed(self._modules.values())).out_features)
            )], dtype=torch.long)
        else:
            action = self(state).max(1).indices.view(1, 1).clone().detach()

        return action

    def learn(self, network=None):
        """
        Q-learning algorithm; a value-based method, with respect to the last game played.

        Returns
        -------
        loss : float

        Raises
        ------
        ValueError
            If no reference network is passed.

        Notes
        -----
        In order for the Agent to best learn the optimal actions, it is common to evaluate the
        expected future rewards. Then, the Agent can adjust its predicted action values so that
        this expected reward is maximized.
        """
        if not network:
            raise ValueError("No reference network passed.")

        memory = random.sample(self.memory, min(self.batch_size, len(self.memory)))
        batch = self.Memory(*zip(*memory))

        # EXPECTED FUTURE REWARDS
        # ------------------------------------------------------------------------------------------
        # The expected reward given an action is the sum of all future (discounted) rewards. This is
        # achieved by reversely adding the observed reward and the discounted cumulative future
        # rewards. The rewards are then standardized.

        _reward = 0
        rewards = torch.cat(batch.reward)
        for i in reversed(range(len(rewards))):
            _reward = _reward * self.discount + rewards[i]
            rewards[i] = _reward
        rewards = ((rewards - rewards.mean()) / (rewards.std() + 1e-9)).view(-1, 1)

        # Q-LEARNING
        # ------------------------------------------------------------------------------------------
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
        # where Q' is a copy of the Agent, which is updated every X steps.

        actions = torch.cat([tensor.flatten() for tensor in batch.action])

        actual = self(torch.cat(batch.state)).gather(1, actions.view(-1, 1))

        optimal = (rewards +
                   self.gamma * network(torch.cat(batch.new_state)).max(1).values.view(-1, 1))
        optimal[-1] = rewards[-1]

        # BACKPROPAGATION
        # ------------------------------------------------------------------------------------------

        loss = torch.nn.functional.mse_loss(actual, optimal)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # EXPLORATION RATE DECAY
        # ------------------------------------------------------------------------------------------

        self.explore["rate"] = max(self.explore["decay"] * self.explore["rate"],
                                   self.explore["min"])

        return loss.item()

    def memorize(self, *args):
        """
        Append state, action, new_state and reward to Agent memory.

        Parameters
        ----------
        *args : list
            Positional arguments to memorize.
        """
        self.memory.append(self.Memory(*args))
