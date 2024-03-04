"""
Value-based agent for reinforcement learning.

Useful for environments with `ram` state spaces (from `gymnasium`).
"""

from collections import deque, namedtuple
import random

import numpy as np
import torch


class DeepQ(torch.nn.Module):
    """Value-based agent for reinforcement learning."""
    Memory = namedtuple("Memory",
                        ["state", "action", "new_state", "reward", "steps"])

    def __init__(self,
                 network,
                 optimizer,
                 batch_size=32,
                 **other):
        """
        Value-based agent for reinforcement learning.

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

            optimizer : torch.optim.X
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        self.explore = {
            "rate": other.get("exploration_rate", 0.9),
            "decay": other.get("exploration_decay", 0.999),
            "min": other.get("exploration_min", 0.01),

            "discount": other.get("discount", 0.99),
            "gamma": other.get("gamma", 0.95),
        }

        self.optimizer = optimizer["optimizer"](self.parameters(), lr=optimizer["lr"],
                                                **optimizer.get("hyperparameters", {}))

        self.batch_size = batch_size
        self.memory = deque(maxlen=other.get("memory", 2500))
        self.game = []

        self.to(self.device)

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
        state = state.to(self.device) / torch.tensor(255,
                                                     dtype=torch.float32, device=self.device)
        _output = torch.relu(self.layer_0(state))

        for i in range(1, len(self._modules) - 1):
            _output = torch.relu(getattr(self, f"layer_{i}")(_output))

        output = getattr(self, f"layer_{len(self._modules)-1}")(_output)

        return output

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
            action = torch.tensor([self(state).argmax()], dtype=torch.long)

        return action

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
        memory = random.sample(self.memory, min(self.batch_size, len(self.memory)))

        states = torch.cat([torch.stack(game.state) for game in memory])
        actions = torch.cat([torch.stack(game.action) for game in memory]).to(self.device)
        new_states = torch.cat([torch.stack(game.new_state) for game in memory])
        rewards = torch.cat([torch.stack(game.reward) for game in memory]).to(self.device)

        steps = [game.steps for game in memory]
        steps = [sum(steps[:i+1])-1 for i in range(len(steps))]

        # EXPECTED FUTURE REWARDS
        # ------------------------------------------------------------------------------------------
        # The expected reward given an action is the sum of all future (discounted) rewards. This is
        # achieved by reversely adding the observed reward and the discounted cumulative future
        # rewards. The rewards are then standardized.

        _reward = 0
        for i in reversed(range(len(rewards))):
            _reward = 0 if i in steps else _reward
            _reward = _reward * self.explore["discount"] + rewards[i]
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
        # where Q' is a copy of the agent, which is updated every C steps.

        actual = self(states).gather(1, actions.view(-1, 1))

        with torch.no_grad():
            optimal = (rewards +
                       self.explore["gamma"] * network(new_states).max(1).values.view(-1, 1))

        # As Google DeepMind suggests, the optimal Q-value is set to r if the game is over.
        for step in steps:
            optimal[step] = rewards[step]

        # BACKPROPAGATION
        # ------------------------------------------------------------------------------------------

        loss = torch.nn.functional.smooth_l1_loss(actual, optimal)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # EXPLORATION RATE DECAY
        # ------------------------------------------------------------------------------------------

        self.explore["rate"] = max(self.explore["decay"] * self.explore["rate"],
                                   self.explore["min"])

        del states, actions, new_states, rewards, _reward, actual, optimal

        return (loss.item() / steps[-1]) * 10000

    def remember(self, *args):
        """
        Append state, action, new_state and reward to agents memory of the current game.

        Parameters
        ----------
        *args : list
            Positional arguments to memorize.
        """
        self.game.append(args)

    def memorize(self, steps):
        """
        Append game to agent memory for mini-batch training.

        Parameters
        ----------
        steps : int
            Number of steps in the game (i.e., game length).
        """
        self.memory.append(self.Memory(*zip(*self.game), steps))
        self.game = []
