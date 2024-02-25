"""
Value-based agents for reinforcement learning using the Apple M chip framework (MLX).

In order to be able to backpropagate properly, the agent has been split into two classes: one for
the network and one for the agent. The network class is a simple feedforward neural network, while
the agent class contains the learning algorithm etc.

It is therefore not recommended to use the `Network` class on its own, but rather through
the "wrapper" class `DeepQ`.
"""

from collections import deque, namedtuple
import random

import numpy as np
from mlx import nn
from mlx import core as mx


class Network(nn.Module):
    """Agent network."""
    def __init__(self, network):
        """
        Base agent for reinforcement learning.

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
        """
        super().__init__()

        # ARCHITECTURE
        # ------------------------------------------------------------------------------------------

        if "nodes" not in network:
            network["nodes"] = [25]

        self.layers = []
        for (_in, _out) in zip([network["inputs"]] + network["nodes"],
                               network["nodes"] + [network["outputs"]]):
            self.layers.append(nn.Linear(_in, _out))

    def __call__(self, state):
        """
        Forward pass with nonmodified output.

        Parameters
        ----------
        state : mlx.core.array
            Observed state.

        Returns
        -------
        output : mlx.core.array
        """
        for layer in self.layers[:-1]:
            state = nn.relu(layer(state))
        output = self.layers[-1](state)

        return output


class DeepQ:
    """Value-based agent for reinforcement learning."""
    Memory = namedtuple("Memory",
                        ["state", "action", "new_state", "reward", "steps"])

    def __init__(self,
                 network,
                 optimizer,
                 batch_size=32,
                 **other):
        """
        Base agent for reinforcement learning.

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

            optim : mlx.optimizers.X
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
        self.agent = Network(network)

        # LEARNING
        # ------------------------------------------------------------------------------------------
        # Default discount factor is 0.99, as suggested by the Google DeepMind paper "Human-level
        # control through deep reinforcement learning" (2015).

        self.gradient = nn.value_and_grad(self.agent, self.loss)

        self.explore = {
            "rate": other.get("exploration_rate", 0.9),
            "decay": other.get("exploration_decay", 0.999),
            "min": other.get("exploration_min", 0.01),

            "discount": other.get("discount", 0.99),
            "gamma": other.get("gamma", 0.95),
        }

        self.optimizer = optimizer["optim"](learning_rate=optimizer["lr"],
                                            **optimizer.get("hyperparameters", {}))

        self.memory = {
            "batch_size": batch_size,
            "memory": deque(maxlen=other.get("memory", 2500)),
            "game": [],
        }

    def action(self, state):
        """
        Greedy action selection with stochastic exploration.

        Parameters
        ----------
        state : mlx.core.array
            Observed state.

        Returns
        -------
        action : int
            Selected action.
        """
        if np.random.rand() < self.explore["rate"]:
            action = mx.array(np.random.choice(range(self.agent.layers[-1].weight.shape[0])))
        else:
            action = self.agent(state).argmax(axis=0)

        return action

    def learn(self, network):
        """
        Q-learning algorithm; a value-based method.

        Parameters
        ----------
        network : nn.Module
            Reference network for Q-learning.

        Returns
        -------
        loss : float

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

        states = mx.concatenate([mx.array(mx.array(game.state)) for game in memory])
        actions = mx.concatenate([mx.array(game.action) for game in memory]).reshape(-1, 1)
        new_states = mx.concatenate([mx.array(game.new_state) for game in memory])
        rewards = mx.concatenate([mx.array(game.reward) for game in memory])

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

        mean = mx.mean(rewards)
        std = mx.sqrt(mx.sum((rewards - mean) ** 2)/rewards.shape[0])
        rewards = (rewards - mean) / (std + 1e-9)

        # GRADIENT
        # ------------------------------------------------------------------------------------------
        # See the `loss` method for more information.

        loss, gradients = self.gradient(states, actions, new_states, rewards, steps, network)

        # BACKPROPAGATION
        # ------------------------------------------------------------------------------------------

        self.optimizer.update(self.agent, gradients)

        # EXPLORATION RATE DECAY
        # ------------------------------------------------------------------------------------------

        self.explore["rate"] = max(self.explore["decay"] * self.explore["rate"],
                                   self.explore["min"])

        return loss.item() * (1000 / steps[-1])

    def loss(self, *data):
        r"""
        Compute the policy gradient through Q-learning.

        Parameters
        ----------
        data : list
            Must contain the following items in this order:

            states : mlx.core.array
                Observed states.
            actions : mlx.core.array
                Selected actions.
            new_states : mlx.core.array
                Observed states after action.
            rewards : mlx.core.array
                Expected rewards.
            steps : list
                Number of steps in each game.
            network : nn.Module
                Reference network for Q-learning.

        Returns
        -------
        loss : mlx.core.array

        Notes
        -----
            The Q-learning implementation is based on the letter by Google DeepMind ("Human-level
            control through deep reinforcement learning", 2015). Which is based on the Bellman
            equation for the optimal action-value function. The optimal action-value function is

        .. math::
                Q*(s, a) = Q(s, a) + \alpha * (r + \gamma * \max_a' Q'(s', a') - Q(s, a))

            where :math:`Q*(s, a)` is the optimal action-value function, :math:`Q(s, a)` is the
            current action-value, :math:`Q'(s', a')` is the approximated action-value,
            :math:`\alpha` is the learning rate, :math:`r` is the expected reward and
            :math:`\gamma` is the discount factor.

            DeepMind further simplifies this equation to:

        .. math::
                Q*(s, a) = r + gamma * max_a' Q'(s', a')

            where :math:`Q'` is a copy of the agent, which is updated every :math:`C` steps.
        """
        states, actions, new_states, rewards, steps, network = data

        # Here, the documentation for MLX was useful:
        actual = mx.take_along_axis(self.agent(states),                                 # noqa
                                    indices=actions, axis=1).squeeze()

        optimal = (rewards.reshape(-1) +                                                      # noqa
                   self.explore["gamma"] * network(new_states).max(axis=1))                   # noqa

        # As Google DeepMind suggests, the optimal Q-value is set to r if the game is over.
        for step in steps:
            optimal[step] = rewards[step]

        loss = nn.losses.mse_loss(actual, optimal)

        return loss

    def remember(self, *args):
        """
        Append state, new_state and reward to agents memory of the current game.

        Parameters
        ----------
        *args : list
            Positional arguments to memorize.
        """
        self.memory["game"].append(args)

    def memorize(self, steps):
        """
        Append game to agent memory for mini-batch training.

        Parameters
        ----------
        steps : int
            Number of steps in the game (i.e., game length).
        """
        self.memory["memory"].append(self.Memory(*zip(*self.memory["game"]), steps))
        self.memory["game"] = []
