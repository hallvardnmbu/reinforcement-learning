"""Policy-based agent for reinforcement learning."""

import numpy as np
from mlx import nn
from mlx import core as mx


class PolicyGradient(nn.Module):
    """Policy-based agent for reinforcement learning."""
    def __init__(self,
                 network,
                 optimizer,
                 **other):
        """
        Policy-based agent for reinforcement learning.

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

            optim : mlx.optim.X
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

        self.layers = []
        for (_in, _out) in zip([network["inputs"]] + network["nodes"],
                               network["nodes"] + [network["outputs"]]):
            self.layers.append(nn.Linear(_in, _out))

        # LEARNING
        # ------------------------------------------------------------------------------------------
        # Default discount factor is 0.99, as suggested by the Google DeepMind paper "Human-level
        # control through deep reinforcement learning" (2015).

        self.discount = other.get("discount", 0.99)
        self.optimizer = optimizer["optim"](learning_rate=optimizer["lr"],
                                            **optimizer.get("hyperparameters", {}))

        self.memory = {key: [] for key in ["logarithm", "reward"]}

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

    def action(self, state):
        """
        Stochastic action selection.

        Parameters
        ----------
        state : mlx.core.array
            Observed state.

        Returns
        -------
        action : int
            Selected action.
        logarithm : mlx.core.array
            Logarithm of the selected action probability.
        """
        actions = nn.softmax(self(state))

        # Issue with MLX softmax, probabilities do not sum to exactly one.
        # ---
        probabilities = [round(action, 5) for action in actions.tolist()]
        rest = 1 - sum(probabilities)
        probabilities[0] = round(probabilities[0] + rest, 5)
        # ---

        if sum(probabilities) != 1.0:
            print(probabilities)

        action = np.random.choice(range(actions.shape[0]), 1,
                                  p=probabilities)[0]

        logarithm = np.log(actions.tolist()[action])

        return action, logarithm

    def learn(self):
        """
        REINFORCE algorithm; a policy-based gradient method, with respect to the last game played.

        Returns
        -------
        gradient : float

        Notes
        -----
        In order for the agent to best learn the optimal actions, it is common to evaluate the
        expected future rewards. Then, the agent can adjust its predicted action probabilities
        (policy) so that this expected reward is maximized. This is done through the REINFORCE
        algorithm, which computes the policy gradient. Algorithm modified from:

         https://medium.com/@thechrisyoon/deriving-policy-gradients-and-implementing-reinforce
         -f887949bd63
        """
        rewards = mx.array(self.memory["reward"])

        # EXPECTED FUTURE REWARDS
        # ------------------------------------------------------------------------------------------
        # The expected reward given an action is the sum of all future (discounted) rewards. This is
        # achieved by reversely adding the observed reward and the discounted cumulative future
        # rewards. The rewards are then standardized.

        _reward = 0
        for i in reversed(range(len(rewards))):
            _reward = _reward * self.discount + rewards[i]
            rewards[i] = _reward

        mean = rewards.mean(axis=0)
        std = mx.sqrt(mx.sum(rewards - mean)/len(rewards))
        rewards = (rewards - mean) / (std + 1e-9)

        # POLICY GRADIENT
        # ------------------------------------------------------------------------------------------
        # The policy gradient is the gradient of the expected reward with respect to the action
        # taken (policy). This is computed by multiplying the logarithm of the selected action
        # probability (see `action` method) with the standardized expected reward — previously
        # calculated. The overall gradient is then the sum of all these products.

        gradient = mx.zeros_like(rewards)
        for i, (logarithm, reward) in enumerate(zip(self.memory["logarithm"], rewards)):
            gradient[i] = -logarithm * reward
        gradient = gradient.sum(axis=0)

        # BACKPROPAGATION
        # ------------------------------------------------------------------------------------------
        # The gradient is then used to update the agent's policy. This is done by backpropagating
        # with the optimizer using the gradient.

        self.optimizer.update(self, gradient)

        self.memory = {key: [] for key in self.memory.keys()}

        return gradient.item()

    def memorize(self, *args):
        """
        Append state, action and reward to agent memory.

        Parameters
        ----------
        *args : list
            Positional arguments to memorize.
        """
        logarithm, reward = args

        self.memory["logarithm"].append(logarithm)
        self.memory["reward"].append(reward)
