"""
Policy-based agent for reinforcement learning using the Apple M chip framework (MLX).

In order to be able to backpropagate properly, the agent has been split into two classes: one for
the network and one for the agent. The network class is a simple feedforward neural network, while
the agent class contains the learning algorithm etc.

It is therefore not recommended to use the `Network` class on its own, but rather through the
"wrapper" class `PolicyGradient`.
"""

from mlx import nn
from mlx import core as mx


class Network(nn.Module):
    """Agent network."""
    def __init__(self, network):
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


class PolicyGradient:
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
        self.agent = Network(network)

        # LEARNING
        # ------------------------------------------------------------------------------------------
        # Default discount factor is 0.99, as suggested by the Google DeepMind paper "Human-level
        # control through deep reinforcement learning" (2015).

        self.discount = other.get("discount", 0.99)
        self.gradient = nn.value_and_grad(self.agent, self.loss)
        self.optimizer = optimizer["optim"](learning_rate=optimizer["lr"],
                                            **optimizer.get("hyperparameters", {}))

        self.memory = {key: [] for key in ["state", "action", "reward"]}

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
        """
        action = mx.random.categorical(self.agent(state)).item()

        return action

    def learn(self):
        """
        REINFORCE algorithm; a policy-based gradient method, with respect to the last game played.

        Returns
        -------
        loss : float

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
        states = mx.array(self.memory["state"])
        actions = mx.array(self.memory["action"])

        # EXPECTED FUTURE REWARDS
        # ------------------------------------------------------------------------------------------
        # The expected reward given an action is the sum of all future (discounted) rewards. This is
        # achieved by reversely adding the observed reward and the discounted cumulative future
        # rewards. The rewards are then standardized.

        _reward = 0
        for i in reversed(range(len(rewards))):
            _reward = _reward * self.discount + rewards[i]
            rewards[i] = _reward

        mean = mx.mean(rewards)
        std = mx.sqrt(mx.sum((rewards - mean) ** 2)/rewards.shape[0])
        rewards = (rewards - mean) / (std + 1e-9)

        # POLICY GRADIENT
        # ------------------------------------------------------------------------------------------

        loss, gradients = self.gradient(rewards, states, actions)

        # BACKPROPAGATION
        # ------------------------------------------------------------------------------------------
        # The gradient is then used to update the agent's policy. This is done by backpropagating
        # with the optimizer using the gradient.

        self.optimizer.update(self.agent, gradients)

        loss = loss.item() * (100 / rewards.shape[0])
        self.memory = {key: [] for key in self.memory.keys()}

        return loss

    def loss(self, rewards, states, actions):
        """
        Compute the policy gradient.

        Parameters
        ----------
        rewards : mlx.core.array
            Standardized expected rewards.
        states : mlx.core.array
            Observed states.
        actions : mlx.core.array
            Selected actions.

        Returns
        -------
        loss : mlx.core.array

        Notes
        -----
            The policy gradient is the gradient of the expected reward with respect to the action
            taken (policy). This is computed by multiplying the logarithm of the selected action
            probability (see `action` method) with the standardized expected reward — previously
            calculated. The overall gradient is then the sum of all these products.
        """
        probabilities = nn.softmax(self.agent(states))[mx.arange(actions.shape[0]), actions]
        logarithms = mx.log(probabilities)

        loss = mx.sum(-logarithms * rewards)

        return loss

    def memorize(self, *args):
        """
        Append state, action and reward to agent memory.

        Parameters
        ----------
        *args : list
            Positional arguments to memorize.
        """
        state, action, reward = args

        self.memory["state"].append(state)
        self.memory["action"].append(action)
        self.memory["reward"].append(reward)
