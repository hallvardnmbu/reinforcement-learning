"""Reinforcement learning Agents."""

from abc import ABC, abstractmethod
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
        # --------------------------------------------------

        if "nodes" not in network:
            network["nodes"] = [25]

        for i, (_in, _out) in enumerate(zip([network["inputs"]] + network["nodes"],
                                            network["nodes"] + [network["outputs"]])):
            setattr(self, f"layer_{i}", torch.nn.Linear(_in, _out, dtype=torch.float32))

        # LEARNING
        # --------------------------------------------------

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

    def learn(self):
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

         https://medium.com/@thechrisyoon/deriving-policy-gradients-and-implementing-reinforce-f887949bd63
        """
        rewards = torch.tensor(self.memory["reward"], dtype=torch.float32)

        # EXPECTED FUTURE REWARDS
        # --------------------------------------------------
        # The expected reward given an action is the sum of all future (discounted) rewards. This is
        # achieved by reversely adding the observed reward and the discounted cumulative future
        # rewards. The rewards are then standardized.

        _reward = 0
        for i in reversed(range(len(rewards))):
            _reward = _reward * self.discount + rewards[i]
            rewards[i] = _reward
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-9)

        # POLICY GRADIENT
        # --------------------------------------------------
        # The policy gradient is the gradient of the expected reward with respect to the action
        # taken (policy). This is computed by multiplying the logarithm of the selected action
        # probability (see `action` method) with the standardized expected reward — previously
        # calculated. The overall gradient is then the sum of all these products.

        gradient = torch.zeros_like(rewards)
        for i, (logarithm, reward) in enumerate(zip(self.memory["logarithm"], rewards)):
            gradient[i] = -logarithm * reward
        gradient = gradient.sum()

        # BACKPROPAGATION
        # --------------------------------------------------
        # The gradient is then used to update the Agent's policy. This is done by backpropagating
        # with the optimizer using the gradient.

        self.optimizer.zero_grad()
        gradient.backward()
        self.optimizer.step()

        self.memory = {key: [] for key in self.memory.keys()}

        return gradient.item()

    def memorize(self, *args):
        """
        Append observation, action and reward to Agent memory.

        Parameters
        ----------
        *args : list
            Positional arguments to memorize.
        """
        logarithm, reward = args

        self.memory["logarithm"].append(logarithm)
        self.memory["reward"].append(reward)


class ValueAgent(Agent):
    """Value-based Agent for reinforcement learning."""
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
            TODO: Add the others here.
        """
        super().__init__(network, optimizer, **other)

        self.q_learning = {
            "learning-rate": other.get("learning_rate", 0.1),
            "exploration-rate": other.get("exploration_rate", 1),
            "exploration-decay": other.get("exploration_decay", 0.001),
            "exploration-min": other.get("exploration_min", 0.01),
        }

        self.memory["state"] = []
        self.memory["action"] = []
        self.memory["q-value"] = []
        self.memory["reward"] = []

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
        actions = self(state)
        if np.random.rand() < self.q_learning["exploration-rate"]:
            action = np.random.choice(range(next(reversed(self._modules.values())).out_features))
        else:
            action = torch.argmax(actions).item()

        return action, actions

    def learn(self):
        """
        Q-learning algorithm; a value-based method, with respect to the last game played.

        Returns
        -------
        loss : float

        Notes
        -----
        In order for the Agent to best learn the optimal actions, it is common to evaluate the
        expected future rewards. Then, the Agent can adjust its predicted action values (
        Q-values) so that this expected reward is maximized. This is done through the Q-learning algorithm, which computes the loss. Algorithm modified from:

         https://towardsdatascience.com/q-learning-algorithm-from-explanation-to-implementation
         -cdbeda2ea187
        """
        rewards = torch.tensor(self.memory["reward"], dtype=torch.float32)
        q_values = torch.stack(self.memory["q-value"])

        # EXPECTED FUTURE REWARDS
        # --------------------------------------------------
        # The expected reward given an action is the sum of all future (discounted) rewards. This is
        # achieved by reversely adding the observed reward and the discounted cumulative future
        # rewards. The rewards are then standardized.

        _reward = 0
        for i in reversed(range(len(rewards))):
            _reward = _reward * self.discount + rewards[i]
            rewards[i] = _reward
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-9)

        # Q-LEARNING
        # --------------------------------------------------

        targets = torch.zeros_like(q_values)
        for i in range(len(rewards)-1):
            targets[i] = (rewards[i].item() +
                          self.q_learning["learning-rate"] * torch.max(q_values[i+1]))
        targets[-1] = rewards[-1].item()

        loss = torch.nn.functional.mse_loss(q_values, targets)

        # BACKPROPAGATION
        # --------------------------------------------------

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # EXPLORATION RATE DECAY
        # --------------------------------------------------

        if self.q_learning["exploration-rate"] > self.q_learning["exploration-min"]:
            self.q_learning["exploration-rate"] *= 1 - self.q_learning["exploration-decay"]
        else:
            self.q_learning["exploration-rate"] = self.q_learning["exploration-min"]

        self.memory = {key: [] for key in self.memory.keys()}

        return loss.item()

    def memorize(self, *args):
        """
        Append observation, action, q-value and reward to Agent memory.

        Parameters
        ----------
        *args : list
            Positional arguments to memorize.
        """
        state, action, q_value, reward = args

        self.memory["state"].append(state)
        self.memory["action"].append(action)
        self.memory["q-value"].append(q_value)
        self.memory["reward"].append(reward)
