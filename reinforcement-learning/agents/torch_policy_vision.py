"""Policy-based agent for reinforcement learning."""

import numpy as np
import torch


class VisionPolicyGradient(torch.nn.Module):
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

            input_channels : int
                Number of input channels.
            outputs : int
                Number of output nodes (actions).
            kernels : list of tuple of int
                Kernel size for each layer.
            channels : list, optional
                Number of channels for each hidden layer.
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

        if "channels" not in network:
            network["channels"] = [15]
        if "kernels" not in network:
            network["kernels"] = [(3, 3)] * (len(network["channels"]) + 1)

        if len(network["kernels"]) != len(network["channels"])+1:
            print("Number of kernels must be equal to the number of layers.\n"
                  "Using default kernel size (3, 3) for all layers.")
            network["kernels"] = [(3, 3)] * (len(network["channels"]) + 1)

        for i, (_in, _out, _kernel) in enumerate(
                zip(
                    [network["input_channels"]] + network["channels"][:-1],
                    network["channels"],
                    network["kernels"]
                )
        ):
            setattr(self, f"layer_{i}", torch.nn.Conv2d(_in, _out, kernel_size=_kernel))

        with torch.no_grad():
            _output = torch.zeros([1, network["input_channels"], 210, 160])
            for layer in self._modules.values():
                _output = layer(_output)
            _output = _output.view(_output.size(0), -1).shape[1]

        setattr(self, f"layer_{len(network['channels'])}",
                torch.nn.Linear(_output, network["outputs"], dtype=torch.float32))

        # LEARNING
        # ------------------------------------------------------------------------------------------
        # Default discount factor is 0.99, as suggested by the Google DeepMind paper "Human-level
        # control through deep reinforcement learning" (2015).

        self.discount = other.get("discount", 0.99)
        self.optimizer = optimizer["optim"](self.parameters(), lr=optimizer["lr"],
                                            **optimizer.get("hyperparameters", {}))

        self.memory = {key: [] for key in ["logarithm", "reward"]}

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
        _output = _output.view(_output.size(0), -1).flatten()

        output = getattr(self, f"layer_{len(self._modules)-1}")(_output)

        return output

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
        In order for the agent to best learn the optimal actions, it is common to evaluate the
        expected future rewards. Then, the agent can adjust its predicted action probabilities
        (policy) so that this expected reward is maximized. This is done through the REINFORCE
        algorithm, which computes the policy gradient. Algorithm modified from:

         https://medium.com/@thechrisyoon/deriving-policy-gradients-and-implementing-reinforce
         -f887949bd63
        """
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
        # The gradient is then used to update the agent's policy. This is done by backpropagating
        # with the optimizer using the gradient.

        self.optimizer.zero_grad()
        gradient.backward()
        self.optimizer.step()

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
