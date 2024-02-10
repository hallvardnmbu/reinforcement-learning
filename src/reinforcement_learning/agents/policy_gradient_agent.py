import numpy as np
import torch
from reinforcement_learning.agents import Agent


class PolicyGradientAgent(Agent):
    """Policy-based Agent for reinforcement learning."""
    def __init__(self,
                 inputs=4,
                 outputs=2,
                 optimizer=torch.optim.RMSprop,
                 lr=0.00025,
                 ):
        """
        Policy-based Agent for reinforcement learning.

        Parameters
        ----------
        inputs : int, optional
            Number of input nodes (observations).
        outputs : int, optional
            Number of output nodes (actions).
        optimizer : torch.optim.X, optional
            Optimizer for the Agent to learn.
        lr : float, optional
            Learning rate for the optimizer.
        """
        super().__init__(inputs, outputs, optimizer, lr)

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

        self.memory["logarithm"] = []
        self.memory["reward"] = []

        return gradient.item()

    def memorize(self, *args, **kwargs):
        """
        Append observation, action and reward to Agent memory.

        Parameters
        ----------
        *args : list
            Positional arguments to memorize.
        **kwargs : dict
            Keyword arguments to memorize.
        """
        logarithm, reward = args

        self.memory["logarithm"].append(logarithm)
        self.memory["reward"].append(reward)
