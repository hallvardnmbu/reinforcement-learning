"""Deep reinforcement learning agent base class."""

from abc import ABC, abstractmethod
import torch


class Agent(ABC, torch.nn.Module):
    """Base Agent for reinforcement learning."""
    def __init__(self,
                 inputs=4,
                 outputs=2,
                 optimizer=torch.optim.RMSprop,
                 lr=0.00025,
                 ):
        """
        Base Agent for reinforcement learning.

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
        super().__init__()

        # ARCHITECTURE
        # --------------------------------------------------

        self.layer_in = torch.nn.Linear(inputs, 20)
        self.layer_hidden = torch.nn.Linear(20, 80)
        self.layer_out = torch.nn.Linear(80, outputs)

        # LEARNING
        # --------------------------------------------------
        # discount : float
        #     Discount factor for future rewards.
        #     --> 0: only consider immediate rewards
        #     --> 1: consider all future rewards equally

        self.discount = 0.99
        self.optimizer = optimizer(self.parameters(), lr=lr)

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
        _output = torch.relu(self.layer_in(state))
        _output = torch.relu(self.layer_hidden(_output))
        output = self.layer_out(_output)

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
    def memorize(self, *args, **kwargs):
        """
        Abstract method for memorizing.

        Parameters
        ----------
        *args : list
            Positional arguments to memorize.
        **kwargs : dict
            Keyword arguments to memorize.
        """
