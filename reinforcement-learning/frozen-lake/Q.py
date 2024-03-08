"""Classic-style tabular Q learning agent for small, discrete-state games."""

import numpy as np
import pandas as pd


class TabularQAgent:
    """Tabular Q-learning agent."""
    def __init__(self, space, lr, gamma, exploration):
        """
        Classic-style tabular Q learning agent for small, discrete-state games.

        Parameters
        ----------
        space : dict
            The number of possible states and actions in the environment.
            Must contain the following keys:

            states : int
                The number of possible states in the environment.
            actions : int
                The number of possible actions in the environment.
        lr : float
            The learning rate for updating the Q-table.
        gamma : float
            The discount factor for future rewards.
        exploration : dict
            The exploration rate for selecting random actions.
            Must contain the following keys:

            rate : float
                The exploration rate for selecting random actions.
            decay : float
                The rate at which the exploration rate decays over time.
            min : float
                The minimum exploration rate.
        """
        self.actions = space['actions']

        self.lr = lr
        self.gamma = gamma

        self.exploration = exploration

        self.table = pd.DataFrame(data=np.zeros(shape=(space["actions"], space["states"])),
                                  columns=[f"State {i}" for i in range(space["states"])],
                                  index=[f"Action {i}" for i in range(space["actions"])])

    def action(self, state):
        """
        Selects an action to take based on the given state.
        Uses simple greedy-epsilon algorithm.

        Parameters
        ----------
        state : int
            The current state of the environment.

        Returns
        -------
        int
            The selected action to take.
        """
        if np.random.random() > self.exploration['rate']:
            action = int(self.table.iloc[:, state].idxmax().split(" ")[-1])
        else:
            action = np.random.choice(range(self.actions))

        self.exploration['rate'] = max(self.exploration['min'],
                                       self.exploration['rate'] - self.exploration['decay'])

        return action

    def learn(self, old_state, action, reward, new_state):
        """
        Updates the Q-table based on the agent's experience.

        Parameters
        ----------
        old_state : int
            The state the agent was in before taking the action.
        action : int
            The action taken by the agent.
        reward : float
            The reward received for taking the action.
        new_state : int
            The state the agent is in after taking the action.
        """
        observed = reward + self.gamma * self.table.iloc[:, new_state].max()
        expected = self.table.at[f"Action {action}", f"State {old_state}"]

        updated = expected + self.lr * (observed - expected)

        self.table.at[f"Action {action}", f"State {old_state}"] = updated
