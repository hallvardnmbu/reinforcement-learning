import pandas as pd
import numpy as np


class TabularQAgent:
    """Classic-style tabular Q learning agent for small, discrete-state games."""
    def __init__(self, state_space, action_space, lr,
                 gamma, epsilon, epsilon_decay, epsilon_minimum):
        """
        Classic-style tabular Q learning agent for small, discrete-state games.

        Parameters
        ----------
        state_space : int
            The number of possible states in the environment.
        action_space : int
            The number of possible actions in the environment.
        lr : float
            The learning rate for updating the Q-table.
        gamma : float
            The discount factor for future rewards.
        epsilon : float
            The exploration rate for selecting random actions.
        epsilon_decay : float
            The rate at which the exploration rate decays over time.
        epsilon_minimum : float
            The minimum exploration rate.

        Returns:
        - None
        """
        self.state_space = state_space
        self.action_space = action_space
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_minimum = epsilon_minimum
        self.Q_table = pd.DataFrame(data=np.zeros(shape=(action_space, state_space)),
                                    columns=[f"State {i}" for i in range(state_space)],
                                    index=[f"Action {i}" for i in range(action_space)])

    def action(self, state):
        """
        Selects an action to take based on the given state.
        Uses simple greedy-epsilon algorithm.

        Parameters:
            state (numpy.ndarray): The current state of the environment.

        Returns:
            int: The selected action to take.
        """
        if np.random.random() > self.epsilon:
            action = self.greedy_action(state)
        else:
            action = np.random.choice(range(self.action_space))
        self.decrement_epsilon()
        return action

    def decrement_epsilon(self):
        """
        Decrements the value of epsilon by the epsilon_decay rate,
        but ensures it doesn't go below epsilon_minimum.
        """
        self.epsilon = max(self.epsilon_minimum, self.epsilon - self.epsilon_decay)

    def greedy_action(self, state):
        """
        Selects the action with the highest Q-value for the given state.

        Parameters:
            state (int): The current state.

        Returns:
            int: The selected action.
        """
        action = self.Q_table.iloc[:, state].idxmax()
        return int(action.split(" ")[-1])

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

        Returns
        -------
        None
            This method updates the Q-table in-place and has no return value.
        """
        Q_observed = reward + self.gamma * self.Q_table.iloc[:, new_state].max()
        Q_expected = self.Q_table.at[f"Action {action}", f"State {old_state}"]
        new_Q_val = Q_expected + self.lr * (Q_observed - Q_expected)
        self.Q_table.at[f"Action {action}", f"State {old_state}"] = new_Q_val
