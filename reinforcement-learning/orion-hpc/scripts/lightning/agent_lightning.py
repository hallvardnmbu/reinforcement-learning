import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from memory_lightning import ReplayBuffer
from base_CNN import BaseCNN
import copy
from torch.utils.data import DataLoader
from memory_lightning import CustomBatchSampler


class VisionDeepQ(pl.LightningModule):
    def __init__(self,
                 environment,
                 network=BaseCNN(),
                 capacity=5000,
                 batch_size=32,
                 lr=1e-3,
                 sync_rate=100,
                 loss_fn=nn.MSELoss()):
        super().__init__()
        self.environment = environment
        self.network = network  # Assuming VisionDeepQ is defined
        self.batch_size = batch_size
        self.lr = lr
        self.sync_rate = sync_rate
        self.loss_fn = loss_fn
        self.target_network = copy.deepcopy(network)
        self.replay_buffer = ReplayBuffer(capacity=capacity)
        self.populate_replay_buffer()

        # To ensure the target network's weights are frozen during backpropagation
        for param in self.target_network.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        # Directly sample from the replay buffer
        experiences = self.replay_buffer.sample(self.batch_size)
        states, actions, next_states, rewards, dones = experiences

        q_values = self.network(states).gather(1, actions.long().unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(dim=1)[0]
            expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.loss_fn(q_values, expected_q_values)
        self.log('train_loss', loss)
        return loss

    def train_dataloader(self):
        # Instantiate your CustomBatchSampler with the replay buffer
        custom_batch_sampler = CustomBatchSampler(self.replay_buffer, batch_size=self.batch_size)

        # Create the DataLoader with your custom batch sampler
        return DataLoader(self.replay_buffer, batch_sampler=custom_batch_sampler, num_workers=4, persistent_workers=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.network.parameters(), lr=self.lr)
        return optimizer

    def on_train_batch_end(self, outputs, batch, batch_idx, unused=0):
        if (batch_idx + 1) % self.sync_rate == 0:
            self.target_network.load_state_dict(self.network.state_dict())

    def populate_replay_buffer(self, prepopulate_steps=1000):
        state = self.environment.reset()
        for _ in range(prepopulate_steps):
            action = self.environment.action_space.sample()  # Take a random action
            next_state, reward, terminated, truncated, _ = self.environment.step(action)
            done = terminated or truncated
            self.replay_buffer.append(state, action, next_state, reward, done)
            state = next_state if not done else self.environment.reset()
