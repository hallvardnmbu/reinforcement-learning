import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import Trainer
from torchvision import transforms
import gymnasium as gym
import torch

from agent_lightning import VisionDeepQ  # Ensure this is your Lightning module


if __name__ == "__main__":
    # Initialize environment
    env = gym.make('ALE/Tetris-v5', render_mode="rgb_array",
                   obs_type="grayscale", frameskip=4, repeat_action_probability=0.25)

    model = VisionDeepQ(capacity=10000,
                        batch_size=64,
                        lr=1e-4,
                        sync_rate=10,
                        environment=env)

    logger = CSVLogger("logs", name="VisionDeepQ")
    trainer = Trainer(max_epochs=200,
                      log_every_n_steps=10,
                      logger=logger,
                      callbacks=[ModelCheckpoint(dirpath="checkpoints/")],
                      accelerator="auto")

    trainer.fit(model)