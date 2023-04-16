from datetime import datetime

import tensorflow as tf
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback,self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        value = np.random.random()
        self.logger.record("random_value", value)
        #self.locals.
        #self.logger.record("reward", self.training_env.env_method('get_rewards'))
        episode_rewards = self.training_env.get_attr('episode_rewards')[-1]
        mean_episode_reward = np.mean(episode_rewards)

        self.logger.record("reward", mean_episode_reward)
        return True

"""    def on_step_end(self, step):
        # Log scalar value (here a random variable)
        value = np.random.random()
        self.logger.record("random_value", value)
        # self.locals."""

