import datetime
import inspect
import numpy as np
import os
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from stable_baselines3 import PPO, A2C, DQN, HER, DDPG, SAC, TD3, PPO

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.logger import configure
from stable_baselines3.common.logger import TensorBoardOutputFormat

from CybORG.Evaluation.tensorboardcallback import TensorboardCallback
from CybORG import CybORG

from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent


class RLagent(BaseAgent):

    def __init__(self, env, agent_type, model_name=None, verbose=1):
        super().__init__()
        self.agent_type = agent_type

        if model_name is None:
            if agent_type == "PPO":
                self.model = PPO('MlpPolicy', env, tensorboard_log="./tb_logs/", verbose=verbose) # n_steps the minimum number of game rounds
            elif agent_type == "A2C":
                self.model = A2C('MlpPolicy', env, tensorboard_log="./tb_logs/", verbose=verbose)
            elif agent_type == "DQN":
                self.model = DQN('MlpPolicy', env, tensorboard_log="./tb_logs/", verbose=verbose, exploration_fraction=0.90)
            else:
                raise Exception("Unknown Agent Type {}".format(agent_type))
        else:
            # print('loading...')
            self.load(agent_type, model_name)

    def train(self, timesteps, log_name):
        #path = str(inspect.getfile(CybORG))[:-10] + '/Evaluation/evals/'
        #self.model.set_logger(configure(path, ["stdout", "csv", "tensorboard"]))
        self.model.learn(timesteps)# tb_log_name=log_name)


    def get_action(self, observation, action_space):

        if self.agent_type == "DQN":
            action, _states = self.model.predict(observation, deterministic=True)
        else:
            action, _states = self.model.predict(observation)
            # if action > 40:
            #   print(action)
            # print("Returning action...", action)
        return action

    def predict(self, observations, state, episode_start, deterministic):
        return self.model.predict(observations, state=state, episode_start=episode_start, deterministic=deterministic)

    def save(self, name=None):
        #print("***saved")
        if name is None:
            name = "{}-{}".format(datetime.datetime.now(), self.agent_type)
        self.model.save(name)

    def load(self, agent_type, model_name):
        #print(f"Loading up RL algorithm {agent_type} with name {model_name}")
        if agent_type == "PPO":
            #print("Loading PPO Agent")
            self.model = PPO.load(model_name)
        elif agent_type == "A2C":
            self.model = A2C.load(model_name)
        elif agent_type == "DQN":
            self.model = DQN.load(model_name)

        self.agent_type = "{} ({})".format(self.agent_type, model_name)

    def __str_(self):
        return self.agent_type
