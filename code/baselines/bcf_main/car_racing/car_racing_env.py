import numpy as np
import os

from custom_envs.observation_manager import CarRacingObservationManager
from utils.tools import make_racing_env

PATH = os.path.dirname(os.path.realpath(__file__))

# ========================================================
# ENV CLASS
# Gym-style wrapper for the Car Racing Environment from Max Schier


class CarRacingEnv:
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, lambda_context=False):

        # create an instance of the normal environment
        self.env = make_racing_env(seed=42, training_tracks_path='custom_envs/CarEnv/SavedTracks/training_track')()

        # create an observation manager
        self.obs_man = CarRacingObservationManager(lambda_context)

        # set action space and observation space of the env
        self.action_space = self.env.action_space
        self.observation_space = self.obs_man.obs_space_rl

    def reset(self):
        obs, _ = self.env.reset()
        return self.obs_man.get_rl_state(obs)

    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        return self.obs_man.get_rl_state(next_obs), reward, truncated or terminated, info

    def seed(self, seed):
        np.random.seed(seed)
        self.env.reset(seed=seed)
        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)

    def get_controller_obs(self):
        obs = np.concatenate([v.flatten() for k, v in self.env.last_observations.items()]).astype(np.float32)
        return self.obs_man.get_ctrl_state(obs)
