from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import torch

import gymnasium as gym

from utils.tools import make_racing_env, make_env


class Algorithm(Enum):
    RL = 'rl'
    CTRL = 'ctrl'
    MIXED = 'mixed'


class ObservationManager(ABC):

    @abstractmethod
    def get_rl_state(self, state: torch.Tensor | np.ndarray):
        pass

    @abstractmethod
    def get_ctrl_state(self, state: torch.Tensor | np.ndarray):
        pass

    @property
    @abstractmethod
    def obs_shape_rl(self):
        pass

    @property
    @abstractmethod
    def obs_shape_ctrl(self):
        pass

    @property
    @abstractmethod
    def obs_space_rl(self):
        pass

    @property
    @abstractmethod
    def augmented_env(self):
        pass


class CarRacingObservationManager(ObservationManager):

    def __init__(self, augmented_env=False):
        self._augmented_env = augmented_env
        self._obs_space_rl = self.__construct_rl_obs_space()

    def get_ctrl_state(self, state: torch.Tensor | np.ndarray):
        indices = self.__get_indices(Algorithm.CTRL)
        if len(state.shape) > 1:
            return state[:, indices]
        else:
            return state[indices]

    def get_rl_state(self, state: torch.Tensor | np.ndarray):
        indices = self.__get_indices(Algorithm.RL)
        if len(state.shape) > 1:
            return state[:, indices]
        else:
            return state[indices]

    @property
    def obs_shape_ctrl(self):
        return (len(self.__get_indices(Algorithm.CTRL)),)

    @property
    def obs_shape_rl(self):
        return (len(self.__get_indices(Algorithm.RL)),)

    @property
    def obs_space_rl(self):
        return self._obs_space_rl

    @property
    def augmented_env(self):
        return self._augmented_env

    def __get_indices(self, algo_type: Algorithm):
        if algo_type == Algorithm.RL:
            if self.augmented_env:
                indices = list(range(44))
                indices.append(48)
            else:
                indices = list(range(44))
        elif algo_type == Algorithm.CTRL:
            indices = list(range(44, 48))
        else:
            raise ValueError('Mixed Agent not supported')

        return indices

    def __construct_rl_obs_space(self):
        dummy_env = make_racing_env(seed=0)()
        indices = self.__get_indices(Algorithm.RL)
        low = dummy_env.observation_space.low
        high = dummy_env.observation_space.high
        if self.augmented_env:
            low = np.append(low, 0)
            high = np.append(high, 1)

        low = low[indices]
        high = high[indices]

        obs_space = gym.spaces.Box(low=low, high=high, shape=high.shape)

        return obs_space


class CartPoleObservationManager(ObservationManager):

    def __init__(self, augmented_env):
        self._augmented_env = augmented_env
        self._obs_space_rl = self.__construct_rl_obs_space()

    def get_rl_state(self, state: torch.Tensor | np.ndarray):
        return state

    def get_ctrl_state(self, state: torch.Tensor | np.ndarray):
        if len(state.shape) > 1:
            if self.augmented_env:
                return state[:, :-1]
            else:
                return state
        else:
            if self.augmented_env:
                return state[:-1]
            else:
                return state

    @property
    def obs_shape_rl(self):
        return (4 + self.augmented_env,)

    @property
    def obs_shape_ctrl(self):
        return (4,)

    @property
    def obs_space_rl(self):
        return self._obs_space_rl

    def __construct_rl_obs_space(self):
        dummy_env = make_env(env_id='CustomCartPole-v1', seed=0)()
        low = dummy_env.observation_space.low
        high = dummy_env.observation_space.high
        if self.augmented_env:
            low = np.append(low, 0)
            high = np.append(high, 1)

        obs_space = gym.spaces.Box(low=low, high=high, shape=high.shape)

        return obs_space

    @property
    def augmented_env(self):
        return self._augmented_env

