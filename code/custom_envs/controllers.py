from abc import ABC, abstractmethod
from collections import deque

import numpy
import scipy
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import gymnasium as gym
import control as ct
import numpy as np

from custom_envs.observation_manager import CarRacingObservationManager, ObservationManager, CartPoleObservationManager
from utils.tools import compute_distance_between_vectors, make_racing_env


class Controller(nn.Module, ABC):

    def __init__(self, obs_man: ObservationManager):
        self.obs_man = obs_man
        super(Controller, self).__init__()

    @abstractmethod
    def get_action(self, state, greedy=True):
        pass


class AttenuatedController(Controller):
    def __init__(self, controller: Controller, attenuation_factor):
        super(AttenuatedController, self).__init__(controller.obs_man)
        self.attenuation_factor = attenuation_factor
        self.controller = controller

    def get_action(self, state, greedy=True):
        return self.attenuation_factor * self.controller.get_action(state, greedy)


class GainScheduleCarRacingController(Controller):

    def __init__(self, obs_man: ObservationManager, k_angle, k_cross_track, k_velocity, k_curvature_rad, max_target_vel,
                 vel_gain_half, vel_gain_dec=1, adapt_vel_gain = True):
        super(GainScheduleCarRacingController, self).__init__(obs_man=obs_man)
        env = make_racing_env(seed=0)()
        self.k_angle = k_angle
        self.k_cross_track = k_cross_track
        self.k_velocity = k_velocity
        self.k_curvature_rad = k_curvature_rad
        self.max_target_vel = max_target_vel
        self.vel_gain_half = vel_gain_half
        self.vel_gain_dec = vel_gain_dec
        self.adapt_vel_gain = adapt_vel_gain

        self.action_size = np.array(env.action_space.shape).prod()
        self.observation_size = np.array(self.obs_man.obs_shape_ctrl).prod()
        self.max_steer_angle = env.unwrapped.vehicle_model.steering_controller.max_angle

        self.max_action = nn.Parameter(torch.tensor(env.action_space.high))
        self.min_action = nn.Parameter(torch.tensor(env.action_space.low))

    def get_action(self, state, greedy=True):
        device = state.device
        if len(state.shape) == 1:
            out = torch.tensor(self.__get_action(state), dtype=torch.float32, device=device)
            out = torch.clip(out, self.min_action, self.max_action)
            return out
        else:
            returns = []
            for i in range(state.shape[0]):
                single_state = state[i]
                out = torch.tensor(self.__get_action(single_state), dtype=torch.float32, device=device)
                out = torch.clip(out, self.min_action, self.max_action)
                returns.append(out)
            return torch.stack(returns)

    def __get_action(self, state):
        cross_track_error = state[0]
        heading_angle_error = state[1]
        velocity = state[2]
        curvature_radius = state[3]

        if self.adapt_vel_gain:
            k_vel = self._k_vel_gain_schedule(velocity)
        else:
            k_vel = self.k_velocity
        vel_desired = self._get_desired_velocity(curvature_radius.item())

        # linearized stanley controller for lateral control
        steering_control = (self.k_cross_track / self.max_steer_angle * cross_track_error / (velocity + 1) +
                            self.k_angle / self.max_steer_angle * heading_angle_error)
        #steering_control = (self.k_cross_track / self.max_steer_angle * cross_track_error +
        #                    self.k_angle / self.max_steer_angle * heading_angle_error)

        # longitudinal control
        brake_control = -1
        acc_control = -1
        if velocity > vel_desired:
            brake_control = k_vel * (velocity - vel_desired) - 1
        else:
            acc_control = k_vel * (vel_desired - velocity) - 1

        return [steering_control, acc_control, brake_control]

    def _k_vel_gain_schedule(self, velocity):
        return -self.k_velocity * (1/(1+torch.exp(-self.vel_gain_dec*(velocity-self.vel_gain_half)))) + self.k_velocity

    def _get_desired_velocity(self, curvature_radius):
        return min(self.k_curvature_rad * curvature_radius, self.max_target_vel)


class LinearGainScheduleRacingController(GainScheduleCarRacingController):
    def __init__(self, obs_man: ObservationManager, k_angle, k_cross_track, k_velocity, k_curvature_rad, max_target_vel,
                 vel_low=8, vel_high=28, gain_low=0.2):
        super(LinearGainScheduleRacingController, self).__init__(obs_man, k_angle, k_cross_track, k_velocity, k_curvature_rad,
                                                                 max_target_vel, vel_gain_half=None, vel_gain_dec=None,
                                                                 adapt_vel_gain=True)

        self.vel_low = vel_low
        self.vel_high = vel_high
        self.gain_low = gain_low
        self.gain_high = 1.0

    def _k_vel_gain_schedule(self, velocity):
        m = (self.gain_high - self.gain_low) / (self.vel_low - self.vel_high)
        lin = m * (velocity - self.vel_low) + self.gain_high
        return self.k_velocity * torch.clip(lin, self.gain_low, self.gain_high)


class CustomCarRacingController(Controller):

    def __init__(self, obs_man: ObservationManager, k_angle, k_cross_track, k_velocity, k_curvature_rad, max_target_vel):
        super(CustomCarRacingController, self).__init__(obs_man=obs_man)
        env = make_racing_env(seed=0)()
        self.k_angle = k_angle
        self.k_cross_track = k_cross_track
        self.k_velocity = k_velocity
        self.k_curvature_rad = k_curvature_rad
        self.max_target_vel = max_target_vel

        #self.observation_size = np.array(env.observation_space.shape).prod() + self.augmented_env
        self.action_size = np.array(env.action_space.shape).prod()
        self.observation_size = np.array(self.obs_man.obs_shape_ctrl).prod()
        self.max_steer_angle = env.unwrapped.vehicle_model.steering_controller.max_angle

        self.pick_layer, self.feature_layer, self.feature_activation = self.init_feature_layers()
        self.acc_control_layer = self.init_acc_control_layer()
        self.brake_control_layer = self.init_brake_control_layer()

        self.max_action = nn.Parameter(torch.tensor(env.action_space.high))
        self.min_action = nn.Parameter(torch.tensor(env.action_space.low))

    def get_action(self, state, greedy=True):
        #return self.fc(state)
        #state = self.obs_man.get_ctrl_state(state)
        f = self.compute_features(state)

        # working with a batch of states
        if len(f.shape) > 1:
            velocity_error = -f[:, 4] + f[:, 3] - f[:, 2]  # v_target - v
        else:
            velocity_error = -f[4] + f[3] - f[2]

        acc_mask = torch.where(velocity_error > 0, 1.0, 0.0)
        brake_mask = 1 - acc_mask

        out = self.acc_control_layer(f) * acc_mask.view(-1, 1) + self.brake_control_layer(f) * brake_mask.view(-1, 1)
        out = torch.clip(out, self.min_action, self.max_action)

        return out

    def compute_features(self, state):
        picked = self.pick_layer(state)
        feature = self.feature_activation(self.feature_layer(state))

        return torch.cat([picked, feature], dim=-1)

    def init_feature_layers(self):
        pick_out_layer = nn.Linear(self.observation_size, 4)
        weights = np.zeros(shape=pick_out_layer.weight.shape)
        weights[0, -4] = 1.0  # pick out cross_track_err
        weights[1, -3] = 1.0  # pick out steering_err
        weights[2, -2] = 1.0  # pick out velocity
        bias = np.zeros(shape=(4,))
        bias[3] = self.max_target_vel  # pick out max_target_velocity
        weights = torch.tensor(weights)
        bias = torch.tensor(bias)

        with torch.no_grad():
            pick_out_layer.weight.copy_(weights)
            pick_out_layer.bias.copy_(bias)

        # computing feature representation for clipped target velocity
        target_vel_layer = nn.Linear(self.observation_size, 1)
        weights2 = np.zeros(shape=target_vel_layer.weight.shape)
        weights2[0, -1] = -self.k_curvature_rad
        bias2 = np.zeros(shape=(1,))
        bias2[0] = self.max_target_vel
        weights2 = torch.tensor(weights2)
        bias2 = torch.tensor(bias2)

        with torch.no_grad():
            target_vel_layer.weight.copy_(weights2)
            target_vel_layer.bias.copy_(bias2)

        return pick_out_layer, target_vel_layer, F.relu

    def init_acc_control_layer(self):
        control_layer = nn.Linear(5, self.action_size)
        weights = np.zeros(shape=control_layer.weight.shape)
        weights[0, 0] = self.k_cross_track / self.max_steer_angle
        weights[0, 1] = self.k_angle / self.max_steer_angle
        weights[1, 2] = -self.k_velocity
        weights[1, 3] = self.k_velocity
        weights[1, 4] = -self.k_velocity
        bias = np.zeros(shape=(self.action_size,))
        bias[1] = -1
        bias[2] = -1
        weights = torch.tensor(weights)
        bias = torch.tensor(bias)

        with torch.no_grad():
            control_layer.weight.copy_(weights)
            control_layer.bias.copy_(bias)

        return control_layer

    def init_brake_control_layer(self):
        control_layer = nn.Linear(5, self.action_size)
        weights = np.zeros(shape=control_layer.weight.shape)
        weights[0, 0] = self.k_cross_track / self.max_steer_angle
        weights[0, 1] = self.k_angle / self.max_steer_angle
        weights[2, 2] = self.k_velocity
        weights[2, 3] = -self.k_velocity
        weights[2, 4] = self.k_velocity
        bias = np.zeros(shape=(self.action_size,))
        bias[1] = -1
        bias[2] = -1
        weights = torch.tensor(weights)
        bias = torch.tensor(bias)

        with torch.no_grad():
            control_layer.weight.copy_(weights)
            control_layer.bias.copy_(bias)

        return control_layer


class CartPoleController(Controller):

    def __init__(self, obs_man: CartPoleObservationManager, place_poles=False, K=None):
        super(CartPoleController, self).__init__(obs_man=obs_man)

        env = gym.make('CustomCartPole-v1')
        cartpole = ct.NonlinearIOSystem(env.unwrapped.ct_sys_update, env.unwrapped.ct_sys_output, states=4, name='cartpole',
                                        inputs=['action'], outputs=['x', 'x_dot', 'theta', 'theta_dot'])
        linsys = cartpole.linearize(x0=env.unwrapped.goal_state, u0=np.array([0.]))
        linsys_d = linsys.sample(env.unwrapped.tau)

        cost_x = env.unwrapped.cost_x
        cost_x_dot = env.unwrapped.cost_x_dot
        cost_theta = env.unwrapped.cost_theta
        cost_theta_dot = env.unwrapped.cost_theta_dot
        cost_control = env.unwrapped.cost_control

        Q = np.diag([cost_x, cost_x_dot, cost_theta, cost_theta_dot])
        R = np.diag([cost_control])

        if place_poles:
            self.K = self.place_poles(linsys_d)
        elif K:
            self.K = torch.tensor(K)
        else:
            self.K = torch.tensor(ct.lqr(linsys_d, Q, R)[0])

        self.fc = nn.Linear(np.array(self.obs_man.obs_shape_ctrl).prod(), 1)
        weights = self.K

        with torch.no_grad():
            self.fc.weight.copy_(-weights)
            init.constant_(self.fc.bias, 0)

        self.max_action = nn.Parameter(torch.tensor(env.action_space.high))
        self.min_action = nn.Parameter(torch.tensor(env.action_space.low))

    def place_poles(self, linsys):
        desired_poles = [0.35114616 + 0.1j, 0.35114616 - 0.1j, 0.97800293 + 0.01737253j, 0.97800293 - 0.01737253j]
        K = ct.place(linsys.A, linsys.B, desired_poles)
        return torch.tensor(K[0])

    def forward(self, x):
        output = self.fc(x)
        output = torch.clip(output, self.min_action, self.max_action)

        return output

    def get_action(self, state, greedy=True):
        return self.forward(state)


class SillyCartPoleController(Controller):

    def __init__(self, obs_man: CartPoleObservationManager, fixed_action: float = 2):
        super(SillyCartPoleController, self).__init__(obs_man=obs_man)
        env = gym.make('CustomCartPole-v1')
        max_action = env.action_space.high.squeeze()
        min_action = env.action_space.low.squeeze()
        fixed_action = np.clip(fixed_action, min_action, max_action)
        self.fc = nn.Linear(np.array(self.obs_man.obs_shape_ctrl).prod(), 1)
        init.constant_(self.fc.weight, 0)
        init.constant_(self.fc.bias, fixed_action)

    def forward(self, x):
        return self.fc(x)

    def get_action(self, state, greedy=True):
        return self.forward(state)