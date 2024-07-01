import gymnasium as gym
import numpy as np
from .FreeDriveProblem import FreeDriveProblem


class ComplexRacingProblem(FreeDriveProblem):
    @property
    def state_observation_space(self):
        return gym.spaces.Box(-np.inf, np.inf, (16,))

    def observe_state(self, env):
        v_state = env.vehicle_state
        assert len(v_state.shape) == 2
        assert v_state.shape[1] == 7

        omega, beta = v_state[0, 5:]

        v_x, v_y = env.vehicle_model.v_loc_[0] if env.vehicle_model.v_loc_ is not None else (0, 0)

        p_x, p_y = env.vehicle_state[0, :2]
        theta = env.vehicle_state[0, 2]

        front_slip = 1 if env.vehicle_model.front_slip_ else 0
        rear_slip = 1 if env.vehicle_model.rear_slip_ else 0

        result = np.array([
            v_x / env.vehicle_model.top_speed,  # longitudinal velocity
            v_y / env.vehicle_model.top_speed,  # lateral velocity
            omega,  # steering angular velocity
            beta / env.vehicle_model.steering_controller.max_angle,  # steering angle
            p_x,  # vehicle position x
            p_y,  # vehicle position y
            theta,  # vehicle orientation
            200,  # R value of vehicle color
            41,  # G value of vehicle color
            29,  # B value of vehicle color
            env.traveled_distance,  # traveled distance so far this episode
            front_slip,  # 1 if front wheels are slipping, else 0
            rear_slip,  # 1 if rear wheels are slipping, else 0
            env.acc_x,  # longitudinal acceleration
            env.acc_y,  # lateral acceleration
            env.torque  # torque
        ])

        return result


class RacingProblem(FreeDriveProblem):
    @property
    def state_observation_space(self):
        return gym.spaces.Box(-np.inf, np.inf, (4,))

    def observe_state(self, env):
        v_state = env.vehicle_state
        assert len(v_state.shape) == 2
        assert v_state.shape[1] == 7

        omega, beta = v_state[0, 5:]

        v_x, v_y = env.vehicle_model.v_loc_[0] if env.vehicle_model.v_loc_ is not None else (0, 0)

        p_x, p_y = env.vehicle_state[0, :2]
        theta = env.vehicle_state[0, 2]

        front_slip = 1 if env.vehicle_model.front_slip_ else 0
        rear_slip = 1 if env.vehicle_model.rear_slip_ else 0

        result = np.array([
            v_x / env.vehicle_model.top_speed,  # longitudinal velocity
            v_y / env.vehicle_model.top_speed,  # lateral velocity
            omega,  # wheel angular velocity
            beta / env.vehicle_model.steering_controller.max_angle,  # steering angle
            # p_x,  # vehicle position x
            # p_y,  # vehicle position y
            # theta,  # vehicle orientation
            # env.acc_x,  # longitudinal acceleration
            # env.acc_y,  # lateral acceleration
            # env.torque  # torque
        ])

        return result
