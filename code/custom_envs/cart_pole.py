import gymnasium as gym
import numpy as np
import math
from typing import Optional, Union

from gymnasium.error import DependencyNotInstalled
from gymnasium import logger, spaces


class CartPoleEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode: Optional[str] = None):
        self.gravity = 9.81
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.cart_friction = 0.0
        self.pole_friction = 0.0
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        self.theta_ddot_old = 0.

        self.max_torque = 10

        # Angle at which to fail the episode
        self.theta_threshold_radians = 24 * 2 * math.pi / 360
        self.x_threshold = 4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        #self.action_space = spaces.Box(-100, 100)
        self.action_space = spaces.Box(-self.max_torque, self.max_torque, shape=(1,))
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

        # Reward settings
        self.cost_x = 10
        self.cost_x_dot = 1
        self.cost_theta = 10
        self.cost_theta_dot = 1
        self.cost_control = 10
        self.survival_bias = 100

        # Goal state
        self.goal_state = np.array([0, 0, 0, 0])  # for x, x_dot, theta, theta_dot, changing x will modify environmnet and controller to learn new setpoint

        self.steps_beyond_terminated = None

    def get_reward(self, state, action):
        '''
        Penalizes the deviation from state and control effort + survival bias
        :param state: next_state of the agent
        :param action: last taken action of the agent
        :return: the reward at that timestep
        '''
        if isinstance(action, np.ndarray):
            # If action is an array, apply np.clip() to its first element
            force = action.item()
        else:
            # If action is a scalar, apply np.clip() directly
            force = action

        #TODO: Modify to have different reward function for RL part then prior controller
        state_diff = (np.array([2, 0, 0, 0]) - state)**2
        state_cost = np.sum(state_diff * np.array([self.cost_x, self.cost_x_dot, self.cost_theta, self.cost_theta_dot]))
        reward = self.survival_bias - force**2 * self.cost_control - state_cost

        # scaling of reward
        reward = reward / self.survival_bias

        return reward

    def state_update(self, action, state):  #need to have state as input to match syntax of control library

        x, x_dot, theta, theta_dot = state
        if isinstance(action, np.ndarray):
            # If action is an array, apply np.clip() to its first element
            force = np.clip(action, -self.max_torque, self.max_torque).item()
        else:
            # If action is a scalar, apply np.clip() directly
            force = np.clip(action, -self.max_torque, self.max_torque)

        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        normal_force_cart = self.total_mass*self.gravity - self.masspole*self.length*(self.theta_ddot_old*sintheta + theta_dot**2*costheta)

        numerator_theta_ddot = self.gravity*sintheta + costheta*((-force-self.masspole*self.length*theta_dot**2*(sintheta + self.cart_friction*np.sign(normal_force_cart*x_dot)*costheta))/(self.total_mass) + self.cart_friction*self.gravity*np.sign(normal_force_cart*x_dot))-(self.pole_friction*theta_dot)/(self.masspole*self.length)
        denominator_theta_ddot = self.length*(4./3. - (self.masspole*costheta)/(self.total_mass)*(costheta-self.cart_friction*np.sign(normal_force_cart*x_dot)))
        theta_ddot = numerator_theta_ddot/denominator_theta_ddot

        x_ddot = (force+self.masspole*self.length*(theta_dot**2*sintheta - self.theta_ddot_old*costheta) - self.cart_friction*normal_force_cart*np.sign(normal_force_cart*x_dot))/self.total_mass

        self.theta_ddot_old = theta_ddot

        x_dot = x_dot if np.isscalar(x_dot) else x_dot.item()
        x_ddot = x_ddot if np.isscalar(x_ddot) else x_ddot.item()
        theta_dot = theta_dot if np.isscalar(theta_dot) else theta_dot.item()
        theta_ddot = theta_ddot if np.isscalar(theta_ddot) else theta_ddot.item()
        state_dot = np.array([x_dot, x_ddot, theta_dot, theta_ddot])
        return state_dot

    def ct_sys_update(self, t, x, u, params):
        x_dot = self.state_update(u[0], x)
        return x_dot

    def ct_sys_output(self, t, x, u, params):
        return x

    def step(self, action):
        state_dot = self.state_update(action, self.state)
        self.state += state_dot*self.tau

        x, x_dot, theta, theta_dot = self.state

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not terminated:
            # reward = 1.0
            reward = self.get_reward(self.state, action)
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = self.get_reward(self.state, action)
            # reward = 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0.0

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        return_info: bool = False,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.

        if options is not None and "start_state" in options.keys():
            self.state = options['start_state']
        #if start_state is not None:
        #    self.state = start_state
        #else:
        #    self.state = self.np_random.uniform(low=low, high=high, size=(4,))
        else:
            x_position = self.np_random.uniform(low=-self.x_threshold/4, high=self.x_threshold/4)
            x_velocity = 0
            angle = self.np_random.uniform(low=-self.theta_threshold_radians/2, high=self.theta_threshold_radians/2)
            angle_velocity = 0
            self.state = np.array([x_position, x_velocity, angle, angle_velocity])

        self.steps_beyond_terminated = None
        self.theta_ddot_old = 0.

        if self.render_mode == "human":
            self.render()

        return np.array(self.state, dtype=np.float32), {}

    def render(self, mode=None):
        if mode == None:
            mode = self.render_mode
        if mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False