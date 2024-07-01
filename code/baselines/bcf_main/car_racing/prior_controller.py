import torch

from custom_envs.controllers import LinearGainScheduleRacingController


# ===============================================
# Wrapper Class for the Path Following Controller with Adaptive Speeds

class PathFollowingController:
    def __init__(self, env):
        self.env = env
        self.controller = LinearGainScheduleRacingController(k_angle=1.0, k_cross_track=-0.5, k_velocity=0.25,
                                                             k_curvature_rad=0.4, max_target_vel=8, vel_low=8,
                                                             vel_high=28, gain_low=0.2, obs_man=env.obs_man)

    def compute_action(self):
        obs = self.env.get_controller_obs()
        torch_obs = torch.tensor(obs)
        with torch.no_grad():
            action = self.controller.get_action(torch_obs)
        return action.squeeze().cpu().numpy()

