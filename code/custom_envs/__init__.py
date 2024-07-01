from gymnasium.envs.registration import register
from custom_envs.CarEnv.Configs import RACING_FAST

register(id="CustomCartPole-v1", entry_point='custom_envs.cart_pole:CartPoleEnv', max_episode_steps=500)
register(id='CarEnv-v1', entry_point='custom_envs.CarEnv.Env:CarEnv', kwargs={'config': RACING_FAST})