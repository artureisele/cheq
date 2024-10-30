import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from custom_envs.controllers import CartPoleController
from custom_envs.controllers import CartPoleObservationManager

def main():
    # Create the environment
    env = gym.make('CustomCartPole-v1', render_mode="human")
    obs_man = CartPoleObservationManager(env)  # Replace with your observation manager for CartPole
    controller = CartPoleController(obs_man=obs_man)
    device = torch.device("cuda:0")
    # Reset the environment and get the initial state
    state, info= env.reset()
    done = False

    while not done:
        # Convert the state to tensor for the controller
        state_tensor = torch.tensor(state, dtype=torch.float32)
        # Get action from the controller
        action = controller.get_action(state_tensor)

        # Step the environment
        state, reward, done, truncated, info = env.step(action.item())  # Controller outputs tensor, so we convert to item
        print(reward)
    env.close()

if __name__ == "__main__":
    main()
