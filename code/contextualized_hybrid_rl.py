from datetime import datetime

import numpy as np
import torch

import wandb
import gymnasium as gym
import hydra
from omegaconf import DictConfig, OmegaConf
from agents import MixedAgent
from custom_envs.observation_manager import Algorithm
from utils.logger import Logger
from utils.plotter import Plotter
from utils.tools import inject_weight_into_state, seed_experiment, \
    get_kwargs


def train(config):
    # READ RELEVANT CONFIG DATA
    env_config = config['env']
    log_config = config['logging']
    train_config = config['training']
    eval_config = config['evaluation']

    device = train_config['device'] if train_config['device'] != 'auto' else (
        "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    print(f'Used device {device}')

    seed = train_config['seed']
    number_steps = train_config['number_steps']

    eval_agent = eval_config['eval_agent']
    eval_count = eval_config['eval_count']
    eval_frequency = eval_config['eval_frequency']

    wandb_prj_name = log_config['wandb_project_name']
    capture_video = log_config['capture_video']
    model_save_frequency = log_config['model_save_frequency']

    env_id = env_config['env_id']
    exp_type = config['exp_type']
    run_name = config['run_name']

    augmented_env = True

    seed_experiment(seed)

    envs = gym.vector.SyncVectorEnv(
        [env_config['make_env_function']['type'](seed=seed, **get_kwargs(env_config['make_env_function']))])
    envs.single_observation_space.dtype = np.float32

    obs_man = env_config['observation_manager']['type'](augmented_env=augmented_env,
                                                        **get_kwargs(env_config['observation_manager']))

    agent_trainer = train_config['agent_trainer']['type'](envs, obs_man, device,
                                                          **get_kwargs(train_config['agent_trainer']))

    timestamp = datetime.now().strftime('%y%m%d_%H%M%S')

    # name prefix of output files
    run_name = f"{run_name}_{env_id}_s={seed}_{timestamp}"

    # init list to track agent's performance throughout training
    last_evaluated_episode = None  # stores the episode_step of when the agent's performance was last evaluated
    eval_max_return = -float('inf')

    # Eval
    eval_env = env_config['make_env_function']['type'](seed=seed, **get_kwargs(env_config['make_env_function']))()
    evaluator = env_config['evaluator']['type'](eval_env, device=device, obs_man=obs_man, eval_count=eval_count,
                                                greedy_eval=True,
                                                **get_kwargs(env_config['evaluator']))
    logger = Logger(run_name=run_name, exp_name=exp_type)
    plotter = Plotter(logger, augmented_env=augmented_env)

    controller = env_config['controller']['type'](obs_man=obs_man,device = device, **get_kwargs(env_config['controller'])).to(device)

    mixed_agent = MixedAgent(agent_trainer.agent, controller, obs_man=obs_man)

    weight_scheduler = train_config['weight_scheduler']['type'](**get_kwargs(train_config['weight_scheduler']))
    uncertainty_evaluator = train_config['uncertainty_evaluator']['type'](agent=agent_trainer.agent, device=device,
                                                                          **get_kwargs(train_config['uncertainty_evaluator']))


    # evaluating controllers
    suboptimal_controller_return = evaluator.evaluate_agent_on_start_states(controller, agent_type=Algorithm.CTRL)
    print(suboptimal_controller_return)

    # last inits
    global_step = 0
    episode_step = 0

    obs, _ = envs.reset(seed=seed)

    mixing_weights = []
    uncertainties = []

    no_fails = 0

    logger.init_wandb_logging(wandb_prj_name=wandb_prj_name,
                              config={
                                  "config": config,
                                  "hyperparams": agent_trainer.get_hyperparams_dict(),
                                  "timestamp": timestamp
                              })

    episode_actions_pi = []
    episode_actions_c = []

    # TRAINING LOOP
    for global_step in range(number_steps):

        weight = weight_scheduler.get_weight()

        mixing_weights.append(weight)
        if augmented_env:
            obs = inject_weight_into_state(obs, weight)
        state = torch.Tensor(obs).to(device)

        with torch.no_grad():
            actions_pi = agent_trainer.get_exploration_action(state, global_step)
            actions_c = controller.get_action(obs_man.get_ctrl_state(state))
            actions_c = actions_c.detach().cpu().numpy().clip(envs.single_action_space.low,
                                                              envs.single_action_space.high)

        mixing_component = weight

        actions = mixing_component * actions_pi + (1 - mixing_component) * actions_c

        # execute the game and log data
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        real_next_obs = inject_weight_into_state(next_obs, mixing_component) if augmented_env else next_obs.copy()

        # compute the uncertainty for the next timestep
        uncertainty = uncertainty_evaluator.get_uncertainty(state=obs_man.get_rl_state(state), action=actions_pi,
                                                            state_next=obs_man.get_rl_state(real_next_obs), reward=rewards[0])
        weight_scheduler.adapt_weight(uncertainty, global_step)

        uncertainties.append(uncertainty)
        episode_actions_pi.append(actions_pi)
        episode_actions_c.append(actions_c)

        wandb.log({
            "rollout/step_lambda": mixing_component,
            "rollout/uncertainty": uncertainty}, step=global_step)

        # record rewards for plotting purposes at the end of every episode
        if 'final_info' in infos:
            for info in infos['final_info']:

                if info['episode']['l'] < 600:  # fail criterion for Car Racing Environment
                    no_fails += 1

                wandb.log(
                    {"rollout/episodic_return": info["episode"]["r"],
                     "rollout/episodic_length": info["episode"]["l"],
                     "rollout/mixing_coeff_mean": np.mean(mixing_weights),
                     "rollout/mixing_coeff_distribution": wandb.Histogram(np_histogram=np.histogram(mixing_weights, bins=20, range=(0.,1.))),
                     "rollout/uncertainty_distribution_large": wandb.Histogram(np_histogram=np.histogram(uncertainties, bins=20)),
                     "rollout/uncertainty_distribution_narrow": wandb.Histogram(np_histogram=np.histogram(uncertainties, bins=20, range=(0, 1))),
                     "rollout/mean_uncertainty": np.mean(uncertainties),
                     "rollout/fails": no_fails,
                     "Charts/episode_step": episode_step}, step=global_step
                )

                episode_step += 1

                print(f"Step:{global_step}, Episode: {episode_step}, Reward: {info['episode']['r']}")

                # generate average performance statistics of current learned agent
                if eval_agent and episode_step % eval_frequency == 0 and last_evaluated_episode != episode_step and global_step >= \
                        agent_trainer.get_learning_starts()[0]:

                    # save model
                    logger.save_model(mixed_agent, f'agent_{global_step}')

                    eval_weights = np.unique(np.linspace(weight_scheduler.lambda_min, weight_scheduler.lambda_max, num=3))
                    eval_weights = np.round(eval_weights, decimals=2)

                    tracked_returns = []
                    for eval_weight in eval_weights:
                        tracked_returns.append(
                            evaluator.evaluate_agent_on_start_states(mixed_agent, agent_type=Algorithm.MIXED, weight=eval_weight))

                    adapted_return = (
                        evaluator.evaluate_weight_adapted_agent_on_start_states(mixed_agent, weight_scheduler,
                                                                                uncertainty_evaluator))

                    if global_step >= agent_trainer.get_learning_starts()[1] or last_evaluated_episode is None:
                        pass

                    wandb.log({
                        "rollout/evaluation_return_agent_controller": suboptimal_controller_return,
                    }, step=global_step)

                    wandb.log({'evaluation/evaluation_return_adapted': adapted_return}, step=global_step)

                    for j, eval_weight in enumerate(eval_weights):

                        wandb.log({
                            f"evaluation/evaluation_return_{eval_weight}": tracked_returns[j]
                        }, step=global_step)

                    last_evaluated_episode = episode_step

                    print("Adapted Performance: ", adapted_return, eval_max_return)
                    if adapted_return > eval_max_return:
                        eval_max_return = adapted_return
                        logger.save_model(mixed_agent)
                        if capture_video:
                            frames = evaluator.collect_video_frames(mixed_agent, agent_type=Algorithm.MIXED, weight=eval_weights[-1],
                                                                    random_start_state=False)
                            video_file = plotter.create_video_from_frames(frames, episode_step, fps=30)
                            wandb.log({'video': wandb.Video(video_file, fps=4, format='gif')}, step=global_step)

                weight_scheduler.episode_weight_reset()

                mixing_weights = []
                uncertainties = []
                episode_actions_c = []
                episode_actions_pi = []

        # handle `final observation`
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = inject_weight_into_state(infos['final_observation'][idx],
                                                              mixing_component) if augmented_env else \
                infos['final_observation'][idx]

        # save data to replay buffer
        agent_trainer.add_to_replay_buffer(obs, real_next_obs, actions_pi, rewards, terminations, infos)

        obs = next_obs

        # execute agent trainer train method for gradient descends
        agent_trainer.train_and_log(global_step, episode_step)

        if global_step % model_save_frequency == 0:
            logger.save_model(mixed_agent, f'agent_{global_step}')

    envs.close()

    if wandb.run is not None:
        wandb.finish(quiet=True)
        wandb.init(mode="disabled")

    logger.save_model(mixed_agent, model_name='final_agent')