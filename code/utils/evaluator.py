import copy
from abc import abstractmethod, ABC

import torch
import numpy as np

from custom_envs.CarEnv.Track.Generator import make_full_environment
from custom_envs.CarEnv.Configs import RACING_FAST
from custom_envs.observation_manager import ObservationManager, Algorithm
from utils.tools import inject_weight_into_state, WeightScheduler, UncertaintyEvaluator

class Evaluator(ABC):

    def __init__(self,
                 env,
                 obs_man: ObservationManager,
                 device=torch.device('cpu'),
                 eval_count=10,
                 seed=42,
                 greedy_eval=True,
                 ):

        self.eval_count = eval_count
        self.seed = seed
        self.greedy_eval = greedy_eval
        self.obs_man = obs_man
        self.device = device

        # setting the eval_env
        self.single_eval_env = env
        self.single_eval_env.unwrapped.render_mode = 'rgb_array'

        # generate the test start states
        self.eval_start_states = self.init_eval_start_states()


    def _convert_state_for_agent(self, state, agent_type):
        if agent_type == Algorithm.RL:
            return self.obs_man.get_rl_state(state)
        elif agent_type == Algorithm.CTRL:
            return self.obs_man.get_ctrl_state(state)
        else:
            return state

    @abstractmethod
    def init_eval_start_states(self):
        pass

    @abstractmethod
    def init_start_state(self, state_no):
        pass

    def evaluate_uncertainty_on_start_states(self, agent_trainer, weight, cv=False):

        states = [inject_weight_into_state(self.eval_start_states[j], weight) for j in range(self.eval_count)]
        states = np.array(states)
        states = torch.tensor(states, device=self.device)
        states = self.obs_man.get_rl_state(states)

        with torch.no_grad():
            actions = agent_trainer.agent.get_action(states)
            mean, std = agent_trainer.get_q_net_std(states, actions)

        if cv:
            return torch.abs(std/mean).mean().item()
        else:
            return std.mean().item()

    def evaluate_static_on_start_states(self, agent_trainer, weight):
        return self.evaluate_static_on_states(agent_trainer, weight, self.eval_start_states)

    def evaluate_static_on_states(self, agent_trainer, weight, states):

        states = [inject_weight_into_state(state, weight) for state in states]
        states = np.array(states)
        states = torch.tensor(states, device=self.device)
        states = self.obs_man.get_rl_state(states)

        with torch.no_grad():
            actions = agent_trainer.agent.get_action(states)
            mean, std = agent_trainer.get_q_net_std(states, actions)

        return mean.mean().item(), std.mean().item()

    def evaluate_agent_on_start_states(self, agent, agent_type, weight=1):

        done = False
        # env = gym.make(env_id)
        total_reward = [0] * len(self.eval_start_states)
        for i in range(len(self.eval_start_states)):
            state = self.init_start_state(i)
            if self.obs_man.augmented_env:
                state = inject_weight_into_state(state, weight)
            state = torch.tensor(state, device=self.device)
            state = self._convert_state_for_agent(state, agent_type)
            while not done:
                with torch.no_grad():
                    action = agent.get_action(state, self.greedy_eval)
                action = action.squeeze().cpu().numpy()
                next_state, reward, terminated, truncated, _ = self.single_eval_env.step(action)
                done = terminated or truncated
                if self.obs_man.augmented_env:
                    next_state = inject_weight_into_state(next_state, weight)
                # state = torch.tensor(next_state, dtype=torch.float32)
                state = torch.tensor(next_state, device=self.device)
                state = self._convert_state_for_agent(state, agent_type)
                total_reward[i] += reward
            done = False

        return np.mean(total_reward)

    def evaluate_weight_adapted_agent_on_start_states(self, mixed_agent, weight_scheduler: WeightScheduler,
                                                      uncertainty_evaluator: UncertaintyEvaluator):

        done = False
        # env = gym.make(env_id)
        total_reward = [0] * len(self.eval_start_states)
        for i in range(len(self.eval_start_states)):
            state = self.init_start_state(i)
            eval_weight_scheduler = copy.deepcopy(weight_scheduler)
            eval_weight_scheduler.episode_weight_reset()
            weight = eval_weight_scheduler.get_weight()
            if self.obs_man.augmented_env:
                state = inject_weight_into_state(state, weight)
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            while not done:
                with torch.no_grad():
                    action_pi = mixed_agent.get_rl_action(state, greedy=self.greedy_eval)
                    action_c = mixed_agent.get_control_action(state)
                action = action_pi * weight + action_c * (1-weight)
                action = action.squeeze().cpu().numpy()
                next_state, reward, terminated, truncated, _ = self.single_eval_env.step(action)
                done = terminated or truncated

                # compute the next uncertainty
                unc_next_state = inject_weight_into_state(next_state, weight) if self.obs_man.augmented_env else next_state
                unc_next_state = unc_next_state[np.newaxis]
                uncertainty = uncertainty_evaluator.get_uncertainty(self.obs_man.get_rl_state(state),
                                                                    action_pi,
                                                                    self.obs_man.get_rl_state(unc_next_state),
                                                                    reward)
                eval_weight_scheduler.adapt_weight(uncertainty, 0, force=True)
                weight = eval_weight_scheduler.get_weight()
                if self.obs_man.augmented_env:
                    next_state = inject_weight_into_state(next_state, weight)
                state = torch.tensor(next_state, device=self.device)
                state = state.unsqueeze(0)
                total_reward[i] += reward
            done = False

        return np.mean(total_reward)

    def evaluate_mixed_agent_on_start_states_diff_weights(self, mixed_agent, action_weight, eval_weight):

        done = False
        # env = gym.make(env_id)
        total_reward = [0] * len(self.eval_start_states)
        for i in range(len(self.eval_start_states)):
            state = self.init_start_state(i)
            if self.obs_man.augmented_env:
                state = inject_weight_into_state(state, action_weight)
            state = torch.tensor(state, device=self.device)
            while not done:
                with torch.no_grad():
                    action_pi = mixed_agent.get_rl_action(state, self.greedy_eval)
                    action_c = mixed_agent.get_control_action(state)
                action = action_pi * eval_weight + action_c * (1-eval_weight)
                action = action.squeeze().cpu().numpy()
                next_state, reward, terminated, truncated, _ = self.single_eval_env.step(action)
                done = terminated or truncated
                if self.obs_man.augmented_env:
                    next_state = inject_weight_into_state(next_state, action_weight)
                state = torch.tensor(next_state, device=self.device)
                total_reward[i] += reward
            done = False

        return np.mean(total_reward)

    def evaluate_actions_over_episode(self, agents, agent_types, weight=0.5):

        return self.evaluate_actions_over_episode_from_state(agents, agent_types, 0, weight=weight)

    def evaluate_actions_over_episode_from_state(self, agents, agent_types, start_state_no, weight=0.5):
        actions = []
        for _ in agents:
            actions.append([])

        states = []

        done = False
        state = self.init_start_state(start_state_no)
        if self.obs_man.augmented_env:
            state = inject_weight_into_state(state, weight)
        states.append(state.copy())
        state = torch.tensor(state, device=self.device)
        while not done:
            with torch.no_grad():
                for idx, agent in enumerate(agents):
                    agent_state = self._convert_state_for_agent(state, agent_types[idx])
                    action = agent.get_action(agent_state, greedy=True)
                    action = action.squeeze().cpu().numpy()
                    actions[idx].append(action)
            next_state, reward, terminated, truncated, _ = self.single_eval_env.step(actions[0][-1])
            done = terminated or truncated
            if self.obs_man.augmented_env:
                next_state = inject_weight_into_state(next_state, weight)
            # state = torch.tensor(next_state, dtype=torch.float32)
            states.append(next_state.copy())
            state = torch.tensor(next_state, device=self.device)

        return actions, states

    def collect_video_frames(self, agent, agent_type, weight=0.5, random_start_state=True):

        frames = []

        done = False
        if random_start_state:
            state, _ = self.single_eval_env.reset()
        else:
            state = self.init_start_state(0)
        if self.obs_man.augmented_env:
            state = inject_weight_into_state(state, weight)
        state = torch.tensor(state, device=self.device)
        while not done:
            with torch.no_grad():
                state = self._convert_state_for_agent(state, agent_type)
                action = agent.get_action(state, greedy=True)
                action = action.squeeze().cpu().numpy()

            next_state, reward, terminated, truncated, _ = self.single_eval_env.step(action)
            done = terminated or truncated
            if self.obs_man.augmented_env:
                next_state = inject_weight_into_state(next_state, weight)
            state = torch.tensor(next_state, device=self.device)
            # state = state.unsqueeze(0)
            out = self.single_eval_env.render()
            frames.append(out)

        return frames


class CarRacingEvaluator(Evaluator):

    def init_eval_start_states(self):
        eval_start_states = []
        if self.single_eval_env.unwrapped.training_tracks is None:
            for i in range(self.eval_count):
                track = make_full_environment(width=RACING_FAST['problem']['track_width'],
                                              extends=(
                                                  RACING_FAST['problem']['extend'], RACING_FAST['problem']['extend']),
                                              cone_width=RACING_FAST['problem']['cone_width'],
                                              rng=self.single_eval_env.np_random)
                eval_start_states.append(track)
        else:
            # pick uniformly from the set of training tracks
            self.eval_count = min(self.eval_count, len(self.single_eval_env.unwrapped.training_tracks))
            indices = np.arange(len(self.single_eval_env.unwrapped.training_tracks))
            np.random.shuffle(indices)
            eval_start_states = [self.single_eval_env.unwrapped.training_tracks[i] for i in
                                      indices[:self.eval_count]]

            return eval_start_states

    def init_start_state(self, state_no):
        state, _ = self.single_eval_env.reset(options={'predefined_track': self.eval_start_states[state_no]})
        return state

    def evaluate_failure_rate(self, agent, agent_type, no_runs):

        fail_count = 0

        for i in range(no_runs):
            state, _ = self.single_eval_env.reset()
            state = torch.tensor(state, device=self.device)
            done = False
            while not done:
                with torch.no_grad():
                    state = self._convert_state_for_agent(state, agent_type)
                    action = agent.get_action(state, greedy=True)
                    action = action.squeeze().cpu().numpy()

                next_state, reward, terminated, truncated, info = self.single_eval_env.step(action)
                done = terminated or truncated
                if done and info['Done.Reason'] == 'LeftTrack':
                    fail_count += 1
                state = torch.tensor(next_state, device=self.device)
        return fail_count


class CartPoleEvaluator(Evaluator):

    def init_eval_start_states(self):
        eval_start_states = [self.single_eval_env.reset(seed=self.seed+i)[0] for i in range(self.eval_count)]
        return eval_start_states

    def init_start_state(self, state_no):
        state, _ = self.single_eval_env.reset(options={'start_state': np.copy(self.eval_start_states[state_no])})
        return state

    def evaluate_static_on_goal_states(self, agent_trainer, weight, perturbation, no_samples):
        # construct states perturbed with the given perturbation
        goal_state = np.array([0., 0., 0., 0.], dtype=np.float32)
        eval_states = np.random.uniform(low=goal_state - perturbation, high=goal_state + perturbation,
                                        size=(no_samples, 4)).astype(dtype=np.float32)
        return self.evaluate_static_on_states(agent_trainer, weight, eval_states)
