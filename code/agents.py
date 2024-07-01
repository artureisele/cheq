import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn.init as init
import control as ct
import gymnasium as gym

from custom_envs.observation_manager import ObservationManager


class QNetwork(nn.Module):
    '''
    Q-network
    '''
    def __init__(self, env, obs_man: ObservationManager, hidden_layer_size=64):
        super().__init__()
        self.fc1 = nn.Linear(np.array(obs_man.obs_shape_rl).prod() + np.prod(env.single_action_space.shape), hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fc4 = nn.Linear(hidden_layer_size, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc4(x)
        return x


class SACActor(nn.Module):
    """Actor Network for SAC"""
    def __init__(self, env, obs_man: ObservationManager, hidden_layer_size=16, attenuate_actor=False, action_scale=1):
        super().__init__()
        self.LOG_STD_MIN = -5
        self.LOG_STD_MAX = 2
        self.fc1 = nn.Linear(np.array(obs_man.obs_shape_rl).prod(), hidden_layer_size)

        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fc_mean = nn.Linear(hidden_layer_size, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(hidden_layer_size, np.prod(env.single_action_space.shape))

        # because Residual Learning attenuating weights of final layer
        if attenuate_actor:
            with torch.no_grad():
                self.fc_mean.weight *= 1e-3
                self.fc_mean.bias *= 1e-3

        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((action_scale * env.action_space.high - action_scale * env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((action_scale * env.action_space.high + action_scale * env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class EnsembleSACAgent(nn.Module):

    def __init__(self, env, obs_man: ObservationManager, ensemble_size=2, use_rpf=False, rpf_scale=1, hidden_layer_size_q=64, hidden_layer_size_actor=16, attenuate_actor=False, action_scale=1):
        super().__init__()

        self.ensemble_size = ensemble_size
        self.use_rpf = use_rpf
        self.rpf_scale = rpf_scale

        self.ensemble = nn.ModuleList([QNetwork(env, obs_man, hidden_layer_size_q) for _ in range(self.ensemble_size)])
        self.ensemble_target = nn.ModuleList([QNetwork(env, obs_man, hidden_layer_size_q) for _ in range(self.ensemble_size)])

        if self.use_rpf:
            self.rpf_ensemble = nn.ModuleList([QNetwork(env, obs_man, hidden_layer_size_q) for _ in range(self.ensemble_size)])

        self.actor_net = SACActor(env, obs_man, hidden_layer_size_actor, attenuate_actor, action_scale)

        for j in range(self.ensemble_size):
            self.ensemble_target[j].load_state_dict(self.ensemble[j].state_dict())

    def get_ensemble_std(self, state, action):
        q_values = []

        for j in range(self.ensemble_size):
            q_values.append(self.get_ensemble_q_value(state, action, j).unsqueeze(-1))

        q_values = torch.cat(q_values, dim=-1)
        mean = torch.mean(q_values, dim=-1)
        std = torch.std(q_values, dim=-1)

        return mean, std

    def get_ensemble_q_value(self, x, a, j):
        if self.use_rpf:
            return self.ensemble[j](x, a) + self.rpf_scale * self.rpf_ensemble[j](x, a)
        else:
            return self.ensemble[j](x, a)

    def get_q1_value(self, x, a):
        return self.get_ensemble_q_value(x, a, 0)

    def get_q2_value(self, x, a):
        return self.get_ensemble_q_value(x, a, 1)

    def get_ensemble_q_target(self, x, a, j):
        if self.use_rpf:
            return self.ensemble_target[j](x, a) + self.rpf_scale * self.rpf_ensemble[j](x, a)
        else:
            return self.ensemble_target[j](x, a)

    def get_target_q1_value(self, x, a):
        return self.get_ensemble_q_target(x, a, 0)

    def get_target_q2_value(self, x, a):
        return self.get_ensemble_q_target(x, a, 1)

    def get_action(self, x, greedy=False):
        mean, log_std = self.actor_net(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        if not greedy:
            x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        else:
            x_t = mean  # greedy action

        y_t = torch.tanh(x_t)
        action = y_t * self.actor_net.action_scale + self.actor_net.action_bias

        return action

    def get_action_and_logprob(self, x):
        mean, log_std = self.actor_net(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.actor_net.action_scale + self.actor_net.action_bias
        log_prob = normal.log_prob(x_t)

        # Enforcing Action Bound
        log_prob -= torch.log(self.actor_net.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.actor_net.action_scale + self.actor_net.action_bias
        return action, log_prob, mean

    def update_target_networks(self, tau):
        for j in range(self.ensemble_size):
           for param, target_param in zip(self.ensemble[j].parameters(), self.ensemble_target[j].parameters()):
               target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


class MixedAgent(nn.Module):
    def __init__(self, rl_agent, controller, obs_man: ObservationManager):
        super(MixedAgent, self).__init__()
        self.controller = controller
        self.rl_agent = rl_agent
        self.obs_man = obs_man

    def get_action(self, state, greedy=True):
        # should do the mixing later on
        # retrieve weight from state
        if len(state.shape) == 1:
            weight = state[-1]
        else:
            # batch of states
            weight = state[:, -1]

        return weight * self.rl_agent.get_action(self.obs_man.get_rl_state(state), greedy) + (1-weight) * self.controller.get_action(self.obs_man.get_ctrl_state(state))

    def get_rl_action(self, state, greedy=True):
        return self.rl_agent.get_action(self.obs_man.get_rl_state(state), greedy)

    def get_control_action(self, state):
        return self.controller.get_action(self.obs_man.get_ctrl_state(state))
