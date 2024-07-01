""" 
Author: Krishan Rana, SpinningUp, Ramil Sabirov
Project: Bayesian Controller Fusion

"""

import numpy as np
import numpy.random
import torch
from torch.optim import Adam
from torch.distributions.normal import Normal
import time

from baselines.bcf_main import sac_core as core
from copy import deepcopy
import itertools
import math
import wandb
import collections
import sys, os
import random

from utils.tools import inject_weight_into_state


#---------------------------------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------------- SAC -------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------------------#

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.mu_prior_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.mu_prior2_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done, mu_prior, next_mu_prior):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.mu_prior_buf[self.ptr] = mu_prior
        self.mu_prior2_buf[self.ptr] = next_mu_prior
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs],
                     mu_prior=self.mu_prior_buf[idxs],
                     mu_prior2=self.mu_prior2_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32).to(agents[0].device) for k,v in batch.items()}


class SAC():
    def __init__ (self, env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
                  steps_per_epoch=10000, epochs=100, replay_size=int(1e6), gamma=0.99,
                  polyak=0.995, lr=1e-3, alpha=0.2, beta=0.3, batch_size=100, start_steps=5000,
                  update_after=256, update_every=50, num_test_episodes=10, max_ep_len=1000,
                  logger_kwargs=dict(), save_freq=1, use_kl_loss=False, epsilon=1e-5, target_KL_div=0, target_entropy=0.3,
                  lambda_context=False):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.polyak = polyak
        self.num_test_episodes = num_test_episodes
        self.start_steps = start_steps
        self.batch_size = batch_size
        self.max_ep_len = max_ep_len
        self.update_every = update_every
        self.update_after = update_after
        self.steps_per_epoch = steps_per_epoch
        self.use_kl_loss = use_kl_loss
        self.target_KL_div = target_KL_div
        self.target_entropy = target_entropy
        self.a_lr = 3e-4

        self.env, self.test_env = env_fn(), env_fn()
        obs_dim = self.env.observation_space.shape
        act_dim = self.env.action_space.shape[0]

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        act_limit = self.env.action_space.high[0]

        # Create actor-critic module and target networks
        self.ac = actor_critic(self.env.observation_space, self.env.action_space, **ac_kwargs,
                               lambda_res_formulation=lambda_context).to(self.device)
        self.ac_targ = deepcopy(self.ac).to(self.device)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=lr)
        self.q_optimizer = Adam(self.q_params, lr=lr)

        # Set up automatic KL div temperature tuning for alpha 
        self.alpha = torch.tensor([[10.0]], requires_grad=True, device=self.device)
        self.alpha_optim = Adam([self.alpha], lr=self.a_lr)

        # Set up automatic entropy temperature tuning for beta
        self.log_beta = torch.tensor([[-0.01]], requires_grad=True, device=self.device)
        self.beta = self.log_beta.exp()
        self.beta_optimizer = Adam([self.log_beta], lr=self.a_lr)


    def compute_loss_q(self, data):
    # Set up function for computing SAC Q-losses    
        o, a, r, o2, d, mu_prior2 = data['obs'], data['act'], data['rew'], data['obs2'], data['done'], data['mu_prior2']

        q1 = self.ac.q1(o,a)
        q2 = self.ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2, mu_policy2, sigma_policy2 = self.ac.pi(o2)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            KL_loss = compute_kld_univariate_gaussians(mu_prior2, torch.tensor(sigma_prior).to(self.device), mu_policy2, sigma_policy2).sum(axis=-1)
            
            if self.use_kl_loss:
                # KL minimisation regularisation
                backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * KL_loss)
            else:
                # Maximum entropy backup
                backup = r + self.gamma * (1 - d) * (q_pi_targ - self.beta * logp_a2)
                
        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        return loss_q, q_pi_targ

    
    def compute_loss_pi(self, data):
    # Set up function for computing SAC pi loss
        o, mu_prior = data['obs'], data['mu_prior']
        pi, logp_pi, mu_policy, sigma_policy = self.ac.pi(o)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)
        KL_loss = compute_kld_univariate_gaussians(mu_prior, torch.tensor(sigma_prior).to(self.device), mu_policy, sigma_policy).sum(axis=-1)

        if self.use_kl_loss:
            # Entropy-regularized policy loss
            loss_pi = (self.alpha * KL_loss - q_pi).mean()
        else:
            loss_pi = (self.beta * logp_pi - q_pi).mean()
            
        return loss_pi, logp_pi, KL_loss


    def update(self, data, pi_updates):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, q_pi_targ = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        for _ in range(pi_updates):

            # Next run one gradient descent step for pi.
            self.pi_optimizer.zero_grad()
            loss_pi, logp_pi, KL_div = self.compute_loss_pi(data)
            loss_pi.backward()
            self.pi_optimizer.step()

            # KL temperature update
            alpha_loss = self.alpha * (self.target_KL_div - KL_div).detach().mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            # Entropy temperature update
            self.beta_optimizer.zero_grad()
            beta_loss = (-self.log_beta * (self.target_entropy + logp_pi).detach()).mean()
            beta_loss.backward()
            self.beta_optimizer.step()
            self.beta = self.log_beta.exp()

        # Unfreeze Q-networks so you can optimize it at next step.
        for p in self.q_params:
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        metrics_dict = {'loss_q': loss_q.item(),
                     'q_pi_targ_max': q_pi_targ.max(),
                     'q_pi_targ_min': q_pi_targ.min(),
                     'q_pi_targ_mean': q_pi_targ.mean(),
                     'target_KL_div': self.target_KL_div}

        if pi_updates > 0:
            metrics_dict['loss_pi'] = loss_pi.item()
            metrics_dict['entropy'] = logp_pi.mean().item()
            metrics_dict['KL_div'] = KL_div.mean().item()

        return metrics_dict


    def get_action(self, o, deterministic=False):
        act, mu, std =  self.ac.act(torch.as_tensor(o, dtype=torch.float32).to(self.device), deterministic)
        return act, mu, std


#---------------------------------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------------- Helpers ---------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------------------#

def test_agent(use_single_agent=True):
      
    global total_steps
    global test_steps
    ep_ret = 0.0
    ep_len = 0.0
    step_counter = 0
    
    for j in range(agents[0].num_test_episodes):
        o, d = agents[0].test_env.reset(), False
        while not (d or (step_counter == agents[0].max_ep_len)):

            if METHOD == "BCF":
                ensemble_actions = [get_distr(o, p.ac) for p in agents]
                mu, sigma = fuse_ensembles_stochastic(ensemble_actions)
                dist = Normal(torch.tensor(mu.detach()), torch.tensor(sigma.detach()))
                action = torch.tanh(dist.sample()).numpy()
                wandb.log({'evaluation/ensemble_std_1': sigma[0],
                'evaluation/ensemble_std_2': sigma[1],
                'evaluation/ensemble_mu_1': mu[0],
                'evaluation/ensemble_mu_2': mu[1]}, total_steps)
            else:
                raise NotImplementedError()
                
            wandb.log({'evaluation/test_steps': test_steps}, total_steps)
            o, r, d, _ = agents[0].test_env.step(action)
            ep_ret += r
            ep_len += 1
            test_steps += 1
            step_counter += 1
        step_counter = 0
    avg_ret = ep_ret/agents[0].num_test_episodes
    avg_len = ep_len/agents[0].num_test_episodes

    agents[0].test_env.reset()
    
    return {'evaluation/rewards_eval': avg_ret,
            'evaluation/len_eval': avg_len}

def test_agent_2(episode_start_weight, adaptive):
    global total_steps
    global test_steps
    ep_ret = 0.0
    ep_len = 0.0
    step_counter = 0

    for j in range(agents[0].num_test_episodes):
        o, d = agents[0].test_env.reset(), False
        o_old = o
        while not (d or (step_counter == agents[0].max_ep_len)):

            if METHOD == "BCF":
                with torch.no_grad():
                    ensemble_actions = [get_distr(o, p.ac) for p in agents]
                mu, sigma = fuse_ensembles_stochastic(ensemble_actions)

                if not adaptive:
                    action = torch.tanh(mu).numpy()  # greedy action
                else:
                    mu_prior = prior.compute_action()
                    mu_mix, sigma_mix = fuse_controllers(prior_mu=mu_prior, prior_sigma=sigma_prior, policy_sigma=sigma, policy_mu=mu)
                    action = torch.tanh(mu_mix).numpy()

                wandb.log({'evaluation/ensemble_std_1': sigma[0],
                           'evaluation/ensemble_std_2': sigma[1],
                           'evaluation/ensemble_mu_1': mu[0],
                           'evaluation/ensemble_mu_2': mu[1]}, total_steps)
            else:
                raise NotImplementedError()

            o, r, d, _ = agents[0].test_env.step(action)
            ep_ret += r
            ep_len += 1
            # total_steps += 1
            test_steps += 1
            step_counter += 1
        step_counter = 0
    avg_ret = ep_ret / agents[0].num_test_episodes
    avg_len = ep_len / agents[0].num_test_episodes

    agents[0].test_env.reset()

    return avg_ret, avg_len


def test_agent_adapted(episode_start_weight):

    assert lambda_context and METHOD == 'BCF', 'Adapted test method is only implemented for BCF with lambda context'

    global total_steps
    global test_steps
    ep_ret = 0.0
    ep_len = 0.0
    step_counter = 0

    for j in range(agents[0].num_test_episodes):
        o, d = agents[0].test_env.reset(), False
        weight = episode_start_weight
        while not (d or (step_counter == agents[0].max_ep_len)):

            # ensemble_actions = ray.get([get_distr.remote(o,p.ac) for p in agents])
            o = inject_weight_into_state(o, weight)

            if METHOD == 'BCF':
                ensemble_actions = [get_distr(o, p.ac) for p in agents]
                mu, sigma = fuse_ensembles_stochastic(ensemble_actions)
                mu_prior = prior.compute_action()
                dist = Normal(torch.tensor(mu.detach()), torch.tensor(sigma.detach()))
                # policy_action = torch.tanh(dist.sample()).numpy()
                policy_action = torch.tanh(mu).numpy()

                action = weight * policy_action + (1-weight) * mu_prior

                weight = agg_function(compute_weight(sigma.detach().numpy(), sigma_prior))

                wandb.log({'evaluation/ensemble_std_1': sigma[0],
                           'evaluation/ensemble_std_2': sigma[1],
                           'evaluation/ensemble_mu_1': mu[0],
                           'evaluation/ensemble_mu_2': mu[1]}, total_steps)


            wandb.log({'evaluation/test_steps': test_steps}, total_steps)
            o_old = o
            o, r, d, _ = agents[0].test_env.step(action)
            ep_ret += r
            ep_len += 1
            # total_steps += 1
            test_steps += 1
            step_counter += 1

        step_counter = 0
    avg_ret = ep_ret / agents[0].num_test_episodes
    avg_len = ep_len / agents[0].num_test_episodes

    agents[0].test_env.reset()

    return {'evaluation/rewards_eval': avg_ret,
            'evaluation/len_eval': avg_len}

def evaluate_prior_agent():

    ep_ret = 0.0
    ep_len = 0.0

    step_counter = 0
    for  i in range(agents[0].num_test_episodes):
        o, d = agents[0].test_env.reset(), False
        while not (d or (step_counter == agents[0].max_ep_len)):
            act = prior.compute_action()
            o, r, d, p = env.step(act)
            ep_ret += r
            ep_len += 1
            step_counter += 1
        step_counter = 0
    avg_ret = ep_ret/agents[0].num_test_episodes
    avg_len = ep_len/agents[0].num_test_episodes

    return {'rewards_eval': avg_ret,
            'len_eval': avg_len}
                    
def fuse_controllers(prior_mu, prior_sigma, policy_mu, policy_sigma):
    # The policy mu and sigma are from the stochastic SAC output
    # The sigma from prior is fixed
    mu = (np.power(policy_sigma, 2) * prior_mu + np.power(prior_sigma,2) * policy_mu)/(np.power(policy_sigma,2) + np.power(prior_sigma,2))
    sigma = np.sqrt((np.power(prior_sigma,2) * np.power(policy_sigma,2))/(np.power(policy_sigma, 2) + np.power(prior_sigma,2)))
    return mu, sigma

def inverse_sigmoid_gating_function(k, C, x):
    val = 1 / (1 + math.exp(k*(x - C))) 
    return val

def compute_kld_univariate_gaussians(mu_prior, sigma_prior, mu_policy, sigma_policy):
    # Computes the analytical KL divergence between two univariate gaussians
    kl = torch.log(sigma_policy/sigma_prior) + ((sigma_prior**2 + (mu_prior - mu_policy)**2)/(2*sigma_policy**2)) - 1/2
    return kl

def get_distr(state, agent):
    state = torch.tensor(state).unsqueeze(0).to(device)
    act, mu, std = agent.act(state, False)
    return [mu.detach().squeeze(0).cpu().numpy(), std.detach().squeeze(0).cpu().numpy()]

def fuse_ensembles_deterministic(ensemble_actions):
    actions = torch.tensor([ensemble_actions[i][0] for i in range (NUM_AGENTS)])
    mu = torch.mean(actions, dim=0)
    var = torch.var(actions, dim=0)
    sigma = np.sqrt(var)
    return mu, sigma

def fuse_ensembles_stochastic(ensemble_actions):
    mu = (np.sum(np.array([ensemble_actions[i][0] for i in range(NUM_AGENTS)]), axis=0))/NUM_AGENTS
    var = (np.sum(np.array([(ensemble_actions[i][1]**2 + ensemble_actions[i][0]**2)-mu**2 for i in range(NUM_AGENTS)]), axis=0))/NUM_AGENTS
    sigma = np.sqrt(var)
    return torch.from_numpy(mu), torch.from_numpy(sigma)

def write_logs(logs, t):
    wandb.log(logs, t)

def save_ensemble(timestep, best=False):
    if not best:
        checkpoint_folder_path = save_dir + f"{timestep}/"
        os.makedirs(checkpoint_folder_path, exist_ok=True)
    for idx, agnt in enumerate(agents):
        # torch.save(agnt.ac.pi, save_dir + wandb.run.name + "_" + str(idx) + postfix + ".pth")
        if not best:
            torch.save(agnt.ac, checkpoint_folder_path + 'agent_' + str(idx) + ".pth")
        else:
            torch.save(agnt.ac, save_dir + 'best_agent_' + str(idx) + ".pth")

def arg(tag, default):
    HYPERS[tag] = type(default)((sys.argv[sys.argv.index(tag)+1])) if tag in sys.argv else default
    return HYPERS[tag]


def compute_weight(RL_sigma, prior_sigma):
    return sigma_prior**2 / (RL_sigma**2 + prior_sigma**2)


def tanh_maximum_likelihood_gaussian(ensemble_actions):
    actions = np.array([ensemble_actions[i][0] for i in range(NUM_AGENTS)])
    actions = np.tanh(actions)
    mu = np.mean(actions, axis=0)
    sigma = np.std(actions, axis=0)

    return mu, sigma

def bcf_weight_to_sigma(weight):
    return np.sqrt((1-weight)*sigma_prior**2/weight)


#---------------------------------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------------- Run -------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------------------#

def run(agents, env):

    episode_start_weight = 0.2
    episode_start_weight_warmup = WARMUP_WEIGHT

    # last_weight = episode_start_weight
    next_weight = episode_start_weight
    cur_weight = episode_start_weight

    # Prepare for interaction with environment
    global total_steps
    o, ep_ret, ep_len = env.reset(), 0, 0
    agent = random.choice(agents)
    r = 0

    if lambda_context:
        o = inject_weight_into_state(o, cur_weight)

    o_old = o

    episode_weights = [[] for _ in range(len(env.action_space.low))]
    episode_weights_agg = []

    td_errors = []
    uncertainties = [[] for _ in range(len(env.action_space.low))]

    curr_best_eval_return = -math.inf

    no_fails = 0
    episode_step = 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(NUM_STEPS):

        mu_prior = prior.compute_action()
        
        if t > agent.update_after:
            policy_action, mu_policy, std_policy = agent.get_action(o)

            if METHOD == "BCF":
                ensemble_actions = [get_distr(o, p.ac) for p in agents]
                mu_ensemble, sigma_ensemble = fuse_ensembles_stochastic(ensemble_actions)

                if not lambda_context:
                    if t <= agent.start_steps:  # overwrite the sigma such that the policy has fixed low weight
                        sigma_ensemble = torch.tensor([bcf_weight_to_sigma(episode_start_weight)]*len(env.action_space.low))

                    mu_mcf, std_mcf = fuse_controllers(mu_prior, sigma_prior, mu_ensemble.cpu().numpy(),
                                                       sigma_ensemble.cpu().numpy())
                    w = SIGMA_PRIOR ** 2 / (sigma_ensemble ** 2 + SIGMA_PRIOR ** 2)
                    agg_weight = agg_function(w.numpy()) if isinstance(w, torch.Tensor) else agg_function(w)
                    episode_weights_agg.append(agg_weight)

                    dist_hybrid = Normal(torch.tensor(mu_mcf).double().detach(),
                                         torch.tensor(std_mcf).double().detach())
                    a = dist_hybrid.sample()
                    a = torch.tanh(a).numpy()
                else:
                    dist_ensemble = Normal(torch.tensor(mu_ensemble).double().detach(),
                                            torch.tensor(sigma_ensemble).double().detach())
                    policy_action = dist_ensemble.sample()
                    policy_action = torch.tanh(policy_action).numpy()
                    a = policy_action * cur_weight + mu_prior * (1 - cur_weight)

                    episode_weights_agg.append(cur_weight)
                    w = SIGMA_PRIOR ** 2 / (sigma_ensemble ** 2 + SIGMA_PRIOR ** 2)

                    if t <= agent.start_steps:
                        next_weight = np.random.uniform(episode_start_weight, episode_start_weight_warmup)
                    else:
                        if isinstance(w, torch.Tensor):
                            next_weight = agg_function(w.numpy())
                        else:
                            next_weight = agg_function(w)
                # computing the weight
                for i in range(len(sigma_ensemble)):
                    episode_weights[i].append(w[i])
                    uncertainties[i].append(sigma_ensemble[i])
            else:
                raise NotImplementedError()
                
        else:
            a = env.action_space.sample()
            if METHOD == 'BCF':
                a = cur_weight * a + (1-cur_weight) * mu_prior
                episode_weights_agg.append(cur_weight)

                if lambda_context:
                    policy_action = a
                    # sample the next weight
                    next_weight = np.random.uniform(episode_start_weight, episode_start_weight_warmup)
            else:
                raise NotImplementedError()
        
        o2, r, d, _ = env.step(a)
        mu_prior2 = prior.compute_action()
        ep_ret += r
        ep_len += 1
        total_steps += 1
    
        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == agent.max_ep_len else d

        # Store experience to replay buffer
        if METHOD == "residual":
            replay_buffer.store(o, policy_action, r, o2, d, mu_prior, mu_prior2)
        else:
            if lambda_context:
                o2_i = inject_weight_into_state(o2, cur_weight)
                replay_buffer.store(o, policy_action, r, o2_i, d, mu_prior, mu_prior2)
            else:
                replay_buffer.store(o, a, r, o2, d, mu_prior, mu_prior2)

        # setting up values for the next iteration
        o = o2
        if lambda_context:
            o = inject_weight_into_state(o, next_weight)
        cur_weight = next_weight

        # Update handling
        if t >= agent.update_after:
            for ag in agents:
                batch = replay_buffer.sample_batch(agent.batch_size)
                policy_updates = POLICY_FREQUENCY if t % POLICY_FREQUENCY == 0 else 0
                metrics = ag.update(batch, policy_updates)
            if t % 100 == 0:
                write_logs(metrics, total_steps)

        if t % MODEL_SAVE_FREQ == 0:
            save_ensemble(t)

        # End of trajectory handling
        if d or (ep_len == agent.max_ep_len):

            if d and ep_len < agent.max_ep_len:
                no_fails += 1
            write_logs({'ep_rewards': ep_ret}, total_steps)
            write_logs({'ep_length': ep_len}, total_steps)
            write_logs({'no_fails': no_fails}, total_steps)

            if METHOD == 'BCF':
                for i in range(len(episode_weights)):
                    write_logs({f'weighting/mean_weight_dim={i+1}': np.mean(episode_weights[i]),
                                f'weighting/distr_weight_dim={i+1}': wandb.Histogram(np_histogram=np.histogram(episode_weights[i], bins=20, range=(0.,1.))),
                                f'mean_uncertainty_dim={i+1}': np.mean(uncertainties[i]),
                                f'uncertainty_distribution_dim={i+1}': wandb.Histogram(np_histogram=np.histogram(uncertainties[i], bins=20)),
                                f'uncertainty_distribution_narrow_dim={i+1}': wandb.Histogram(np_histogram=np.histogram(uncertainties[i], bins=20, range=(0., 2)))}, total_steps)
                    episode_weights[i].clear()
                    uncertainties[i].clear()

                write_logs(
                    {f'weighting/mean_agg_weight': np.mean(episode_weights_agg),
                            f'weighting/distr_weight': wandb.Histogram(np_histogram=np.histogram(episode_weights_agg, bins=20, range=(0., 1.)))}, total_steps)
                episode_weights_agg.clear()
            else:
                raise NotImplementedError()

            if episode_step % EVALUATION_FREQ_EPISODES == 0:
                # save_ensemble(total_steps)
                if lambda_context:
                    metrics = test_agent_adapted(episode_start_weight)
                    compare_return = metrics['evaluation/rewards_eval']
                else:
                    avg_ret_adaptive, avg_len_adaptive = test_agent_2(episode_start_weight, adaptive=True)
                    avg_ret_pure, avg_len_pure = test_agent_2(episode_start_weight, adaptive=False)
                    metrics = {'evaluation/rewards_eval_ada': avg_ret_adaptive,
                                'evaluation/len_eval_ada': avg_len_adaptive,
                               'evaluation/rewards_eval': avg_ret_pure,
                               'evaluation/len_eval': avg_len_pure}
                    compare_return = avg_ret_adaptive
                write_logs(metrics, total_steps)

                if compare_return > curr_best_eval_return:
                    curr_best_eval_return = compare_return
                    save_ensemble(t, best=True)

            episode_step += 1
            o, ep_ret, ep_len, r = env.reset(), 0, 0, 0
            agent = random.choice(agents)

            if lambda_context:
                cur_weight = episode_start_weight
                o = inject_weight_into_state(o, cur_weight)

def update_hypers():
    HYPERS['TASK'] = TASK
    HYPERS['METHOD'] = METHOD
    HYPERS['REWARD'] = REWARD
    HYPERS['USE_KL'] = USE_KL
    HYPERS['ALPHA'] = ALPHA
    HYPERS['BETA'] = BETA
    HYPERS['EPSILON'] = EPSILON
    HYPERS['SEED'] = SEED
    HYPERS['NUM_AGENTS'] = NUM_AGENTS
    HYPERS['NUM_STEPS'] = NUM_STEPS
    HYPERS['TARGET_KL_DIV'] = TARGET_KL_DIV
    HYPERS['TARGET_ENTROPY'] = TARGET_ENTROPY
    HYPERS['SIGMA_PRIOR'] = SIGMA_PRIOR
    HYPERS['PRIOR_CONTROLLER'] = PRIOR_CONTROLLER
    HYPERS['ENV'] = ENV

    HYPERS['GAMMA'] = GAMMA
    HYPERS['POLYAK'] = POLYAK
    HYPERS['LEARNING_RATE'] = LEARNING_RATE
    HYPERS['BATCH_SIZE'] = BATCH_SIZE
    HYPERS['START_STEPS'] = START_STEPS
    HYPERS['TRAINING_START'] = TRAINING_START
    HYPERS['NUM_TEST_EPISODES'] = NUM_TEST_EPISODES
    HYPERS['STEPS_PER_EPOCH'] = STEPS_PER_EPOCH
    HYPERS['EVAL_FREQ_EPISODES'] = EVALUATION_FREQ_EPISODES
    HYPERS['MODEL_SAVE_FREQ'] = MODEL_SAVE_FREQ
    HYPERS['POLICY_FREQUENCY'] = POLICY_FREQUENCY

    HYPERS['WEIGHT_AGGREGATE'] = WEIGHT_AGGREGATE
    HYPERS['WARMUP_WEIGHT'] = WARMUP_WEIGHT

    HYPERS['RES_ADA_FORMULATION'] = lambda_context

    HYPERS['TIME_STAMP'] = TIME_STAMP


def init_run(name, exp_type):
    global replay_buffer
    global total_steps
    global test_steps
    global agents
    global agg_function
    global save_dir

    assert not lambda_context or METHOD == 'BCF', \
        'lambda_context is not yet implemented for other methods apart from BCF'

    torch.set_num_threads(torch.get_num_threads())
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    env.seed(SEED)
    obs_dim = (env.observation_space.shape[0] + lambda_context,)
    act_dim = env.action_space.shape[0]
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=int(1e6))
    total_steps = 0
    test_steps = 0

    if WEIGHT_AGGREGATE == 'mean':
        agg_function = lambda x: np.mean(x)
    elif WEIGHT_AGGREGATE == 'min':
        agg_function = lambda x: np.min(x)
    else:
        raise NotImplementedError(f'Unknown weight aggregate {WEIGHT_AGGREGATE}')

    # Initialise an ensemble of agents
    agents = [SAC(lambda: env,
                  actor_critic=core.MLPActorCritic,
                  gamma=GAMMA,
                  polyak=POLYAK,
                  lr=LEARNING_RATE,
                  batch_size=BATCH_SIZE,
                  start_steps=START_STEPS,
                  update_after=TRAINING_START,
                  num_test_episodes=NUM_TEST_EPISODES,
                  max_ep_len=max_ep_steps,
                  alpha=ALPHA,
                  beta=BETA,
                  epsilon=EPSILON,
                  use_kl_loss=USE_KL,
                  target_entropy=TARGET_ENTROPY,
                  target_KL_div=TARGET_KL_DIV,
                  steps_per_epoch=STEPS_PER_EPOCH,
                  lambda_context=lambda_context) for _ in range(NUM_AGENTS)]


    # FOLDER handling
    run_name = name + '_' + TASK + '_' + METHOD + '_SEED=' + str(SEED) + "_" + TIME_STAMP
    save_dir = f"logs/{exp_type}/{run_name}/models/"
    os.makedirs(save_dir)

    # WANDB handling
    wandb.login()
    wandb.init(
        dir=f'logs/{exp_type}/{run_name}',
        project=PROJECT_LOG,
        name=run_name
               )

    wandb.config.update(HYPERS)


lambda_context = False

# BCF baseline benchmark
HYPERS = collections.OrderedDict()
PROJECT_LOG = "GPO_2021"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TASK = 'racing'
METHOD = "BCF"
REWARD = "sparse"
USE_KL = int(False)
ALPHA = 0.5
BETA = 0.1
EPSILON = 2e-4
SEED = 0
NUM_AGENTS = 5  # Number of agents in ensemble
NUM_STEPS = int(1e6)
TARGET_KL_DIV = 10e-3
TARGET_ENTROPY = -3  # -7
SIGMA_PRIOR = 2.0
PRIOR_CONTROLLER = "APF"

TIME_STAMP = time.asctime().replace(' ', '_').replace(':', '-')

ENV = "CarRacingEnv" if TASK == 'racing' else 'PointGoalNavigation'
NUM_AGENTS = NUM_AGENTS if METHOD == "BCF" else 1

# SAC Parameters
GAMMA = 0.99
POLYAK = 0.995
LEARNING_RATE = 3e-4 #1e-3
BATCH_SIZE = 256 #100
START_STEPS = 5000 #100
STEPS_PER_EPOCH = 10000
TRAINING_START = 256
NUM_TEST_EPISODES = 1 if TASK == 'racing' else 10

EVALUATION_FREQ_EPISODES = 1
MODEL_SAVE_FREQ = 50000

POLICY_FREQUENCY = 20
WEIGHT_AGGREGATE = 'mean'

WARMUP_WEIGHT = 0.3

# will be set by main
env = None
sigma_prior = None
max_ep_steps = None
prior = None

update_hypers()

# these global variables will need to be set by init_run
replay_buffer = None
total_steps = None
test_steps = None
agents = None
save_dir = None
agg_function = None
