# train.py
# Script to train policies in Isaac Gym
#
# Copyright (c) 2018-2021, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.




import argparse
import copy
import inspect
import logging
import logging.config
import os
import re
import sys
import warnings
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from distutils.command.config import config
from enum import Enum
from gc import callbacks
from os.path import dirname, join, normpath, realpath, splitext
from pathlib import Path
from textwrap import dedent
from traceback import print_exc, print_exception
from types import FrameType, TracebackType
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, MutableSequence,
                    Optional, Sequence, Tuple, Union, cast)

import isaacgym
import yaml
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

from isaacgymenvs.tasks.franka_cabinet import FrankaCabinet
from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict


class A2CAgent:
    def __init__(self, base_name, config, env):
        self.vec_env = env

        self.experiment_name = config.get('full_experiment_name', None)
        self.config = config
        self.multi_gpu = False
        self.rank = 0
        self.rank_size = 1
        self.network_path = config.get('network_path', "./nn/")
        self.log_path = config.get('log_path', "runs/")
        self.num_actors = config['num_actors']


        self.observation_space = env.observation_space
        action_space = env.action_space
        
        self.value_size = 1
        self.num_agents = 1

        self.ppo_device = config.get('device', 'cuda:0')
        self.weight_decay = config.get('weight_decay', 0.0)
        self.use_action_masks = config.get('use_action_masks', False)
        self.is_train = config.get('is_train', True)
        self.central_value_config = self.config.get('central_value_config', None)
        self.has_central_value = self.central_value_config is not None
        self.truncate_grads = self.config.get('truncate_grads', False)
        self.self_play_config = self.config.get('self_play_config', None)
        self.has_self_play_config = self.self_play_config is not None
        self.self_play = config.get('self_play', False)
        self.save_freq = config.get('save_frequency', 0)
        self.save_best_after = config.get('save_best_after', 100)
        self.print_stats = config.get('print_stats', True)
        self.rnn_states = None
        self.name = base_name
        self.ppo = config['ppo']
        self.max_epochs = self.config.get('max_epochs', 1e6)
        self.is_adaptive_lr = config['lr_schedule'] == 'adaptive'
        self.linear_lr = config['lr_schedule'] == 'linear'
        self.schedule_type = config.get('schedule_type', 'legacy')
        self.kl_threshold = config['kl_threshold']
        self.scheduler = schedulers.AdaptiveScheduler(self.kl_threshold)
        self.e_clip = config['e_clip']
        self.clip_value = config['clip_value']
        self.network = config['network']
        self.rewards_shaper = config['reward_shaper']
        self.horizon_length = config['horizon_length']
        self.seq_len = self.config.get('seq_length', 4)
        self.normalize_advantage = config['normalize_advantage']
        self.normalize_input = self.config['normalize_input']
        self.normalize_value = self.config.get('normalize_value', False)
        self.truncate_grads = self.config.get('truncate_grads', False)
        self.has_phasic_policy_gradients = False
        self.obs_shape = self.observation_space.shape
        self.critic_coef = config['critic_coef']
        self.grad_norm = config['grad_norm']
        self.gamma = self.config['gamma']
        self.tau = self.config['tau']
        self.games_to_track = self.config.get('games_to_track', 100)
        self.game_rewards = torch_ext.AverageMeter(self.value_size, self.games_to_track).to(self.ppo_device)
        self.game_lengths = torch_ext.AverageMeter(1, self.games_to_track).to(self.ppo_device)
        self.obs = None
        self.games_num = self.config['minibatch_size'] // self.seq_len # it is used only for current rnn implementation
        self.batch_size = self.horizon_length * self.num_actors * self.num_agents
        self.batch_size_envs = self.horizon_length * self.num_actors
        self.minibatch_size = self.config['minibatch_size']
        self.mini_epochs_num = self.config['mini_epochs']
        self.num_minibatches = self.batch_size // self.minibatch_size
        assert(self.batch_size % self.minibatch_size == 0)
        self.mixed_precision = self.config.get('mixed_precision', False)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
        self.last_lr = self.config['learning_rate']
        self.frame = 0
        self.update_time = 0
        self.mean_rewards = self.last_mean_rewards = -100500
        self.play_time = 0
        self.epoch_num = 0
        self.train_dir = config.get('train_dir', 'runs')
        self.experiment_dir = os.path.join(self.train_dir, self.experiment_name)
        self.nn_dir = os.path.join(self.experiment_dir, 'nn')
        self.summaries_dir = os.path.join(self.experiment_dir, 'summaries')

        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.summaries_dir, exist_ok=True)

        self.entropy_coef = self.config['entropy_coef']
        self.writer = SummaryWriter(self.summaries_dir)
        self.value_bootstrap = self.config.get('value_bootstrap')
        self.value_mean_std = RunningMeanStd((1,)).to(self.ppo_device)
        self.last_rnn_indices = None
        self.last_state_indices = None
        self.is_discrete = False
        self.actions_num = action_space.shape[0]
        self.bounds_loss_coef = config.get('bounds_loss_coef', None)
        self.clip_actions = config.get('clip_actions', True)
        self.actions_low = torch.from_numpy(action_space.low.copy()).float().to(self.ppo_device)
        self.actions_high = torch.from_numpy(action_space.high.copy()).float().to(self.ppo_device)
        obs_shape = self.obs_shape
        config = {
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'num_seqs' : self.num_actors * self.num_agents,
            'value_size': 1
        }
        self.model = self.network.build(config)
        self.model.to(self.ppo_device)
        self.states = None
        self.is_rnn = False
        self.last_lr = float(self.last_lr)
        self.optimizer = optim.Adam(self.model.parameters(), float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay)
        self.running_mean_std = RunningMeanStd(obs_shape).to(self.ppo_device)
        self.use_experimental_cv = self.config.get('use_experimental_cv', True)
        self.dataset = datasets.PPODataset(self.batch_size, self.minibatch_size, self.is_discrete, self.is_rnn, self.ppo_device, self.seq_len)
        self.has_value_loss = (self.has_central_value and self.use_experimental_cv) \
                            or (not self.has_phasic_policy_gradients and not self.has_central_value) 

        batch_size = self.num_agents * self.num_actors


        env_info = {}
        env_info['action_space'] = env.action_space
        env_info['observation_space'] = env.observation_space
        if env.num_states > 0:
            env_info['state_space'] = env.state_space
    
        algo_info = {
            'num_actors' : self.num_actors,
            'horizon_length' : self.horizon_length,
            'has_central_value' : self.has_central_value,
            'use_action_masks' : self.use_action_masks
        }
        self.experience_buffer = ExperienceBuffer(env_info, algo_info, self.ppo_device)

        current_rewards_shape = (batch_size, self.value_size)
        self.current_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.ppo_device)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self.ppo_device)
        self.dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.ppo_device)
        self.update_list = ['actions', 'neglogpacs', 'values', 'mus', 'sigmas']
        self.tensor_list = self.update_list + ['obses', 'states', 'dones']


    def set_eval(self):
        self.model.eval()
        self.running_mean_std.eval()
        self.value_mean_std.eval()

    def set_train(self):
        self.model.train()
        self.running_mean_std.train()
        self.value_mean_std.train()

    def update_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        

    def get_action_values(self, obs):
        processed_obs = self.running_mean_std(obs)
        self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : processed_obs,
            'rnn_states' : self.rnn_states
        }

        with torch.no_grad():
            res_dict = self.model(input_dict)
        res_dict['values'] = self.value_mean_std(res_dict['values'], True)
        return res_dict

    def get_values(self, obs):
        with torch.no_grad():
            self.model.eval()
            processed_obs = self.running_mean_std(obs)
            input_dict = {
                'is_train': False,
                'prev_actions': None, 
                'obs' : processed_obs,
                'rnn_states' : self.rnn_states
            }
            result = self.model(input_dict)
            value = result['values']
            return self.value_mean_std(value, True)

    @property
    def device(self):
        return self.ppo_device




    def discount_values(self, fdones, last_extrinsic_values, mb_fdones, mb_extrinsic_values, mb_rewards):
        lastgaelam = 0
        mb_advs = torch.zeros_like(mb_rewards)

        for t in reversed(range(self.horizon_length)):
            if t == self.horizon_length - 1:
                nextnonterminal = 1.0 - fdones
                nextvalues = last_extrinsic_values
            else:
                nextnonterminal = 1.0 - mb_fdones[t+1]
                nextvalues = mb_extrinsic_values[t+1]
            nextnonterminal = nextnonterminal.unsqueeze(1)

            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_extrinsic_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.tau * nextnonterminal * lastgaelam
        return mb_advs

    def play_steps(self):
        update_list = self.update_list

        for n in range(self.horizon_length):
            res_dict = self.get_action_values(self.obs)

            self.experience_buffer.update_data('obses', n, self.obs)
            self.experience_buffer.update_data('dones', n, self.dones)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k]) 

            clamped_actions = torch.clamp(res_dict['actions'], -1.0, 1.0)
            obs, rewards, dones, _ = self.vec_env.step(clamped_actions)
            self.obs, rewards, self.dones, _ = obs, rewards.unsqueeze(1).to(self.ppo_device), dones.to(self.ppo_device), _
            shaped_rewards = self.rewards_shaper(rewards)

            self.experience_buffer.update_data('rewards', n, shaped_rewards)

            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]

            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        last_values = self.get_values(self.obs)

        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_advs = self.discount_values(fdones, last_values, mb_fdones, mb_values, mb_rewards)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = swap_and_flatten01(mb_returns)
        return batch_dict

    def prepare_dataset(self, batch_dict):
        obses = batch_dict['obses']
        returns = batch_dict['returns']
        values = batch_dict['values']
        actions = batch_dict['actions']
        neglogpacs = batch_dict['neglogpacs']
        mus = batch_dict['mus']
        sigmas = batch_dict['sigmas']

        advantages = returns - values

        if self.normalize_value:
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)

        advantages = torch.sum(advantages, axis=1)

        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_dict = {}
        dataset_dict['obs'] = obses
        dataset_dict['returns'] = returns
        dataset_dict['old_values'] = values
        dataset_dict['actions'] = actions
        dataset_dict['old_logp_actions'] = neglogpacs
        dataset_dict['mu'] = mus
        dataset_dict['sigma'] = sigmas

        dataset_dict['advantages'] = advantages
        self.dataset.values_dict = dataset_dict

    def train_actor_critic(self, input_dict):
        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch = self.running_mean_std(obs_batch)

        curr_e_clip = self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch, 
            'obs' : obs_batch,
        }

        rnn_masks = None
            
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            a_loss = common_losses.actor_loss(old_action_log_probs_batch, action_log_probs, advantage, self.ppo, curr_e_clip)

            if self.has_value_loss:
                c_loss = common_losses.critic_loss(value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)

            b_loss = self.bound_loss(mu)
            losses, sum_mask = torch_ext.apply_masks([a_loss.unsqueeze(1), c_loss, entropy.unsqueeze(1), b_loss.unsqueeze(1)], rnn_masks)
            a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3]

            loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef
            
            for param in self.model.parameters():
                param.grad = None

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()    

        with torch.no_grad():
            reduce_kl = not self.is_rnn
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
                    
        return kl_dist, mu.detach(), sigma.detach()

    def bound_loss(self, mu):
        if self.bounds_loss_coef is not None:
            soft_bound = 1.1
            mu_loss_high = torch.clamp_max(mu - soft_bound, 0.0)**2
            mu_loss_low = torch.clamp_max(-mu + soft_bound, 0.0)**2
            b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
        else:
            b_loss = 0
        return b_loss


if __name__ == "__main__":
    import copy
    import os
    import time
    from datetime import datetime
    from time import sleep

    import numpy as np
    import torch
    from rl_games.algos_torch import (central_value, models, network_builder,
                                      ppg_aux, torch_ext)
    from rl_games.algos_torch.running_mean_std import (RunningMeanStd,
                                                       RunningMeanStdObs)
    from rl_games.common import (common_losses, datasets, env_configurations,
                                 schedulers, tr_helpers, vecenv)
    from rl_games.common.experience import ExperienceBuffer
    from tensorboardX import SummaryWriter
    from torch import nn, optim
    def swap_and_flatten01(arr):
        """
        swap and then flatten axes 0 and 1
        """
        if arr is None:
            return arr
        s = arr.size()
        return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])

    sleep(5)

    with open('cfg/config.yaml') as f:
        yaml_cfg = yaml.safe_load(f)
    train_params, task_params, sim_device, graphics_device_id, headless = yaml_cfg['train']['params'], yaml_cfg['task'], yaml_cfg['sim_device'], yaml_cfg['graphics_device_id'], yaml_cfg['headless']
    seed, network_cfg, agent_cfg = train_params['seed'], train_params['network'], train_params['config']
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    a2c_builder = network_builder.A2CBuilder()
    a2c_builder.load(network_cfg)
    agent_cfg['network'] = models.ModelA2CContinuousLogStd(a2c_builder)
    agent_cfg['reward_shaper'] = tr_helpers.DefaultRewardsShaper(**agent_cfg['reward_shaper'])
    agent_cfg['features'] = {}
    
    env = FrankaCabinet(cfg=task_params, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)
    agent = A2CAgent(base_name='my_a2c_agent', config=agent_cfg, env=env)  
    agent.obs = agent.vec_env.reset()
    while True:
        agent.model.eval()
        agent.running_mean_std.eval()
        agent.value_mean_std.eval()
        with torch.no_grad():
            batch_dict = agent.play_steps()
        agent.model.train()
        agent.running_mean_std.train()
        agent.value_mean_std.train()
        agent.prepare_dataset(batch_dict)
        for _ in range(0, agent.mini_epochs_num):
            for i in range(len(agent.dataset)):
                kl, cmu, csigma= agent.train_actor_critic(agent.dataset[i])
                agent.dataset.update_mu_sigma(cmu, csigma)   
                agent.last_lr, agent.entropy_coef = agent.scheduler.update(agent.last_lr, agent.entropy_coef, agent.epoch_num, 0, kl.item())
                agent.update_lr(agent.last_lr)
        if agent.game_rewards.current_size > 0:
            print(agent.game_rewards.get_mean()[0])
