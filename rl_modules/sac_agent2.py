import torch
import numpy as np
from mpi_utils.mpi_utils import sync_networks
from rl_modules.replay_buffer import MultiBuffer
from rl_modules.sac_models import QNetworkFlat, GaussianPolicyFlat, ConfigNetwork, QNetworkDisentangled, GaussianPolicyDisentangled
from mpi_utils.normalizer import normalizer
from her_modules.her import her_sampler
from rl_modules.sac_deepset_models import DeepSetSAC
from updates import update_flat, update_disentangled, update_deepsets



"""
SAC with HER (MPI-version)
"""

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class SACAgent:
    def __init__(self, args, compute_rew, goal_sampler):

        self.args = args
        self.alpha = args.alpha
        self.env_params = args.env_params

        self.goal_sampler = goal_sampler

        # create the network
        self.architecture = self.args.architecture
        if self.architecture == 'disentangled':
            self.actor_network = GaussianPolicyDisentangled(self.env_params)
            self.critic_network = QNetworkDisentangled(self.env_params)
            self.configuration_network = ConfigNetwork(self.env_params)
            # sync the networks across the CPUs
            sync_networks(self.actor_network)
            sync_networks(self.critic_network)
            sync_networks(self.configuration_network)
            # build up the target network
            self.critic_target_network = QNetworkDisentangled(self.env_params)
            hard_update(self.critic_target_network, self.critic_network)
            sync_networks(self.critic_target_network)
            # create the optimizer
            self.policy_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
            self.critic_optim = torch.optim.Adam(list(self.critic_network.parameters()) + list(self.configuration_network.parameters())
                                                 , lr=self.args.lr_critic)
            # self.configuration_optim = torch.optim.Adam(self.configuration_network.parameters(), lr=self.args.lr_critic)
        elif self.architecture == 'flat':
            self.actor_network = GaussianPolicyFlat(self.env_params)
            self.critic_network = QNetworkFlat(self.env_params)
            # sync the networks across the CPUs
            sync_networks(self.actor_network)
            sync_networks(self.critic_network)
            # build up the target network
            self.critic_target_network = QNetworkFlat(self.env_params)
            hard_update(self.critic_target_network, self.critic_network)
            sync_networks(self.critic_target_network)
            # create the optimizer
            self.policy_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
            self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)
        elif self.architecture == 'deepsets':
            self.model = DeepSetSAC(self.env_params, self.args.deepsets_attention, self.args.double_critic_attention, args)
            # sync the networks across the CPUs
            sync_networks(self.model.rho_actor)
            sync_networks(self.model.rho_critic)
            sync_networks(self.model.single_phi_actor)
            sync_networks(self.model.single_phi_critic)
            if self.args.deepsets_attention:
                sync_networks(self.model.attention_actor)
                sync_networks(self.model.attention_critic_1)
                sync_networks(self.model.attention_critic_2)

            hard_update(self.model.single_phi_target_critic, self.model.single_phi_critic)
            hard_update(self.model.rho_target_critic, self.model.rho_critic)
            sync_networks(self.model.single_phi_target_critic)
            sync_networks(self.model.rho_target_critic)
            # create the optimizer
            if self.args.deepsets_attention:
                self.policy_optim = torch.optim.Adam(list(self.model.single_phi_actor.parameters()) +
                                                     list(self.model.rho_actor.parameters()) +
                                                     list(self.model.attention_actor.parameters()),
                                                     lr=self.args.lr_actor)

                self.critic_optim = torch.optim.Adam(list(self.model.single_phi_critic.parameters()) +
                                                     list(self.model.rho_critic.parameters()) +
                                                     list(self.model.attention_critic_1.parameters()) +
                                                     list(self.model.attention_critic_2.parameters()),
                                                     lr=self.args.lr_critic)

            else:
                self.policy_optim = torch.optim.Adam(list(self.model.single_phi_actor.parameters()) +
                                                     list(self.model.rho_actor.parameters()),
                                                     lr=self.args.lr_actor)
                self.critic_optim = torch.optim.Adam(list(self.model.single_phi_critic.parameters()) +
                                                     list(self.model.rho_critic.parameters()),
                                                     lr=self.args.lr_critic)

        else:
            raise NotImplementedError

        # create the normalizer
        self.o_norm = normalizer(size=self.env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(size=self.env_params['goal'], default_clip_range=self.args.clip_range)

        # if use GPU
        if self.args.cuda:
            self.actor_network.cuda()
            self.critic_network.cuda()
            self.critic_target_network.cuda()

        # Target Entropy
        if self.args.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(self.env_params['action'])).item()
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.args.lr_entropy)

        # her sampler
        self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, compute_rew)

        # create the replay buffer
        self.buffer = MultiBuffer(env_params=self.env_params,
                                  buffer_size=self.args.buffer_size,
                                  sample_func=self.her_module.sample_her_transitions,
                                  multi_head=self.args.multihead_buffer,
                                  goal_sampler=self.goal_sampler
                                  )



    def act(self, obs, ag, g, no_noise):
        with torch.no_grad():
            # normalize policy inputs 
            obs_norm = self.o_norm.normalize(obs)
            g_norm = torch.tensor(self.g_norm.normalize(g), dtype=torch.float32).unsqueeze(0)
            ag_norm = torch.tensor(self.g_norm.normalize(ag), dtype=torch.float32).unsqueeze(0)

            if self.architecture == 'deepsets':
                obs_tensor = torch.tensor(obs_norm, dtype=torch.float32).unsqueeze(0)
                self.model.policy_forward_pass(obs_tensor, ag_norm, g_norm, no_noise=no_noise)
                action = self.model.pi_tensor.numpy()[0]
                
            elif self.architecture == 'disentangled':
                z_ag = self.configuration_network(ag_norm)[0]
                z_g = self.configuration_network(g_norm)[0]
                input_tensor = torch.tensor(np.concatenate([obs_norm, z_ag, z_g]), dtype=torch.float32).unsqueeze(0)
                action = self._select_actions(input_tensor, no_noise=no_noise)
            else:
                input_tensor = self._preproc_inputs(obs, g)  # PROCESSING TO CHECK
                action = self._select_actions(input_tensor, no_noise=no_noise)
                
        return action.copy()
        
    
    def store(self, episodes):
        self.buffer.store_episode(episode_batch=episodes)

    # pre_process the inputs
    def _preproc_inputs(self, obs, g):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        # concatenate the stuffs
        inputs = np.concatenate([obs_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
        return inputs

    def train(self):
        # train the network
        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = self._update_network()

        # soft update
        if self.architecture == 'deepsets':
            self._soft_update_target_network(self.model.single_phi_target_critic, self.model.single_phi_critic)
            self._soft_update_target_network(self.model.rho_target_critic, self.model.rho_critic)
        else:
            self._soft_update_target_network(self.critic_target_network, self.critic_network)

    def _select_actions(self, state, no_noise=False):
        if not no_noise:
            action, _, _ = self.actor_network.sample(state)
        else:
            _, _, action = self.actor_network.sample(state)
        return action.detach().cpu().numpy()[0]

    # update the normalizer
    def _update_normalizer(self, episode):

        mb_obs = episode['obs']
        mb_ag = episode['ag']
        mb_g = episode['g']
        mb_actions = episode['act']
        mb_obs_next = mb_obs[1:, :]
        mb_ag_next = mb_ag[1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[0]
        # create the new buffer to store them
        buffer_temp = {'obs': np.expand_dims(mb_obs, 0),
                       'ag': np.expand_dims(mb_ag, 0),
                       'g': np.expand_dims(mb_g, 0),
                       'actions': np.expand_dims(mb_actions, 0),
                       'obs_next': np.expand_dims(mb_obs_next, 0),
                       'ag_next': np.expand_dims(mb_ag_next, 0),
                       }
        transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        # update
        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    # update the network
    def _update_network(self):

        # sample from buffer, this is done with LP is multi-head is true
        transitions = self.buffer.sample(self.args.batch_size)

        # pre-process the observation and goal
        o, o_next, g, ag, ag_next, actions, rewards = transitions['obs'], transitions['obs_next'], transitions['g'], transitions['ag'], \
                                                      transitions['ag_next'], transitions['actions'], transitions['r']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)
        _, transitions['ag'] = self._preproc_og(o, ag)
        _, transitions['ag_next'] = self._preproc_og(o, ag_next)

        # apply normalization
        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        ag_norm = self.g_norm.normalize(transitions['ag'])
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        ag_next_norm = self.g_norm.normalize(transitions['ag_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])

        if self.architecture == 'flat':
            critic_1_loss, critic_2_loss, actor_loss, alpha_loss, alpha_tlogs = update_flat(self.actor_network, self.critic_network, self.critic_target_network,
                                                                           self.policy_optim, self.critic_optim, self.alpha, self.log_alpha,
                                                                           self.target_entropy, self.alpha_optim, obs_norm, g_norm, obs_next_norm,
                                                                           actions, rewards, self.args)
        elif self.architecture == 'disentangled':
            critic_1_loss, critic_2_loss, actor_loss, alpha_loss, alpha_tlogs,  = update_disentangled(self.actor_network, self.critic_network,
                                                                                   self.critic_target_network, self.configuration_network,
                                                                                   self.policy_optim, self.critic_optim, self.alpha,
                                                                                   self.log_alpha, self.target_entropy, self.alpha_optim,
                                                                                   obs_norm, ag_norm, g_norm, obs_next_norm, ag_next_norm, g_next_norm,
                                                                                   actions, rewards, self.args)
        elif self.architecture == 'deepsets':
            critic_1_loss, critic_2_loss, actor_loss, alpha_loss, alpha_tlogs = update_deepsets(self.model, self.policy_optim, self.critic_optim,
                                                                               self.alpha, self.log_alpha, self.target_entropy,
                                                                               self.alpha_optim, obs_norm, ag_norm, g_norm, obs_next_norm,
                                                                               ag_next_norm, actions, rewards, self.args)
        else:
            raise NotImplementedError

        return critic_1_loss, critic_2_loss, actor_loss, alpha_loss, alpha_tlogs


    def save(self, model_path, epoch):
        # Store model
        if self.args.architecture == 'flat':
            torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std,
                        self.actor_network.state_dict(), self.critic_network.state_dict()],
                       model_path + '/model_{}.pt'.format(epoch))
        elif self.args.architecture == 'disentangled':
            torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std,
                        self.actor_network.state_dict(), self.critic_network.state_dict(),
                        self.configuration_network.state_dict()],
                       model_path + '/model_{}.pt'.format(epoch))
        elif self.args.architecture == 'deepsets':
            if self.args.deepsets_attention and not self.args.double_critic_attention:
                torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std,
                            self.model.single_phi_actor.state_dict(), self.model.single_phi_critic.state_dict(),
                            self.model.rho_actor.state_dict(), self.model.rho_critic.state_dict(),
                            self.model.attention_actor.state_dict(), self.model.attention_critic_1.state_dict()],
                           model_path + '/model_{}.pt'.format(epoch))
            elif self.args.deepsets_attention and self.args.double_critic_attention:
                torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std,
                            self.model.single_phi_actor.state_dict(), self.model.single_phi_critic.state_dict(),
                            self.model.rho_actor.state_dict(), self.model.rho_critic.state_dict(),
                            self.model.attention_actor.state_dict(), self.model.attention_critic_1.state_dict(),
                            self.model.attention_critic_2.state_dict()],
                           model_path + '/model_{}.pt'.format(epoch))
            else:
                torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std,
                            self.model.single_phi_actor.state_dict(), self.model.single_phi_critic.state_dict(),
                            self.model.rho_actor.state_dict(), self.model.rho_critic.state_dict()],
                           model_path + '/model_{}.pt'.format(epoch))
        else:
            raise NotImplementedError


    def load(self, model_path, args):

        if args.architecture == 'deepsets':
            o_mean, o_std, g_mean, g_std, phi_a, phi_c, rho_a, rho_c = torch.load(model_path, map_location=lambda storage, loc: storage)
            self.model.single_phi_actor.load_state_dict(phi_a)
            self.model.single_phi_critic.load_state_dict(phi_c)
            self.model.rho_actor.load_state_dict(rho_a)
            self.model.rho_critic.load_state_dict(rho_c)
            self.o_norm.mean = o_mean
            self.o_norm.std = o_std
            self.g_norm.mean = g_mean
            self.g_norm.std = g_std
        else:
            o_mean, o_std, g_mean, g_std, model, _, config = torch.load(model_path, map_location=lambda storage, loc: storage)
            self.actor_network.load_state_dict(model)
            self.actor_network.eval()
            self.configuration_network.load_state_dict(config)
            self.o_norm.mean = o_mean
            self.o_norm.std = o_std
            self.g_norm.mean = g_mean
            self.g_norm.std = g_std
