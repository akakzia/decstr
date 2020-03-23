import torch
import os
from datetime import datetime
import numpy as np
from mpi4py import MPI
from tensorboardX import SummaryWriter
from mpi_utils.mpi_utils import sync_networks, sync_grads
from rl_modules.replay_buffer import MultiHeadBuffer, replay_buffer
from rl_modules.sac_models import QNetworkFlat, GaussianPolicyFlat, ConfigNetwork, QNetworkDisentangled, GaussianPolicyDisentangled
from mpi_utils.normalizer import normalizer
from her_modules.her import her_sampler
from queues import CompetenceQueue
from utils import hard_update, rollout, load_models, generate_goals, init_storage
import pickle as pkl
import datetime
from rl_modules.sac_deepset_models import DeepSetSAC
from stats import save_plot
from updates import update_flat, update_disentangled, update_deepsets
from logger import log_results
from evaluation import eval_agent
"""
SAC with HER (MPI-version)
"""


class SACAgent:
    def __init__(self, args, env, env_params):
        self.args = args
        self.env = env
        self.env_params = env_params
        self.alpha = args.alpha

        # To store stats for plot curves
        self.overall_stats = []

        # Competence progress setup
        self.goals = generate_goals(nb_objects=3, sym=1, asym=1)
        self.num_buckets = len(self.goals.keys())
        self.competence_computers = [CompetenceQueue(window=self.args.queue_length) for _ in range(self.num_buckets)]
        self.p = 1 / self.num_buckets * np.ones([self.num_buckets])
        self.C = np.zeros([self.num_buckets])  # competence
        self.CP = np.zeros([self.num_buckets])  # learning progress

        # create the network
        self.architecture = self.args.architecture
        if self.architecture == 'disentangled':
            self.actor_network = GaussianPolicyDisentangled(env_params)
            self.critic_network = QNetworkDisentangled(env_params)
            self.configuration_network = ConfigNetwork(env_params)
            # sync the networks across the CPUs
            sync_networks(self.actor_network)
            sync_networks(self.critic_network)
            sync_networks(self.configuration_network)
            # build up the target network
            self.critic_target_network = QNetworkDisentangled(env_params)
            hard_update(self.critic_target_network, self.critic_network)
            sync_networks(self.critic_target_network)
            # create the optimizer
            self.policy_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
            self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)
            self.configuration_optim = torch.optim.Adam(self.configuration_network.parameters(), lr=self.args.lr_critic)
        elif self.architecture == 'flat':
            self.actor_network = GaussianPolicyFlat(env_params)
            self.critic_network = QNetworkFlat(env_params)
            # sync the networks across the CPUs
            sync_networks(self.actor_network)
            sync_networks(self.critic_network)
            # build up the target network
            self.critic_target_network = QNetworkFlat(env_params)
            hard_update(self.critic_target_network, self.critic_network)
            sync_networks(self.critic_target_network)
            # create the optimizer
            self.policy_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
            self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)
        elif self.architecture == 'deepsets':
            self.model = DeepSetSAC(env_params, self.args.deepsets_attention, self.args.double_critic_attention)
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

            # self.attention_optim = torch.optim.Adam(self.model.attention.parameters(), lr=self.args.lr_critic)
        else:
            raise NotImplementedError

        # create the normalizer
        self.o_norm = normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(size=env_params['goal'], default_clip_range=self.args.clip_range)

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
        self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, self.args.multi_criteria_her,
                                      self.env.compute_reward)
        # create the replay buffer
        if self.args.multihead_buffer:
            self.buffer = MultiHeadBuffer(self.env_params, self.args.buffer_size, self.num_buckets,
                                          self.her_module.sample_her_transitions)
        else:
            self.buffer = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)
        # create the dict for store the model
        if MPI.COMM_WORLD.Get_rank() == 0:
            self.model_path, self.eval_path = init_storage(self.args)

    def learn(self):
        """
        train the network

        """
        """# for tensorboardX
        global writer

        if MPI.COMM_WORLD.Get_rank() == 0:
            writer = SummaryWriter(logdir='runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                                                        self.args.env_name,
                                                                        self.args.folder_prefix, "autotune" if self.args.automatic_entropy_tuning else ""))"""

        # start to collect samples
        #updates = 0
        for epoch in range(self.args.n_epochs):
            for _ in range(self.args.n_cycles):
                # Initialize dictionaries to contain successes relative to each bucket (local ~ per CPU | global ~ all CPUs gathered)
                mb_successes_local = {i: [] for i in range(self.num_buckets)}
                mb_successes_global = {i: [] for i in range(self.num_buckets)}

                mb_obs, mb_ag, mb_g, mb_actions, mb_buckets = [], [], [], [], []
                for _ in range(self.args.num_rollouts_per_mpi):
                    ep_obs, ep_ag, ep_g, ep_actions, ep_success, \
                    eval, current_goal, bucket = rollout(self.env, self.env_params, self, self.args, self.goals, animated=False)
                    mb_obs.append(ep_obs)
                    mb_ag.append(ep_ag)
                    mb_g.append(ep_g)
                    mb_actions.append(ep_actions)
                    mb_buckets.append(bucket)
                    if self.args.curriculum_learning and eval:
                        mb_successes_local[bucket].append(ep_success)
                # convert them into arrays
                mb_obs = np.array(mb_obs)
                mb_ag = np.array(mb_ag)
                mb_g = np.array(mb_g)
                mb_actions = np.array(mb_actions)
                if self.args.curriculum_learning:
                    # Estimate learning progress
                    for i, value in enumerate(mb_successes_local.values()):
                        mb_successes_global[i] = MPI.COMM_WORLD.gather(value, root=0)

                    if eval and MPI.COMM_WORLD.Get_rank() == 0:
                        for i, value in enumerate(mb_successes_global.values()):
                            value = sum(value, [])
                            self.competence_computers[i].update(value)
                        self._update_p(self.args.curriculum_eps)
                    self.p = MPI.COMM_WORLD.bcast(self.p, root=0)
                    self.CP = MPI.COMM_WORLD.bcast(self.CP, root=0)
                # store the episodes
                if self.args.multihead_buffer:
                    for i in range(self.args.num_rollouts_per_mpi):
                        self.buffer.store_episode([mb_obs[i], mb_ag[i], mb_g[i], mb_actions[i]], mb_buckets[i])
                else:
                    self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
                self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])
                for _ in range(self.args.n_batches):
                    # train the network
                    """if self.args.architecture == 'flat':
                        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = self._update_network()
                    else:
                        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = self._update_network_disentangled()"""
                    if self.CP[mb_buckets[0]] > self.CP[mb_buckets[1]]:  # update according to highest LP
                        up_bucket = mb_buckets[0]
                    else:
                        up_bucket = mb_buckets[1]
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = self._update_network(up_bucket, epoch)
                    """if MPI.COMM_WORLD.Get_rank() == 0:
                        writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                        writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                        writer.add_scalar('loss/policy', policy_loss, updates)
                        writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                        writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                    updates += 1"""
                # soft update
                if self.architecture == 'deepsets':
                    self._soft_update_target_network(self.model.single_phi_target_critic, self.model.single_phi_critic)
                    self._soft_update_target_network(self.model.rho_target_critic, self.model.rho_critic)
                else:
                    self._soft_update_target_network(self.critic_target_network, self.critic_network)
            if epoch % self.args.save_freq == 0 and self.args.evaluations:
                res = eval_agent(self, curriculum=self.args.curriculum_learning, separate_goals=self.args.separate_goals)
            log_results(self, epoch, res, evaluations=self.args.evaluations, frequency=self.args.save_freq, store_model=True,
                        store_stats=False, separate_goals=self.args.separate_goals)

    # Update LP probability
    def _update_p(self, epsilon=0.4):
        self.C = np.array([self._get_c()]).squeeze()
        # compute competence progress for each task
        self.CP = np.array([self._get_cp()]).squeeze()

        if self.CP.sum() == 0:
            self.p = (1 / self.num_buckets) * np.ones([self.num_buckets])
        else:
            self.p = epsilon * (1 / self.num_buckets) * np.ones([self.num_buckets]) + \
                     (1 - epsilon) * np.power(self.CP, self.args.curriculum_nu) / np.power(self.CP, self.args.curriculum_nu).sum()
            """self.p = epsilon * (1 - self.C) / ((1 - self.C).sum()) + \
                     (1 - epsilon) * self.CP / self.CP.sum()"""

        if self.p.sum() > 1:
            self.p[np.argmax(self.p)] -= self.p.sum() - 1
        elif self.p.sum() < 1:
            self.p[-1] = 1 - self.p[:-1].sum()

    def _get_cp(self):
        # addition for active goal task selection
        # extract measures of competence progress for all tasks
        return [cq.CP for cq in self.competence_computers]

    def _get_c(self):
        # addition for active goal task selection
        # extract measures of competence for all tasks
        return [cq.C for cq in self.competence_computers]

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

    def _select_actions(self, state, eval=False):
        if not eval:
            action, _, _ = self.actor_network.sample(state)
        else:
            _, _, action = self.actor_network.sample(state)
        return action.detach().cpu().numpy()[0]

    # update the normalizer
    def _update_normalizer(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]
        # create the new buffer to store them
        buffer_temp = {'obs': mb_obs,
                       'ag': mb_ag,
                       'g': mb_g,
                       'actions': mb_actions,
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
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
    def _update_network(self, head, e):
        # sample the episodes
        if self.args.multihead_buffer:
            # Starting from the 5th epoch, heads are selected according to p
            if e > 5:
                head = np.random.choice(np.arange(self.num_buckets), 1, p=self.p)[0]
            transitions = self.buffer.sample(self.args.batch_size, head)
        else:
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
            critic_1_loss, critic_2_loss, actor_loss, alpha_loss, alpha_tlogs = update_disentangled(self.actor_network, self.critic_network,
                                                                                   self.critic_target_network, self.configuration_network,
                                                                                   self.policy_optim, self.critic_optim, self.configuration_optim,
                                                                                   self.alpha, self.log_alpha, self.target_entropy, self.alpha_optim,
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

    def log_results(self, epoch, overall_stats, evaluations=True, frequency=10, store_model=True, store_stats=False,
                    separate_goals=False):
        # start to do the evaluation
        if epoch % frequency == 0 and evaluations:
            res = self._eval_agent(curriculum=self.args.curriculum_learning, separate_goals=separate_goals)
            # success_rate = self._eval_agent()
            overall_stats.append(res)
        if MPI.COMM_WORLD.Get_rank() == 0:
            if self.args.curriculum_learning:
                print('[{}] Epoch: {} |==============================|'.format(datetime.now(), epoch))
                for i in range(len(self.goals)):
                    print('Bucket ', i, '| p = {:.3f} | LP = {:.3f}| C = {:.3f}'.format(self.p[i], self.CP[i], self.C[i]))
                if epoch % frequency == 0 and evaluations:
                    print('===========================================')
                    for i in range(len(self.goals)):
                        print('Bucket', i, 'eval success rate is: {:.3f}'.format(res[i]))
                    print('===========================================')
                if epoch % frequency == 0 and store_stats:
                    torch.save([self.p, self.C, self.CP], self.eval_path + '/LP_{}.pt'.format(epoch))
            elif not separate_goals and epoch % frequency == 0:
                print('[{}] epoch is: {}, eval success rate is: {:.3f}'.format(datetime.now(), epoch, res))
            else:
                print('[{}] Epoch: {} |==============================|'.format(datetime.now(), epoch))
                if epoch % frequency == 0 and evaluations:
                    for goal in res[0].keys():
                        print('Goal: {}, eval success rate is: {:.3f}'.format(goal, res[0][goal]))
            if epoch % frequency == 0 and store_model:
                if self.args.architecture == 'flat':
                    torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std,
                                self.actor_network.state_dict(), self.critic_network.state_dict()],
                               self.model_path + '/model_{}.pt'.format(epoch))
                else:
                    """torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std,
                                self.actor_network.state_dict(), self.critic_network.state_dict(),
                                self.configuration_network.state_dict()],
                               self.model_path + '/model_{}.pt'.format(epoch))"""
                    torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std,
                                self.model.single_phi_actor.state_dict(), self.model.single_phi_critic.state_dict(),
                                self.model.rho_actor.state_dict(), self.model.rho_critic.state_dict(),
                                self.model.attention.state_dict()],
                               self.model_path + '/model_{}.pt'.format(epoch))
            if epoch % frequency == 0 and store_stats:
                with open(os.path.join(self.eval_path, 'evaluations.pkl'), 'wb') as f:
                    pkl.dump(overall_stats, f)
        return overall_stats

    # do the evaluation
    def _eval_agent(self, curriculum=False, separate_goals=False):
        if not curriculum and separate_goals:
            goals = self.goals
            per_goal_sr = {}
            per_goal_std = {}
            for goal in goals:
                total_success_rate = []
                for _ in range(self.args.n_test_rollouts):
                    per_success_rate = []
                    observation = self.env.reset_goal(goal)
                    obs = observation['observation']
                    observation['desired_goal'] = goal
                    ag = observation['achieved_goal']
                    for _ in range(self.env_params['max_timesteps']):
                        with torch.no_grad():
                            if self.architecture == 'disentangled':
                                g_norm = torch.tensor(self.g_norm.normalize_goal(goal), dtype=torch.float32).unsqueeze(0)
                                ag_norm = torch.tensor(self.g_norm.normalize_goal(ag), dtype=torch.float32).unsqueeze(0)
                                # config_inputs = np.concatenate([ag, g])
                                # config_inputs = torch.tensor(config_inputs, dtype=torch.float32).unsqueeze(0)
                                config_z = self.configuration_network(ag_norm, g_norm)[0]
                                input_tensor = torch.tensor(np.concatenate([self.o_norm.normalize(obs),
                                                                            config_z]), dtype=torch.float32).unsqueeze(0)
                            else:
                                input_tensor = self._preproc_inputs(obs, ag, goal)  # PROCESSING TO CHECK
                            action = self._select_actions(input_tensor, eval=True)
                        observation_new, _, _, info = self.env.step(action)
                        obs = observation_new['observation']
                        goal = observation_new['desired_goal']
                        per_success_rate.append(info['is_success'])
                    total_success_rate.append(per_success_rate)
                total_success_rate = np.array(total_success_rate)
                local_success_rate = np.mean(total_success_rate[:, -1])
                global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
                per_goal_sr[tuple(goal)] = global_success_rate / MPI.COMM_WORLD.Get_size()
                per_goal_std[tuple(goal)] = np.sqrt(MPI.COMM_WORLD.allreduce(pow(local_success_rate - per_goal_sr[tuple(goal)], 2),
                                                    op=MPI.SUM) / MPI.COMM_WORLD.Get_size())
            return per_goal_sr, per_goal_std
        if curriculum:
            stats = []
            res = []
            for i in range(self.num_buckets):
                total_success_rate = []
                for _ in range(self.args.n_test_rollouts):
                    per_success_rate = []
                    goal = self.goals[i][np.random.choice(len(self.goals[i]))]
                    observation = self.env.reset_goal(np.array(goal))
                    obs = observation['observation']
                    g = observation['desired_goal']
                    ag = observation['achieved_goal']
                    for _ in range(self.env_params['max_timesteps']):
                        with torch.no_grad():
                            if self.architecture == 'disentangled':
                                g_norm = torch.tensor(self.g_norm.normalize(g), dtype=torch.float32).unsqueeze(0)
                                ag_norm = torch.tensor(self.g_norm.normalize(ag), dtype=torch.float32).unsqueeze(0)
                                # config_inputs = np.concatenate([ag, g])
                                # config_inputs = torch.tensor(config_inputs, dtype=torch.float32).unsqueeze(0)
                                #z_ag = self.configuration_network(ag_norm)[0]
                                #z_g = self.configuration_network(g_norm)[0]
                                """input_tensor = torch.tensor(np.concatenate([self.o_norm.normalize(obs),
                                                                            z_ag, z_g]), dtype=torch.float32).unsqueeze(0)"""
                                obs_tensor = torch.tensor(self.o_norm.normalize(obs), dtype=torch.float32).unsqueeze(0)
                                self.model.forward_pass(obs_tensor, ag_norm, g_norm)
                                action = self.model.pi_tensor.numpy()
                            else:
                                input_tensor = self._preproc_inputs(obs, ag, g)  # PROCESSING TO CHECK
                            # action = self._select_actions(input_tensor, eval=True)
                        observation_new, _, _, info = self.env.step(action)
                        obs = observation_new['observation']
                        g = observation_new['desired_goal']
                        per_success_rate.append(info['is_success'])
                    total_success_rate.append(per_success_rate)
                total_success_rate = np.array(total_success_rate)
                local_success_rate = np.mean(total_success_rate[:, -1])
                global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
                mean = global_success_rate / MPI.COMM_WORLD.Get_size()
                std = np.sqrt(MPI.COMM_WORLD.allreduce(pow(local_success_rate - mean, 2), op=MPI.SUM) / MPI.COMM_WORLD.Get_size())
                stats.append((mean, std))
                res.append(mean)
            self.overall_stats.append(stats)
            if MPI.COMM_WORLD.Get_rank() == 0:
                save_plot(np.array(self.overall_stats))
            return res
        if not curriculum and not separate_goals:
            total_success_rate = []
            for _ in range(self.args.n_test_rollouts):
                per_success_rate = []
                #goal = self.goals[np.random.choice(len(self.goals))]
                observation = self.env.reset()
                obs = observation['observation']
                g = observation['desired_goal']
                ag = observation['achieved_goal']
                for _ in range(self.env_params['max_timesteps']):
                    with torch.no_grad():
                        if self.architecture == 'disentangled':
                            g_norm = torch.tensor(self.g_norm.normalize(g), dtype=torch.float32).unsqueeze(0)
                            ag_norm = torch.tensor(self.g_norm.normalize(ag), dtype=torch.float32).unsqueeze(0)
                            # config_inputs = np.concatenate([ag, g])
                            # config_inputs = torch.tensor(config_inputs, dtype=torch.float32).unsqueeze(0)
                            #config_z = self.configuration_network(ag_norm, g_norm)[0]
                            z_ag = self.configuration_network(ag_norm)[0]
                            z_g = self.configuration_network(g_norm)[0]
                            input_tensor = torch.tensor(np.concatenate([self.o_norm.normalize(obs),
                                                                        z_ag, z_g]), dtype=torch.float32).unsqueeze(0)
                        else:
                            input_tensor = self._preproc_inputs(obs, ag, g)  # PROCESSING TO CHECK
                        action = self._select_actions(input_tensor, eval=True)
                    observation_new, _, _, info = self.env.step(action)
                    obs = observation_new['observation']
                    g = observation_new['desired_goal']
                    per_success_rate.append(info['is_success'])
                total_success_rate.append(per_success_rate)
            total_success_rate = np.array(total_success_rate)
            local_success_rate = np.mean(total_success_rate[:, -1])
            global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
            return global_success_rate / MPI.COMM_WORLD.Get_size()
