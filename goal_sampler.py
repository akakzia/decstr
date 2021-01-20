from collections import deque
import numpy as np
from utils import  generate_all_goals_in_goal_space, generate_goals
from mpi4py import MPI
import os
import pickle
import pandas as pd
from mpi_utils import logger


class GoalSampler:
    def __init__(self, args):
        self.continuous = args.algo == 'continuous'
        self.curriculum_learning = args.curriculum_learning
        self.automatic_buckets = args.automatic_buckets
        self.num_buckets = args.num_buckets
        self.queue_length = args.queue_length
        self.num_rollouts_per_mpi = args.num_rollouts_per_mpi
        self.rank = MPI.COMM_WORLD.Get_rank()
        self.self_eval_prob = args.self_eval_prob  # probability to perform self evaluation
        assert 0 <= self.self_eval_prob <= 1, "Self-evaluation probability must be in [0, 1]."

        # Get expert buckets
        buckets = generate_goals()
        if not self.curriculum_learning: self.num_buckets = 0
        elif not self.automatic_buckets: self.num_buckets = len(buckets)
        all_goals = generate_all_goals_in_goal_space().astype(np.float32)
        valid_goals = []
        for k in buckets.keys():
            # if k < 4:
            valid_goals += buckets[k]
        self.valid_goals = np.array(valid_goals)
        self.all_goals = np.array(all_goals)
        self.num_goals = self.all_goals.shape[0]
        self.all_goals_str = [str(g) for g in self.all_goals]
        self.valid_goals_str = [str(vg) for vg in self.valid_goals]
        self.goal_dim = self.all_goals.shape[1]

        # initialize dict to convert from the oracle id to goals and vice versa.
        # oracle id is position in the all_goal array
        self.oracle_id_to_g = dict(zip(range(self.num_goals), self.all_goals))
        self.g_str_to_oracle_id = dict(zip(self.all_goals_str, range(self.num_goals)))
        self.valid_goals_oracle_ids = np.array([self.g_str_to_oracle_id[str(vg)] for vg in self.valid_goals])

        self.rew_counters = dict(zip(range(len(self.all_goals)), [0 for _ in range(len(self.all_goals))]))
        self.target_counters = dict(zip(range(len(self.all_goals)), [0 for _ in range(len(self.all_goals))]))

        if self.curriculum_learning:
            # initialize deques of successes and failures for all goals
            self.successes_and_failures = []
            # fifth bucket contains not discovered goals
            self.LP = np.zeros([self.num_buckets])
            self.C = np.zeros([self.num_buckets])
            self.p = np.ones([self.num_buckets]) / self.num_buckets
            if self.automatic_buckets:
                self.discovered_goals = []
                self.discovered_goals_str = []
                self.discovered_goals_oracle_id = []

                # initialize empty buckets, the last one contains all
                self.buckets = dict(zip(range(self.num_buckets), [[] for _ in range(self.num_buckets)]))
            else:
                self.buckets = dict()
                for k in list(buckets.keys()):
                    self.buckets[k] = [self.g_str_to_oracle_id[str(np.array(g))] for g in buckets[k]]
                self.discovered_goals_oracle_id = []
                for k in self.buckets.keys():
                    self.discovered_goals_oracle_id += self.buckets[k]
                self.discovered_goals = self.all_goals[np.array(self.discovered_goals_oracle_id)].tolist()
                self.discovered_goals_str = [str(np.array(g)) for g in self.discovered_goals]

        else:
            self.discovered_goals = []
            self.discovered_goals_str = []
            self.discovered_goals_oracle_id = []

        self.init_stats()

    def sample_goal(self, n_goals, evaluation):
        """
        Sample n_goals goals to be targeted during rollouts
        evaluation controls whether or not to sample the goal uniformly or according to curriculum
        """
        if evaluation:
            goals = np.random.choice(self.valid_goals, size=self.num_rollouts_per_mpi)
            self_eval = False
        else:
            # if no goal has been discovered or if not all buckets are filled in the case of automatic buckets
            cond1 = (self.curriculum_learning and self.automatic_buckets and len(self.discovered_goals) < self.num_buckets)
            if len(self.discovered_goals) == 0 or cond1:
                # sample randomly in the goal space
                goals = np.random.randint(0, 2, size=(n_goals, self.goal_dim)).astype(np.float32)
                self_eval = False

            # if goals have been discovered
            else:
                # if no curriculum learning
                if not self.curriculum_learning:
                    # sample uniformly from discovered goals
                    goal_ids = np.random.choice(range(len(self.discovered_goals)), size=n_goals)
                    goals = np.array(self.discovered_goals)[goal_ids]
                    self_eval = False
                else:
                    # decide whether to self evaluate
                    self_eval = True if np.random.random() < self.self_eval_prob else False
                    # if self-evaluation then sample randomly from discovered goals
                    if self_eval:
                        buckets = np.random.choice(range(self.num_buckets), size=n_goals)
                    # if no self evaluation
                    else:
                        buckets = np.random.choice(range(self.num_buckets), p=self.p, size=n_goals)
                    goals = []
                    for i_b, b in enumerate(buckets):
                        goals.append(self.all_goals[np.random.choice(self.buckets[b])])
                    goals = np.array(goals)
        return goals, self_eval

    def update(self, episodes, t):
        """
        Update discovered goals list from episodes
        Update list of successes and failures for LP curriculum
        Label each episode with the last ag (for buffer storage)
        """
        all_episodes = MPI.COMM_WORLD.gather(episodes, root=0)

        if self.rank == 0:
            all_episode_list = []
            for eps in all_episodes:
                all_episode_list += eps

            for e in all_episode_list:
                reached_oracle_id = self.g_str_to_oracle_id[str(e['ag_binary'][-1])]
                target_oracle_id = self.g_str_to_oracle_id[str(e['g_binary'][0])]
                self.rew_counters[reached_oracle_id] += 1
                self.target_counters[target_oracle_id] += 1
            # find out if new goals were discovered
            # label each episode with the oracle id of the last ag (to know where to store it in buffers)
            if not self.curriculum_learning or self.automatic_buckets:
                new_goal_found = False
                for e in all_episode_list:

                    id_ag_end = self.g_str_to_oracle_id[str(e['ag_binary'][-1])]

                    if str(e['ag_binary'][-1]) not in self.discovered_goals_str:
                        if str(e['ag_binary'][-1]) not in self.valid_goals_str:
                            stop = 1
                        else:
                            new_goal_found = True
                            self.discovered_goals.append(e['ag_binary'][-1].copy())
                            self.discovered_goals_str.append(str(e['ag_binary'][-1]))
                            self.discovered_goals_oracle_id.append(id_ag_end)

                # update buckets
                if self.automatic_buckets and new_goal_found:
                    self.update_buckets()
                    self.update_LP()

            if self.curriculum_learning:
                # update list of successes and failures
                for e in all_episode_list:
                    if e['self_eval']:
                        oracle_id = self.g_str_to_oracle_id[str(e['g_binary'][0])]
                        if not self.continuous and str(e['g_binary'][0]) == str(e['ag_binary'][-1]):
                            success = 1
                        elif self.continuous and (e['rewards'][-1] == 3.):
                            success = 1
                        else:
                            success = 0
                        if oracle_id in self.discovered_goals_oracle_id:
                            self.successes_and_failures.append([t, success, oracle_id])
        self.sync()
        for e in episodes:
            last_ag = e['ag_binary'][-1]
            oracle_id = self.g_str_to_oracle_id[str(last_ag)]
            e['last_ag_oracle_id'] = oracle_id

        return episodes

    def update_buckets(self):
        """
        Dispatch the discovered goals in the buckets chronologically
        """
        discovered = np.array(self.discovered_goals_oracle_id).copy()

        j = 0
        portion_length = len(discovered) // self.num_buckets
        k = len(discovered) %  self.num_buckets
        for i in range(self.num_buckets):
            if k > 0:
                l = portion_length + 1
                k -= 1
            else:
                l = portion_length
            self.buckets[i] = discovered[j:j + l].tolist()
            j += l

    def update_LP(self):

        if len(self.successes_and_failures) > 0:
            # organize the successes and failures per bucket
            succ_fail_per_bucket = [[] for _ in range(self.num_buckets)]
            for k in self.buckets.keys():
                for sf in np.flip(self.successes_and_failures, axis=0):
                    end = sf[-1]
                    if end in self.buckets[k]:
                        succ_fail_per_bucket[k].append(sf[:2])
                        if len(succ_fail_per_bucket[k]) == self.queue_length:
                            break

            # compute C, LP per bucket
            for k in self.buckets.keys():
                n_points = len(succ_fail_per_bucket[k])
                if n_points > 100:
                    sf = np.array(succ_fail_per_bucket[k])
                    self.C[k] = np.mean(sf[n_points // 2:, 1])
                    self.LP[k] = np.abs(np.sum(sf[n_points // 2:, 1]) - np.sum(sf[: n_points // 2, 1])) / n_points
                else:
                    self.C[k] = 0
                    self.LP[k] = 0

            # compute p
            if np.sum((1 - self.C) * self.LP) == 0:
                self.p = np.ones([self.num_buckets]) / self.num_buckets
            else:
                self.p = (1 - self.C) * self.LP / np.sum((1 - self.C) * self.LP)
                # self.p = self.LP / self.LP.sum()
                # self.p = self.epsilon * (1 - self.C) / (1 - self.C).sum() + (1 - self.epsilon) * self.LP / self.LP.sum()

            if self.p.sum() > 1:
                self.p[np.argmax(self.p)] -= self.p.sum() - 1
            elif self.p.sum() < 1:
                self.p[-1] = 1 - self.p[:-1].sum()

    def sync(self):
        if self.curriculum_learning:
            self.p = MPI.COMM_WORLD.bcast(self.p, root=0)
            self.LP = MPI.COMM_WORLD.bcast(self.LP, root=0)
            self.C = MPI.COMM_WORLD.bcast(self.C, root=0)
            self.buckets = MPI.COMM_WORLD.bcast(self.buckets, root=0)
        self.discovered_goals = MPI.COMM_WORLD.bcast(self.discovered_goals, root=0)
        self.discovered_goals_str = MPI.COMM_WORLD.bcast(self.discovered_goals_str, root=0)
        self.discovered_goals_oracle_id = MPI.COMM_WORLD.bcast(self.discovered_goals_oracle_id, root=0)

    def build_batch(self, batch_size):
        # only consider buckets filled with discovered goals
        if self.curriculum_learning:
            LP = self.LP
            C = self.C
            if np.sum((1 - C) * LP) == 0:
                p = np.ones([self.num_buckets]) / self.num_buckets
            else:
                p = (1 - C) * LP / np.sum((1 - C) * LP)
                # p = self.epsilon * np.ones([self.num_buckets]) / self.num_buckets + (1 - self.epsilon) * LP / LP.sum()
                # p = (1 - np.power(C, beta)) * np.power(LP, nu) / np.sum((1 - np.power(C, beta)) * np.power(LP, nu))
                # p = self.epsilon * (1 - C) / (1 - C).sum() + (1 - self.epsilon) * LP / LP.sum()
            if p.sum() > 1:
                p[np.argmax(p)] -= p.sum() - 1
            elif p.sum() < 1:
                p[-1] = 1 - p[:-1].sum()
            buckets = np.random.choice(range(self.num_buckets), p=p, size=batch_size)
            # buckets = np.random.choice(range(self.num_buckets), p=p) * np.ones(batch_size)
            goal_ids = []
            for b in buckets:
                if len(self.buckets[b]) > 0:
                    goal_ids.append(np.random.choice(self.buckets[b]))
                else:
                    goal_ids.append(3000) # this will lead the buffer to sample a random episode
            assert len(goal_ids) == batch_size
        else:
            goal_ids = np.random.choice(self.discovered_goals_oracle_id, size=batch_size)
        return goal_ids

    def init_stats(self):
        self.stats = dict()
        for i in range(self.valid_goals.shape[0]):
            self.stats['{}_in_bucket'.format(i)] = []  # track the bucket allocation of each valid goal
            self.stats['Eval_SR_{}'.format(i)] = []  # track the offline success rate of each valid goal
            self.stats['Av_Rew_{}'.format(i)] = []  # track the offline average reward of each valid goal
            self.stats['#Rew_{}'.format(i)] = []  # track the number of rewards obtained by each valid goal (last_ag = g)
            self.stats['#Target_{}'.format(i)] = []  # track the number of times each goal was used as a target.

        for i in range(self.num_buckets):
            self.stats['B_{}_LP'.format(i)] = []
            self.stats['B_{}_C'.format(i)] = []
            self.stats['B_{}_p'.format(i)] = []
        self.stats['epoch'] = []
        self.stats['episodes'] = []
        self.stats['global_sr'] = []
        self.stats['nb_discovered'] = []
        keys = ['goal_sampler', 'rollout', 'gs_update', 'store', 'norm_update',
                  'policy_train', 'lp_update', 'eval', 'epoch', 'total']
        for k in keys:
            self.stats['t_{}'.format(k)] = []

    def save(self, epoch, episode_count, av_res, av_rew, global_sr, time_dict):
        self.stats['epoch'].append(epoch)
        self.stats['episodes'].append(episode_count)
        self.stats['global_sr'].append(global_sr)
        for k in time_dict.keys():
            self.stats['t_{}'.format(k)].append(time_dict[k])
        self.stats['nb_discovered'].append(len(self.discovered_goals_oracle_id))
        for g_id, oracle_id in enumerate(self.valid_goals_oracle_ids):
            if self.curriculum_learning:
                found = False
                for k in self.buckets.keys():
                    if oracle_id in self.buckets[k]:
                        self.stats['{}_in_bucket'.format(g_id)].append(k)
                        found = True
                        break
                if not found:
                    self.stats['{}_in_bucket'.format(g_id)].append(np.nan)
            else:
                # set bucket 1 if discovered, 0 otherwise
                if oracle_id in self.discovered_goals_oracle_id:
                    self.stats['{}_in_bucket'.format(g_id)].append(1)
                else:
                    self.stats['{}_in_bucket'.format(g_id)].append(0)
            self.stats['Eval_SR_{}'.format(g_id)].append(av_res[g_id])
            self.stats['Av_Rew_{}'.format(g_id)].append(av_rew[g_id])
            self.stats['#Rew_{}'.format(g_id)].append(self.rew_counters[oracle_id])
            self.stats['#Target_{}'.format(g_id)].append(self.target_counters[oracle_id])

        for i in range(self.num_buckets):
            self.stats['B_{}_LP'.format(i)].append(self.LP[i])
            self.stats['B_{}_C'.format(i)].append(self.C[i])
            self.stats['B_{}_p'.format(i)].append(self.p[i])

    def save_bucket_contents(self, bucket_path, epoch):
        # save the contents of buckets
        if self.curriculum_learning and self.automatic_buckets:
            with open(bucket_path + '/bucket_ep_{}.pkl'.format(epoch), 'wb') as f:
                pickle.dump(self.buckets, f)
