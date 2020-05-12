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

        self.curriculum_learning = args.curriculum_learning
        self.automatic_buckets = args.automatic_buckets
        self.num_buckets = args.num_buckets
        self.queue_length = 2 * args.queue_length
        self.rank = MPI.COMM_WORLD.Get_rank()
        self.use_pairs = args.use_pairs

        self.epsilon = args.curriculum_eps

        buckets = generate_goals(nb_objects=3, sym=1, asym=1)
        # If no curriculum is used then 0 buckets
        # If no automatic buckets, then number of buckets = number of predefined buckets
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
                self.discovered_pairs_oracle_ids = []

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
                self.discovered_pairs_oracle_ids = []

        else:
            self.discovered_goals = []
            self.discovered_goals_str = []
            self.discovered_goals_oracle_id = []
            self.discovered_pairs_oracle_ids = []

        self.init_stats()
        stop = 1

    def sample_goal(self, n_goals, evaluation):

        inits = [None] * n_goals
        if evaluation:
            goals = np.random.choice(self.valid_goals, size=2)
            self_eval = False
        else:
            # if no goal has been discovered or if not all buckets are filled in the case of automatic buckets
            cond1 = (self.curriculum_learning and self.automatic_buckets and len(self.discovered_goals) < self.num_buckets)
            cond2 = (self.curriculum_learning and self.automatic_buckets and len(self.discovered_pairs_oracle_ids) < self.num_buckets)
            if self.use_pairs:
                cond = cond2
            else:
                cond = cond1
            if len(self.discovered_goals) == 0 or cond:
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
                    self_eval = True if np.random.random() < 0.1 else False
                    # if self-evaluation then sample randomly from discovered goals
                    if self_eval:
                        # goal_ids = np.random.choice(range(len(self.discovered_goals)), size=n_goals)
                        # goals = np.array(self.discovered_goals)[goal_ids]
                        buckets = np.random.choice(range(self.num_buckets), size=n_goals)
                    # if no self evaluation
                    else:
                        buckets = np.random.choice(range(self.num_buckets), p=self.p, size=n_goals)
                    goals = []
                    for i_b, b in enumerate(buckets):
                        if self.use_pairs:
                            ind = np.random.choice(range(len(self.buckets[b])))
                            bucket = self.buckets[b][ind]
                            inits[i_b] = self.all_goals[bucket[0]]
                            goals.append(self.all_goals[bucket[1]])
                        else:
                            goals.append(self.all_goals[np.random.choice(self.buckets[b])])
                    goals = np.array(goals)
        return inits, goals, self_eval


    def update(self, episodes, t):
        all_episodes = MPI.COMM_WORLD.gather(episodes, root=0)

        if self.rank == 0:
            all_episode_list = []
            for eps in all_episodes:
                all_episode_list += eps
            # logger.info('Len eps' + str(len(all_episode_list)))
            # find out if new goals were discovered
            # label each episode with the oracle id of the last ag (to know where to store it in buffers)
            if not self.curriculum_learning or self.automatic_buckets:
                for e in all_episode_list:
                    # if we're looking for pairs
                    id_ag_0 = self.g_str_to_oracle_id[str(e['ag'][0])]

                    for ag in e['ag']:
                        # check if ag is a new goal
                        if str(ag) not in self.discovered_goals_str:
                            if str(ag) not in self.valid_goals_str:
                                stop = 1
                            # it is, update info
                            else:
                                self.discovered_goals.append(ag.copy())
                                self.discovered_goals_str.append(str(ag))
                                self.discovered_goals_oracle_id.append(self.g_str_to_oracle_id[str(ag)])

                        # update discovered pairs
                        id_ag = self.g_str_to_oracle_id[str(ag)]
                        if id_ag_0 != id_ag and [id_ag_0, id_ag] not in self.discovered_pairs_oracle_ids:
                            self.discovered_pairs_oracle_ids.append([id_ag_0, id_ag])

                # update buckets
                if self.automatic_buckets:
                    self.update_buckets()

            if self.curriculum_learning:
                # update list of successes and failures
                for e in all_episode_list:
                    if e['self_eval']:
                        oracle_id_init = self.g_str_to_oracle_id[str(e['ag'][0])]
                        oracle_id = self.g_str_to_oracle_id[str(e['g'][0])]
                        if str(e['g'][0]) == str(e['ag'][-1]):
                            success = 1
                        else:
                            success = 0
                        if self.use_pairs:
                            if [oracle_id_init, oracle_id] in self.discovered_pairs_oracle_ids:
                                self.successes_and_failures.append([t, success, oracle_id_init, oracle_id])
                        else:
                            if oracle_id in self.discovered_goals_oracle_id:
                                self.successes_and_failures.append([t, success, oracle_id])

        self.sync()
        for e in episodes:
            last_ag = e['ag'][-1]
            oracle_id = self.g_str_to_oracle_id[str(last_ag)]
            e['last_ag_oracle_id'] = oracle_id
            # g = e['g'][0]
            # oracle_id = self.g_str_to_oracle_id[str(g)]
            # e['g_oracle_id'] = oracle_id

        return episodes

    def update_buckets(self):

        if self.use_pairs:
            discovered = np.array(self.discovered_pairs_oracle_ids).copy()
        else:
            discovered = np.array(self.discovered_goals_oracle_id).copy()

        # Dispatch the discovered goals (or pairs) in the buckets chronologically
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
                    if self.use_pairs:
                        end = sf[2:].tolist()
                    else:
                        end = sf[-1]
                    if end in self.buckets[k]:
                        succ_fail_per_bucket[k].append(sf[:2])
                        if len(succ_fail_per_bucket[k]) == self.queue_length:
                            break

            # compute C, LP per bucket
            for k in self.buckets.keys():
                n_points = len(succ_fail_per_bucket[k])
                if n_points > 4:
                    sf = np.array(succ_fail_per_bucket[k])
                    self.C[k] = np.mean(sf[n_points // 2:, 1])
                    self.LP[k] = np.abs(np.sum(sf[n_points // 2:, 1]) - np.sum(sf[: n_points // 2, 1])) / n_points
                    # self.LP[k] = np.abs(np.mean(sf[n_points // 2:, 1]) - np.mean(sf[: n_points // 2, 1]))

            # compute p
            if self.LP.sum() == 0:
                self.p = np.ones([self.num_buckets]) / self.num_buckets
            else:
                # self.p = self.LP / self.LP.sum()
                self.p = self.epsilon * (1 - self.C) / (1 - self.C).sum() + (1 - self.epsilon) * self.LP / self.LP.sum()

            if self.p.sum() > 1:
                self.p[np.argmax(self.p)] -= self.p.sum() - 1
            elif self.p.sum() < 1:
                self.p[-1] = 1 - self.p[:-1].sum()

    def sync(self):
        # logger.info('R{}be, p{}'.format(self.rank, self.p))
        # logger.info('R{}be, dg{}'.format(self.rank, self.discovered_goals_str))
        # logger.info('R{}be, b{}, {}, {}, {}'.format(self.rank, self.buckets[0], self.buckets[1], self.buckets[2], self.buckets[3]))
        if self.curriculum_learning:
            self.p = MPI.COMM_WORLD.bcast(self.p, root=0)
            self.LP = MPI.COMM_WORLD.bcast(self.LP, root=0)
            self.C = MPI.COMM_WORLD.bcast(self.C, root=0)
            self.buckets = MPI.COMM_WORLD.bcast(self.buckets, root=0)
        self.discovered_goals = MPI.COMM_WORLD.bcast(self.discovered_goals, root=0)
        self.discovered_goals_str = MPI.COMM_WORLD.bcast(self.discovered_goals_str, root=0)
        self.discovered_goals_oracle_id = MPI.COMM_WORLD.bcast(self.discovered_goals_oracle_id, root=0)
        # logger.info('R{}af, p{}'.format(self.rank, self.p))
        # logger.info('R{}af, b{}, {}, {}, {}'.format(self.rank, self.buckets[0], self.buckets[1], self.buckets[2], self.buckets[3]))
        # logger.info('R{}af, dg{}'.format(self.rank, self.discovered_goals_str))

    def build_batch(self, batch_size):
        # only consider buckets filled with discovered goals
        LP = self.LP
        C = self.C
        if LP.sum() == 0:
            p = np.ones([self.num_buckets]) / self.num_buckets
        else:
            # p = self.epsilon * np.ones([self.num_buckets]) / self.num_buckets + (1 - self.epsilon) * LP / LP.sum()
            p = self.epsilon * (1 - C) / (1 - C).sum() + (1 - self.epsilon) * LP / LP.sum()
        if p.sum() > 1:
            p[np.argmax(self.p)] -= p.sum() - 1
        elif p.sum() < 1:
            p[-1] = 1 - p[:-1].sum()
        buckets = np.random.choice(range(self.num_buckets), p=p, size=batch_size)
        # buckets = np.random.choice(range(self.num_buckets), p=p) * np.ones(batch_size)
        goal_ids = []
        for b in buckets:
            if len(self.buckets[b]) > 0:
                if self.use_pairs:
                    goal_ids.append(np.random.choice(np.array(self.buckets[b])[:, 1]))
                else:
                    goal_ids.append(np.random.choice(self.buckets[b]))
            else:
                goal_ids.append(3000) # this will lead the buffer to sample a random episode
        assert len(goal_ids) == batch_size
        return goal_ids

    def init_stats(self):
        self.stats = dict()
        for i in range(self.valid_goals.shape[0]):
            self.stats['{}_in_bucket'.format(i)] = []
            self.stats['Eval_SR_{}'.format(i)] = []
        for i in range(self.num_buckets):
            self.stats['B_{}_LP'.format(i)] = []
            self.stats['B_{}_C'.format(i)] = []
            self.stats['B_{}_p'.format(i)] = []
        self.stats['epoch'] = []
        self.stats['episodes'] = []
        self.stats['z_global_sr'] = []
        self.stats['nb_discovered'] = []
        keys = ['goal_sampler', 'rollout', 'gs_update', 'store', 'norm_update',
                  'policy_train', 'lp_update', 'eval', 'epoch', 'total']
        for k in keys:
            self.stats['t_{}'.format(k)] = []

    def save(self, eval_path, epoch, episode_count, av_res, global_sr, time_dict):
        self.stats['epoch'].append(epoch)
        self.stats['episodes'].append(episode_count)
        self.stats['z_global_sr'].append(global_sr)
        for k in time_dict.keys():
            self.stats['t_{}'.format(k)].append(time_dict[k])
        self.stats['nb_discovered'].append(len(self.discovered_goals_oracle_id))
        for g_id, oracle_id in enumerate(self.valid_goals_oracle_ids):
            if self.curriculum_learning:
                found = False
                if not self.use_pairs:
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
        for i in range(self.num_buckets):
            self.stats['B_{}_LP'.format(i)].append(self.LP[i])
            self.stats['B_{}_C'.format(i)].append(self.C[i])
            self.stats['B_{}_p'.format(i)].append(self.p[i])

        data = pd.DataFrame(self.stats.copy())
        data.to_csv(os.path.join(eval_path, 'evaluations.csv'))

    def save_bucket_contents(self, bucket_path, epoch):
        if self.curriculum_learning:
            with open(bucket_path + '/bucket_ep_{}.pkl'.format(epoch), 'wb') as f:
                pickle.dump(self.buckets, f)
