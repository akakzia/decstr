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

        self.epsilon = 0.2

        buckets = generate_goals(nb_objects=3, sym=1, asym=1)
        all_goals = generate_all_goals_in_goal_space().astype(np.float32)
        valid_goals = []
        for k in buckets.keys():
            if k < 4:
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
            self.successes_and_failures = [deque(maxlen=self.queue_length) for _ in range(self.num_goals)]
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
                self.buckets = buckets
        else:
            self.discovered_goals = []
            self.discovered_goals_str = []
            self.discovered_goals_oracle_id = []

        self.init_stats()

        stop = 1

    def sample_goal(self, n_goals, evaluation):

        if evaluation:
            goals = np.random.choice(self.valid_goals, size=2)
            self_eval = False
        else:
            # if no goal has been discovered or if not all buckets are filled in the case of automatic buckets
            if len(self.discovered_goals) == 0 or (self.curriculum_learning and self.automatic_buckets and len(self.discovered_goals) < self.num_buckets):
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
                        goal_ids = np.random.choice(range(len(self.discovered_goals)), size=n_goals)
                        goals = np.array(self.discovered_goals)[goal_ids]
                    # if no self evaluation
                    else:
                        buckets = np.random.choice(range(self.num_buckets), p=self.p, size=n_goals)
                        goals = []
                        for b in buckets:
                            goals.append(self.all_goals[np.random.choice(self.buckets[b])])
                        goals = np.array(goals)
        return goals, self_eval


    def update(self, episodes):
        if self.curriculum_learning:

            all_episodes = MPI.COMM_WORLD.gather(episodes, root=0)

            if self.rank == 0:
                all_episode_list = []
                for eps in all_episodes:
                    all_episode_list += eps
                # logger.info('Len eps' + str(len(all_episode_list)))
                # find out if new goals were discovered
                # label each episode with the oracle id of the last ag (to know where to store it in buffers)
                for e in episodes:
                    for ag in e['ag']:
                        # check if ag is a new goal
                        if str(ag) not in self.discovered_goals_str:
                            if str(ag) not in self.valid_goals_str:
                                stop = 1
                            # it is, update info
                            self.discovered_goals.append(ag.copy())
                            self.discovered_goals_str.append(str(ag))
                            self.discovered_goals_oracle_id.append(self.g_str_to_oracle_id[str(ag)])

                # update buckets
                if self.automatic_buckets:
                    self.update_buckets()

                # update LP
                update = False
                for e in episodes:
                    if e['self_eval']:
                        oracle_id = self.g_str_to_oracle_id[str(e['g'][0])]
                        if str(e['g'][0]) == str(e['ag'][-1]):
                            success = 1
                        else:
                            success = 0
                        self.successes_and_failures[oracle_id].append(success)
                        update = True

                # if new successes and failures, update LP, C, and p
                if update:
                    self.update_LP()

            self.sync()
            for e in episodes:
                last_ag = e['ag'][-1]
                oracle_id = self.g_str_to_oracle_id[str(last_ag)]
                e['last_ag_oracle_id'] = oracle_id
        else:
            for e in episodes:
                e['last_ag_oracle_id'] = 0

        return episodes

    def update_buckets(self):
        discovered_ids = np.array(self.discovered_goals_oracle_id).copy()

        # Dispatch the discovered goals in the buckets chronologically
        j = 0
        portion_length = len(discovered_ids) // self.num_buckets
        k = len(discovered_ids) %  self.num_buckets
        for i in range(self.num_buckets):
            if k > 0:
                l = portion_length + 1
                k -= 1
            else:
                l = portion_length
            self.buckets[i] = discovered_ids[j:j + l].tolist()
            j += l

    def update_LP(self):
        LPs = []
        Cs = []
        for s_and_f in self.successes_and_failures:
            if len(s_and_f) > 4:
                LPs.append(np.abs(np.mean(s_and_f[:len(s_and_f) // 2]) - np.mean(s_and_f[len(s_and_f) // 2:])))
                Cs.append(np.mean(s_and_f[len(s_and_f) // 2:]))
            else:
                LPs.append(0)
                Cs.append(0)
        for k in list(self.buckets.keys()):
            inds = np.array(self.buckets[k])
            self.LP[k] = np.mean(np.array(LPs)[inds])
            self.C[k] = np.mean(np.array(Cs)[inds])

        if self.LP.sum() == 0:
            self.p = np.ones([self.num_buckets]) / self.num_buckets
        else:
            self.p = self.LP / self.LP.sum()

        if self.p.sum() > 1:
            self.p[np.argmax(self.p)] -= self.p.sum() - 1
        elif self.p.sum() < 1:
            self.p[-1] = 1 - self.p[:-1].sum()

    def sync(self):
        # logger.info('R{}be, p{}'.format(self.rank, self.p))
        # logger.info('R{}be, dg{}'.format(self.rank, self.discovered_goals_str))
        # logger.info('R{}be, b{}, {}, {}, {}'.format(self.rank, self.buckets[0], self.buckets[1], self.buckets[2], self.buckets[3]))
        self.p = MPI.COMM_WORLD.bcast(self.p, root=0)
        self.LP = MPI.COMM_WORLD.bcast(self.LP, root=0)
        self.C = MPI.COMM_WORLD.bcast(self.C, root=0)
        self.discovered_goals = MPI.COMM_WORLD.bcast(self.discovered_goals, root=0)
        self.discovered_goals_str = MPI.COMM_WORLD.bcast(self.discovered_goals_str, root=0)
        self.discovered_goals_oracle_id = MPI.COMM_WORLD.bcast(self.discovered_goals_oracle_id, root=0)
        self.buckets = MPI.COMM_WORLD.bcast(self.buckets, root=0)
        # logger.info('R{}af, p{}'.format(self.rank, self.p))
        # logger.info('R{}af, b{}, {}, {}, {}'.format(self.rank, self.buckets[0], self.buckets[1], self.buckets[2], self.buckets[3]))
        # logger.info('R{}af, dg{}'.format(self.rank, self.discovered_goals_str))

    def build_batch(self, batch_size):
        # only consider buckets filled with discovered goals
        LP = self.LP[:-1]
        if LP.sum() == 0:
            p = np.ones([self.num_buckets]) / self.num_buckets
        else:
            p = self.epsilon * np.ones([self.num_buckets]) / self.num_buckets + (1 - self.epsilon) * LP / LP.sum()
        if p.sum() > 1:
            p[np.argmax(self.p)] -= p.sum() - 1
        elif p.sum() < 1:
            p[-1] = 1 - p[:-1].sum()
        buckets = np.random.choice(range(self.num_buckets), p=p, size=batch_size)
        goal_ids = []
        for b in buckets:
            if len(self.buckets[b]) > 0:
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
        self.stats['time_epoch'] = []
        self.stats['total_time'] = []
        self.stats['z_global_sr'] = []
        self.stats['nb_discovered'] = []

    def save(self, eval_path, epoch, episode_count, av_res, global_sr, t_epoch, t_total):
        self.stats['epoch'].append(epoch)
        self.stats['episodes'].append(episode_count)
        self.stats['z_global_sr'].append(global_sr)
        self.stats['time_epoch'].append(t_epoch)
        self.stats['total_time'].append(t_total)
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
        for i in range(self.num_buckets):
            self.stats['B_{}_LP'.format(i)].append(self.LP[i])
            self.stats['B_{}_C'.format(i)].append(self.C[i])
            self.stats['B_{}_p'.format(i)].append(self.p[i])

        data = pd.DataFrame(self.stats.copy())
        data.to_csv(os.path.join(eval_path, 'evaluations.csv'))

    def save_bucket_contents(self, bucket_path, epoch):
        with open(bucket_path + 'bucket_ep_{}'.format(epoch), 'wb') as f:
            pickle.dump(self.buckets, f)
