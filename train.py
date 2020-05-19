import numpy as np
from mpi4py import MPI
import env
import gym
import os, sys
from arguments import get_args
from rl_modules.sac_agent2 import SACAgent
import random
import torch
from rollout import RolloutWorker
from goal_sampler import GoalSampler
from utils import init_storage
import time
from mpi_utils import logger

def get_env_params(env):
    obs = env.reset()

    # close the environment
    params = {'obs': obs['observation'].shape[0], 'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0], 'action_max': env.action_space.high[0],
              'max_timesteps': env._max_episode_steps}
    return params

def launch(args):

    rank = MPI.COMM_WORLD.Get_rank()

    t_total_init = time.time()
    # Make the environment
    env = gym.make(args.env_name)

    # set random seeds for reproduce
    env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.cuda:
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())

    # get saving paths
    if rank == 0:
        logdir, model_path, bucket_path = init_storage(args)
        logger.configure(dir=logdir)

    args.env_params = get_env_params(env)

    logger.info(vars(args))
    # def goal sampler:
    goal_sampler = GoalSampler(args)

    # create the sac agent to interact with the environment
    if args.agent == "SAC":
        policy = SACAgent(args, env.compute_reward, goal_sampler)
    else:
        raise NotImplementedError

    # def rollout worker
    rollout_worker = RolloutWorker(env, policy, goal_sampler,  args)

    # start to collect samples
    episode_count = 0
    for epoch in range(args.n_epochs):
        t_init = time.time()
        # setup time_tracking
        time_dict = dict(goal_sampler=0,
                         rollout=0,
                         gs_update=0,
                         store=0,
                         norm_update=0,
                         policy_train=0,
                         lp_update=0,
                         eval=0,
                         epoch=0)

        if rank==0: logger.info('\n\nEpoch #{}'.format(epoch))
        for _ in range(args.n_cycles):
            # sample goal
            t_i = time.time()
            inits, goals, self_eval = goal_sampler.sample_goal(n_goals=args.num_rollouts_per_mpi, evaluation=False)
            time_dict['goal_sampler'] += time.time() - t_i

            # collect episodes
            t_i = time.time()
            # Add condition on number of discovered goals to make sure no stacks are discovered at early stage
            # 12 is chosen heuristically, being greater than 8 (to allow discovering close configs first, and
            # some random above configs
            if epoch < 100:# and args.automatic_buckets:
                biased_init = False
            else:
                biased_init = args.biased_init
            episodes = rollout_worker.generate_rollout(inits=inits,
                                                       goals=goals,
                                                       self_eval=self_eval,
                                                       true_eval=False,
                                                       biased_init=biased_init)
            time_dict['rollout'] += time.time() - t_i

            # update goal sampler (add new discovered goals to the list
            # label episodes with the id of the last ag
            t_i = time.time()
            episodes = goal_sampler.update(episodes, episode_count)
            time_dict['gs_update'] += time.time() - t_i

            # store episodes
            t_i = time.time()
            policy.store(episodes)
            time_dict['store'] += time.time() - t_i

            # update normalizer
            t_i = time.time()
            for e in episodes:
                policy._update_normalizer(e)
            time_dict['norm_update'] += time.time() - t_i

            # train policy
            t_i = time.time()
            for _ in range(args.n_batches):
                policy.train()
            time_dict['policy_train'] += time.time() - t_i

            episode_count += args.num_rollouts_per_mpi * args.num_workers

        t_i = time.time()
        if goal_sampler.curriculum_learning:
            goal_sampler.update_LP()
        time_dict['lp_update'] += time.time() - t_i
        time_dict['epoch'] += time.time() -t_init
        time_dict['total'] = time.time() - t_total_init

        if args.evaluations:
            t_i = time.time()
            if rank==0: logger.info('\tRunning eval ..')
            eval_goals = goal_sampler.valid_goals
            episodes = rollout_worker.generate_rollout(inits=[None] * len(eval_goals),
                                                       goals=eval_goals,
                                                       self_eval=True,
                                                       true_eval=True)

            results = np.array([str(e['g'][0]) == str(e['ag'][-1]) for e in episodes]).astype(np.int)
            all_results = MPI.COMM_WORLD.gather(results, root=0)
            time_dict['eval'] += time.time() - t_i

            if rank == 0:
                av_res = np.array(all_results).mean(axis=0)
                global_sr = np.mean(av_res)
                log_and_save(logdir, goal_sampler, epoch, episode_count, av_res, global_sr,time_dict)
                if epoch % args.save_freq == 0:
                    policy.save(model_path, epoch)
                    goal_sampler.save_bucket_contents(bucket_path, epoch)
                if rank==0: logger.info('\tEpoch #{}: SR: {}'.format(epoch, global_sr))


def log_and_save( logdir, goal_sampler, epoch, episode_count, av_res, global_sr, time_dict):
    goal_sampler.save(logdir, epoch, episode_count, av_res, global_sr, time_dict)
    for k, l in goal_sampler.stats.items():
        logger.record_tabular(k, l[-1])
    logger.dump_tabular()



if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'

    # get the params
    args = get_args()

    launch(args)
