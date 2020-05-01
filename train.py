import numpy as np
import env
import gym
import os, sys
from arguments import get_args
from mpi4py import MPI
from rl_modules.sac_agent2 import SACAgent
import random
import torch
from rollout import RolloutWorker
from goal_sampler import GoalSampler
from utils import init_storage
from mpi_utils.mpi_utils import fork
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
    logdir, model_path = init_storage(args)
    if rank == 0:
        logger.configure(dir=logdir)

    args.env_params = get_env_params(env)

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
        if rank==0: logger.info('\n\nEpoch #{}'.format(epoch))
        for _ in range(args.n_cycles):
            print(rank)
            # sample goal
            goals, self_eval = goal_sampler.sample_goal(n_goals=args.num_rollouts_per_mpi, evaluation=False)

            # collect episodes
            episodes = rollout_worker.generate_rollout(goals=goals,
                                                       self_eval=self_eval,
                                                       true_eval=False)

            # update goal sampler (add new discovered goals to the list
            # label episodes with the id of the last ag
            episodes = goal_sampler.update(episodes)

            # store episodes
            policy.store(episodes)

            # update normalizer
            for e in episodes:
                policy._update_normalizer(e)

            # train policy
            for _ in range(args.n_batches):
                policy.train()

            episode_count += args.num_rollouts_per_mpi * args.num_workers

        t_epoch = time.time() - t_init
        t_total = time.time() - t_total_init
        if args.evaluations:
            if rank==0: logger.info('\tRunning eval ..')
            eval_goals = goal_sampler.valid_goals
            episodes = rollout_worker.generate_rollout(goals=eval_goals,
                                                       self_eval=True,
                                                       true_eval=True)

            results = np.array([str(e['g'][0]) == str(e['ag'][-1]) for e in episodes]).astype(np.int)
            all_results = MPI.COMM_WORLD.gather(results, root=0)
            if rank == 0:
                av_res = np.array(all_results).mean(axis=0)
                global_sr = np.mean(av_res)
                log_and_save(logdir, goal_sampler, epoch, episode_count, av_res, global_sr, t_epoch, t_total)
                if epoch % args.save_freq == 0:
                    policy.save(model_path, epoch)
                if rank==0: logger.info('\tEpoch #{}: SR: {}'.format(epoch, global_sr))


def log_and_save( logdir, goal_sampler, epoch, episode_count, av_res, global_sr, t_epoch, t_total):
    goal_sampler.save(logdir, epoch, episode_count, av_res, global_sr, t_epoch, t_total)
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
