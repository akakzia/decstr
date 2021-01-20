import numpy as np
from mpi4py import MPI
import env
import gym
import os
from arguments import get_args
from rl_modules.rl_agent import RLAgent
import random
import torch
from rollout import RolloutWorker
from temporary_lg_goal_sampler import LanguageGoalSampler
from goal_sampler import GoalSampler
from utils import init_storage, get_instruction
import time
from mpi_utils import logger
from language.build_dataset import sentence_from_configuration

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
    if args.algo == 'continuous':
        args.env_name = 'FetchManipulate3ObjectsContinuous-v0'
        args.multi_criteria_her = True
    else:
        args.env_name = 'FetchManipulate3Objects-v0'
    env = gym.make(args.env_name)

    # set random seeds for reproducibility
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
        logger.info(vars(args))

    args.env_params = get_env_params(env)


    if args.algo == 'language':
        language_goal = get_instruction()
        goal_sampler = LanguageGoalSampler(args)
    else:
        language_goal = None
        goal_sampler = GoalSampler(args)

    # Initialize RL Agent
    if args.agent == "SAC":
        policy = RLAgent(args, env.compute_reward, goal_sampler)
    else:
        raise NotImplementedError

    # Initialize Rollout Worker
    rollout_worker = RolloutWorker(env, policy, goal_sampler,  args)

    # Main interaction loop
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

        # log current epoch
        if rank == 0: logger.info('\n\nEpoch #{}'.format(epoch))

        # Cycles loop
        for _ in range(args.n_cycles):

            # Sample goals
            t_i = time.time()
            goals, self_eval = goal_sampler.sample_goal(n_goals=args.num_rollouts_per_mpi, evaluation=False)
            if args.algo == 'language':
                language_goal_ep = np.random.choice(language_goal, size=args.num_rollouts_per_mpi)
            else:
                language_goal_ep = None
            time_dict['goal_sampler'] += time.time() - t_i

            # Control biased initializations
            if epoch < args.start_biased_init:
                biased_init = False
            else:
                biased_init = args.biased_init

            # Environment interactions
            t_i = time.time()
            episodes = rollout_worker.generate_rollout(goals=goals,  # list of goal configurations
                                                       self_eval=self_eval,  # whether the agent performs self-evaluations
                                                       true_eval=False,  # these are not offline evaluation episodes
                                                       biased_init=biased_init,
                                                       language_goal=language_goal_ep)  # whether initializations should be biased.
            time_dict['rollout'] += time.time() - t_i

            # Goal Sampler updates
            t_i = time.time()
            episodes = goal_sampler.update(episodes, episode_count)
            time_dict['gs_update'] += time.time() - t_i

            # Storing episodes
            t_i = time.time()
            policy.store(episodes)
            time_dict['store'] += time.time() - t_i

            # Updating observation normalization
            t_i = time.time()
            for e in episodes:
                policy._update_normalizer(e)
            time_dict['norm_update'] += time.time() - t_i

            # Policy updates
            t_i = time.time()
            for _ in range(args.n_batches):
                policy.train()
            time_dict['policy_train'] += time.time() - t_i
            episode_count += args.num_rollouts_per_mpi * args.num_workers

        # Updating Learning Progress
        t_i = time.time()
        if goal_sampler.curriculum_learning and rank == 0:
            goal_sampler.update_LP()
        goal_sampler.sync()

        time_dict['lp_update'] += time.time() - t_i
        time_dict['epoch'] += time.time() -t_init
        time_dict['total'] = time.time() - t_total_init

        if args.evaluations:
            if rank==0: logger.info('\tRunning eval ..')
            # Performing evaluations
            t_i = time.time()
            if args.algo == 'language':
                ids = np.random.choice(np.arange(35), size=len(language_goal))
                eval_goals = goal_sampler.valid_goals[ids]
            else:
                eval_goals = goal_sampler.valid_goals
            episodes = rollout_worker.generate_rollout(goals=eval_goals,
                                                       self_eval=True,  # this parameter is overridden by true_eval
                                                       true_eval=True,  # this is offline evaluations
                                                       biased_init=False,
                                                       language_goal=language_goal)

            # Extract the results
            if args.algo == 'continuous':
                results = np.array([e['rewards'][-1] == 3. for e in episodes]).astype(np.int)
            elif args.algo == 'language':
                results = np.array([e['language_goal'] in sentence_from_configuration(config=e['ag'][-1], all=True) for e in episodes]).astype(np.int)
            else:
                results = np.array([str(e['g'][0]) == str(e['ag'][-1]) for e in episodes]).astype(np.int)
            rewards = np.array([e['rewards'][-1] for e in episodes])
            all_results = MPI.COMM_WORLD.gather(results, root=0)
            all_rewards = MPI.COMM_WORLD.gather(rewards, root=0)
            time_dict['eval'] += time.time() - t_i

            # Logs
            if rank == 0:
                assert len(all_results) == args.num_workers  # MPI test
                av_res = np.array(all_results).mean(axis=0)
                av_rewards = np.array(all_rewards).mean(axis=0)
                global_sr = np.mean(av_res)
                log_and_save(goal_sampler, epoch, episode_count, av_res, av_rewards, global_sr, time_dict)

                # Saving policy models
                if epoch % args.save_freq == 0:
                    policy.save(model_path, epoch)
                    goal_sampler.save_bucket_contents(bucket_path, epoch)
                if rank==0: logger.info('\tEpoch #{}: SR: {}'.format(epoch, global_sr))


def log_and_save( goal_sampler, epoch, episode_count, av_res, av_rew, global_sr, time_dict):
    goal_sampler.save(epoch, episode_count, av_res, av_rew, global_sr, time_dict)
    for k, l in goal_sampler.stats.items():
        logger.record_tabular(k, l[-1])
    logger.dump_tabular()


if __name__ == '__main__':
    # Prevent hyperthreading between MPI processes
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'

    # Get parameters
    args = get_args()

    launch(args)
