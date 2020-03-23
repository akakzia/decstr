import torch
from mpi4py import MPI
import numpy as np
from stats import save_plot


# do the evaluation
def eval_agent(agent, curriculum=False, separate_goals=False):
    # Curriculum evaluations with buckets of ascending complexity
    if curriculum:
        stats = []
        res = []
        for i in range(agent.num_buckets):
            total_success_rate = []
            for _ in range(agent.args.n_test_rollouts):
                per_success_rate = []
                goal = agent.goals[i][np.random.choice(len(agent.goals[i]))]
                observation = agent.env.reset_goal(np.array(goal))
                obs = observation['observation']
                g = observation['desired_goal']
                ag = observation['achieved_goal']
                for _ in range(agent.env_params['max_timesteps']):
                    with torch.no_grad():
                        g_norm = torch.tensor(agent.g_norm.normalize(g), dtype=torch.float32).unsqueeze(0)
                        ag_norm = torch.tensor(agent.g_norm.normalize(ag), dtype=torch.float32).unsqueeze(0)
                        if agent.architecture == 'deepsets':
                            obs_tensor = torch.tensor(agent.o_norm.normalize(obs), dtype=torch.float32).unsqueeze(0)
                            agent.model.forward_pass(obs_tensor, ag_norm, g_norm, eval=True)
                            action = agent.model.pi_tensor.numpy()[0]
                        elif agent.architecture == 'disentangled':
                            z_ag = agent.configuration_network(ag_norm)[0]
                            z_g = agent.configuration_network(g_norm)[0]
                            input_tensor = torch.tensor(np.concatenate([agent.o_norm.normalize(obs), z_ag, z_g]),
                                                        dtype=torch.float32).unsqueeze(0)
                            action = agent._select_actions(input_tensor, eval=True)
                        else:
                            input_tensor = agent._preproc_inputs(obs, g)  # PROCESSING TO CHECK
                            action = agent._select_actions(input_tensor, eval=True)
                    observation_new, _, _, info = agent.env.step(action)
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
        agent.overall_stats.append(stats)
        if MPI.COMM_WORLD.Get_rank() == 0:
            save_plot(np.array(agent.overall_stats), agent.args)
        return res

    #
    elif separate_goals:
        goals = agent.goals
        per_goal_sr = {}
        per_goal_std = {}
        for goal in goals:
            total_success_rate = []
            for _ in range(agent.args.n_test_rollouts):
                per_success_rate = []
                observation = agent.env.reset_goal(goal)
                obs = observation['observation']
                observation['desired_goal'] = goal
                ag = observation['achieved_goal']
                for _ in range(agent.env_params['max_timesteps']):
                    with torch.no_grad():
                        if agent.architecture == 'disentangled':
                            g_norm = torch.tensor(agent.g_norm.normalize_goal(goal), dtype=torch.float32).unsqueeze(0)
                            ag_norm = torch.tensor(agent.g_norm.normalize_goal(ag), dtype=torch.float32).unsqueeze(0)
                            # config_inputs = np.concatenate([ag, g])
                            # config_inputs = torch.tensor(config_inputs, dtype=torch.float32).unsqueeze(0)
                            config_z = agent.configuration_network(ag_norm, g_norm)[0]
                            input_tensor = torch.tensor(np.concatenate([agent.o_norm.normalize(obs),
                                                                        config_z]), dtype=torch.float32).unsqueeze(0)
                        else:
                            input_tensor = agent._preproc_inputs(obs, goal)  # PROCESSING TO CHECK
                        action = agent._select_actions(input_tensor, eval=True)
                    observation_new, _, _, info = agent.env.step(action)
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

    else:
        total_success_rate = []
        for _ in range(agent.args.n_test_rollouts):
            per_success_rate = []
            #goal = agent.goals[np.random.choice(len(agent.goals))]
            observation = agent.env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            ag = observation['achieved_goal']
            for _ in range(agent.env_params['max_timesteps']):
                with torch.no_grad():
                    if agent.architecture == 'disentangled':
                        g_norm = torch.tensor(agent.g_norm.normalize(g), dtype=torch.float32).unsqueeze(0)
                        ag_norm = torch.tensor(agent.g_norm.normalize(ag), dtype=torch.float32).unsqueeze(0)
                        # config_inputs = np.concatenate([ag, g])
                        # config_inputs = torch.tensor(config_inputs, dtype=torch.float32).unsqueeze(0)
                        #config_z = agent.configuration_network(ag_norm, g_norm)[0]
                        z_ag = agent.configuration_network(ag_norm)[0]
                        z_g = agent.configuration_network(g_norm)[0]
                        input_tensor = torch.tensor(np.concatenate([agent.o_norm.normalize(obs),
                                                                    z_ag, z_g]), dtype=torch.float32).unsqueeze(0)
                    else:
                        input_tensor = agent._preproc_inputs(obs, g)  # PROCESSING TO CHECK
                    action = agent._select_actions(input_tensor, eval=True)
                observation_new, _, _, info = agent.env.step(action)
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                per_success_rate.append(info['is_success'])
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        return global_success_rate / MPI.COMM_WORLD.Get_size()