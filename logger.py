import torch
from mpi4py import MPI
from datetime import datetime
import os
import pickle as pkl


def log_results(agent, epoch, res, evaluations=True, frequency=10, store_model=True, store_stats=False, separate_goals=False):
    # start to do the evaluation
    """if epoch % frequency == 0 and evaluations:
        res = agent._eval_agent(curriculum=agent.args.curriculum_learning, separate_goals=separate_goals)
        # success_rate = agent._eval_agent()
        overall_stats.append(res)"""
    if MPI.COMM_WORLD.Get_rank() == 0:
        if agent.args.curriculum_learning:
            print('[{}] Epoch: {} |==============================|'.format(datetime.now(), epoch))
            for i in range(len(agent.goals)):
                print('Bucket ', i, '| p = {:.3f} | LP = {:.3f}| C = {:.3f}'.format(agent.p[i], agent.CP[i], agent.C[i]))
            if epoch % frequency == 0 and evaluations:
                print('===========================================')
                for i in range(agent.num_buckets):
                    print('Bucket', i, 'eval success rate is: {:.3f}'.format(res[i]))
                print('===========================================')
            if epoch % frequency == 0 and store_stats:
                torch.save([agent.p, agent.C, agent.CP], agent.eval_path + '/LP_{}.pt'.format(epoch))
        elif not separate_goals and epoch % frequency == 0:
            print('[{}] epoch is: {}, eval success rate is: {:.3f}'.format(datetime.now(), epoch, res))
        else:
            print('[{}] Epoch: {} |==============================|'.format(datetime.now(), epoch))
            if epoch % frequency == 0 and evaluations:
                for goal in res[0].keys():
                    print('Goal: {}, eval success rate is: {:.3f}'.format(goal, res[0][goal]))
        if epoch % frequency == 0 and store_model:
            if agent.args.architecture == 'flat':
                torch.save([agent.o_norm.mean, agent.o_norm.std, agent.g_norm.mean, agent.g_norm.std,
                            agent.actor_network.state_dict(), agent.critic_network.state_dict()],
                           agent.model_path + '/model_{}.pt'.format(epoch))
            elif agent.args.architecture == 'disentangled':
                torch.save([agent.o_norm.mean, agent.o_norm.std, agent.g_norm.mean, agent.g_norm.std,
                            agent.actor_network.state_dict(), agent.critic_network.state_dict(),
                            agent.configuration_network.state_dict()],
                           agent.model_path + '/model_{}.pt'.format(epoch))
            elif agent.args.architecture == 'deepsets':
                torch.save([agent.o_norm.mean, agent.o_norm.std, agent.g_norm.mean, agent.g_norm.std,
                            agent.model.single_phi_actor.state_dict(), agent.model.single_phi_critic.state_dict(),
                            agent.model.rho_actor.state_dict(), agent.model.rho_critic.state_dict(),
                            agent.model.attention.state_dict()],
                           agent.model_path + '/model_{}.pt'.format(epoch))
            else:
                raise NotImplementedError
        if epoch % frequency == 0 and store_stats:
            with open(os.path.join(agent.eval_path, 'evaluations.pkl'), 'wb') as f:
                pkl.dump(agent.overall_stats, f)
