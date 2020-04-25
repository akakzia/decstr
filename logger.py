import torch
from mpi4py import MPI
from datetime import datetime
import os
import pickle as pkl


def log_results(agent, epoch, res, evaluations=True, frequency=10, store_model=True, store_stats=False):
    # start to do the evaluation
    if agent.args.curriculum_learning:
        print('[{}] Epoch: {} |==============================|'.format(datetime.now(), epoch))
        for i in range(len(agent.buckets)):
            print('Bucket ', i, '| p = {:.3f} | LP = {:.3f}| C = {:.3f}'.format(agent.p[i], agent.CP[i], agent.C[i]))
        if epoch % frequency == 0 and evaluations:
            print('===========================================')
            for i in range(agent.num_buckets):
                print('Bucket', i, 'eval success rate is: {:.3f}'.format(res[i]))
            print('===========================================')
        if epoch % frequency == 0 and store_stats:
            torch.save([agent.p, agent.C, agent.CP], agent.eval_path + '/LP_{}.pt'.format(epoch))

    else:
        print('[{}] Epoch: {} |==============================|'.format(datetime.now(), epoch))
        if epoch % frequency == 0 and evaluations:
            if epoch % frequency == 0 and evaluations:
                print('===========================================')
                for i in range(agent.num_buckets):
                    print('Bucket', i, 'eval success rate is: {:.3f}'.format(res[i]))
                print('===========================================')

    # Store model
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
            if agent.args.deepsets_attention and not agent.args.double_critic_attention:
                torch.save([agent.o_norm.mean, agent.o_norm.std, agent.g_norm.mean, agent.g_norm.std,
                            agent.model.single_phi_actor.state_dict(), agent.model.single_phi_critic.state_dict(),
                            agent.model.rho_actor.state_dict(), agent.model.rho_critic.state_dict(),
                            agent.model.attention_actor.state_dict(), agent.model.attention_critic_1.state_dict()],
                           agent.model_path + '/model_{}.pt'.format(epoch))
            elif agent.args.deepsets_attention and agent.args.double_critic_attention:
                torch.save([agent.o_norm.mean, agent.o_norm.std, agent.g_norm.mean, agent.g_norm.std,
                            agent.model.single_phi_actor.state_dict(), agent.model.single_phi_critic.state_dict(),
                            agent.model.rho_actor.state_dict(), agent.model.rho_critic.state_dict(),
                            agent.model.attention_actor.state_dict(), agent.model.attention_critic_1.state_dict(),
                            agent.model.attention_critic_2.state_dict()],
                           agent.model_path + '/model_{}.pt'.format(epoch))
            else:
                torch.save([agent.o_norm.mean, agent.o_norm.std, agent.g_norm.mean, agent.g_norm.std,
                            agent.model.single_phi_actor.state_dict(), agent.model.single_phi_critic.state_dict(),
                            agent.model.rho_actor.state_dict(), agent.model.rho_critic.state_dict()],
                           agent.model_path + '/model_{}.pt'.format(epoch))
        else:
            raise NotImplementedError

    # Store stats
    if epoch % frequency == 0 and store_stats:
        with open(os.path.join(agent.eval_path, 'evaluations.pkl'), 'wb') as f:
            pkl.dump(agent.overall_stats, f)
