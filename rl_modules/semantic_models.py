import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from itertools import permutations
import numpy as np
from rl_modules.networks import PhiActorDeepSet, PhiCriticDeepSet, RhoActorDeepSet, RhoCriticDeepSet

epsilon = 1e-6


class SemanticCritic(nn.Module):
    def __init__(self, nb_objects, dim_phi_critic_input, dim_phi_critic_output, dim_rho_critic_input, dim_rho_critic_output, nb_permutations,
                 dim_body, dim_object, symmetry_trick, combinations_trick):
        super(SemanticCritic, self).__init__()

        self.nb_permutations = nb_permutations
        self.one_hot_encodings = [torch.tensor([1., 0., 0.]), torch.tensor([0., 1., 0.]), torch.tensor([0., 0., 1.])]
        self.nb_objects = nb_objects
        self.dim_body = dim_body
        self.dim_object = dim_object
        self.symmetry_trick = symmetry_trick
        self.first_inds = np.array([0, 1, 2, 3, 5, 7])
        self.second_inds = np.array([0, 1, 2, 4, 6, 8])
        self.combinations_trick = combinations_trick
        self.single_phi_critic = PhiCriticDeepSet(dim_phi_critic_input, 256, dim_phi_critic_output)
        self.rho_critic = RhoCriticDeepSet(dim_rho_critic_input, dim_rho_critic_output)

    def forward(self, obs, ag, g, anchor_g, act):

        batch_size = obs.shape[0]
        assert batch_size == len(ag)

        obs_body = obs[:, :self.dim_body]
        obs_objects = [torch.cat((torch.cat(batch_size * [self.one_hot_encodings[i]]).reshape(obs_body.shape[0], self.nb_objects),
                                  obs[:, self.dim_body + self.dim_object * i: self.dim_body + self.dim_object * (i + 1)]), dim=1)
                       for i in range(self.nb_objects)]

        if self.symmetry_trick:
            all_inputs = []
            for i in range(self.nb_objects):
                for j in range(self.nb_objects):
                    if i < j:
                        all_inputs.append(torch.cat([ag[:, self.first_inds], g[:, self.first_inds], obs_body, obs_objects[i], obs_objects[j],
                                                     act], dim=1))
                    elif j < i:
                        all_inputs.append(torch.cat([ag[:, self.second_inds], g[:, self.second_inds], obs_body, obs_objects[i], obs_objects[j],
                                                     act], dim=1))

            inp = torch.stack(all_inputs)

        elif self.combinations_trick:
            # Get indexes of atomic goals and corresponding object tuple
            extractors = [torch.zeros((anchor_g.shape[1], 1)) for _ in range(anchor_g.shape[1])]
            for i in range(len(extractors)):
                extractors[i][i, :] = 1.

            # The trick is to create selector matrices that, when multiplied with goals retrieves certain bits. Then the sign of the difference
            # between bits gives which objet goes above the the other

            idxs_bits = [torch.empty(anchor_g.shape[0], 2) for _ in range(3)]
            idxs_objects = [torch.empty(anchor_g.shape[0], 2) for _ in range(3)]

            for i, ((o1, o2), (j, k)) in enumerate(zip([(0, 1), (0, 2), (1, 2)], [(3, 5), (4, 7), (6, 8)])):
                stacked = torch.cat([extractors[j], extractors[k]], dim=1)
                multiplied_matrix = torch.matmul(anchor_g, stacked)
                selector = multiplied_matrix[:, 0] - multiplied_matrix[:, 1]

                idxs_bits[i] = torch.tensor([i, k]).repeat(anchor_g.shape[0], 1).long()
                idxs_bits[i][selector >= 0] = torch.Tensor([i, j]).long()

                idxs_objects[i] = torch.tensor([o2, o1]).repeat(anchor_g.shape[0], 1).long()
                idxs_objects[i][selector >= 0] = torch.Tensor([o1, o2]).long()

            # Gather 2 bits achieved goal
            ag_1_2 = ag.gather(1, idxs_bits[0])
            ag_1_3 = ag.gather(1, idxs_bits[1])
            ag_2_3 = ag.gather(1, idxs_bits[2])

            # Gather 2 bits goal
            g_1_2 = g.gather(1, idxs_bits[0])
            g_1_3 = g.gather(1, idxs_bits[1])
            g_2_3 = g.gather(1, idxs_bits[2])

            obs_object_tensor = torch.stack(obs_objects)

            obs_objects_pairs_list = []
            for idxs_objects in idxs_objects:
                permuted_idxs = idxs_objects.unsqueeze(0).permute(2, 1, 0)
                permuted_idxs = permuted_idxs.repeat(1, 1, obs_object_tensor.shape[2])
                obs_objects_pair = obs_object_tensor.gather(0, permuted_idxs)
                obs_objects_pairs_list.append(obs_objects_pair)

            input_1_2 = torch.cat([ag_1_2, g_1_2, obs_body, obs_objects_pairs_list[0][0, :, :], obs_objects_pairs_list[0][1, :, :], act], dim=1)
            input_1_3 = torch.cat([ag_1_3, g_1_3, obs_body, obs_objects_pairs_list[1][0, :, :], obs_objects_pairs_list[1][1, :, :], act], dim=1)
            input_2_3 = torch.cat([ag_2_3, g_2_3, obs_body, obs_objects_pairs_list[2][0, :, :], obs_objects_pairs_list[2][1, :, :], act], dim=1)

            inp = torch.stack([input_1_2, input_1_3, input_2_3])
        else:
            # Parallelization by stacking input tensors
            inp = torch.stack([torch.cat([ag, g, obs_body, x[0], x[1], act], dim=1) for x in permutations(obs_objects, 2)])

        output_phi_critic_1, output_phi_critic_2 = self.single_phi_critic(inp)
        output_phi_critic_1 = output_phi_critic_1.sum(dim=0)
        output_phi_critic_2 = output_phi_critic_2.sum(dim=0)
        q1_pi_tensor, q2_pi_tensor = self.rho_critic(output_phi_critic_1, output_phi_critic_2)
        return q1_pi_tensor, q2_pi_tensor


class SemanticActor(nn.Module):
    def __init__(self, nb_objects, dim_phi_actor_input, dim_phi_actor_output, dim_rho_actor_input, dim_rho_actor_output,
                 dim_body, dim_object, symmetry_trick, combinations_trick):
        super(SemanticActor, self).__init__()

        self.nb_objects = nb_objects
        self.dim_body = dim_body
        self.dim_object = dim_object
        self.symmetry_trick = symmetry_trick
        self.first_inds = np.array([0, 1, 2, 3, 5, 7])
        self.second_inds = np.array([0, 1, 2, 4, 6, 8])
        self.combinations_trick = combinations_trick
        self.single_phi_actor = PhiActorDeepSet(dim_phi_actor_input, 256, dim_phi_actor_output)
        self.rho_actor = RhoActorDeepSet(dim_rho_actor_input, dim_rho_actor_output)
        self.one_hot_encodings = [torch.tensor([1., 0., 0.]), torch.tensor([0., 1., 0.]), torch.tensor([0., 0., 1.])]

    def forward(self, obs, ag, g, anchor_g):
        batch_size = obs.shape[0]
        assert batch_size == ag.shape[0]

        obs_body = obs[:, :self.dim_body]
        obs_objects = [torch.cat((torch.cat(batch_size * [self.one_hot_encodings[i]]).reshape(obs_body.shape[0], self.nb_objects),
                                  obs[:, self.dim_body + self.dim_object * i: self.dim_body + self.dim_object * (i + 1)]), dim=1)
                       for i in range(self.nb_objects)]

        if self.symmetry_trick:
            all_inputs = []
            for i in range(self.nb_objects):
                for j in range(self.nb_objects):
                    if i < j:
                        all_inputs.append(torch.cat([ag[:, self.first_inds], obs_body, g[:, self.first_inds], obs_objects[i], obs_objects[j]], dim=1))
                    elif j < i:
                        all_inputs.append(torch.cat([ag[:, self.second_inds], obs_body, g[:, self.second_inds], obs_objects[i], obs_objects[j]], dim=1))

            inp = torch.stack(all_inputs)

        elif self.combinations_trick:
            # Get indexes of atomic goals and corresponding object tuple
            extractors = [torch.zeros((anchor_g.shape[1], 1)) for _ in range(anchor_g.shape[1])]
            for i in range(len(extractors)):
                extractors[i][i, :] = 1.

            # The trick is to create selector matrices that, when multiplied with goals retrieves certain bits. Then the sign of the difference
            # between bits gives which objet goes above the the other

            idxs_bits = [torch.empty(anchor_g.shape[0], 2) for _ in range(3)]
            idxs_objects = [torch.empty(anchor_g.shape[0], 2) for _ in range(3)]

            for i, ((o1, o2), (j, k)) in enumerate(zip([(0, 1), (0, 2), (1, 2)], [(3, 5), (4, 7), (6, 8)])):
                stacked = torch.cat([extractors[j], extractors[k]], dim=1)
                multiplied_matrix = torch.matmul(anchor_g, stacked)
                selector = multiplied_matrix[:, 0] - multiplied_matrix[:, 1]

                idxs_bits[i] = torch.tensor([i, k]).repeat(anchor_g.shape[0], 1).long()
                idxs_bits[i][selector >= 0] = torch.Tensor([i, j]).long()

                idxs_objects[i] = torch.tensor([o2, o1]).repeat(anchor_g.shape[0], 1).long()
                idxs_objects[i][selector >= 0] = torch.Tensor([o1, o2]).long()

            # Gather 2 bits achieved goal
            ag_1_2 = ag.gather(1, idxs_bits[0])
            ag_1_3 = ag.gather(1, idxs_bits[1])
            ag_2_3 = ag.gather(1, idxs_bits[2])

            # Gather 2 bits goal
            g_1_2 = g.gather(1, idxs_bits[0])
            g_1_3 = g.gather(1, idxs_bits[1])
            g_2_3 = g.gather(1, idxs_bits[2])

            obs_object_tensor = torch.stack(obs_objects)

            obs_objects_pairs_list = []
            for idxs_objects in idxs_objects:
                permuted_idxs = idxs_objects.unsqueeze(0).permute(2, 1, 0)
                permuted_idxs = permuted_idxs.repeat(1, 1, obs_object_tensor.shape[2])
                obs_objects_pair = obs_object_tensor.gather(0, permuted_idxs)
                obs_objects_pairs_list.append(obs_objects_pair)

            input_1_2 = torch.cat([ag_1_2, torch.cat([g_1_2, obs_body], dim=1), obs_objects_pairs_list[0][0, :, :],
                                   obs_objects_pairs_list[0][1, :, :]], dim=1)
            input_1_3 = torch.cat([ag_1_3, torch.cat([g_1_3, obs_body], dim=1), obs_objects_pairs_list[1][0, :, :],
                                   obs_objects_pairs_list[1][1, :, :]], dim=1)
            input_2_3 = torch.cat([ag_2_3, torch.cat([g_2_3, obs_body], dim=1), obs_objects_pairs_list[2][0, :, :],
                                   obs_objects_pairs_list[2][1, :, :]], dim=1)

            inp = torch.stack([input_1_2, input_1_3, input_2_3])
        else:
            # Parallelization by stacking input tensors
            inp = torch.stack([torch.cat([ag, g, obs_body, x[0], x[1]], dim=1) for x in permutations(obs_objects, 2)])

        output_phi_actor = self.single_phi_actor(inp)
        output_phi_actor = output_phi_actor.sum(dim=0)
        mean, logstd = self.rho_actor(output_phi_actor)
        return mean, logstd

    def sample(self, obs, ag, g, anchor_g):
        mean, log_std = self.forward(obs, ag, g, anchor_g)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log((1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob, torch.tanh(mean)


class DeepSetSemantic:
    def __init__(self, env_params, args):
        self.observation = None
        self.ag = None
        self.g = None
        self.dim_body = 10
        self.dim_object = 15
        self.dim_act = env_params['action']
        self.nb_objects = 3
        self.n_permutations = len([x for x in permutations(range(self.nb_objects), 2)])

        self.symmetry_trick = args.symmetry_trick
        self.dim_goal = env_params['goal'] if not self.symmetry_trick else 6
        self.combinations_trick = args.combinations_trick
        if self.combinations_trick:
            self.symmetry_trick = False
            self.dim_goal = 2

        self.q1_pi_tensor = None
        self.q2_pi_tensor = None
        self.target_q1_pi_tensor = None
        self.target_q2_pi_tensor = None
        self.pi_tensor = None
        self.log_prob = None

        dim_input_objects = 2 * (self.nb_objects + self.dim_object)
        dim_phi_actor_input = 2 * self.dim_goal + self.dim_body + dim_input_objects
        dim_phi_actor_output = 3 * dim_phi_actor_input
        dim_rho_actor_input = dim_phi_actor_output
        dim_rho_actor_output = self.dim_act

        dim_phi_critic_input = dim_phi_actor_input + self.dim_act
        dim_phi_critic_output = 3 * dim_phi_critic_input
        dim_rho_critic_input = dim_phi_critic_output
        dim_rho_critic_output = 1

        self.critic = SemanticCritic(self.nb_objects, dim_phi_critic_input, dim_phi_critic_output, dim_rho_critic_input, dim_rho_critic_output,
                                     self.n_permutations, self.dim_body, self.dim_object, self.symmetry_trick, self.combinations_trick)
        self.critic_target = SemanticCritic(self.nb_objects, dim_phi_critic_input, dim_phi_critic_output, dim_rho_critic_input,
                                            dim_rho_critic_output, self.n_permutations, self.dim_body, self.dim_object, self.symmetry_trick,
                                            self.combinations_trick)
        self.actor = SemanticActor(self.nb_objects, dim_phi_actor_input, dim_phi_actor_output, dim_rho_actor_input, dim_rho_actor_output,
                                   self.dim_body, self.dim_object, self.symmetry_trick, self.combinations_trick)

    def policy_forward_pass(self, obs, ag, g, anchor_g, no_noise=False):
        if not no_noise:
            self.pi_tensor, self.log_prob, _ = self.actor.sample(obs, ag, g, anchor_g)
        else:
            _, self.log_prob, self.pi_tensor = self.actor.sample(obs, ag, g, anchor_g)

    def forward_pass(self, obs, ag, g, anchor_g, actions=None):

        self.pi_tensor, self.log_prob, _ = self.actor.sample(obs, ag, g, anchor_g)

        if actions is not None:
            self.q1_pi_tensor, self.q2_pi_tensor = self.critic.forward(obs, ag, g, anchor_g, self.pi_tensor)
            return self.critic.forward(obs, ag, g, anchor_g, actions)
        else:
            with torch.no_grad():
                self.target_q1_pi_tensor, self.target_q2_pi_tensor = self.critic_target.forward(obs, ag, g, anchor_g, self.pi_tensor)
            self.q1_pi_tensor, self.q2_pi_tensor = None, None
