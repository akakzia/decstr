import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from itertools import permutations
import numpy as np

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6
LATENT = 3


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class AttentionNetwork(nn.Module):
    def __init__(self, inp, hid, out):
        super(AttentionNetwork, self).__init__()
        self.linear1 = nn.Linear(inp, hid)
        # self.linear2 = nn.Linear(hid, hid)
        self.linear2 = nn.Linear(hid, out)

        self.apply(weights_init_)

    def forward(self, inp):
        x = F.relu(self.linear1(inp))
        # x = self.linear2(x)
        x = nn.Sigmoid()(self.linear2(x))

        return x


class SinglePhiActor(nn.Module):
    def __init__(self, inp, hid, out):
        super(SinglePhiActor, self).__init__()
        self.linear1 = nn.Linear(inp, hid)
        self.linear2 = nn.Linear(hid, out)
        #self.linear3 = nn.Linear(hid, out)

        self.apply(weights_init_)

    def forward(self, inp):
        x = F.relu(self.linear1(inp))
        x = F.relu(self.linear2(x))
        #x = torch.tanh(self.linear3(x))

        return x


class RhoActor(nn.Module):
    def __init__(self, inp, out, action_space=None):
        super(RhoActor, self).__init__()
        self.linear1 = nn.Linear(inp, 256)
        self.linear2 = nn.Linear(256, 256)
        self.mean_linear = nn.Linear(256, out)
        self.log_std_linear = nn.Linear(256, out)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)

    def forward(self, x):
        #x = torch.tanh(inp)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob, torch.tanh(mean)


class SinglePhiCritic(nn.Module):
    def __init__(self, inp, hid, out):
        super(SinglePhiCritic, self).__init__()
        self.linear1 = nn.Linear(inp, hid)
        self.linear2 = nn.Linear(hid, out)
        # self.linear3 = nn.Linear(hid, out)

        self.linear4 = nn.Linear(inp, hid)
        self.linear5 = nn.Linear(hid, out)
        # self.linear6 = nn.Linear(hid, out)

        self.apply(weights_init_)

    def forward(self, inp):
        x1 = F.relu(self.linear1(inp))
        x1 = F.relu(self.linear2(x1))
        # x1 = F.relu(self.linear3(x1))

        x2 = F.relu(self.linear4(inp))
        x2 = F.relu(self.linear5(x2))
        # x2 = F.relu(self.linear6(x2))

        return x1, x2

class RhoCritic(nn.Module):
    def __init__(self, inp, out, action_space=None):
        super(RhoCritic, self).__init__()
        self.linear1 = nn.Linear(inp, 256)  # Added one layer (To Check)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, out)

        self.linear4 = nn.Linear(inp, 256)
        self.linear5 = nn.Linear(256, 256)
        self.linear6 = nn.Linear(256, out)

        self.apply(weights_init_)

    def forward(self, inp1, inp2):
        #x1 = torch.tanh(self.linear1(inp1))
        x1 = F.relu(self.linear1(inp1))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        #x2 = torch.tanh(self.linear2(inp2))
        x2 = F.relu(self.linear4(inp2))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class DeepSetSAC:
    def __init__(self, env_params):
        # A raw version of DeepSet-based SAC without attention mechanism
        self.observation = None
        self.ag = None
        self.g = None
        self.dim_body = 10
        self.dim_object = 15
        self.dim_goal = env_params['goal']
        self.dim_act = env_params['action']
        self.num_blocks = 3

        self.one_hot_encodings = [torch.tensor([1., 0., 0.]), torch.tensor([0., 1., 0.]), torch.tensor([0., 0., 1.])]

        self.q1_pi_tensor = None
        self.q2_pi_tensor = None
        self.target_q1_pi_tensor = None
        self.target_q2_pi_tensor = None
        self.pi_tensor = None
        self.log_prob = None
        # Define the forming blocks of the DeepSet network
        self.attention_actor = AttentionNetwork(self.dim_goal, 256, self.dim_body + self.dim_object + self.num_blocks)
        self.attention_critic = AttentionNetwork(self.dim_goal, 256, self.dim_body + self.dim_object + self.num_blocks)

        self.single_phi_actor = SinglePhiActor(self.dim_goal + self.dim_body + 2 * (self.num_blocks + self.dim_object), 256,
                                               3 * (self.dim_body + (self.num_blocks + self.dim_object)))
        self.rho_actor = RhoActor(3 * (self.dim_body + (self.num_blocks + self.dim_object)), env_params['action'])

        self.single_phi_critic = SinglePhiCritic(self.dim_goal + self.dim_body + 2 * (self.num_blocks + self.dim_object) + env_params['action'], 256,
                                                 3 * (self.dim_body + (self.num_blocks + self.dim_object) + env_params['action']))
        self.rho_critic = RhoCritic(3 * (self.dim_body + (self.num_blocks + self.dim_object) + env_params['action']), 1)

        self.single_phi_target_critic = SinglePhiCritic(self.dim_goal + self.dim_body + 2 * (self.num_blocks + self.dim_object) + env_params['action'], 256,
                                                        3 * (self.dim_body + (self.num_blocks + self.dim_object) + env_params['action']))
        self.rho_target_critic = RhoCritic(3 * (self.dim_body +  (self.num_blocks + self.dim_object) + env_params['action']), 1)

    def forward_pass(self, obs, ag, g):
        self.observation = obs
        self.ag = ag
        self.g = g

        obs_body = self.observation.narrow(-1, start=0, length=self.dim_body)
        obs_objects = [torch.cat((torch.cat(obs_body.shape[0] * [self.one_hot_encodings[i]]).reshape(obs_body.shape[0], self.num_blocks),
                                  self.observation.narrow(-1, start=self.dim_object*i + self.dim_body, length=self.dim_object)),
                                 dim=-1) for i in range(self.num_blocks)]

        input_objects = []

        body_object_pairs_list = []

        # Pass through the attention network
        # input_attention = torch.cat((self.ag, self.g), dim=-1)

        output_attention_actor = self.attention_actor(self.g)
        output_attention_critic = self.attention_critic(self.g)
        # Process the shapes of the one hot encoding tensors according to input batch
        """proc_one_hot = []
        for i in range(self.num_blocks):
            proc_one_hot.append(torch.cat(obs_body.shape[0] * [self.one_hot_encodings[i]]).reshape(obs_body.shape[0], self.num_blocks))"""


        # body_attention
        body_input_actor = obs_body * output_attention_actor[:, :self.dim_body]
        body_input_critic = obs_body * output_attention_critic[:, :self.dim_body]

        # obj attention
        obj_input_actor = [obs_objects[i] * output_attention_actor[:, self.dim_body:] for i in range(self.num_blocks)]
        obj_input_critic = [obs_objects[i] * output_attention_critic[:, self.dim_body:] for i in range(self.num_blocks)]

        input_actor = torch.stack([torch.cat([ag, body_input_actor, x[0], x[1]], dim=1) for x in permutations(obj_input_actor, 2)])
        output_phi_actor = self.single_phi_actor(input_actor).sum(dim=0)
        self.pi_tensor, self.log_prob, _ = self.rho_actor.sample(output_phi_actor)


        input_critic_without_actions = torch.stack([torch.cat([ag, body_input_critic, x[0], x[1]], dim=1) for x in permutations(obj_input_critic, 2)])
        repeat_actions = self.pi_tensor.repeat(input_actor.shape[0], 1, 1)
        input_critic = torch.cat([input_critic_without_actions, repeat_actions], dim=-1)

        with torch.no_grad():
            output_phi_target_critic1, output_phi_target_critic2 = self.single_phi_target_critic(input_critic)
            output_phi_target_critic1 = output_phi_target_critic1.sum(dim=0)
            output_phi_target_critic2 = output_phi_target_critic2.sum(dim=0)
            self.target_q1_pi_tensor, self.target_q2_pi_tensor = self.rho_target_critic(output_phi_target_critic1, output_phi_target_critic2)

        output_phi_critic1, output_phi_critic2 = self.single_phi_critic(input_critic)
        output_phi_critic1 = output_phi_critic1.sum(dim=0)
        output_phi_critic2 = output_phi_critic2.sum(dim=0)
        self.q1_pi_tensor, self.q2_pi_tensor = self.rho_critic(output_phi_critic1, output_phi_critic2)

    def forward_with_actions(self, obs, ag, g, actions):
        self.observation = obs
        self.ag = ag
        self.g = g

        obs_body = self.observation.narrow(-1, start=0, length=self.dim_body)
        obs_objects = [torch.cat((torch.cat(obs_body.shape[0] * [self.one_hot_encodings[i]]).reshape(obs_body.shape[0], self.num_blocks),
                                  self.observation.narrow(-1, start=self.dim_object * i + self.dim_body, length=self.dim_object)),
                                 dim=-1) for i in range(self.num_blocks)]

        input_objects = []

        body_object_pairs_list = []

        # Pass through the attention network
        # input_attention = torch.cat((self.ag, self.g), dim=-1)

        output_attention_actor = self.attention_actor(self.g)
        output_attention_critic = self.attention_critic(self.g)
        # Process the shapes of the one hot encoding tensors according to input batch
        """proc_one_hot = []
        for i in range(self.num_blocks):
            proc_one_hot.append(torch.cat(obs_body.shape[0] * [self.one_hot_encodings[i]]).reshape(obs_body.shape[0], self.num_blocks))"""

        # body_attention
        body_input_critic = obs_body * output_attention_critic[:, :self.dim_body]

        # obj attention
        obj_input_critic = [obs_objects[i] * output_attention_critic[:, self.dim_body:] for i in range(self.num_blocks)]

        input_critic_without_actions = torch.stack([torch.cat([ag, body_input_critic, x[0], x[1]], dim=1) for x in permutations(obj_input_critic, 2)])
        repeat_actions = actions.repeat(input_critic_without_actions.shape[0], 1, 1)
        input_critic = torch.cat([input_critic_without_actions, repeat_actions], dim=-1)


        output_phi_critic1, output_phi_critic2 = self.single_phi_critic(input_critic)
        output_phi_critic1 = output_phi_critic1.sum(dim=0)
        output_phi_critic2 = output_phi_critic2.sum(dim=0)
        q1_pi_tensor, q2_pi_tensor = self.rho_critic(output_phi_critic1, output_phi_critic2)

        return q1_pi_tensor, q2_pi_tensor


        # # Actor part
        # combinations_tensor = torch.stack([torch.cat(x).reshape(obs_body.shape[0], 36) for x in permutations(obs_objects, 2)], dim=0)
        # attention_tensor = output_attention.repeat(combinations_tensor.shape[0], 1, 2)
        #
        #
        # attended_combinations_tensor = combinations_tensor * attention_tensor
        # obs_body_repeated = obs_body.repeat(combinations_tensor.shape[0], 1, 1)
        # body_att_objects = torch.cat((obs_body_repeated, attended_combinations_tensor), dim=-1)
        # output_phi_actor = self.single_phi_actor(body_att_objects)
        # output_phi_actor = output_phi_actor.sum(dim=0)
        # self.pi_tensor, self.log_prob, _ = self.rho_actor.sample(output_phi_actor)
        # # Critic part
        # repeat_pi = self.pi_tensor.repeat(combinations_tensor.shape[0], 1, 1)
        # body_att_objects_actions = torch.cat((body_att_objects, repeat_pi), dim=-1)
        # with torch.no_grad():
        #     output_phi_target_critic1, output_phi_target_critic2 = self.single_phi_target_critic(body_att_objects_actions)
        #     output_phi_target_critic1 = output_phi_target_critic1.sum(dim=0)
        #     output_phi_target_critic2 = output_phi_target_critic2.sum(dim=0)
        # output_phi_critic1, output_phi_critic2 = self.single_phi_critic(body_att_objects_actions)
        # output_phi_critic1 = output_phi_critic1.sum(dim=0)
        # output_phi_critic2 = output_phi_critic2.sum(dim=0)
        # with torch.no_grad():
        #     self.target_q1_pi_tensor,  self.target_q2_pi_tensor = self.rho_target_critic(output_phi_target_critic1, output_phi_target_critic2)
        # self.q1_pi_tensor, self.q2_pi_tensor = self.rho_critic(output_phi_critic1, output_phi_critic2)

        # """print('dz')
        # for i, j in combinations(torch.arange(self.num_blocks).numpy(), 2):
        #     # Get each object features
        #     obs_object_i = self.observation.narrow(-1, start=int(self.dim_object*i + self.dim_body), length=int(self.dim_object))
        #     obs_object_j = self.observation.narrow(-1, start=int(self.dim_object*j + self.dim_body), length=int(self.dim_object))
        #
        #     obs_object_i = torch.cat((proc_one_hot[i], obs_object_i), dim=-1)
        #     obs_object_j = torch.cat((proc_one_hot[j], obs_object_j), dim=-1)
        #
        #     # Element wise product of attention and each object's features
        #     attended_obs_object_i = obs_object_i * output_attention
        #     attended_obs_object_j = obs_object_j * output_attention
        #
        #     # Concat the input of the Phi Networks [body, attended_object_i, attended_object_j]
        #     obs_pair_objects = torch.cat((attended_obs_object_i, attended_obs_object_j), dim=-1)
        #     body_objects_input = torch.cat((obs_body, obs_pair_objects), dim=-1)
        #
        #     input_pair_objects = self.single_phi_actor(body_objects_input)
        #
        #     input_objects.append(input_pair_objects)
        #
        #     body_object_pairs_list.append(body_objects_input)
        #
        # #input_pi = torch.stack(input_objects).sum(dim=0).sum(dim=0)
        # input_pi = sum(input_objects)
        # self.pi_tensor, self.log_prob, _ = self.rho_actor.sample(input_pi)
        #
        # input_objects_action_list_1 = []
        #
        # input_objects_action_list_2 = []
        #
        # input_objects_action_list_target_1 = []
        #
        # input_objects_action_list_target_2 = []
        #
        # # Critic part
        # #pi = self.pi_tensor.unsqueeze(0).repeat(obs_body.shape[0], 1)
        # for body_objects_input in body_object_pairs_list:
        #     body_objects_action_input = torch.cat((body_objects_input, self.pi_tensor), dim=-1)
        #
        #     with torch.no_grad():
        #         input_target_phi_1, input_target_phi_2 = self.single_phi_target_critic(body_objects_action_input)
        #
        #     input_objects_action_1, input_objects_action_2 = self.single_phi_critic(body_objects_action_input)
        #
        #     input_objects_action_list_1.append(input_objects_action_1)
        #
        #     input_objects_action_list_2.append(input_objects_action_2)
        #
        #     input_objects_action_list_target_1.append(input_target_phi_1)
        #
        #     input_objects_action_list_target_2.append(input_target_phi_2)
        #
        # #input_critic_1 = torch.stack(input_objects_action_list_1).sum(dim=0).sum(dim=0)
        # #input_critic_2 = torch.stack(input_objects_action_list_2).sum(dim=0).sum(dim=0)
        # #input_target_critic_1 = torch.stack(input_objects_action_list_target_1).sum(dim=0).sum(dim=0)
        # #input_target_critic_2 = torch.stack(input_objects_action_list_target_2).sum(dim=0).sum(dim=0)
        # input_critic_1 = sum(input_objects_action_list_1)
        # input_critic_2 = sum(input_objects_action_list_2)
        # input_target_critic_1 = sum(input_objects_action_list_target_1)
        # input_target_critic_2 = sum(input_objects_action_list_target_2)
        # self.q1_pi_tensor, self.q2_pi_tensor = self.rho_critic(input_critic_1, input_critic_2)
        #
        # with torch.no_grad():
        #     self.target_q1_pi_tensor,  self.target_q2_pi_tensor = self.rho_target_critic(input_target_critic_1, input_target_critic_2)"""

    # def forward_with_actions(self, obs, actions, ag, g):
    #     self.observation = obs
    #     self.ag = ag
    #     self.g = g
    #
    #     obs_body = self.observation.narrow(-1, start=0, length=self.dim_body)
    #     obs_objects = [torch.cat((torch.cat(obs_body.shape[0] * [self.one_hot_encodings[i]]).reshape(obs_body.shape[0], self.num_blocks),
    #                               self.observation.narrow(-1, start=self.dim_object * i + self.dim_body, length=self.dim_object)),
    #                              dim=-1) for i in range(self.num_blocks)]
    #
    #     # Pass through the attention network
    #     input_attention = torch.cat((self.ag, self.g), dim=-1)
    #     output_attention = self.attention(input_attention)
    #
    #     input_objects_action_list_1 = []
    #
    #     input_objects_action_list_2 = []
    #
    #     proc_one_hot = []
    #
    #     combinations_tensor = torch.stack([torch.cat(x).reshape(obs_body.shape[0], 36) for x in permutations(obs_objects, 2)], dim=0)
    #     attention_tensor = output_attention.repeat(combinations_tensor.shape[0], 1, 2)
    #     attended_combinations_tensor = combinations_tensor * attention_tensor
    #     obs_body_repeated = obs_body.repeat(combinations_tensor.shape[0], 1, 1)
    #     body_att_objects = torch.cat((obs_body_repeated, attended_combinations_tensor), dim=-1)
    #     # Critic part
    #     repeat_actions = actions.repeat(combinations_tensor.shape[0], 1, 1)
    #     body_att_objects_actions = torch.cat((body_att_objects, repeat_actions), dim=-1)
    #     output_phi_critic1, output_phi_critic2 = self.single_phi_critic(body_att_objects_actions)
    #     output_phi_critic1 = output_phi_critic1.sum(dim=0)
    #     output_phi_critic2 = output_phi_critic2.sum(dim=0)
    #     q1_pi_tensor, q2_pi_tensor = self.rho_critic(output_phi_critic1, output_phi_critic2)
    #
    #     """for i in range(self.num_blocks):
    #         proc_one_hot.append(torch.cat(obs_body.shape[0] * [self.one_hot_encodings[i]]).reshape(obs_body.shape[0], self.num_blocks))"""
    #
    #     # Critic part
    #     """for i, j in combinations(torch.arange(self.num_blocks).numpy(), 2):
    #         # Get each object features
    #         obs_object_i = self.observation.narrow(-1, start=int(self.dim_object * i + self.dim_body), length=int(self.dim_object))
    #         obs_object_j = self.observation.narrow(-1, start=int(self.dim_object * j + self.dim_body), length=int(self.dim_object))
    #
    #         obs_object_i = torch.cat((proc_one_hot[i], obs_object_i), dim=-1)
    #         obs_object_j = torch.cat((proc_one_hot[j], obs_object_j), dim=-1)
    #
    #         # Element wise product of attention and each object's features
    #         attended_obs_object_i = obs_object_i * output_attention
    #         attended_obs_object_j = obs_object_j * output_attention
    #
    #         # Concat the input of the Phi Networks [body, attended_object_i, attended_object_j]
    #         obs_pair_objects = torch.cat((attended_obs_object_i, attended_obs_object_j), dim=-1)
    #         body_objects_action_input = torch.cat((obs_body, obs_pair_objects, actions), dim=-1)
    #
    #         input_objects_action_1, input_objects_action_2 = self.single_phi_critic(body_objects_action_input)
    #
    #         input_objects_action_list_1.append(input_objects_action_1)
    #
    #         input_objects_action_list_2.append(input_objects_action_2)
    #
    #     input_critic_1 = torch.stack(input_objects_action_list_1).sum(dim=0)
    #     input_critic_2 = torch.stack(input_objects_action_list_2).sum(dim=0)
    #     q1_pi_tensor, q2_pi_tensor = self.rho_critic(input_critic_1, input_critic_2)"""
    #
    #     return q1_pi_tensor, q2_pi_tensor
