import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from itertools import permutations

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
        self.linear2 = nn.Linear(hid, hid)
        self.linear3 = nn.Linear(hid, out)

        self.apply(weights_init_)

    def forward(self, inp):
        x = self.linear1(inp)
        x = self.linear2(x)
        x = nn.Sigmoid()(self.linear3(x))

        return x


class SinglePhiActor(nn.Module):
    def __init__(self, inp, hid, out):
        super(SinglePhiActor, self).__init__()
        self.linear1 = nn.Linear(inp, hid)
        self.linear2 = nn.Linear(hid, hid)
        self.linear3 = nn.Linear(hid, out)

        self.apply(weights_init_)

    def forward(self, inp):
        x = self.linear1(inp)
        x = self.linear2(x)
        x = F.relu(self.linear3(x))

        return x


class RhoActor(nn.Module):
    def __init__(self, inp, out, action_space=None):
        super(RhoActor, self).__init__()
        self.mean_linear = nn.Linear(inp, out)
        self.log_std_linear = nn.Linear(inp, out)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)

    def forward(self, inp):
        x = torch.tanh(inp)

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
        self.linear2 = nn.Linear(hid, hid)
        self.linear3 = nn.Linear(hid, out)

        self.linear4 = nn.Linear(inp, hid)
        self.linear5 = nn.Linear(hid, hid)
        self.linear6 = nn.Linear(hid, out)

        self.apply(weights_init_)

    def forward(self, inp):
        x1 = self.linear1(inp)
        x1 = self.linear2(x1)
        x1 = F.relu(self.linear3(x1))

        x2 = self.linear4(inp)
        x2 = self.linear5(x2)
        x2 = F.relu(self.linear6(x2))

        return x1, x2


class RhoCritic(nn.Module):
    def __init__(self, inp, out, action_space=None):
        super(RhoCritic, self).__init__()
        self.linear1 = nn.Linear(inp, out)  # Added one layer (To Check)

        self.linear2 = nn.Linear(inp, out)

        self.apply(weights_init_)

    def forward(self, inp1, inp2):
        x1 = torch.tanh(self.linear1(inp1))

        x2 = torch.tanh(self.linear2(inp2))

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
        self.num_blocks = 3

        self.one_hot_encodings = [torch.tensor([1., 0., 0.]), torch.tensor([0., 1., 0.]), torch.tensor([0., 0., 1.])]

        self.q1_pi_tensor = None
        self.q2_pi_tensor = None
        self.target_q1_pi_tensor = None
        self.target_q2_pi_tensor = None
        self.pi_tensor = None
        self.log_prob = None
        # Define the forming blocks of the DeepSet network
        self.attention = AttentionNetwork(2 * self.dim_goal, 256, self.dim_object + self.num_blocks)

        self.single_phi_actor = SinglePhiActor(self.dim_body + 2 * (self.num_blocks + self.dim_object), 256,
                                               3 * (self.dim_body + 2 * (self.num_blocks + self.dim_object)))
        self.rho_actor = RhoActor(3 * (self.dim_body + 2 * (self.num_blocks + self.dim_object)), env_params['action'])

        self.single_phi_critic = SinglePhiCritic(self.dim_body + 2 * (self.num_blocks + self.dim_object) + env_params['action'], 256,
                                                 3 * (self.dim_body + 2 * (self.num_blocks + self.dim_object) + env_params['action']))
        self.rho_critic = RhoCritic(3 * (self.dim_body + 2 * (self.num_blocks + self.dim_object) + env_params['action']), 1)

        self.single_phi_target_critic = SinglePhiCritic(self.dim_body + 2 * (self.num_blocks + self.dim_object) + env_params['action'], 256,
                                                        3 * (self.dim_body + 2 * (self.num_blocks + self.dim_object) + env_params['action']))
        self.rho_target_critic = RhoCritic(3 * (self.dim_body + 2 * (self.num_blocks + self.dim_object) + env_params['action']), 1)

    def forward_pass(self, obs, ag, g):
        self.observation = obs
        self.ag = ag
        self.g = g

        obs_body = self.observation.narrow(-1, start=0, length=self.dim_body)

        input_objects = []

        body_object_pairs_list = []

        # Pass through the attention network
        input_attention = torch.cat((self.ag, self.g), dim=-1)
        output_attention = self.attention(input_attention)

        # Process the shapes of the one hot encoding tensors according to input batch
        proc_one_hot = []
        for i in range(self.num_blocks):
            proc_one_hot.append(torch.cat(obs_body.shape[0] * [self.one_hot_encodings[i]]).reshape(obs_body.shape[0], self.num_blocks))

        # Actor part
        for i, j in permutations(torch.arange(self.num_blocks).numpy(), 2):
            # Get each object features
            obs_object_i = self.observation.narrow(-1, start=int(self.dim_object*i + self.dim_body), length=int(self.dim_object))
            obs_object_j = self.observation.narrow(-1, start=int(self.dim_object*j + self.dim_body), length=int(self.dim_object))

            obs_object_i = torch.cat((proc_one_hot[i], obs_object_i), dim=-1)
            obs_object_j = torch.cat((proc_one_hot[j], obs_object_j), dim=-1)

            # Element wise product of attention and each object's features
            attended_obs_object_i = obs_object_i * output_attention
            attended_obs_object_j = obs_object_j * output_attention

            # Concat the input of the Phi Networks [body, attended_object_i, attended_object_j]
            obs_pair_objects = torch.cat((attended_obs_object_i, attended_obs_object_j), dim=-1)
            body_objects_input = torch.cat((obs_body, obs_pair_objects), dim=-1)

            input_pair_objects = self.single_phi_actor(body_objects_input)

            input_objects.append(input_pair_objects)

            body_object_pairs_list.append(body_objects_input)

        input_pi = torch.stack(input_objects).sum(dim=0).sum(dim=0)
        self.pi_tensor, self.log_prob, _ = self.rho_actor.sample(input_pi)

        input_objects_action_list_1 = []

        input_objects_action_list_2 = []

        input_objects_action_list_target_1 = []

        input_objects_action_list_target_2 = []

        # Critic part
        pi = self.pi_tensor.unsqueeze(0).repeat(obs_body.shape[0], 1)
        for body_objects_input in body_object_pairs_list:
            body_objects_action_input = torch.cat((body_objects_input, pi), dim=-1)

            with torch.no_grad():
                input_target_phi_1, input_target_phi_2 = self.single_phi_target_critic(body_objects_action_input)

            input_objects_action_1, input_objects_action_2 = self.single_phi_critic(body_objects_action_input)

            input_objects_action_list_1.append(input_objects_action_1)

            input_objects_action_list_2.append(input_objects_action_2)

            input_objects_action_list_target_1.append(input_target_phi_1)

            input_objects_action_list_target_2.append(input_target_phi_2)

        input_critic_1 = torch.stack(input_objects_action_list_1).sum(dim=0).sum(dim=0)
        input_critic_2 = torch.stack(input_objects_action_list_2).sum(dim=0).sum(dim=0)
        input_target_critic_1 = torch.stack(input_objects_action_list_target_1).sum(dim=0).sum(dim=0)
        input_target_critic_2 = torch.stack(input_objects_action_list_target_2).sum(dim=0).sum(dim=0)
        self.q1_pi_tensor, self.q2_pi_tensor = self.rho_critic(input_critic_1, input_critic_2)

        with torch.no_grad():
            self.target_q1_pi_tensor,  self.target_q2_pi_tensor = self.rho_target_critic(input_target_critic_1, input_target_critic_2)

    def forward_with_actions(self, obs, actions, ag, g):
        self.observation = obs
        self.ag = ag
        self.g = g

        obs_body = self.observation.narrow(-1, start=0, length=self.dim_body)

        # Pass through the attention network
        input_attention = torch.cat((self.ag, self.g), dim=-1)
        output_attention = self.attention(input_attention)

        input_objects_action_list_1 = []

        input_objects_action_list_2 = []

        proc_one_hot = []
        for i in range(self.num_blocks):
            proc_one_hot.append(torch.cat(obs_body.shape[0] * [self.one_hot_encodings[i]]).reshape(obs_body.shape[0], self.num_blocks))

        # Critic part
        for i, j in permutations(torch.arange(self.num_blocks).numpy(), 2):
            # Get each object features
            obs_object_i = self.observation.narrow(-1, start=int(self.dim_object * i + self.dim_body), length=int(self.dim_object))
            obs_object_j = self.observation.narrow(-1, start=int(self.dim_object * j + self.dim_body), length=int(self.dim_object))

            obs_object_i = torch.cat((proc_one_hot[i], obs_object_i), dim=-1)
            obs_object_j = torch.cat((proc_one_hot[j], obs_object_j), dim=-1)

            # Element wise product of attention and each object's features
            attended_obs_object_i = obs_object_i * output_attention
            attended_obs_object_j = obs_object_j * output_attention

            # Concat the input of the Phi Networks [body, attended_object_i, attended_object_j]
            obs_pair_objects = torch.cat((attended_obs_object_i, attended_obs_object_j), dim=-1)
            body_objects_action_input = torch.cat((obs_body, obs_pair_objects, actions), dim=-1)

            input_objects_action_1, input_objects_action_2 = self.single_phi_critic(body_objects_action_input)

            input_objects_action_list_1.append(input_objects_action_1)

            input_objects_action_list_2.append(input_objects_action_2)

        input_critic_1 = torch.stack(input_objects_action_list_1).sum(dim=0)
        input_critic_2 = torch.stack(input_objects_action_list_2).sum(dim=0)
        q1_pi_tensor, q2_pi_tensor = self.rho_critic(input_critic_1, input_critic_2)

        return q1_pi_tensor, q2_pi_tensor
