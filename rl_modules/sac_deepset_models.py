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


"""class AttentionCriticNetwork(nn.Module):
    def __init__(self, inp, hid, out):
        super(AttentionCriticNetwork, self).__init__()
        self.linear1 = nn.Linear(inp, hid)
        self.linear2 = nn.Linear(hid, out)

        self.linear3 = nn.Linear(inp, hid)
        self.linear4 = nn.Linear(hid, out)

        self.apply(weights_init_)

    def forward(self, inp, mode):
        x1 = F.relu(self.linear1(inp))
        x1 = nn.Sigmoid()(self.linear2(x1))

        # If using an attention network for each critic
        if mode == 'double':
            x2 = F.relu(self.linear3(inp))
            x2 = nn.Sigmoid()(self.linear4(x2))

            return x1, x2
        # If using a single attention network for both critics
        else:
            return x1"""


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
        #self.linear2 = nn.Linear(256, 256)
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
        x = F.relu(self.linear1(x))
        #x = F.relu(self.linear2(x))

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
        self.linear1 = nn.Linear(inp, 256)
        #self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, out)

        self.linear4 = nn.Linear(inp, 256)
        #self.linear5 = nn.Linear(256, 256)
        self.linear6 = nn.Linear(256, out)

        self.apply(weights_init_)

    def forward(self, inp1, inp2):
        x1 = F.relu(self.linear1(inp1))
        #x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(inp2))
        #x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class DeepSetSAC:
    def __init__(self, env_params, use_attention, double_critic_attention):
        # A raw version of DeepSet-based SAC without attention mechanism
        self.observation = None
        self.ag = None
        self.g = None
        self.dim_body = 10
        self.dim_object = 15
        self.dim_goal = env_params['goal']
        self.dim_act = env_params['action']
        self.num_blocks = 3
        self.n_permutations = len([x for x in permutations(range(self.num_blocks), 2)])

        # Whether to use attention networks or concatenate goal to input
        self.use_attention = use_attention
        # double_critic_attention = double_critic_attention
        self.one_hot_encodings = [torch.tensor([1., 0., 0.]), torch.tensor([0., 1., 0.]), torch.tensor([0., 0., 1.])]

        self.q1_pi_tensor = None
        self.q2_pi_tensor = None
        self.target_q1_pi_tensor = None
        self.target_q2_pi_tensor = None
        self.pi_tensor = None
        self.log_prob = None

        # Define dimensions according to parameters use_attention
        # if attention not used, then concatenate [g, ag] in input ==> dimension = 2 * dim_goal
        dim_input_goals = self.dim_goal if use_attention else 2 * self.dim_goal

        dim_input_objects = 2 * (self.num_blocks + self.dim_object)

        dim_phi_actor_input = dim_input_goals + self.dim_body + dim_input_objects
        dim_phi_actor_output = 3 * (self.dim_body + (self.num_blocks + self.dim_object))

        dim_rho_actor_input = 3 * (self.dim_body + (self.num_blocks + self.dim_object))
        dim_rho_actor_output = self.dim_act

        dim_phi_critic_input = dim_input_goals + self.dim_body + dim_input_objects + self.dim_act
        dim_phi_critic_output = 3 * (self.dim_body + (self.num_blocks + self.dim_object) + self.dim_act)

        dim_rho_critic_input = 3 * (self.dim_body + (self.num_blocks + self.dim_object) + self.dim_act)
        dim_rho_critic_output = 1

        if use_attention:
            self.attention_actor = AttentionNetwork(self.dim_goal, 256, self.dim_body + self.dim_object + self.num_blocks)
            self.attention_critic_1 = AttentionNetwork(self.dim_goal, 256, self.dim_body + self.dim_object + self.num_blocks)
            # if self.double_critic_attention:
            #    self.attention_critic_2 = AttentionNetwork(self.dim_goal, 256, self.dim_body + self.dim_object + self.num_blocks)
            self.attention_critic_2 = AttentionNetwork(self.dim_goal, 256, self.dim_body + self.dim_object + self.num_blocks)

        self.single_phi_actor = SinglePhiActor(dim_phi_actor_input, 256, dim_phi_actor_output)
        self.rho_actor = RhoActor(dim_rho_actor_input, dim_rho_actor_output)

        self.single_phi_critic = SinglePhiCritic(dim_phi_critic_input, 256, dim_phi_critic_output)
        self.rho_critic = RhoCritic(dim_rho_critic_input, dim_rho_critic_output)

        self.single_phi_target_critic = SinglePhiCritic(dim_phi_critic_input, 256, dim_phi_critic_output)
        self.rho_target_critic = RhoCritic(dim_rho_critic_input, dim_rho_critic_output)

    def policy_forward_pass(self, obs, ag, g, eval=False):
        self.observation = obs
        self.ag = ag
        self.g = g

        obs_body = self.observation.narrow(-1, start=0, length=self.dim_body)
        obs_objects = [torch.cat((torch.cat(obs_body.shape[0] * [self.one_hot_encodings[i]]).reshape(obs_body.shape[0], self.num_blocks),
                                  self.observation.narrow(-1, start=self.dim_object*i + self.dim_body, length=self.dim_object)),
                                 dim=-1) for i in range(self.num_blocks)]

        if self.use_attention:
            # Pass through the attention network
            output_attention_actor = self.attention_actor(self.g)
            # body_attention_actor
            body_input_actor = obs_body * output_attention_actor[:, :self.dim_body]
            # object attention actor
            obj_input_actor = [obs_objects[i] * output_attention_actor[:, self.dim_body:] for i in range(self.num_blocks)]
            """if not self.double_critic_attention:
                output_attention_critic = self.attention_critic_1(self.g)
                # body attention critic ( same inputs for both critics)
                body_input_critic_1 = obs_body * output_attention_critic[:, :self.dim_body]
                body_input_critic_2 = obs_body * output_attention_critic[:, :self.dim_body]
                # object attention critic
                obj_input_critic_1 = [obs_objects[i] * output_attention_critic[:, self.dim_body:] for i in range(self.num_blocks)]
                obj_input_critic_2 = [obs_objects[i] * output_attention_critic[:, self.dim_body:] for i in range(self.num_blocks)]"""

        else:
            body_input_actor = torch.cat([self.g, obs_body], dim=1)
            obj_input_actor = [obs_objects[i] for i in range(self.num_blocks)]

        # Parallelization by stacking input tensors
        input_actor = torch.stack([torch.cat([ag, body_input_actor, x[0], x[1]], dim=1) for x in permutations(obj_input_actor, 2)])
        #input_actor = torch.stack([torch.cat([ag, body_input_actor, x[0], x[1]], dim=1) for x in combinations(obj_input_actor, 2)])

        output_phi_actor = self.single_phi_actor(input_actor).sum(dim=0)
        # self.pi_tensor, self.log_prob, _ = self.rho_actor.sample(output_phi_actor)
        if not eval:
            self.pi_tensor, self.log_prob, _ = self.rho_actor.sample(output_phi_actor)
        else:
            _, self.log_prob, self.pi_tensor = self.rho_actor.sample(output_phi_actor)

    def forward_pass(self, obs, ag, g, eval=False, actions=None):
        batch_size = obs.shape[0]
        self.observation = obs
        self.ag = ag
        self.g = g
        obs_body = self.observation[:, :self.dim_body]
        obs_objects = [torch.cat((torch.cat(batch_size * [self.one_hot_encodings[i]]).reshape(obs_body.shape[0], self.num_blocks),
                               obs[:, self.dim_body + self.dim_object * i: self.dim_body + self.dim_object * (i + 1)]), dim=1)
                              for i in range(self.num_blocks)]

        if self.use_attention:
            # Pass through the attention network
            output_attention_actor = self.attention_actor(self.g)
            # body_attention_actor
            body_input_actor = obs_body * output_attention_actor[:, :self.dim_body]
            # object attention actor
            obj_input_actor = [obs_objects[i] * output_attention_actor[:, self.dim_body:] for i in range(self.num_blocks)]
            """if not self.double_critic_attention:
                output_attention_critic = self.attention_critic_1(self.g)
                # body attention critic ( same inputs for both critics)
                body_input_critic_1 = obs_body * output_attention_critic[:, :self.dim_body]
                body_input_critic_2 = obs_body * output_attention_critic[:, :self.dim_body]
                # object attention critic
                obj_input_critic_1 = [obs_objects[i] * output_attention_critic[:, self.dim_body:] for i in range(self.num_blocks)]
                obj_input_critic_2 = [obs_objects[i] * output_attention_critic[:, self.dim_body:] for i in range(self.num_blocks)]"""

            output_attention_critic_1 = self.attention_critic_1(self.g)
            output_attention_critic_2 = self.attention_critic_2(self.g)
            # body attention critic for each critic
            body_input_critic_1 = obs_body * output_attention_critic_1[:, :self.dim_body]
            body_input_critic_2 = obs_body * output_attention_critic_2[:, :self.dim_body]
            # object attention critic for each critic
            obj_input_critic_1 = [obs_objects[i] * output_attention_critic_1[:, self.dim_body:] for i in range(self.num_blocks)]
            obj_input_critic_2 = [obs_objects[i] * output_attention_critic_2[:, self.dim_body:] for i in range(self.num_blocks)]
        else:
            body_input = torch.cat([self.g, obs_body], dim=1)
            obj_input = [obs_objects[i] for i in range(self.num_blocks)]

        # Parallelization by stacking input tensors
        input_actor = torch.stack([torch.cat([ag, body_input, x[0], x[1]], dim=1) for x in permutations(obj_input, 2)])
        #input_actor = torch.stack([torch.cat([ag, body_input, x[0], x[1]], dim=1) for x in combinations(obj_input, 2)])

        output_phi_actor = self.single_phi_actor(input_actor).sum(dim=0)
        # self.pi_tensor, self.log_prob, _ = self.rho_actor.sample(output_phi_actor)
        if not eval:
            self.pi_tensor, self.log_prob, _ = self.rho_actor.sample(output_phi_actor)
        else:
            _, self.log_prob, self.pi_tensor = self.rho_actor.sample(output_phi_actor)

        # The critic part
        repeat_pol_actions = self.pi_tensor.repeat(self.n_permutations, 1, 1)
        input_critic = torch.cat([input_actor, repeat_pol_actions], dim=-1)
        if actions is not None:
            repeat_actions = actions.repeat(self.n_permutations, 1, 1)
            input_critic_with_act = torch.cat([input_actor, repeat_actions], dim=-1)
            input_critic = torch.cat([input_critic, input_critic_with_act], dim=0)

        with torch.no_grad():
            output_phi_target_critic_1, output_phi_target_critic_2 = self.single_phi_target_critic(input_critic[:self.n_permutations])
            output_phi_target_critic_1 = output_phi_target_critic_1.sum(dim=0)
            output_phi_target_critic_2 = output_phi_target_critic_2.sum(dim=0)
            self.target_q1_pi_tensor, self.target_q2_pi_tensor = self.rho_target_critic(output_phi_target_critic_1, output_phi_target_critic_2)

        output_phi_critic_1, output_phi_critic_2 = self.single_phi_critic(input_critic)
        if actions is not None:
            output_phi_critic_1 = torch.stack([output_phi_critic_1[:self.n_permutations].sum(dim=0),
                                               output_phi_critic_1[self.n_permutations:].sum(dim=0)])
            output_phi_critic_2 = torch.stack([output_phi_critic_2[:self.n_permutations].sum(dim=0),
                                               output_phi_critic_2[self.n_permutations:].sum(dim=0)])
            q1_pi_tensor, q2_pi_tensor = self.rho_critic(output_phi_critic_1, output_phi_critic_2)
            self.q1_pi_tensor, self.q2_pi_tensor = q1_pi_tensor[0], q2_pi_tensor[0]
            return q1_pi_tensor[1], q2_pi_tensor[1]
        else:
            output_phi_critic_1 = output_phi_critic_1.sum(dim=0)
            output_phi_critic_2 = output_phi_critic_2.sum(dim=0)
            self.q1_pi_tensor, self.q2_pi_tensor = self.rho_critic(output_phi_critic_1, output_phi_critic_2)




