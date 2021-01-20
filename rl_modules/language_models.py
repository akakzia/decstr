import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from itertools import permutations, combinations
import numpy as np
from language.utils import OneHotEncoder, analyze_inst, Vocab
from utils import get_instruction
from rl_modules.networks import PhiActorDeepSet, PhiCriticDeepSet, RhoActorDeepSet, RhoCriticDeepSet

epsilon = 1e-6


class LanguageCritic(nn.Module):
    def __init__(self, nb_objects,  dim_phi_critic_input, dim_phi_critic_output,
                 dim_rho_critic_input, dim_rho_critic_output, one_hot_language,
                 vocab_size, embedding_size, nb_permutations, dim_body, dim_object):
        super(LanguageCritic, self).__init__()
        self.critic_sentence_encoder = nn.RNN(input_size=vocab_size,
                                              hidden_size=embedding_size,
                                              num_layers=1,
                                              nonlinearity='tanh',
                                              bias=True,
                                              batch_first=True)

        self.nb_permutations = nb_permutations
        self.one_hot_language = one_hot_language
        self.one_hot_encodings = [torch.tensor([1., 0., 0.]), torch.tensor([0., 1., 0.]), torch.tensor([0., 0., 1.])]
        self.nb_objects = nb_objects
        self.dim_body = dim_body
        self.dim_object = dim_object
        self.single_phi_critic = PhiCriticDeepSet(dim_phi_critic_input, 256, dim_phi_critic_output)
        self.rho_critic = RhoCriticDeepSet(dim_rho_critic_input, dim_rho_critic_output)

    def forward(self, obs, act, language_goals):

        batch_size = obs.shape[0]
        assert batch_size == len(language_goals)

        # encode language goals
        encodings = torch.tensor(np.array([self.one_hot_language[lg] for lg in language_goals]), dtype=torch.float32)
        l_emb = self.critic_sentence_encoder.forward(encodings)[0][:, -1, :]

        obs_body = obs[:, :self.dim_body]
        obs_objects = [torch.cat((torch.cat(batch_size * [self.one_hot_encodings[i]]).reshape(obs_body.shape[0], self.nb_objects),
                                  obs[:, self.dim_body + self.dim_object * i: self.dim_body + self.dim_object * (i + 1)]), dim=1)
                       for i in range(self.nb_objects)]

        # Parallelization by stacking input tensors
        inp = torch.stack([torch.cat([l_emb, obs_body, act, x[0], x[1]], dim=1) for x in permutations(obs_objects, 2)])

        output_phi_critic_1, output_phi_critic_2 = self.single_phi_critic(inp)
        output_phi_critic_1 = output_phi_critic_1.sum(dim=0)
        output_phi_critic_2 = output_phi_critic_2.sum(dim=0)
        q1_pi_tensor, q2_pi_tensor = self.rho_critic(output_phi_critic_1, output_phi_critic_2)
        return q1_pi_tensor, q2_pi_tensor

    def encode_language(self, language_goals):
        if isinstance(language_goals, str):
            encodings = torch.tensor(self.one_hot_language[language_goals], dtype=torch.float32).unsqueeze(dim=0)
        else:
            encodings = torch.tensor(np.array([self.one_hot_language[lg] for lg in language_goals]), dtype=torch.float32)
        l_emb = self.critic_sentence_encoder.forward(encodings)[0][:, -1, :]
        return l_emb


class LanguageActor(nn.Module):
    def __init__(self, nb_objects,  dim_phi_actor_input, dim_phi_actor_output,
                 dim_rho_actor_input, dim_rho_actor_output, dim_body, dim_object):
        super(LanguageActor, self).__init__()

        self.nb_objects = nb_objects
        self.dim_body = dim_body
        self.dim_object = dim_object
        self.single_phi_actor = PhiActorDeepSet(dim_phi_actor_input, 256, dim_phi_actor_output)
        self.rho_actor = RhoActorDeepSet(dim_rho_actor_input, dim_rho_actor_output)
        self.one_hot_encodings = [torch.tensor([1., 0., 0.]), torch.tensor([0., 1., 0.]), torch.tensor([0., 0., 1.])]

    def forward(self, obs, l_emb):
        batch_size = obs.shape[0]
        assert batch_size == l_emb.shape[0]

        obs_body = obs[:, :self.dim_body]
        obs_objects = [torch.cat((torch.cat(batch_size * [self.one_hot_encodings[i]]).reshape(obs_body.shape[0], self.nb_objects),
                                  obs[:, self.dim_body + self.dim_object * i: self.dim_body + self.dim_object * (i + 1)]), dim=1)
                       for i in range(self.nb_objects)]

        # Parallelization by stacking input tensors
        inp = torch.stack([torch.cat([l_emb, obs_body, x[0], x[1]], dim=1) for x in permutations(obs_objects, 2)])

        output_phi_actor = self.single_phi_actor(inp)
        output_phi_actor = output_phi_actor.sum(dim=0)
        mean, logstd = self.rho_actor(output_phi_actor)
        return mean, logstd

    def sample(self, obs, l_emb):
        mean, log_std = self.forward(obs, l_emb)
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


class DeepSetLanguage:
    def __init__(self, env_params, args):
        self.observation = None
        self.ag = None
        self.g = None
        self.dim_body = 10
        self.dim_object = 15
        self.dim_goal = env_params['goal']
        self.dim_act = env_params['action']
        self.num_blocks = 3
        self.n_permutations = len([x for x in permutations(range(self.num_blocks), 2)])
        self.instructions = get_instruction()
        self.nb_instructions = len(self.instructions)
        self.embedding_size = args.embedding_size
        self.one_hot_encodings = [torch.tensor([1., 0., 0.]), torch.tensor([0., 1., 0.]), torch.tensor([0., 0., 1.])]

        split_instructions, max_seq_length, word_set = analyze_inst(self.instructions)
        vocab = Vocab(word_set)
        self.one_hot_encoder = OneHotEncoder(vocab, max_seq_length)
        self.one_hot_language = dict(zip(self.instructions, [self.one_hot_encoder.encode(s) for s in split_instructions]))

        self.q1_pi_tensor = None
        self.q2_pi_tensor = None
        self.target_q1_pi_tensor = None
        self.target_q2_pi_tensor = None
        self.pi_tensor = None
        self.log_prob = None

        dim_input_objects = 2 * (self.num_blocks + self.dim_object)
        dim_phi_actor_input = self.embedding_size + self.dim_body + dim_input_objects
        dim_phi_actor_output = 3 * dim_phi_actor_input
        dim_rho_actor_input = dim_phi_actor_output
        dim_rho_actor_output = self.dim_act

        dim_phi_critic_input = dim_phi_actor_input + self.dim_act
        dim_phi_critic_output = 3 * dim_phi_critic_input
        dim_rho_critic_input = dim_phi_critic_output
        dim_rho_critic_output = 1

        self.critic = LanguageCritic(self.num_blocks, dim_phi_critic_input, dim_phi_critic_output, dim_rho_critic_input, dim_rho_critic_output,
                             self.one_hot_language, vocab.size, self.embedding_size, self.n_permutations, self.dim_body, self.dim_object)
        self.critic_target = LanguageCritic(self.num_blocks, dim_phi_critic_input, dim_phi_critic_output, dim_rho_critic_input, dim_rho_critic_output,
                                    self.one_hot_language, vocab.size, self.embedding_size, self.n_permutations, self.dim_body, self.dim_object)
        self.actor = LanguageActor(self.num_blocks, dim_phi_actor_input, dim_phi_actor_output, dim_rho_actor_input, dim_rho_actor_output, self.dim_body,
                           self.dim_object)

    def policy_forward_pass(self, obs, no_noise=False, language_goal=None):

        l_emb = self.critic.encode_language(language_goal)
        if not no_noise:
            self.pi_tensor, self.log_prob, _ = self.actor.sample(obs, l_emb)
        else:
            _, self.log_prob, self.pi_tensor = self.actor.sample(obs, l_emb)

    def forward_pass(self, obs, actions=None, language_goals=None):

        l_emb = self.critic.encode_language(language_goals)
        self.pi_tensor, self.log_prob, _ = self.actor.sample(obs, l_emb)

        if actions is not None:
            self.q1_pi_tensor, self.q2_pi_tensor = self.critic.forward(obs, self.pi_tensor, language_goals)
            return self.critic.forward(obs, actions, language_goals)
        else:
            with torch.no_grad():
                self.target_q1_pi_tensor, self.target_q2_pi_tensor = self.critic_target.forward(obs, self.pi_tensor, language_goals)
            self.q1_pi_tensor, self.q2_pi_tensor = None, None
