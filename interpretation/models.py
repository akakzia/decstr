import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch.autograd import Variable
import numpy as np


def to_categorical(y, num_columns):
    """Returns one-hot encoded Variable"""
    y_cat = np.zeros((y.shape[0], num_columns))
    y_cat[range(y.shape[0]), y] = 1.0

    return Variable(torch.FloatTensor(y_cat))

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


class GATLayer(nn.Module):
    """
    Simple PyTorch Implementation of the Graph Attention layer.
    """

    def __init__(self, in_features, out_features, dropout, alpha, aggregation='sum'):
        super(GATLayer, self).__init__()
        assert aggregation == 'sum'
        self.dropout = dropout  # drop prob = 0.6
        self.in_features = in_features  #
        self.out_features = out_features  #
        self.alpha = alpha  # LeakyReLU with negative input slope, alpha = 0.2
        self.aggregation = aggregation

        # Xavier Initialization of Weights
        # Alternatively use weights_init to apply weights of choice
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))  # +2 to add goal
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # LeakyReLU
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input1, input2):
        input = torch.cat([input1, input2], dim=1)
        batch_size = input.size()[0]
        # Linear Transformation
        h = torch.matmul(input, self.W)
        N = input.size()[1]

        # Attention Mechanism
        a_input = torch.cat([h.repeat(1, 1, N).view(batch_size, N * N, -1), h.repeat(1, N, 1)], dim=-1).view(batch_size, N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))

        attention = F.softmax(e, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        h_prime1 = h_prime[:, :6, :]
        h_prime2 = h_prime[:, 6:, :]
        return h_prime1, h_prime2

class SimpleCVAE(nn.Module):

    def __init__(self, inner_sizes=[128, 128], state_size=3, latent_size=1):
        super().__init__()

        assert type(inner_sizes) == list
        assert type(latent_size) == int
        assert type(state_size) == int

        self.latent_size = latent_size
        self.state_size = state_size

        encoder_layer_sizes = [state_size] + inner_sizes
        decoder_layer_sizes = [latent_size] + inner_sizes + [state_size]

        self.encoder = SimpleEncoder(encoder_layer_sizes, latent_size)
        self.decoder = SimpleDecoder(decoder_layer_sizes)

    def forward(self, state):
        batch_size = state.size(0)
        state = torch.cat([state[:, 0], state[:, 0], abs(state[:, 0] - state[:, 1])], dim=-1)
        # state = state.reshape(batch_size, state.size(1) * state.size(2))
        means, log_var = self.encoder(state)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.latent_size])
        z = eps * std + means

        recon_x = self.decoder(z)
        return recon_x, means, log_var, z

    def inference(self, n=1):

        batch_size = n
        z = torch.randn([batch_size, self.latent_size])
        recon_state = self.decoder(z)

        return recon_state


class SimpleEncoder(nn.Module):

    def __init__(self, layer_sizes, latent_size):

        super().__init__()

        self.MLP = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x):

        x = self.MLP(x)
        x = x.sum(0)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class SimpleDecoder(nn.Module):

    def __init__(self, layer_sizes):

        super().__init__()

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i + 2 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())


    def forward(self, z):

        x = self.MLP(z)
        return x


class Encoder(nn.Module):

    def __init__(self, input_size, latent_size, nb_predicates=2):

        super().__init__()
        self.MLP = [nn.Sequential(
            nn.Linear(input_size, input_size)
        ) for _ in range(nb_predicates)]

        self.linear_means = [nn.Linear(input_size, latent_size) for _ in range(nb_predicates)]
        self.linear_log_var = [nn.Linear(input_size, latent_size) for _ in range(nb_predicates)]

        self.nb_predicates = nb_predicates

    def forward(self, x):
        assert type(x) == list
        assert len(x) == self.nb_predicates
        means, log_vars = [], []
        for i, e in enumerate(x):
            e = self.MLP[i](e)
            m = self.linear_means[i](e)
            l_v = self.linear_log_var[i](e)
            means.append(m)
            log_vars.append(l_v)
        return means, log_vars


class Decoder(nn.Module):
    def __init__(self, input_size, output_size, nb_predicates=2):

        super().__init__()
        self.hidden_size = 3 * input_size

        self.attention_layer = GATLayer(self.hidden_size, self.hidden_size, dropout=0.0, alpha=0.2, aggregation='sum')

        # self.MLP = [nn.Sequential(nn.Linear(input_size, self.hidden_size),
        #                           nn.ReLU())
        #             for _ in range(nb_predicates)]
        # self.readout = [nn.Sequential(nn.Linear(self.hidden_size, output_size),
        #                           nn.Sigmoid())
        #             for _ in range(nb_predicates)]

        self.nb_predicates = nb_predicates
        self.MLP_1 = nn.Sequential()
        self.MLP_1.add_module(name="L0", module=nn.Linear(input_size, self.hidden_size))
        self.MLP_1.add_module(name="A0", module=nn.ReLU())

        self.readout_1 = nn.Sequential()
        self.readout_1.add_module(name="L0", module=nn.Linear(self.hidden_size, output_size))

        self.readout_1.add_module(name="sigmoid", module=nn.Sigmoid())

        self.MLP_2 = nn.Sequential()
        self.MLP_2.add_module(name="L0", module=nn.Linear(input_size, self.hidden_size))
        self.MLP_2.add_module(name="A0", module=nn.ReLU())

        self.readout_2 = nn.Sequential()
        self.readout_2.add_module(name="L0", module=nn.Linear(self.hidden_size, output_size))

        self.readout_2.add_module(name="sigmoid", module=nn.Sigmoid())

    def forward(self, z1, z2):
        # assert type(z) == list
        # assert len(z) == self.nb_predicates
        #
        # recon_x = []
        # for i, e in enumerate(z):
        #     e = self.MLP[i](e)
        #     e = self.readout[i](e)
        #     recon_x.append(e)

        x1 = self.MLP_1(z1)
        x2 = self.MLP_2(z2)
        # x1, x2 = self.attention_layer(x1, x2)

        x1 = self.readout_1(x1)
        x2 = self.readout_2(x2)

        return [x1, x2]


class Hulk(MessagePassing):

    def __init__(self, inner_sizes=[128, 128], state_size=3, latent_size=1):
        super().__init__()

        assert type(inner_sizes) == list
        assert type(latent_size) == int
        assert type(state_size) == int

        self.latent_size = latent_size
        self.state_size = state_size

        encoder_layer_sizes = [state_size] + inner_sizes
        decoder_layer_sizes = [latent_size] + inner_sizes + [state_size]

        self.encoder = SimpleEncoder(encoder_layer_sizes, latent_size)
        self.decoder = SimpleDecoder(decoder_layer_sizes)

    def forward(self, state):
        batch_size = state.size(0)
        state = torch.stack([torch.cat([state[:, i, :], state[:, j, :], abs(state[:, i, :] - state[:, j, :])], dim=-1)
                           for i, j in [(0, 1), (0, 2), (1, 2)]])
        # state = state.reshape(batch_size, state.size(1) * state.size(2))
        means, log_var = self.encoder(state)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.latent_size])
        z = eps * std + means

        recon_x = self.decoder(z)
        return recon_x, means, log_var, z

    def inference(self, n=1):
        batch_size = n
        z = torch.randn([batch_size, self.latent_size])
        recon_state = self.decoder(z)

        return recon_state