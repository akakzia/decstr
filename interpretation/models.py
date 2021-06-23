import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch.autograd import Variable
from itertools import permutations
import numpy as np


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


class SimpleNetwork(nn.Module):

    def __init__(self, layer_sizes, latent_size):

        super().__init__()

        self.MLP = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            # self.MLP.add_module(name='B:{:d}'.format(i), module=nn.BatchNorm1d(out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.output_layer = nn.Linear(layer_sizes[-1], latent_size)
        self.bn = nn.BatchNorm1d(latent_size)

    def forward(self, x):
        x = self.MLP(x)
        x = self.output_layer(x)

        return x


class GraphNetwork(nn.Module):

    def __init__(self, layer_sizes, latent_size):

        super().__init__()

        self.MLP1 = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP1.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            # self.MLP.add_module(name='B:{:d}'.format(i), module=nn.BatchNorm1d(out_size))
            self.MLP1.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.MLP2 = nn.Linear(layer_sizes[-1], latent_size)
        self.bn = nn.BatchNorm1d(latent_size)

    def forward(self, x):
        input_network = torch.stack([x[:, i, :] + x[:, j, :] for i, j in [(0, 1), (0, 2), (1, 2)]])
        output_network = self.MLP1(input_network).sum(0)
        z = self.bn(self.MLP2(output_network))

        return z


class SimpleModel(nn.Module):

    def __init__(self, inner_sizes=[128, 128], state_size=9, output_size=2):
        super().__init__()

        assert type(inner_sizes) == list
        assert type(output_size) == int
        assert type(state_size) == int

        self.latent_size = output_size
        self.state_size = state_size

        network_layer_sizes = [state_size] + inner_sizes

        self.network = SimpleNetwork(network_layer_sizes, output_size)

    def forward(self, states, positives, negatives):
        states = states.reshape(-1, self.state_size)
        positives = positives.reshape(-1, self.state_size)
        negatives = negatives.reshape(-1, self.state_size)

        phi_states = self.network(states)
        phi_positives = self.network(positives)
        phi_negatives = self.network(negatives)


        return phi_states, phi_positives, phi_negatives


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


class GnnModel(nn.Module):

    def __init__(self, inner_sizes=[32, 32], state_size=3, output_size=1):
        super().__init__()

        assert type(inner_sizes) == list
        assert type(output_size) == int
        assert type(state_size) == int

        self.output_size = output_size
        self.state_size = state_size

        self.W = nn.Linear(state_size, state_size, bias=False)

        network_layer_sizes = [state_size] + inner_sizes

        self.network = GraphNetwork(network_layer_sizes, output_size)

    def forward(self, states, positives, negatives):
        h_anchor = self.W(states)
        h_positives = self.W(positives)
        h_negatives = self.W(negatives)

        phi_anchor = self.network(h_anchor)
        phi_positives = self.network(h_positives)
        phi_negatives = self.network(h_negatives)

        return phi_anchor, phi_positives, phi_negatives
