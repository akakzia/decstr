import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch.autograd import Variable
from itertools import permutations, combinations
import numpy as np

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class GraphNetwork(nn.Module):

    def __init__(self, layer_sizes, latent_size):

        super().__init__()

        self.one_hot = [torch.Tensor([0, 0, 1]), torch.Tensor([0, 1, 0]), torch.Tensor([1, 0, 0])]

        self.W = nn.Linear(3, 3, bias=False)

        self.MLP_unary = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP_unary.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP_unary.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.MLP_unary.add_module(name="L{:d}".format(len(layer_sizes)+1), module=nn.Linear(layer_sizes[-1], latent_size))
        self.MLP_unary.add_module(name="A{:d}".format(len(layer_sizes)+1), module=nn.Sigmoid())
        self.MLP_unary.apply(init_weights)

        self.MLP_binary = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP_binary.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP_binary.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.MLP_binary.add_module(name="L{:d}".format(len(layer_sizes) + 1), module=nn.Linear(layer_sizes[-1], latent_size))
        self.MLP_binary.add_module(name="A{:d}".format(len(layer_sizes) + 1), module=nn.Sigmoid())
        self.MLP_binary.apply(init_weights)

        self.MLP_ternary = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP_ternary.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            # self.MLP.add_module(name='B:{:d}'.format(i), module=nn.BatchNorm1d(out_size))
            self.MLP_ternary.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.MLP_ternary.add_module(name="L{:d}".format(len(layer_sizes) + 1), module=nn.Linear(layer_sizes[-1], latent_size))
        self.MLP_ternary.add_module(name="A{:d}".format(len(layer_sizes) + 1), module=nn.Sigmoid())
        self.MLP_ternary.apply(init_weights)

    def forward(self, x):
        batch_size = x.size(0)
        # Linear Transformation
        x = self.W(x)

        input_graph = torch.stack([torch.cat([torch.cat(batch_size * [self.one_hot[i]]).reshape(batch_size, 3),
                                              x[:, i, :]], dim=-1) for i in range(x.size(1))], dim=1)

        # Unary predicates
        z_unary = self.MLP_unary(input_graph)

        # Binary predicates
        input_binary = torch.stack([input_graph[:, i, :] + input_graph[:, j, :] for i, j in combinations([0, 1, 2], 2)], dim=1)
        z_binary = self.MLP_binary(input_binary)

        # Ternary predicates
        input_ternary = input_graph.sum(1)
        z_ternary = self.MLP_ternary(input_ternary).unsqueeze(1)

        z = torch.cat([z_unary, z_binary, z_ternary], dim=1).reshape(batch_size, -1)

        return z


class GnnModel(nn.Module):

    def __init__(self, inner_sizes=[32, 32], state_size=3, output_size=1):
        super().__init__()

        assert type(inner_sizes) == list
        assert type(output_size) == int
        assert type(state_size) == int

        self.output_size = output_size
        self.state_size = state_size

        network_layer_sizes = [state_size] + inner_sizes

        self.network = GraphNetwork(network_layer_sizes, output_size)

    def forward(self, states, positives, negatives):
        states = states.reshape(-1, 3, 3)
        positives = positives.reshape(-1, 3, 3)
        negatives = negatives.reshape(-1, 3, 3)
        # aa = [n.shape[0] for n in negatives]
        # len_neg = [0]
        # for i, n in enumerate(negatives):
        #     len_neg.append(len_neg[i] + n.shape[0])
        # negatives = torch.cat(negatives).reshape(-1, 3, 3)
        z_anchor = self.network(states)
        z_positives = self.network(positives)
        z_negatives = self.network(negatives)

        # z_negatives = torch.cat([z_negatives[len_neg[i]:len_neg[i+1]].sum(0) for i in range(states.shape[0])]).reshape(states.shape[0], -1)

        return z_anchor, z_positives, z_negatives