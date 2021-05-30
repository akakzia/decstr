import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from itertools import permutations

class SimpleCVAE(nn.Module):

    def __init__(self, state_size=3, latent_size=1):
        super().__init__()

        assert type(latent_size) == int
        assert type(state_size) == int

        self.latent_size = latent_size

        self.encoder = Encoder(state_size + 1, latent_size)
        self.decoder = Decoder(latent_size + state_size, 1)

    def forward(self, state, p):

        batch_size = state.size(0)
        assert state.size(0) == p.size(0)

        means, log_var = self.encoder(torch.cat((state[:, 0] - state[:, 1], p[:, 0].unsqueeze(-1)), dim=-1))

        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.latent_size])
        z = eps * std + means

        recon_x = self.decoder(torch.cat((z, state[:, 0] - state[:, 1]), dim=1))
        return recon_x, means, log_var, z

    def inference(self, state, n=1):

        batch_size = state.size(0)
        z = torch.randn([batch_size, self.latent_size])
        recon_state = self.decoder(torch.cat((z, state[:, 0] - state[:, 1]), dim=1))

        return recon_state


class Encoder(nn.Module):

    def __init__(self, input_size, latent_size):

        super().__init__()

        self.MLP = nn.Sequential()
        self.MLP.add_module(name="L0", module=nn.Linear(input_size, input_size))
        # self.MLP.add_module(name="A0", module=nn.ReLU())

        self.linear_means = nn.Linear(input_size, latent_size)
        self.linear_log_var = nn.Linear(input_size, latent_size)

    def forward(self, x):
        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(nn.Module):
    def __init__(self, input_size, output_size):

        super().__init__()
        self.hidden_size = 3 * input_size

        self.MLP = nn.Sequential()
        self.MLP.add_module(name="L0", module=nn.Linear(input_size, self.hidden_size))
        self.MLP.add_module(name="A0", module=nn.ReLU())

        self.MLP.add_module(name="L1", module=nn.Linear(self.hidden_size, output_size))

        self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

    def forward(self, z):

        x = self.MLP(z)
        return x


class Hulk(MessagePassing):

    def __init__(self, nodes=3, state_size=3, latent_size=1):
        super().__init__()

        assert type(latent_size) == int
        assert type(state_size) == int

        self.latent_size = latent_size

        self.encoder1 = Encoder(state_size + 1, latent_size)
        self.decoder1 = Decoder(latent_size + state_size, 1)

        self.encoder2 = Encoder(state_size + 1, latent_size)
        self.decoder2 = Decoder(latent_size + state_size, 1)

        self.nb_nodes = nodes
        # This order for reshape later
        self.edge_index = torch.tensor([[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]], dtype=torch.long)

    def forward(self, state, p1, p2):
        batch_size = state.size(0)
        assert state.size(0) == p1.size(0) == p2.size(0)

        #Matrixify
        # Transform predicates to matrices for easy handling in the graph
        p1_m = torch.zeros((p1.size(0), self.nb_nodes, self.nb_nodes))
        p2_m = torch.zeros((p2.size(0), self.nb_nodes, self.nb_nodes))

        old, new = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)], [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]
        for o, n in zip(old, new):
            p1_m[:, n[0], n[1]] = p1[:, o[0], o[1]]
            p2_m[:, n[0], n[1]] = p2[:, o[0], o[1]]


        row, col = self.edge_index
        edge_features_input_p1 = torch.stack([torch.cat((state[:, i] - state[:, j], p1_m[:, i, j].unsqueeze(-1)), dim=-1)
                                              for i, j in zip(row, col)], dim=1)

        edge_features_input_p2 = torch.stack([torch.cat((state[:, i] - state[:, j], p2_m[:, i, j].unsqueeze(-1)), dim=-1)
                                              for i, j in zip(row, col)], dim=1)


        means_p1, log_var_p1 = self.encoder1(edge_features_input_p1)
        means_p2, log_var_p2 = self.encoder2(edge_features_input_p2)

        eps = torch.randn([batch_size, 6, self.latent_size])

        std_p1 = torch.exp(0.5 * log_var_p1)
        z_p1 = eps * std_p1 + means_p1

        std_p2 = torch.exp(0.5 * log_var_p2)
        z_p2 = eps * std_p2 + means_p2

        z_p1_m = torch.zeros((z_p1.size(0), self.nb_nodes, self.nb_nodes))
        new = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]
        for i, n in enumerate(new):
            z_p1_m[:, n[0], n[1]] = z_p1[:, i, 0]
            z_p1_m[:, n[0], n[1]] = z_p1[:, i, 0]

        decoder_edges_features_input_p1 = torch.stack([torch.cat((z_p1_m[:, i, j].unsqueeze(-1), state[:, i] - state[:, j]), dim=-1)
                                              for i, j in zip(row, col)], dim=1)

        z_p2_m = torch.zeros((z_p2.size(0), self.nb_nodes, self.nb_nodes))
        new = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]
        for i, n in enumerate(new):
            z_p2_m[:, n[0], n[1]] = z_p2[:, i, 0]
            z_p2_m[:, n[0], n[1]] = z_p2[:, i, 0]

        decoder_edges_features_input_p2 = torch.stack([torch.cat((z_p2_m[:, i, j].unsqueeze(-1), state[:, i] - state[:, j]), dim=-1)
                                                       for i, j in zip(row, col)], dim=1)


        recon_x_p1 = self.decoder1(decoder_edges_features_input_p1)
        recon_x_p2 = self.decoder2(decoder_edges_features_input_p2)

        recon_x_p1 = recon_x_p1.reshape(recon_x_p1.size(0), self.nb_nodes, 2)
        means_p1 = means_p1.reshape(means_p1.size(0), self.nb_nodes, 2)
        log_var_p1 = log_var_p1.reshape(log_var_p1.size(0), self.nb_nodes, 2)

        recon_x_p2 = recon_x_p2.reshape(recon_x_p2.size(0), self.nb_nodes, 2)
        means_p2 = means_p2.reshape(means_p2.size(0), self.nb_nodes, 2)
        log_var_p2 = log_var_p2.reshape(log_var_p2.size(0), self.nb_nodes, 2)

        return recon_x_p1, means_p1, log_var_p1, z_p1, recon_x_p2, means_p2, log_var_p2, z_p2

    def inference(self, state, predicate_id=0):
        batch_size = state.size(0)
        z = torch.randn([batch_size, 6, self.latent_size])
        if predicate_id == 0:
            recon_state_1 = self.decoder1(torch.stack([torch.cat((z[:, 0, :], state[:, 0] - state[:, 1]), dim=1),
                                                       torch.cat((z[:, 1, :], state[:, 1] - state[:, 0]), dim=1),
                                                       torch.cat((z[:, 2, :], state[:, 0] - state[:, 2]), dim=1),
                                                       torch.cat((z[:, 3, :], state[:, 2] - state[:, 0]), dim=1),
                                                       torch.cat((z[:, 4, :], state[:, 1] - state[:, 2]), dim=1),
                                                       torch.cat((z[:, 5, :], state[:, 2] - state[:, 1]), dim=1),
                                                       ], dim=1))
            recon_state_1 = recon_state_1.reshape(recon_state_1.size(0), self.nb_nodes, 2)
            return recon_state_1
        else:
            recon_state_2 = self.decoder2(torch.stack([torch.cat((z[:, 0, :], state[:, 0] - state[:, 1]), dim=1),
                                                       torch.cat((z[:, 1, :], state[:, 1] - state[:, 0]), dim=1),
                                                       torch.cat((z[:, 2, :], state[:, 0] - state[:, 2]), dim=1),
                                                       torch.cat((z[:, 3, :], state[:, 2] - state[:, 0]), dim=1),
                                                       torch.cat((z[:, 4, :], state[:, 1] - state[:, 2]), dim=1),
                                                       torch.cat((z[:, 5, :], state[:, 2] - state[:, 1]), dim=1),
                                                       ], dim=1))
            recon_state_2 = recon_state_2.reshape(recon_state_2.size(0), self.nb_nodes, 2)
            return recon_state_2
        # row, col = self.edge_index
        # z_1 = torch.randn([batch_size, self.nb_nodes, self.nb_nodes])
        #
        # decoder_input_1 = torch.stack([torch.cat((z_1[:, i, j].unsqueeze(-1), state[:, i] - state[:, j]), dim=1)
        #                                for i, j in zip(row, col)])
        # recon_state_1 = self.decoder1(decoder_input_1)
        #
        # z_2 = torch.randn([batch_size, self.nb_nodes, self.nb_nodes])
        #
        # decoder_input_2 = torch.stack([torch.cat((z_2[:, i, j].unsqueeze(-1), state[:, i] - state[:, j]), dim=1)
        #                                for i, j in zip(row, col)])
        # recon_state_2 = self.decoder1(decoder_input_2)
        #
        # recon_state_1 = recon_state_1.reshape(recon_state_1.size(0), self.nb_nodes, 2)
        # recon_state_2 = recon_state_2.reshape(recon_state_2.size(1), self.nb_nodes, 2)

        return recon_state_1