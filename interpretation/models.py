import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


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

    def __init__(self, nodes=3, state_size=3, latent_size=1):
        super().__init__()

        assert type(latent_size) == int
        assert type(state_size) == int

        self.latent_size = latent_size

        self.encoder = Encoder(state_size + 1, latent_size)
        self.decoder = Decoder(latent_size + state_size, 1)

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

        means, log_vars = self.encoder([edge_features_input_p1, edge_features_input_p2])

        means_p1 = means[0]
        means_p2 = means[1]
        log_var_p1 = log_vars[0]
        log_var_p2 = log_vars[1]

        z = []
        decoder_inputs = []
        for k, (m, l) in enumerate(zip(means, log_vars)):
            eps = torch.randn([batch_size, 6, self.latent_size])

            std = torch.exp(0.5 * l)
            z_c = eps * std + m

            # std_p2 = torch.exp(0.5 * log_var_p2)
            # z_p2 = eps * std_p2 + means_p2

            z_c_m = torch.randn((z_c.size(0), self.nb_nodes, self.nb_nodes))
            new = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]
            for i, n in enumerate(new):
                z_c_m[:, n[0], n[1]] = z_c[:, i, 0]
                # z_c_m[:, n[0], n[1]] = z_c[:, i, 0]

            # z_m.append(z_c_m)
            decoder_inputs.append(torch.stack([torch.cat((z_c_m[:, i, j].unsqueeze(-1), state[:, i] - state[:, j]), dim=-1)
                                                           for i, j in zip(row, col)], dim=1))
            z.append(z_c)


        # decoder_edges_features_input_p1 = torch.stack([torch.cat((z_m[0][:, i, j].unsqueeze(-1), state[:, i] - state[:, j]), dim=-1)
        #                                       for i, j in zip(row, col)], dim=1)

        # z_p2_m = torch.zeros((z_p2.size(0), self.nb_nodes, self.nb_nodes))
        # new = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]
        # for i, n in enumerate(new):
        #     z_p2_m[:, n[0], n[1]] = z_p2[:, i, 0]
        #     z_p2_m[:, n[0], n[1]] = z_p2[:, i, 0]

        # decoder_edges_features_input_p2 = torch.stack([torch.cat((z_m[1][:, i, j].unsqueeze(-1), state[:, i] - state[:, j]), dim=-1)
        #                                                for i, j in zip(row, col)], dim=1)


        # recon_x_p1, recon_x_p2 = self.decoder(decoder_edges_features_input_p1, decoder_edges_features_input_p2)
        recon_x = self.decoder(decoder_inputs[0], decoder_inputs[1])
        for i in range(2):
            recon_x[i] = recon_x[i].reshape(recon_x[i].size(0), self.nb_nodes, 2)
            means[i] = means[i].reshape(means[i].size(0), self.nb_nodes, 2)
            log_vars[i] = log_vars[i].reshape(log_vars[i].size(0), self.nb_nodes, 2)

        return recon_x[0], means[0], log_vars[0], z[0], recon_x[1], means[1], log_vars[1], z[1]

        # recon_x_p1 = recon_x_p1.reshape(recon_x_p1.size(0), self.nb_nodes, 2)
        # means_p1 = means_p1.reshape(means_p1.size(0), self.nb_nodes, 2)
        # log_var_p1 = log_var_p1.reshape(log_var_p1.size(0), self.nb_nodes, 2)
        #
        # recon_x_p2 = recon_x_p2.reshape(recon_x_p2.size(0), self.nb_nodes, 2)
        # means_p2 = means_p2.reshape(means_p2.size(0), self.nb_nodes, 2)
        # log_var_p2 = log_var_p2.reshape(log_var_p2.size(0), self.nb_nodes, 2)

        # return recon_x_p1, means_p1, log_var_p1, z[0], recon_x_p2, means_p2, log_var_p2, z[1]

    def inference(self, state, nb_predicates=2):
        batch_size = state.size(0)
        z1 = torch.randn([batch_size, 6, self.latent_size])
        z2 = torch.randn([batch_size, 6, self.latent_size])
        recon_state = self.decoder(torch.stack([torch.cat((z1[:, 0, :], state[:, 0] - state[:, 1]), dim=1),
                                                       torch.cat((z1[:, 1, :], state[:, 1] - state[:, 0]), dim=1),
                                                       torch.cat((z1[:, 2, :], state[:, 0] - state[:, 2]), dim=1),
                                                       torch.cat((z1[:, 3, :], state[:, 2] - state[:, 0]), dim=1),
                                                       torch.cat((z1[:, 4, :], state[:, 1] - state[:, 2]), dim=1),
                                                       torch.cat((z1[:, 5, :], state[:, 2] - state[:, 1]), dim=1),
                                                       ], dim=1),
                                                     torch.stack([torch.cat((z2[:, 0, :], state[:, 0] - state[:, 1]), dim=1),
                                                                  torch.cat((z2[:, 1, :], state[:, 1] - state[:, 0]), dim=1),
                                                                  torch.cat((z2[:, 2, :], state[:, 0] - state[:, 2]), dim=1),
                                                                  torch.cat((z2[:, 3, :], state[:, 2] - state[:, 0]), dim=1),
                                                                  torch.cat((z2[:, 4, :], state[:, 1] - state[:, 2]), dim=1),
                                                                  torch.cat((z2[:, 5, :], state[:, 2] - state[:, 1]), dim=1),
                                                                  ], dim=1)
                                                     )
        recon_state_1 = recon_state[0].reshape(recon_state[0].size(0), self.nb_nodes, 2)
        recon_state_2 = recon_state[1].reshape(recon_state[1].size(0), self.nb_nodes, 2)
        return recon_state_1, recon_state_2