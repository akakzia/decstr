import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from itertools import combinations


def idx2onehot(idx, n):

    assert torch.max(idx).item() < n
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)

    onehot = torch.zeros(idx.size(0), n)
    onehot.scatter_(1, idx, 1)

    return onehot


class ContextVAE(nn.Module):

    def __init__(self, nb_words, encoder_inner_sizes=[32], decoder_inner_sizes=[32], state_size=9, embedding_size=20,
                 latent_size=9, binary=True, relational=False):
        super().__init__()

        assert type(encoder_inner_sizes) == list
        assert type(decoder_inner_sizes) == list
        assert type(latent_size) == int
        assert type(state_size) == int
        assert type(embedding_size) == int

        self.latent_size = latent_size
        self.state_size = state_size
        self.embedding_size = embedding_size

        self.sentence_encoder = nn.RNN(input_size=nb_words,
                                       hidden_size=embedding_size,
                                       num_layers=1,
                                       nonlinearity='tanh',
                                       bias=True,
                                       batch_first=True)

        self.relational = relational

        encoder_layer_sizes = [3 * 2 + 3 * 2 + embedding_size] + encoder_inner_sizes
        decoder_layer_sizes = [latent_size + 3 + 2*3 + embedding_size] + decoder_inner_sizes + [3]
        self.encoder = RelationalEncoder(encoder_layer_sizes, latent_size)
        self.decoder = RelationalDecoder(decoder_layer_sizes, binary=binary)

    def forward(self, initial_s, sentence, current_s):

        batch_size = current_s.size(0)
        assert current_s.size(0) == initial_s.size(0) == sentence.size(0)

        embeddings = self.sentence_encoder.forward(sentence)[0][:, -1, :]
        means, log_var = self.encoder(initial_s, embeddings, current_s)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.latent_size])
        z = eps * std + means

        recon_x = self.decoder(z, embeddings, initial_s)
        return recon_x, means, log_var, z

    def inference(self, initial_s, sentence, pair=None, n=1):

        batch_size = n
        embeddings = self.sentence_encoder.forward(sentence)[0][:, -1, :]

        if pair is None:
            z = torch.stack([torch.randn([batch_size, self.latent_size]) for _ in range(3)])
        else:
            pair_to_id = {(0, 1): 0, (0, 2): 1, (1, 2): 2, (1, 0): 0, (2, 0): 1, (2, 1): 2}
            i = pair_to_id[pair]
            z = torch.stack([torch.zeros([batch_size, self.latent_size]) if k != i else torch.randn([batch_size, self.latent_size]) for k in range(3)])
        recon_state = self.decoder(z, embeddings, initial_s)

        return recon_state


class RelationalEncoder(MessagePassing):
    def __init__(self, layer_sizes, latent_size):
        super(RelationalEncoder, self).__init__(aggr='add')

        self.MLP = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, initial_s, embeddings, current_s):
        ids_in_config = [[0, 3, 4], [1, 5, 6], [2, 7, 8]]
        batch_size = initial_s.shape[0]
        n_nodes = 3
        one_hot_encodings = [torch.tensor([1., 0., 0.]), torch.tensor([0., 1., 0.]), torch.tensor([0., 0., 1.])]
        edges_index = torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.long)

        row, col = edges_index

        nodes = torch.cat(batch_size * one_hot_encodings).reshape(batch_size, n_nodes, n_nodes)

        inp = torch.stack([torch.cat([nodes[:, :, row[i]], nodes[:, :, col[i]], initial_s[:, ids_in_config[i]],
                         current_s[:, ids_in_config[i]], embeddings], dim=-1) for i in range(row.size(0))])


        # inp = []
        # count = 0
        # for i, j in combinations([k for k in range(n_nodes)], 2):
        #     oh_i = torch.cat(batch_size * [one_hot_encodings[i]]).reshape(batch_size, n_nodes)
        #     oh_j = torch.cat(batch_size * [one_hot_encodings[j]]).reshape(batch_size, n_nodes)
        #     inp.append(torch.cat([oh_i, oh_j, initial_s[:, ids_in_config[count]], current_s[:, ids_in_config[count]], embeddings], dim=-1))
        #     count += 1

        x = self.MLP(inp)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class RelationalDecoder(nn.Module):

    def __init__(self, layer_sizes, binary):

        super().__init__()

        self.MLP = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i + 2 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                if binary:
                    self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

    def forward(self, z, embeddings, initial_s):
        batch_size = initial_s.shape[0]
        n_nodes = 3
        one_hot_encodings = [torch.tensor([1., 0., 0.]), torch.tensor([0., 1., 0.]), torch.tensor([0., 0., 1.])]
        ids_in_config = [[0, 3, 4], [1, 5, 6], [2, 7, 8]]

        inp = []
        count = 0
        for i, j in combinations([k for k in range(n_nodes)], 2):
            oh_i = torch.cat(batch_size * [one_hot_encodings[i]]).reshape(batch_size, n_nodes)
            oh_j = torch.cat(batch_size * [one_hot_encodings[j]]).reshape(batch_size, n_nodes)
            inp.append(torch.cat([oh_i, oh_j, initial_s[:, ids_in_config[count]], z[count], embeddings], dim=-1))
            count += 1

        inp = torch.stack(inp)
        x = self.MLP(inp)

        # flatten edges
        flat_x = torch.cat([x[i] for i in range(3)], dim=-1)[:, [0, 3, 4, 1, 5, 6, 2, 7, 8]]

        return flat_x