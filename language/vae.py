import torch
import torch.nn as nn

def idx2onehot(idx, n):

    assert torch.max(idx).item() < n
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)

    onehot = torch.zeros(idx.size(0), n)
    onehot.scatter_(1, idx, 1)

    return onehot

class ContextVAE(nn.Module):

    def __init__(self, nb_words, inner_sizes=[32], state_size=9, embedding_size=20, latent_size=9, binary=True):

        super().__init__()


        assert type(inner_sizes) == list
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


        encoder_layer_sizes = [state_size * 2 + embedding_size] + inner_sizes
        decoder_layer_sizes = [latent_size + state_size + embedding_size] + inner_sizes + [state_size]
        self.encoder = Encoder(encoder_layer_sizes, latent_size)
        self.decoder = Decoder(decoder_layer_sizes, binary=binary)

    def forward(self, initial_s, sentence, current_s):

        batch_size = current_s.size(0)
        assert current_s.size(0) == initial_s.size(0) == sentence.size(0)


        embeddings = self.sentence_encoder.forward(sentence)[0][:, -1, :]
        means, log_var = self.encoder(torch.cat((initial_s, embeddings, current_s), dim=1))

        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.latent_size])
        z = eps * std + means


        recon_x = self.decoder(torch.cat((z, embeddings, initial_s), dim=1))

        return recon_x, means, log_var, z

    def inference(self, initial_s, sentence, n=1):

        batch_size = n
        z = torch.randn([batch_size, self.latent_size])

        embeddings = self.sentence_encoder.forward(sentence)[0][:, -1, :]
        recon_state = self.decoder(torch.cat((z, embeddings, initial_s), dim=1))

        return recon_state


class Encoder(nn.Module):

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

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(nn.Module):

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


    def forward(self, z):

        x = self.MLP(z)
        return x