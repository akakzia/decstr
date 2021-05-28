import torch
import torch.nn as nn

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