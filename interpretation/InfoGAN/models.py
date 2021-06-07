import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.linear3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.linear1(x)))

        x = F.relu(self.bn2(self.linear2(x)))


        recon = self.linear3(x)
        return recon

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)

        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.bn2(self.linear2(x)))

        return x

class DHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        output = torch.sigmoid(self.linear(x))

        return output

class QHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim_disc, output_dim_con):
        super().__init__()

        self.linear = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(input_dim)

        self.linear_disc = nn.Linear(hidden_dim, output_dim_disc)
        self.linear_mu = nn.Linear(hidden_dim, output_dim_con)
        self.linear_var = nn.Linear(hidden_dim, output_dim_con)

    def forward(self, x):
        x = F.relu(self.bn(self.linear(x)))

        disc_logits = self.linear_disc(x).squeeze()

        mu = self.linear_mu(x).squeeze()
        var = torch.exp(self.linear_var(x).squeeze())

        return disc_logits, mu, var