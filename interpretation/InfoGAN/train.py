import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import random

from models import Generator, Discriminator, DHead, QHead
from arguments import get_args
from utils import *


args = get_args()
random.seed(args.seed)
torch.manual_seed(args.seed)
print("Random Seed: ", args.seed)

# Use GPU if available.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
print(device, " will be used.\n")

dataset = load_data()[0]
dataset = dataset.reshape(dataset.shape[0], dataset.shape[1] * dataset.shape[2])
dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)

args.latent_size = 3
args.num_dis_c = 0
args.dis_c_dim = 0
args.num_con_c = 0

# Initialise the network.
generator_input_dim = args.latent_size + args.dis_c_dim + args.num_con_c
netG = Generator(generator_input_dim, args.hidden_size, args.input_size).to(device)
netG.apply(weights_init)
print(netG)

discriminator = Discriminator(args.input_size, args.hidden_size, args.hidden_size).to(device)
discriminator.apply(weights_init)
print(discriminator)

netD = DHead(args.hidden_size).to(device)
netD.apply(weights_init)
print(netD)

netQ = QHead(args.hidden_size, args.hidden_size, args.dis_c_dim, args.num_con_c).to(device)
netQ.apply(weights_init)
print(netQ)

# Loss for discrimination between real and fake images.
criterionD = nn.BCELoss()
# Loss for discrete latent code.
criterionQ_dis = nn.CrossEntropyLoss()
# Loss for continuous latent code.
criterionQ_con = NormalNLLLoss()

# Adam optimiser is used.
optimD = optim.Adam([{'params': discriminator.parameters()}, {'params': netD.parameters()}], lr=args.learning_rate)
optimG = optim.Adam([{'params': netG.parameters()}, {'params': netQ.parameters()}], lr=args.learning_rate)

# Fixed Noise
z = torch.randn(100, args.latent_size, device=device)
fixed_noise = z
if args.num_dis_c != 0:
    idx = np.arange(args.dis_c_dim).repeat(50)
    dis_c = torch.zeros(100, args.num_dis_c, args.dis_c_dim, device=device)
    for i in range(args.num_dis_c):
        dis_c[torch.arange(0, 100), i, idx] = 1.0

    dis_c = dis_c.view(100, -1)

    fixed_noise = torch.cat((fixed_noise, dis_c), dim=1)

if args.num_con_c != 0:
    con_c = torch.rand(100, args.num_con_c, device=device) * 2 - 1
    fixed_noise = torch.cat((fixed_noise, con_c), dim=1)

real_label = 1
fake_label = 0

# List variables to store results pf training.
G_losses = []
D_losses = []

print("-"*25)
print("Starting Training Loop...\n")
print('Epochs: %d\nBatch Size: %d\nLength of Data Loader: %d' % (args.epochs, args.batch_size, len(dataloader)))
print("-"*25)

start_time = time.time()
iters = 0

for epoch in range(args.epochs):
    epoch_start_time = time.time()

    for i, data in enumerate(dataloader, 0):
        # Get batch size
        b_size = data.size(0)
        # Transfer data tensor to GPU/CPU (device)
        real_data = data.to(device)

        # Updating discriminator and DHead
        optimD.zero_grad()
        # Real data
        label = torch.full((b_size, ), real_label, device=device)
        output1 = discriminator(real_data.float())
        probs_real = netD(output1).view(-1)
        loss_real = criterionD(probs_real, label.float())
        # Calculate gradients.
        loss_real.backward()

        # Fake data
        label.fill_(fake_label)
        noise, idx = noise_sample(args.num_dis_c, args.dis_c_dim, args.num_con_c, args.latent_size, b_size, device)
        fake_data = netG(noise)
        output2 = discriminator(fake_data.detach())
        probs_fake = netD(output2).view(-1)
        loss_fake = criterionD(probs_fake, label.float())
        # Calculate gradients.
        loss_fake.backward()

        # Net Loss for the discriminator
        D_loss = loss_real + loss_fake
        # Update parameters
        optimD.step()

        # Updating Generator and QHead
        optimG.zero_grad()

        # Fake data treated as real.
        output = discriminator(fake_data)
        label.fill_(real_label)
        probs_fake = netD(output).view(-1)
        gen_loss = criterionD(probs_fake, label.float())

        q_logits, q_mu, q_var = netQ(output)
        target = torch.LongTensor(idx).to(device)
        # Calculating loss for discrete latent code.
        dis_loss = 0
        for j in range(args.num_dis_c):
            dis_loss += criterionQ_dis(q_logits[:, j*10 : j*10 + 10], target[j])

        # Calculating loss for continuous latent code.
        con_loss = 0
        if args.num_con_c != 0:
            con_loss = criterionQ_con(noise[:, args.latent_size + args.num_dis_c * args.dis_c_dim : ].view(-1, args.num_con_c), q_mu, q_var)*0.1

        # Net loss for generator.
        G_loss = gen_loss + dis_loss + con_loss
        # Calculate gradients.
        G_loss.backward()
        # Update parameters.
        optimG.step()

        # Check progress of training.
        if i != 0 and i%100 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                  % (epoch+1, args.epochs, i, len(dataloader),
                    D_loss.item(), G_loss.item()))

        # Save the losses for plotting.
        G_losses.append(G_loss.item())
        D_losses.append(D_loss.item())

        iters += 1

    epoch_time = time.time() - epoch_start_time
    print("Time taken for Epoch %d: %.2fs" %(epoch + 1, epoch_time))
    # Generate image after each epoch to check performance of the generator. Used for creating animated gif later.
    # with torch.no_grad():
    #     gen_data = netG(fixed_noise).detach().cpu()
    # img_list.append(vutils.make_grid(gen_data, nrow=10, padding=2, normalize=True))
    #
    # Generate image to check performance of generator.
    with torch.no_grad():
        gen_data = netG(fixed_noise).detach().cpu()
        stop = 1
    #     plt.figure(figsize=(10, 10))
    #     plt.axis("off")
    #     plt.imshow(np.transpose(vutils.make_grid(gen_data, nrow=10, padding=2, normalize=True), (1,2,0)))
    #     plt.savefig("Epoch_%d {}".format(params['dataset']) %(epoch+1))
    #     plt.close('all')
    #
    # # Save network weights.
    # if (epoch+1) % params['save_epoch'] == 0:
    #     torch.save({
    #         'netG' : netG.state_dict(),
    #         'discriminator' : discriminator.state_dict(),
    #         'netD' : netD.state_dict(),
    #         'netQ' : netQ.state_dict(),
    #         'optimD' : optimD.state_dict(),
    #         'optimG' : optimG.state_dict(),
    #         'params' : params
    #         }, 'checkpoint/model_epoch_%d_{}'.format(params['dataset']) %(epoch+1))

training_time = time.time() - start_time
print("-"*50)
print('Training finished!\nTotal Time for Training: %.2fm' %(training_time / 60))
print("-"*50)