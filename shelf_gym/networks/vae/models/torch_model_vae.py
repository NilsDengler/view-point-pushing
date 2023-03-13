import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import (
    Dataset,
    DataLoader,
)
import h5py
from dataset_utils import h5_to_np

class CustomDataset(Dataset):
    def __init__(self, root_dir, width, height):
        self.width = width
        self.height = height
        self.h5f_data = h5py.File(os.path.join(os.path.dirname(__file__), root_dir), "r")
        self.np_data = h5_to_np(self.h5f_data, width, height)
        print("Done with initializing data set of shape:", self.np_data.shape)
        self.h5f_data = "free"
    def __len__(self):
        return self.np_data.shape[0]

    def __getitem__(self, index):
        return torch.from_numpy(self.np_data[index]).float().reshape(1, self.width, self.height)


class Encoder(nn.Module):
    def __init__(self, latent_dims, width, height):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=2, padding=(0, 0)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            #nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 4), stride=2, padding=(1, 1)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 4), stride=2, padding=(0, 1)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
            #nn.MaxPool2d(2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4), stride=2, padding=(1, 1)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4, 4), stride=2, padding=(1, 1)),
            nn.LeakyReLU(0.1),
            nn.Flatten(),
        )

        test = np.ones((32, 1, width, height))
        with torch.no_grad():
            n_features = self.encoder(
                torch.as_tensor(test).float()
            ).shape[1]
        print("encoder features: ", n_features)
        self.fc_mu = nn.Sequential(nn.Linear(in_features=n_features, out_features=latent_dims))
        self.fc_logvar = nn.Sequential(nn.Linear(in_features=n_features, out_features=latent_dims))

    def forward(self, x):
        x = self.encoder(x)
        return self.fc_mu(x), self.fc_logvar(x)


class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(in_features=latent_dims, out_features=128 * 9 * 12)

        self.decoder = nn.Sequential(nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
                                nn.LeakyReLU(0.1),
                                nn.BatchNorm2d(64),
                                nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(3, 4), stride=(2, 2), padding=(1, 1)),
                                nn.LeakyReLU(0.1),
                                nn.BatchNorm2d(64),
                                nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(3, 4), stride=(2, 2), padding=(0, 1)),
                                nn.LeakyReLU(0.1),
                                nn.BatchNorm2d(32),
                                nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
                                nn.LeakyReLU(0.1),
                                nn.BatchNorm2d(32),
                                nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
                                nn.LeakyReLU(0.1),
                                nn.Sigmoid(),
                                )
        test = np.ones((1,128,6,8))
        with torch.no_grad():
           n_features = self.decoder(torch.as_tensor(test).float()).shape
        print("decoder features: ", n_features)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], 128, 9, 12)  # unflatten batch of feature vectors to a batch of multi-channel feature maps
        return self.decoder(x)

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim, beta, width, heigth):
        super(VariationalAutoencoder, self).__init__()
        self.beta = beta
        self.encoder = Encoder(latent_dim, width, heigth)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        loss, recon_loss, kl_loss = self.vae_loss(x_recon, x, latent_mu, latent_logvar)
        return x_recon, latent, loss, recon_loss, kl_loss

    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def vae_loss(self, recon_x, x, mu, logvar):
        # recon_x is the probability of a multivariate Bernoulli distribution p.
        # -log(p(x)) is then the pixel-wise binary cross-entropy.
        # Averaging or not averaging the binary cross-entropy over all pixels here
        # is a subtle detail with big effect on training, since it changes the weight
        # we need to pick for the other loss term by several orders of magnitude.
        # Not averaging is the direct implementation of the negative log likelihood,
        # but averaging makes the weight of the other loss term independent of the image resolution.
        recon_loss = F.binary_cross_entropy(recon_x.view(-1, x.shape[2]), x.view(-1, x.shape[2]), reduction='sum')
        #recon_loss = F.mse_loss(recon_x.view(-1, x.shape[2]), x.view(-1, x.shape[2]))
        # KL-divergence between the prior distribution over latent vectors
        # (the one we are going to sample from when generating new images)
        # and the distribution estimated by the generator for the given image.
        #kldivergence = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = self.beta * kldivergence
        #print("kdl = ", kldivergence, " beta = ", self.beta, " loss = ", kl_loss)
        return recon_loss + kl_loss,  recon_loss, kl_loss




