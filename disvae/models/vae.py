"""
Module containing the main VAE class.
"""
import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
from disvae.utils.initialization import weights_init
from .encoders import get_encoder, DomainEncoder, ConvEncoder
from .decoders import get_decoder, ResDecoder, ConvDecoder
import random
MODELS = ["Burgess"]


def init_specific_model(model_type, img_size, latent_dim):
    """Return an instance of a VAE with encoder and decoder from `model_type`."""
    model_type = model_type.lower().capitalize()
    if model_type not in MODELS:
        err = "Unkown model_type={}. Possible values: {}"
        raise ValueError(err.format(model_type, MODELS))

    encoder = get_encoder(model_type)
    decoder = get_decoder(model_type)
    model = VAE(img_size, encoder, decoder, latent_dim)
    model.model_type = model_type  # store to help reloading
    return model


class VAE(nn.Module):
    def __init__(self, img_size, encoder, decoder, latent_dim):
        """
        Class which defines model and forward pass.

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).
        """
        super(VAE, self).__init__()

        if list(img_size[1:]) not in [[32, 32], [64, 64]]:
            raise RuntimeError("{} sized images not supported. Only (None, 32, 32) and (None, 64, 64) supported. Build your own architecture or reshape images!".format(img_size))

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.num_pixels = self.img_size[1] * self.img_size[2]
        self.encoder = encoder(img_size, self.latent_dim)
        self.decoder = decoder(img_size, self.latent_dim)

        self.reset_parameters()

    def reparameterize(self, mean, logvar):
        """
        Samples from a normal distribution using the reparameterization trick.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (batch_size, latent_dim)

        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (batch_size,
            latent_dim)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + std * eps
        else:
            # Reconstruction mode
            return mean

    def forward(self, x):
        """
        Forward pass of model.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        latent_dist = self.encoder(x)
        latent_sample = self.reparameterize(*latent_dist)
        reconstruct = self.decoder(latent_sample)
        return reconstruct, latent_dist, latent_sample

    def reset_parameters(self):
        self.apply(weights_init)

    def sample_latent(self, x):
        """
        Returns a sample from the latent distribution.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        latent_dist = self.encoder(x)
        latent_sample = self.reparameterize(*latent_dist)
        return latent_sample


class VAEBase(nn.Module):
    def __init__(self, args):
        super(VAEBase, self).__init__()
        self.args = args
        self.domain_list = np.arange(args.num_domains - 1)
        # create the encoder and decoder networks
        self.encoder_d = DomainEncoder(args.num_domains, args.domain_in_features)

        self.encoder_z = ConvEncoder(args.in_features)
        # self.decoder = ConvDecoder(args.in_features + args.domain_in_features)
        
        self.decoder = ResDecoder(args.in_features + args.domain_in_features)


    def reparameterize(self, mean, logvar):
        """
        Samples from a normal distribution using the reparameterization trick.
        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (batch_size, latent_dim)
        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (batch_size,
            latent_dim)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + std * eps
        else:
            # Reconstruction mode
            return mean

    def forward(self, x, domain_labels, domain_id):
        """
        Forward pass of model.
        Parameters
        ----------checkpoint
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        fake_domains = np.delete(self.domain_list, domain_id) if domain_id < self.args.num_domains - 1 else self.domain_list
        fake_domain_labels = torch.tensor(random.choices(fake_domains, k=domain_labels.size()[0])).cuda()
        d_latent_dist = self.encoder_d(domain_labels[:, domain_id])
        d_fake_latent_dist = self.encoder_d(fake_domain_labels)
        z_latent_dist = self.encoder_z(x)
        d_latent_sample = self.reparameterize(*d_latent_dist)
        d_fake_latent_sample = self.reparameterize(*d_fake_latent_dist)
        z_latent_sample = self.reparameterize(*z_latent_dist)
        reconstruct = self.decoder(torch.cat((d_latent_sample, z_latent_sample), dim=1))
        fake_reconstruct = self.decoder(torch.cat((d_fake_latent_sample, z_latent_sample), dim=1))
        
        return reconstruct, d_latent_dist, d_latent_sample, z_latent_dist, z_latent_sample, fake_reconstruct, fake_domain_labels

    def sample_latent(self, x, num_domains, domain_in_features):
        """
        Returns a sample from the latent distribution.
        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        latent_dist = self.encoder_z(x)
        latent_dist_d = self.encoder_d(num_domains, domain_in_features)
        latent_sample_z = self.reparameterize(*latent_dist)
        latent_sample_d = self.reparameterize(*latent_dist_d)
        return latent_sample_z, latent_sample_d


    def reset_parameters(self):
        self.apply(weights_init)

    # def reset_parameters(self):
    #     self.apply(initialization.weights_init)
