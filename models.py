import torch
import torch.nn as nn
import numpy as np


class GaussianParameterizedNetwork(nn.Module):
    def __init__(self, n_features, covariance_type='diag', variance_range_limiter=None):
        super(type(self), self).__init__()
        #self.covariance_type = covariance_type
        self.variance_range_limiter = variance_range_limiter
        if covariance_type not in ['spherical', 'diag']:
            raise ValueError("covariance_type should be in ['spherical', 'diag']")
        if len(n_features) < 2:
            raise ValueError("Please specify at least two layers")
        self.hidden_layers = nn.ModuleList([nn.Linear(n_features[k], n_features[k+1])
                                            for k in range(len(n_features)-2)])
        self.param_mu = nn.Linear(n_features[-2], n_features[-1])
        if covariance_type == 'diag':
            self.param_logvar = nn.Linear(n_features[-2], n_features[-1])
        elif covariance_type == 'spherical':
            self.param_logvar = nn.Linear(n_features[-2], 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        log_var = self.param_logvar(x)
        if self.variance_range_limiter is not None:
            log_var = nn.Tanh()(log_var)*self.variance_range_limiter
        return self.param_mu(x), log_var
    
""" 
class BernoulliParameterizedNetwork(torch.nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim=128):
        super(type(self), self).__init__()
        self.hidden1 = torch.nn.Linear(latent_dim, hidden_dim)
        self.hidden2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.x_plogits = torch.nn.Linear(hidden_dim, output_dim)
        self.activation = torch.nn.Softplus()

    def forward(self, z):
        h = self.activation(self.hidden1(z))
        h = self.activation(self.hidden2(h))
        return self.x_plogits(h)
"""    

def gaussian_log_likelihood(x, mu, logvar):
    C = torch.log(torch.tensor([2*np.pi]))
    return -0.5*(C + logvar + (x.unsqueeze(1) - mu).pow(2)/logvar.exp()).sum(dim=-1)


"""
Do this: https://stackoverflow.com/questions/60974047/importance-weighted-autoencoder-doing-worse-than-vae

interleave instead of new dimension, that way convolutions can be used

"""
class _StochasticAutoencoder(nn.Module):
    def __init__(self, latent_dim, data_dim, hidden_dim=128):
        #super(type(self), self).__init__() 
        nn.Module.__init__(self)
        self.encoder = GaussianParameterizedNetwork(n_features=[data_dim] + [hidden_dim]*2  +[latent_dim],
                                                    covariance_type='diag')
        # TODO: variance_range_limiter, greater than 4 does not work, why?
        # Check upper/lower range reached
        self.decoder = GaussianParameterizedNetwork(n_features=[latent_dim] + [hidden_dim]*2 + [data_dim],
                                                    covariance_type='diag', variance_range_limiter=4)

    def sample(self, mu, std, mc_samples=1):
        batch_size, n_latent = mu.shape
        eps = torch.randn(batch_size, mc_samples, n_latent, 
                          device=std.device, requires_grad=False)
        return eps.mul(std.unsqueeze(1)).add(mu.unsqueeze(1))

    def forward(self, x, mc_samples=1):
        z_mu, z_logvar = self.encoder(x)
        z = self.sample(z_mu, (0.5*z_logvar).exp(), mc_samples)
        return self.decoder(z), (z_mu, z_logvar), z
    
    def negELBO(self, x, mc_samples=1):
        dec_output, enc_output, z = self.forward(x, mc_samples)
        dec_mu, dec_logvar = dec_output
        enc_mu, enc_logvar = enc_output
        logpxz = gaussian_log_likelihood(x, dec_mu, dec_logvar)
        logqzxpz = self.KL_cost(enc_mu, enc_logvar, z)
        C = torch.log(torch.Tensor([mc_samples]))
        loss = torch.sum(torch.logsumexp(logqzxpz - logpxz, dim=1)) + C
        #loss = torch.sum(logqzxpz - logpxz)
        return loss, logpxz.sum(), logqzxpz.sum()
        

# TODO: move criterion out
class VariationalAutoencoder(_StochasticAutoencoder):    
    
    def KL_cost(self, mu, logvar, z):
        return -0.5*(1. + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).unsqueeze_(1)
    
                         
class ImportanceWeightedAutoencoder(_StochasticAutoencoder):

    def KL_cost(self, mu, logvar, z):
        return 0.5*(z.pow(2) - (z - mu.unsqueeze(1)).pow(2)/logvar.exp().unsqueeze(1) - logvar.unsqueeze(1)).sum(dim=-1)
    
    