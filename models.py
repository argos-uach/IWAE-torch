import torch
import torch.nn as nn
import numpy as np


class GaussianLayer(nn.Module):
    def __init__(self, input_dim, output_dim, covariance_type='diagonal', variance_range_limiter=None):
        super(type(self), self).__init__()
        self.variance_range_limiter = variance_range_limiter
        
        if covariance_type not in ['spherical', 'diagonal']:
            raise ValueError("covariance_type should be in ['spherical', 'diagonal']")
        
        self.param_mu = nn.Linear(input_dim, output_dim)    
        if covariance_type == 'diagonal':
            self.param_logvar = nn.Linear(input_dim, output_dim)
        elif covariance_type == 'spherical':
            self.param_logvar = nn.Linear(input_dim, 1)
            
    def forward(self, x):
        log_var = self.param_logvar(x)
        if self.variance_range_limiter is not None:
            log_var = nn.Tanh()(log_var)*self.variance_range_limiter
        return self.param_mu(x), self.param_logvar(x)

   
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
        
        encoder_dim = [data_dim] + [hidden_dim]*2
        self.encoder_network = nn.ModuleList([nn.Linear(input_dim, output_dim)
                                              for input_dim, output_dim in zip(encoder_dim[:-1], encoder_dim[1:])])
        
        # Inferential distribution parameters
        self.q_parameters = GaussianLayer(input_dim=encoder_dim[-1], output_dim=latent_dim, 
                                          covariance_type='diagonal')
        
        
        decoder_dim = [latent_dim] + [hidden_dim]*2
        self.decoder_network = nn.ModuleList([nn.Linear(input_dim, output_dim)
                                              for input_dim, output_dim in zip(decoder_dim[:-1], decoder_dim[1:])])
        # Generative distribution parameters
        self.p_parameters = GaussianLayer(input_dim=decoder_dim[-1], output_dim=data_dim, 
                                          covariance_type='spherical')
        # TODO: variance_range_limiter, greater than 4 does not work, why?
        # Check upper/lower range reached
        self.activation = nn.ReLU()

    def encode(self, x):
        for layer in self.encoder_network:
            x = self.activation(layer(x))
        return self.q_parameters(x)
        
    def sample(self, mu, std, mc_samples=1):
        batch_size, n_latent = mu.shape
        eps = torch.randn(batch_size, mc_samples, n_latent, 
                          device=std.device, requires_grad=False)
        return eps.mul(std.unsqueeze(1)).add(mu.unsqueeze(1))

    def decode(self, z):
        for layer in self.decoder_network:
            z = self.activation(layer(z))
        return self.p_parameters(z)
        
    def forward(self, x, mc_samples=1):
        q_mu, q_logvar = self.encode(x)
        z = self.sample(q_mu, (0.5*q_logvar).exp(), mc_samples)
        return self.decode(z), (q_mu, q_logvar), z
    
    def negELBO(self, x, mc_samples=1):
        p_parameters, q_parameters, z = self.forward(x, mc_samples)
        p_mu, p_logvar = p_parameters
        q_mu, q_logvar = q_parameters
        logpxz = gaussian_log_likelihood(x, p_mu, p_logvar)
        logqzxpz = self.KL_cost(q_mu, q_logvar, z)
        C = torch.log(torch.Tensor([mc_samples]))
        loss = torch.sum(torch.logsumexp(logqzxpz - logpxz, dim=1)) + C
        return loss, logpxz.sum(), logqzxpz.sum()
        

# TODO: move criterion out
class VariationalAutoencoder(_StochasticAutoencoder):    
    
    def KL_cost(self, mu, logvar, z):
        return -0.5*(1. + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).unsqueeze_(1)
    
                         
class ImportanceWeightedAutoencoder(_StochasticAutoencoder):

    def KL_cost(self, mu, logvar, z):
        return 0.5*(z.pow(2) - (z - mu.unsqueeze(1)).pow(2)/logvar.exp().unsqueeze(1) - logvar.unsqueeze(1)).sum(dim=-1)
    
    