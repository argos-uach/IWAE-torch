import torch
import torch.nn as nn
import numpy as np


class StochasticLayer(nn.Module):
    def __init__(self, input_dim, output_dim, distribution='diagonal_gaussian', variance_range_limiter=None):
        super(type(self), self).__init__()
        
        allowed_distributions = ['diagonal_gaussian', 'spherical_gaussian', 'bernoulli_logits']
        if distribution not in allowed_distributions:
            raise ValueError(f'distribution should be in {allowed_distributions}')
        self.distribution = distribution
        self.variance_range_limiter = variance_range_limiter
        
        if distribution == 'diagonal_gaussian':
            self.mu = nn.Linear(input_dim, output_dim)
            self.logvar = nn.Linear(input_dim, output_dim)
        elif distribution == 'spherical_gaussian':
            self.mu = nn.Linear(input_dim, output_dim)
            self.logvar = nn.Linear(input_dim, 1)
        elif distribution == 'bernoulli_logits':
            self.logits = nn.Linear(input_dim, output_dim)
            
    def forward(self, x):
        if self.distribution in ['diagonal_gaussian', 'spherical_gaussian']:
            mu = self.mu(x)
            logvar = self.logvar(x)
            if self.variance_range_limiter is not None:
                logvar = nn.Tanh()(logvar)*self.variance_range_limiter # Make this more clear!
            return mu, logvar
        elif self.distribution in ['bernoulli_logits']:
            return self.logits(x)
    
    def sample(self, params, samples=1):
        if self.distribution in ['diagonal_gaussian', 'spherical_gaussian']:
            mu, logvar = params
            std = (0.5*logvar).exp()
            if samples > 1:
                mu = torch.repeat_interleave(mu, samples, dim=0)
                std = torch.repeat_interleave(std, samples, dim=0)
            with torch.no_grad():
                eps = torch.randn_like(mu)
            return mu + eps*std

    def log_likelihood(self, x, params):
        if self.distribution in ['diagonal_gaussian', 'spherical_gaussian']:
            mu, logvar = params
            C = torch.log(torch.tensor([2*np.pi]))
            return -0.5*(C + logvar + (x - mu).pow(2)/logvar.exp()).sum(dim=-1)
        elif self.distribution in ['bernoulli_logits']:
            return -1.*nn.BCEWithLogitsLoss(reduction='sum')(params, x)
            

class _StochasticAutoencoder(nn.Module):
    def __init__(self, latent_dim, data_dim, hidden_dim=128):
        #super(type(self), self).__init__() 
        nn.Module.__init__(self)
        
        encoder_dim = [data_dim] + [hidden_dim]*2
        self.encoder_network = nn.ModuleList([nn.Linear(input_dim, output_dim)
                                              for input_dim, output_dim in zip(encoder_dim[:-1], 
                                                                               encoder_dim[1:])])        
        # Inferential distribution $q_\phi(z|x)$
        # Only diagonal gaussian is supported for the approximate posterior
        self.approximate_posterior = StochasticLayer(input_dim=encoder_dim[-1], output_dim=latent_dim, 
                                                     distribution='diagonal_gaussian')        
        
        decoder_dim = [latent_dim] + [hidden_dim]*2
        self.decoder_network = nn.ModuleList([nn.Linear(input_dim, output_dim)
                                              for input_dim, output_dim in zip(decoder_dim[:-1], 
                                                                               decoder_dim[1:])])
        # Generative distribution $p_\theta(x|z)$
        self.generative_model = StochasticLayer(input_dim=decoder_dim[-1], output_dim=data_dim, 
                                                distribution='spherical_gaussian')
        # TODO: Continue testing diagonal gaussian decoder and variance range. Check upper/lower range reached
        self.activation = nn.ReLU()

    def encode(self, x):
        for layer in self.encoder_network:
            x = self.activation(layer(x))
        return self.approximate_posterior(x)
        
    def sample_encoder(self, parameters, mc_samples=1):
        return self.approximate_posterior.sample(parameters, mc_samples)

    def decode(self, z):
        for layer in self.decoder_network:
            z = self.activation(layer(z))
        return self.generative_model(z)
    
    def sample_decoder(self, parameters, mc_samples=1):
        with torch.no_grad():
            return self.generative_model.sample(parameters, mc_samples)
        
    def forward(self, x, mc_samples=1):
        q_parameters = self.encode(x)
        z = self.sample_encoder(q_parameters, mc_samples)
        p_parameters = self.decode(z)
        return p_parameters, q_parameters, z
    
    def negELBO(self, x, mc_samples=1):
        p_parameters, q_parameters, z = self.forward(x, mc_samples)
        q_mu, q_logvar = q_parameters
        if mc_samples > 1:
            x = torch.repeat_interleave(x, mc_samples, dim=0)
            q_mu = torch.repeat_interleave(q_mu, mc_samples, dim=0)
            q_logvar = torch.repeat_interleave(q_logvar, mc_samples, dim=0)
        logpxz = self.generative_model.log_likelihood(x, p_parameters)
        logqzxpz = self.KL_cost(q_mu, q_logvar, z)
        negELBO = (logqzxpz - logpxz).reshape(-1, mc_samples)
        loss = torch.sum(torch.logsumexp(negELBO, dim=1)) + torch.log(torch.Tensor([mc_samples]))
        return loss, logpxz.sum(), logqzxpz.sum()
        

# TODO: move criterion out
class VariationalAutoencoder(_StochasticAutoencoder):    
    
    def KL_cost(self, mu, logvar, z):
        return -0.5*(1. + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)
    
                         
class ImportanceWeightedAutoencoder(_StochasticAutoencoder):

    def KL_cost(self, mu, logvar, z):
        return 0.5*(z.pow(2) - (z - mu).pow(2)/logvar.exp() - logvar).sum(dim=-1)
    
    