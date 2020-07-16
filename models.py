import torch
import numpy as np
#from torch.nn import functional as F

"""
def logsumexp(inputs, dim=None, keepdim=True):    
    # From: https://github.com/YosefLab/scVI/issues/13
    return (inputs - F.log_softmax(inputs, dim=dim)).sum(dim, keepdim=keepdim)
"""

class GaussianEncoder(torch.nn.Module):
    def __init__(self, latent_dim, input_dim, hidden_dim):
        super(type(self), self).__init__()
        self.hidden1 = torch.nn.Linear(input_dim, hidden_dim)
        self.hidden2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.z_mu = torch.nn.Linear(hidden_dim, latent_dim)
        self.z_logvar = torch.nn.Linear(hidden_dim, latent_dim)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        h = self.activation(self.hidden1(x))
        h = self.activation(self.hidden2(h))
        return self.z_mu(h), self.z_logvar(h)
    
class GaussianDecoder(torch.nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim):
        super(type(self), self).__init__()
        self.hidden1 = torch.nn.Linear(latent_dim, hidden_dim)
        self.hidden2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.x_mu = torch.nn.Linear(hidden_dim, output_dim)
        self.x_logvar = torch.nn.Linear(hidden_dim, output_dim)
        self.activation = torch.nn.ReLU()
        self.range_limiter = torch.nn.Tanh()

    def forward(self, z):
        h = self.activation(self.hidden1(z))
        h = self.activation(self.hidden2(h))
        return self.x_mu(h), self.range_limiter(self.x_logvar(h))*2
"""    
class BernoulliDecoder(torch.nn.Module):
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
class VariationalAutoencoder(torch.nn.Module):
    def __init__(self, latent_dim, data_dim, hidden_dim=128, decoder_dist='Gaussian'):
        super(type(self), self).__init__() 
        self.encoder = GaussianEncoder(latent_dim, input_dim=data_dim, hidden_dim=hidden_dim)
        if decoder_dist == 'Gaussian':
            self.decoder = GaussianDecoder(latent_dim, output_dim=data_dim, hidden_dim=hidden_dim)

    def sample(self, mu, std, mc_samples=1):
        batch_size, n_latent = mu.shape
        eps = torch.randn(batch_size, mc_samples, n_latent, 
                          device=std.device, requires_grad=False)
        return eps.mul(std.unsqueeze(1)).add(mu.unsqueeze(1))

    def forward(self, x, mc_samples=1):
        z_mu, z_logvar = self.encoder(x)
        z = self.sample(z_mu, (0.5*z_logvar).exp(), mc_samples)
        return self.decoder(z), (z_mu, z_logvar), z


def ELBO_VAE(x, dec_output, enc_output):
    dec_mu, dec_logvar = dec_output
    enc_mu, enc_logvar = enc_output
    mc_samples = dec_mu.shape[1] # number of monte-carlo samples
    C = torch.log(torch.tensor(2*np.pi)) # Gaussian constant factor
    # Log likelihood of the decoder
    logpxz = -0.5*(C + dec_logvar + (x.unsqueeze(1) - dec_mu).pow(2)/dec_logvar.exp()).sum(dim=-1)
    # KL divergence between encoder distribution and standard normal prior
    logqzxpz = -0.5 * (1.0 + enc_logvar - enc_mu.pow(2) - enc_logvar.exp()).sum(dim=-1).unsqueeze_(1)
    
    #ELBO = torch.sum(logsumexp(logqzxpz - logpxz, dim=1) + np.log(mc_samples))
    ELBO = torch.sum(logqzxpz - logpxz) # Only for k=1
    return ELBO, logpxz.sum()/mc_samples, logqzxpz.sum()/logqzxpz.shape[1]


def ELBO_IWAE(x, dec_output, enc_output, z):   
    dec_mu, dec_logvar = dec_output
    enc_mu, enc_logvar = enc_output
    mc_samples = dec_mu.shape[1] # number of monte-carlo samples
    C = torch.log(torch.tensor(2*np.pi)) # Gaussian constant factor
    logpxz = -0.5*(C + dec_logvar + (x.unsqueeze(1) - dec_mu).pow(2)/dec_logvar.exp()).sum(dim=-1)
    logqzxpz = 0.5 * (z.pow(2) - z.sub(enc_mu.unsqueeze(1)).pow(2)/enc_logvar.unsqueeze(1).exp() - enc_logvar.unsqueeze(1)).sum(dim=-1)
    ELBO = torch.sum(logsumexp(logqzxpz - logpxz, dim=1) + torch.log(torch.tensor(mc_samples)))
    return ELBO, logpxz.sum()/mc_samples, logqzxpz.sum()/logqzxpz.shape[1]
