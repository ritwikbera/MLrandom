import torch
import numpy as np 
import random

def sample(mu, var, nb_samples=500):
    out = []
    for i in range(nb_samples):
        out += [
            torch.normal(mu, var.sqrt())
        ]
    return torch.stack(out, dim=0)

def initialize(data, K, var=1):

  m = data.size(0)
  idxs = torch.from_numpy(np.random.choice(m, k, replace=False))
  mu = data[idxs]

  var = torch.Tensor(k, d).fill_(var)
  pi = torch.empty(k).fill_(1. / k)

  return mu, var, pi

log_norm_constant = -0.5 * np.log(2 * np.pi)

def log_gaussian(x, mean=0, logvar=0.):
  if type(logvar) == 'float':
      logvar = x.new(1).fill_(logvar)

  a = (x - mean) ** 2
  log_p = -0.5 * (logvar + a / logvar.exp())
  log_p = log_p + log_norm_constant

  return log_p

def get_likelihoods(X, mu, logvar, log=True):

  # get feature-wise log-likelihoods (K, examples, features)

  # take advantage of broadcasting
  
  log_likelihoods = log_gaussian(
      X[None, :, :], # (1, examples, features)
      mu[:, None, :], # (K, 1, features)
      logvar[:, None, :] # (K, 1, features)
  )

  # sum over the feature dimension
  log_likelihoods = log_likelihoods.sum(-1)

  if not log:
      log_likelihoods.exp_()

  return log_likelihoods

def get_posteriors(log_likelihoods):
  posteriors = log_likelihoods - logsumexp(log_likelihoods, dim=0, keepdim=True)
  return posteriors

def get_parameters(X, log_posteriors, eps=1e-6, min_var=1e-6):
  posteriors = log_posteriors.exp()

  # compute `N_k` the proxy "number of points" assigned to each distribution.
  K = posteriors.size(0)
  N_k = torch.sum(posteriors, dim=1) # (K)
  N_k = N_k.view(K, 1, 1)

  # get the means by taking the weighted combination of points
  # (K, 1, examples) @ (1, examples, features) -> (K, 1, features)
  mu = posteriors[:, None] @ X[None,]
  mu = mu / (N_k + eps)

  # compute the diagonal covar. matrix, by taking a weighted combination of
  # the each point's square distance from the mean
  A = X - mu
  var = posteriors[:, None] @ (A ** 2) # (K, 1, features)
  var = var / (N_k + eps)
  logvar = torch.clamp(var, min=min_var).log()

  # recompute the mixing probabilities
  m = X.size(1) # nb. of training examples
  pi = N_k / N_k.sum()

  return mu.squeeze(1), logvar.squeeze(1), pi.squeeze()