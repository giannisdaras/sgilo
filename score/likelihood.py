# Code adapted from: https://github.com/yang-song/score_sde_pytorch/blob/main/likelihood.py
import torch
import numpy as np
from scipy import integrate
from utils import get_score_fn, from_flattened_numpy, to_flattened_numpy
from torchdiffeq import odeint_adjoint as odeint
# from torchdiffeq import odeint

def get_div_fn(fn):
  """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

  def div_fn(x, t, eps):
    with torch.enable_grad():
      x.requires_grad_(True)
      fn_eps = torch.sum(fn(x, t) * eps)
      grad_fn_eps = torch.autograd.grad(fn_eps, x)[0]
    # x.requires_grad_(False)
    return torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(x.shape))))

  return div_fn


def get_likelihood_fn(sde, inverse_scaler, 
                      hutchinson_type='Rademacher',
                      rtol=1e-5, 
                      atol=1e-5, 
                      method='euler', 
                      eps=1e-5, 
                      evals=100):
  """Create a function to compute the unbiased log-likelihood estimate of a given data point.

  Args:
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    inverse_scaler: The inverse data normalizer.
    hutchinson_type: "Rademacher" or "Gaussian". The type of noise for Hutchinson-Skilling trace estimator.
    rtol: A `float` number. The relative tolerance level of the black-box ODE solver.
    atol: A `float` number. The absolute tolerance level of the black-box ODE solver.
    method: A `str`. The algorithm for the black-box ODE solver.
      See documentation for `scipy.integrate.solve_ivp`.
    eps: A `float` number. The probability flow ODE is integrated to `eps` for numerical stability.

  Returns:
    A function that a batch of data points and returns the log-likelihoods in bits/dim,
      the latent code, and the number of function evaluations cost by computation.
  """
  
  def drift_fn(model, x, t):
    """The drift function of the reverse-time SDE."""
    score_fn = get_score_fn(sde, model, train=False, continuous=True)
    # Probability flow ODE is a special case of Reverse SDE
    rsde = sde.reverse(score_fn, probability_flow=True)
    return rsde.sde(x, t)[0]

  def div_fn(model, x, t, noise):
    return get_div_fn(lambda xx, tt: drift_fn(model, xx, tt))(x, t, noise)
  
  def likelihood_fn(model, data):
    shape = data.shape
    if hutchinson_type == 'Gaussian':
      epsilon = torch.randn_like(data)
    elif hutchinson_type == 'Rademacher':
      epsilon = torch.randint_like(data, low=0, high=2).float() * 2 - 1.
    else:
      raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")

    def ode_func(t, x):
      with torch.enable_grad():
        sample = x[:-1].reshape(shape)
        vec_t = torch.ones(sample.shape[0], device=sample.device) * t
        drift = drift_fn(model, sample, vec_t)
        logp_grad = div_fn(model, sample, vec_t, epsilon)
        return torch.cat([drift.flatten(), logp_grad], dim=0)

    init = torch.cat([data.flatten(), torch.zeros(data.shape[0], device=data.device)], dim=0)   
    solution = odeint(
      ode_func, 
      init, 
      torch.linspace(eps, sde.T, evals, device=data.device), 
      rtol=rtol,
      atol=atol,
      method=method,
      adjoint_params=[init])
    z = solution[-1][:-1].reshape(shape)
    delta_logp = solution[-1][-1]
    prior_logp = sde.prior_logp(z)
    bpd = -(prior_logp + delta_logp) / np.log(2)
    N = np.prod(shape[1:])
    bpd = bpd / N
    # A hack to convert log-likelihoods to bits/dim
    offset = 7. - inverse_scaler(-1.)
    bpd = bpd + offset
    return bpd, z, evals

  return likelihood_fn

