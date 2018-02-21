# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 15:23:59 2018
Affine Invariant MCMC Ensembles for Estimating
Simple Generative Linear Model
@author: jehagerty

"""
# Import Module Dependency Requirements
import numpy as np
import scipy.optimize as op
import emcee
import corner
import matplotlib.pyplot as pl

# Choose the "true" parameters.
m_true = -0.9594
b_true = 4.294
f_true = 0.534

# Generate some synthetic data from the model.
N = 50
x = np.sort(10*np.random.rand(N))
yerr = 0.1+0.5*np.random.rand(N)
y = m_true*x+b_true
y += np.abs(f_true*y) * np.random.randn(N)
y += yerr * np.random.randn(N)

# The ordinary least squares solution to these data 
A = np.vstack((np.ones_like(x), x)).T
C = np.diag(yerr * yerr)
cov = np.linalg.inv(np.dot(A.T, np.linalg.solve(C, A)))
b_ls, m_ls = np.dot(cov, np.dot(A.T, np.linalg.solve(C, y)))

 # The MLE solution for the same data where the error bars are assumed correct, Gaussian and independent.
 # This likelihood function is simply a Gaussian where the variance is underestimated by some fractional amount:
def lnlike(theta, x, y, yerr):
    m, b, lnf = theta
    model = m * x + b
    inv_sigma2 = 1.0/(yerr**2 + model**2*np.exp(2*lnf))
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

# A good way of finding this numerical optimum of this likelihood function is to use the scipy.optimize module:
nll = lambda *args: -lnlike(*args)
result = op.minimize(nll, [m_true, b_true, np.log(f_true)], args=(x, y, yerr))
m_ml, b_ml, lnf_ml = result["x"]   

 # The Bayesian solution for the same data...
# Marginalization & uncertainty estimation using the AIE-MCMC
# Define the log-prior:
def lnprior(theta):
    m, b, lnf = theta
    if -5.0 < m < 0.5 and 0.0 < b < 10.0 and -10.0 < lnf < 1.0:
        return 0.0
    return -np.inf 
 
# The full log-probability distribution function is, and the solution includes the lnlike function from the MLE step:
def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)

# Sample this distribution using AIE-MCMC. We’ll start by initializing the walkers in a tiny 
# Gaussian ball around the MLE result:
ndim, nwalkers = 3, 100
pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

# Set up the sampler:
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))

# Run the AIE-MCMC for 500 steps starting from the tiny ball defined above:
sampler.run_mcmc(pos, 500)

# Flatten the chain, and discard the first 50 samples, due to "burn-in" timing
samples = sampler.chain[:, 50:, :].reshape((-1, ndim)) 

# Plot 1 & 2-Dimensional Projections of the Posterior of the Model Parameters 
fig = corner.corner(samples, labels=["$m$", "$b$", "$\ln\,f$"],
                      truths=[m_true, b_true, np.log(f_true)])
fig.savefig("triangle.png")

# Plot the Projection of the results into the space of the observed data. 
xl = np.array([0, 10])
for m, b, lnf in samples[np.random.randint(len(samples), size=100)]:
    pl.plot(xl, m*xl+b, color="k", alpha=0.1)
pl.plot(xl, m_true*xl+b_true, color="r", lw=2, alpha=0.8)
pl.errorbar(x, y, yerr=yerr, fmt=".k")

# Print 16th 50th and 84th percentiles of the Marginals 
samples[:, 2] = np.exp(samples[:, 2])
m_mcmc, b_mcmc, f_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))
print(m_mcmc, b_mcmc, f_mcmc)