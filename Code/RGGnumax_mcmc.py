#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings("ignore")

import sys
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import os
import emcee
import ptemcee
import corner
import scipy.optimize as opt
import gc
import scipy
import scipy.special as spec
import RGGnumax as RGG
import itertools
from sklearn.externals import joblib
from wavelet_class import *
import fast_fit_class as ffc
import warnings
warnings.filterwarnings('ignore')
from tqdm import *
import time
import json

import RGGmodel as Model
from ptemcee.mpi_pool import MPIPool

# This way home is the same across all users and on all computers!
from os.path import expanduser
home = expanduser("~")

def rebin(f, smoo):
    if smoo < 1.5:
        return f
    smoo = int(smoo)
    m = len(f) // smoo
    ff = f[:m*smoo].reshape((m,smoo)).mean(1)
    return ff

class Likelihood:
    def __init__(self, _f, _obs, _model, smoo=1):
        self.obs = _obs
        self.f = _f
        self.model = _model
        self.sqrt_smoo = np.sqrt(smoo)
        self.smoo = smoo

    def __call__(self, params):
        self.model.set_parameter_vector(params)
        #plt.plot(self.f, self.obs, 'k')
        mod = self.model.compute_value()
        #plt.plot(self.f, mod, 'r')
        #plt.show()
        L = -1.0 * np.sum(np.log(mod) + \
                          self.obs/mod)
        if np.isnan(L) == True:
            return -1.0e30
        elif np.isfinite(L) == False:
            return -1.0e30
        else:
            return L * self.smoo

class Prior:
    def __init__(self, _gaussian, _Model):
        # Bounds now included in model!
        self.gaussian = _gaussian
        self.Model = _Model
        self.names = self.Model.get_parameter_names()

    def __call__(self, p):
        # We'll just put reasonable uniform priors on all the parameters.
        #if not all(b[0] < v < b[1] for v, b in zip(p, self.bounds)):
        #    return -np.inf

        # If any parameters are nans for some reason then return -np.inf
        if np.any(~np.isfinite(p)):
            return -np.inf

        self.Model.set_parameter_vector(p)
        lnprior = 0.0
        for idx, i in enumerate(self.gaussian):
            if i[1] != 0:
                lnprior += -0.5 * (p[idx] - i[0])**2 / i[1]**2

        # Add a prior in on the first component amplitude
        #lnprior += -0.5 * ((np.log(2e7) - 2.0*np.log(p[6])) - p[0])**2 / (np.log(8e6)**2)
        # Add a prior in on the first component timescale
        #sigma = 5.0 * p[6]**0.5
        #sigma = 5.0 * 200.0
        #lnprior += -0.5 * (p[6]**0.95/2.75 - p[1])**2 / sigma**2
        # Add a prior in on the second component amplitude
        #lnprior += -0.5 * ((np.log(1.5e7) - 2.4*np.log(p[6])) - p[3])**2 / (np.log(3e6)**2)
        # Add a prior in on the second component timescale
        #sigma2 = 5.0 * 300.0 #5.0*p[6]
        #lnprior += -0.5 * (p[6]/0.95 - p[4])**2 / (sigma2)**2
        # Add in a prior on numax and width
        # prior is width \approx numax * 0.17
        # Changed prior so on dnu_env not width!
        # Need to add in slightly more restrictive prior based upon
        # the current numax value
        if not (0.1*p[self.names.index("numax")] < p[self.names.index("denv")] < 0.9*p[self.names.index("numax")]):
            return -np.inf

        #if p[1] > p[4]:
        #    return -np.inf

        #if p[10] > p[1]:
        #    return -np.inf

        if not p[self.names.index("hfreq1")] < p[self.names.index("hfreq2")] < p[self.names.index("beta")]:
            return -np.inf

#        if len(p) == 13:
#            if not p[1] < p[4] < p[10]:
#                return -np.inf
#        if len(p) == 10:
#            if not p[1] < p[3] < p[8]:
#                return -np.inf

        #lnprior += -0.5 * (0.66*p[self.names.index("numax")]**0.88 - p[self.names.index("denv")])**2 / (50.0)**2

        # Add prior on height ...
        #sigma = np.log(5e6)
        #lnprior += -0.5 * ((np.log(4e7) - 2.6*np.log(p[self.names.index("numax")])) - p[self.names.index("Henv")])**2 / sigma**2

        if not np.isfinite(lnprior):
            return -np.inf
        #print("PRIORS: ", lnprior, self.Model.log_prior())
        return lnprior + self.Model.log_prior()

class Dummy(object):
    # Dummy likelihood function
    def __init__(self):
        pass

    def __call__(self, p):
        return 0

class LogProb:
    def __init__(self, _like, _prior):
        self.like = _like
        self.prior = _prior

    def __call__(self, p):
        lp = self.prior(p)
        if not np.isfinite(lp):
            return -np.inf
        like = self.like(p)
        return lp + like

class MCMC(object):
    """
    MCMC class object
    """
    def __init__(self, _freq, _power, _kic, _rfreq, _rpower, _prior, _like, _dummy, _ntemps, _nwalkers, _niter, _ndims, _output_dir):
        print("Initialising Class")
        # Initialise class object
        self.freq = _freq
        self.power = _power
        self.kic = _kic
        self.rfreq = _rfreq
        self.rpower = _rpower
        self.ntemps = _ntemps
        self.nwalkers = _nwalkers
        self.niter = _niter
        self.ndims = _ndims
        self.prior = _prior
        self.like = _like
        self.dummy = _dummy
        self.names = self.like.model.get_parameter_names()
        self.output_dir = _output_dir
        #self.width = 30

    def run_dummy(self):
        #print("Fast fit")
        # Fast fit
        #ffit = ffc.FFit(self.freq, self.power, '')
        #numax_est = ffit.run_fit()

        # Wavelet test
        print("... running wavelets ...")
        wav = WAVELET(self.rfreq, self.rpower, self.kic, self.output_dir)
        first_guesses = wav()
        #plt.show()

        self.numax_est = first_guesses[0]
        self.numax_est_err = first_guesses[1]
        self.dnu_est = first_guesses[2]
        self.dnu_est_err = first_guesses[3]

        if self.numax_est < 25.0:
            self.smoo = int(0.1/(self.rfreq[1]-self.rfreq[0]))
            sel = np.where(self.rfreq > 0.5)
            self.freq = rebin(self.rfreq[sel], self.smoo)
            self.power = rebin(self.rpower[sel], self.smoo)
            # Reset the model
            self.like = Likelihood(self.freq, self.power, self.like.model, smoo=self.smoo)
            self.like.model.setup_freqs(self.freq, self.rfreq.max())

            #print(len(self.freq), self.smoo, len(self.rfreq), len(self.like.model.f), len(self.like.f))
            #print(self.like(self.get_start()))


        self.p0 = np.zeros([self.ntemps, self.nwalkers, self.ndims])
        for i in range(self.ntemps):
            for j in range(self.nwalkers):
                self.p0[i,j,:] = self.get_start() + 1.0e-5*np.random.randn(self.ndims)
                #if np.isfinite(self.prior(self.p0[i,j,:]) + self.like(self.p0[i,j,:])) is False:
                while np.isfinite(self.prior(self.p0[i,j,:]) + self.like(self.p0[i,j,:])) == False:
                    #print("Infinite looop!")
                    #print("PRIOR: ", self.prior.Model.log_prior())
                    #print(self.prior.Model.parameter_bounds)
                    #print(self.get_start() + 1.0e-5*np.random.randn(self.ndims))
                    self.p0[i,j,:] = self.get_start() + 1.0e-5*np.random.randn(self.ndims)
                    #print("NUTS!")
                    #print(self.prior.Model.get_parameter_bounds())
                    #print(self.prior(self.p0[i,j,:]))
                    #print(self.like(self.p0[i,j,:]))
                    #print(self.p0[i,j,:])
                    #sys.exit()

    def get_pos(self):
        """ Get position of walkers by running mcmc over priors
        """
        self.run_dummy()

    def get_start(self):
        # Need to rewrite this so it knows which parameters are being frozen and which aren't!
        x = np.zeros(self.ndims)
        whitenoise = np.median(self.like.obs[-10:])*1.428
        numax = self.numax_est
        x[self.names.index("numax")] = self.numax_est
        x[self.names.index("hsig1")] = np.log(2.0e7/(numax**2.0) + np.random.normal(0, 2.0e7/(numax**2.0)/5.0))
        #x[self.names.index("hfreq1")] = 1.0 / (4.0 / numax + np.random.normal(0, 4.0 / numax / 5.0))
        x[self.names.index("hfreq1")] = 0.05 * self.numax_est + np.random.normal(0, self.numax_est/50.0) #np.random.uniform(0.1, 4.0) #
        x[self.names.index("hsig2")] = np.log(1.5e7/numax**2.4 + np.random.normal(0, 1.5e7/numax**2.4 /5.0))
        #x[self.names.index("hfreq2")] = 1.0 / (1.0 / numax + np.random.normal(0, 1.0 / numax / 10.0))
        x[self.names.index("hfreq2")] = 0.3*self.numax_est + np.random.normal(0, 0.01*self.numax_est)
        x[self.names.index("denv")] = 0.66 * numax ** 0.88 + np.random.normal(0, 0.66 * numax ** 0.88 / 10.0)
        x[self.names.index("Henv")] = np.log((200 / x[4])**(1/0.35) + np.random.normal(0, (200 / x[4])**(1/0.35)/5.0))
        x[self.names.index("alpha")] = np.log(0.5e7/numax**2.4 + np.random.normal(0, 0.5e7/numax**2.4 /5.0))
        #x[9] = 8.893120e+05 * numax ** -2.091983
        #x[self.names.index("beta")] = 1.0 / (0.8 / numax + np.random.normal(0, 0.8 / numax / 20.0))#6.945823e-01 *  numax ** 1.274016
        x[self.names.index("beta")] = 1.0 / (1.0 / numax + np.random.normal(0, 1.0 / numax / 20.0))
        #x[10] = 1.0 / (0.5/numax + np.random.normal(0, 0.5/numax / 5.0))
        x[self.names.index("white")] = 0.5 * whitenoise + np.random.normal(0, whitenoise/15.0)
        # Reorder parameters:
        if not x[self.names.index("hfreq1")] < x[self.names.index("hfreq2")] < x[self.names.index("beta")]:
            t = np.array([x[self.names.index("hfreq1")],x[self.names.index("hfreq2")],x[self.names.index("beta")]])
            t = t[np.argsort(t)]
            x[self.names.index("hfreq1")] = t[0]
            x[self.names.index("hfreq2")] = t[1]
            x[self.names.index("beta")] = t[2]
            #print(x)

        if "hexp1" in self.names:
            x[self.names.index("hexp1")] = 4.0 + np.random.normal(0, 0.1)
        if "hexp2" in self.names:
            x[self.names.index("hexp2")] = 4.0 + np.random.normal(0, 0.1)
        if "exp3" in self.names:
            x[self.names.index("exp3")] = 4.0 + np.random.normal(0, 0.1)

        # SCatter numax
        if self.numax_est_err / self.numax_est > 0.1:
            x[self.names.index("numax")] = self.numax_est + np.random.normal(0, 0.1*self.numax_est)
        else:
            x[self.names.index("numax")] = self.numax_est + np.random.normal(0, self.numax_est_err)

        #c = ['b', 'g', 'r', 'y', 'k']
        #plt.plot(self.freq, self.power, 'k')
        #for idx, i in enumerate(self.like.model.compute_sep(abs(x))):
        #    try:
        #        plt.plot(self.freq, i, '--', c=c[idx])
        #    except:
        #        plt.plot(self.freq, i*np.ones_like(self.freq), '--', c=c[idx])
        #plt.show()
        #self.prior.Model.set_parameter_vector(abs(x))

        #print(self.prior.Model.get_parameter_bounds())
        #if np.isfinite(self.prior(abs(x)) + self.like(abs(x))) is False:
        #print("PRIOR: ", self.prior.Model.log_prior())
        #    print("NUTS!")
        #    print(abs(x))
        #if np.isfinite(self.like(abs(x))) is False:
        #    print(abs(x))

        #if not all([np.isfinite(i) for i in abs(x)]):
        #    print("PRIOR: ", self.prior.Model.log_prior())
        #    print(abs(x))
        return np.abs(x)

    def GR_diagnostic(self, sampler_chain):
        """ Function that calculates the Gelman-Rubin statistic for each parameter.
            Identical to calculation from pymc3.
        """
        try:
            m, n, ndims = np.shape(sampler_chain)
            r_hat = []
            for i in range(ndims):
                x = sampler_chain[:,:,i]
                # Calculate between-chain variance
                B_over_n = np.sum((np.mean(x,1) - np.mean(x))**2)/(m-1)

                # Calculate within-chain variances
                W = np.sum([(x[i] - xbar)**2 for i,xbar in enumerate(np.mean(x,1))]) / (m*(n-1))

                # (over) estimate of variance
                s2 = W*(n-1)/n + B_over_n

                # Pooled posterior variance estimate
                V = s2 + B_over_n/m

                # Calculate PSRF
                R = np.sqrt(V/W)
                r_hat.append(R)
        except:
            m, n = np.shape(sampler_chain)
            x = sampler_chain[:,:]
            # Calculate between-chain variance
            B_over_n = np.sum((np.mean(x,1) - np.mean(x))**2)/(m-1)

            # Calculate within-chain variances
            W = np.sum([(x[i] - xbar)**2 for i,xbar in enumerate(np.mean(x,1))]) / (m*(n-1))

            # (over) estimate of variance
            s2 = W*(n-1)/n + B_over_n

            # Pooled posterior variance estimate
            V = s2 + B_over_n/m

            # Calculate PSRF
            R = np.sqrt(V/W)
            r_hat = R

        return r_hat

    def find_bad_chains_logp(self, logp):
        good = np.where(logp > 0.5*np.max(logp))
        bad = np.where(logp < 0.5*np.max(logp))
        return good, bad

    def reassign_bad_chains_logp(self, p, logp, good, bad):
        """
        Reassign bad walkers
        """
        # "Good" and "Bad" walkers
        good = np.argsort(logp[0,:])[-len(logp[0,:])//2:]
        bad = np.argsort(logp[0,:])[:len(logp[0,:])//2]
        for i in range(np.shape(p)[0]):
            for j in range(np.shape(p)[2]):
                new_p = np.median(p[i,good,j])
                new_sigma = 1.4826 * np.median(np.abs(p[i,good,j] - new_p))
                if new_sigma == 0.0:
                    new_sigma = 0.1
                p[i,bad,j] = np.abs(np.random.normal(new_p, new_sigma/20.0, len(bad)))
            for k in range(len(bad)):
                logp[i,bad[k]] = self.like(p[i,bad[k],:]) + self.prior(p[i,bad[k],:])
        return p, logp

    def run_sampler(self):

        # Set up Completion flag
        COMPLETED = False
        # FLAG for number of cycles
        NITER_TOT = 0
        # Define target r_hat
        target_rhat = 1.01
        # Set up MPI pool


        # Set up hdf5 file for saving final iteration
        #if os.path.exists("astero-{0}.h5".format(kicid)):
        #    result = input("MCMC save file exists. Overwrite? (type 'yes'): ")
        #    if result.lower() != "yes":
        #        sys.exit(0)

        #sampler = emcee.PTSampler(self.ntemps, self.nwalkers, self.ndims, self.like, \
        #                          self.prior, threads=4)
        print("Setting up Sampler")
        sampler = ptemcee.Sampler(self.nwalkers, self.ndims, self.like, self.prior,
                                  ntemps=self.ntemps, threads=4)
        print('... burning in ...')
        #for p, lnprob, lnlike in tqdm(sampler.sample(self.p0, iterations=self.niter), total=self.niter):
        for p, lnprob, lnlike in tqdm(sampler.sample(self.p0, adapt=True, iterations=self.niter), total=self.niter):
            pass
        #sys.exit()
        #fig, ax = plt.subplots(self.ndims, sharex=True)

        #for i in range(self.ndims):
        #    ax[i].plot(sampler.chain[0,:,:,i].T, color='k', alpha=0.2)
        #ax[-1].set_xlabel(r'Step')
        #plt.show()
        #fig, ax = plt.subplots(self.ndims, sharex=True)
        """
        for i in range(self.ndims):
            ax[i].plot(sampler.chain[-1,:,:,i].T, color='k', alpha=0.2)
        ax[-1].set_xlabel(r'Step')
        plt.show()


        samples = sampler.chain[-1,:,-1000:,:].reshape((-1, self.ndims))
        fig = corner.corner(samples, labels=self.like.model.get_parameter_names())
        plt.show()
        """
        samples = sampler.chain[0,:,:,:].reshape((-1, self.ndims))
        #print("Burn-in GR: ", self.GR_diagnostic(sampler.chain[0,:,:,:]))
        sampler.reset()
        print('... running fit ...')

        #for p, lnprob, lnlike in tqdm(sampler.sample(p, lnprob0=lnprob, lnlike0=lnlike, iterations=self.niter), total=self.niter):
        for p, lnprob, lnlike in tqdm(sampler.sample(p, adapt=True, iterations=self.niter), total=self.niter):
            pass
        #fig, ax = plt.subplots(self.ndims, sharex=True)
        #for i in range(self.ndims):
        #    ax[i].plot(sampler.chain[0,:,:,i].T, color='k', alpha=0.2)
        #ax[-1].set_xlabel(r'Step')
        #plt.show()
        GR = self.GR_diagnostic(sampler.chain[0,:,:,:])
        samples = sampler.chain[0,:,:,:].reshape((-1, self.ndims))
        logprobGR = self.GR_diagnostic(sampler.logprobability[0,:,:])
        print(GR, logprobGR)
        #RGG.plot_back(self.freq, self.power, samples=samples, func=self.like.model, \
        #          kic='', params=np.median(samples, axis=1), \
        #          save=[], ff=self.freq, pp=self.power, posterior=samples[:,6])
        #plt.show()
        #fig, ax = plt.subplots(self.ndims, sharex=True)

        #for i in range(self.ndims):
        #    ax[i].plot(sampler.chain[0,:,:,i].T, color='k', alpha=0.2)
        #ax[-1].set_xlabel(r'Step')
        #plt.show()
        sampler.reset()
        #print("Final GR: ", GR)
        mean_GR = []
        if np.max(GR) > target_rhat:
            for i in range(20):
                print("Running iteration {0} of {1}".format(i+1, 20))
                # Pruning via clustering
                    # Let's do some pruning!
                    #good, bad = self.find_bad_chains_logp(lnprob)
                    #p, lnprob = self.reassign_bad_chains_logp(p, lnprob, good, bad)
                #for p, lnprob, lnlike in tqdm(sampler.sample(p, lnprob0=lnprob, lnlike0=lnlike, iterations=self.niter), total=self.niter):
                for p, lnprob, lnlike in tqdm(sampler.sample(p, adapt=True, iterations=self.niter), total=self.niter):
                     pass
                # Gelman-Rubin statistic
                GR = self.GR_diagnostic(sampler.chain[0,:,:,:])
                logprobGR = self.GR_diagnostic(sampler.logprobability[0,:,:])
                print(GR, logprobGR)

                #fig, ax = plt.subplots(self.ndims, sharex=True)

                #for i in range(self.ndims):
                #    ax[i].plot(sampler.chain[0,:,:,i].T, color='k', alpha=0.2)
                #ax[-1].set_xlabel(r'Step')
                #plt.show()
                #print("GR: ", GR)
                #print("MEAN GR: ", np.mean(GR))
                mean_GR = np.append(mean_GR, np.mean(GR))
                current_limit = target_rhat + 0.001*i#0.005*i
                # Set current limit for GR
                print("CURRENT LIMIT: ", current_limit)
                print("CURRENT GR: ", GR)
                if i != 19:
                    sampler.reset()


                if np.max(GR) < current_limit:
                    print("running final iteration!")
                    # Let's do some pruning!
                    #good, bad = self.find_bad_chains_logp(lnprob)
                    #p, lnprob = self.reassign_bad_chains_logp(p, lnprob, good, bad)
                    # Final iteration
                    #for p, lnprob, lnlike in tqdm(sampler.sample(p, lnprob0=lnprob, lnlike0=lnlike, iterations=self.niter*2), total=self.niter*2):
                    for p, lnprob, lnlike in tqdm(sampler.sample(p, adapt=True, iterations=self.niter), total=self.niter):
                        pass
                    COMPLETED = True
                    NITER_TOT = i+1

                    return sampler, p, COMPLETED, NITER_TOT, GR
                elif i == 19:
                    print("Fit failed!")
                    print("running final iteration!")
                    # Let's do some pruning!
                    #good, bad = self.find_bad_chains_logp(lnprob)
                    #p, lnprob = self.reassign_bad_chains_logp(p, lnprob, good, bad)
                    # Final iteration
                    #for p, lnprob, lnlike in tqdm(sampler.sample(p, lnprob0=lnprob, lnlike0=lnlike, iterations=self.niter*2), total=self.niter*2):
                    #for p, lnprob, lnlike in tqdm(sampler.sample(p, adapt=True, iterations=self.niter*2), total=self.niter*2):
                    #    time.sleep(0.01)
                    NITER_TOT = i+1

                    return sampler, p, COMPLETED, NITER_TOT, GR
        else:
            print("running final iteration!")
            # Let's do some pruning!
            #good, bad = self.find_bad_chains_logp(lnprob)
            #p, lnprob = self.reassign_bad_chains_logp(p, lnprob, good, bad)
            # Final iteration
            #for p, lnprob, lnlike in tqdm(sampler.sample(p, lnprob0=lnprob, lnlike0=lnlike, iterations=self.niter*2), total=self.niter*2):
            for p, lnprob, lnlike in tqdm(sampler.sample(p, adapt=True, iterations=self.niter), total=self.niter):
                pass
            COMPLETED = True
            NITER_TOT = 0

            return sampler, p, COMPLETED, NITER_TOT, GR
        COMPLETED = True
        NITER_TOT = 0
        return sampler, p, COMPLETED, NITER_TOT, GR


    def perform_fit(self):

        # Get positions of walkers
        self.get_pos()

        # Run fit
        sampler, p, COMPLETED, NITER_TOT, GR = self.run_sampler()
        # Flatten samples from lowest temeprature
        #self.samples = sampler.chain[0,:,-400:,:].reshape((-1, self.ndims))
        self.samples = sampler.chain[0,:,:,:].reshape((-1, self.ndims))
        # Create array of samples of parameter a from height
        tmp_a1 = np.sqrt((np.pi * np.exp(self.samples[:,0]) * self.samples[:,1]) / (2*np.sqrt(2)))
        tmp_a2 = np.sqrt((np.pi * np.exp(self.samples[:,2]) * self.samples[:,3]) / (2*np.sqrt(2)))
        tmp_a3 = np.sqrt((np.pi * np.exp(self.samples[:,7]) * self.samples[:,8]) / (2*np.sqrt(2)))
        self.samples = np.c_[self.samples, tmp_a1, tmp_a2, tmp_a3]
        #self.logprobability = sampler.lnprobability[0,:,-400:].reshape((-1, 1))
        #self.logprobability = sampler.logprobability[0,:,-400:].reshape((-1, 1))
        self.logprobability = sampler.logprobability[0,:,:].reshape((-1, 1))

        self.COMPLETED = COMPLETED
        self.NITER_TOT = NITER_TOT
        self.GR = GR
        self.bw = self.freq[1]-self.freq[0]
        sampler.pool.terminate()


    def run(self):

        self.perform_fit()
