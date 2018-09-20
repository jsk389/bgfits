#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')

import warnings
warnings.filterwarnings("ignore")

import sys

import RGGdata
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import os
import emcee
from emcee import autocorr
import corner
import scipy.optimize as opt
import RGGnumax_mcmc as RGM
import fast_fit_class as ffc
import RGGmodel as Model
import pymc3
import json
import time

from wavelet_class import *


def compute_covariance(samples, burnin=1000, thin=10):
    """ Compute the covariance matrix of a thinned sample of the data """
    new_samples = samples[burnin:, :]
    return np.cov(new_samples[::thin, :].T)

def cov2corr(A):
    """
    covariance matrix to correlation matrix.
    """
    D = np.diag(np.sqrt(np.diag(A)))
    D_inv = np.linalg.inv(D)
    A = np.dot(D_inv, np.dot(A, D_inv))
    return A

def plot_covariance(cov, labels):
    """ Plot covariance matrix """

    fig, ax = plt.subplots(figsize=(16,12))
    heatmap = ax.pcolor(cov, cmap='seismic')
    plt.colorbar(heatmap, label='Correlation')
    plt.yticks(np.arange(0.5, len(labels)+0.5), labels)
    plt.xticks(np.arange(0.5, len(labels)+0.5), labels)
    ax.get_xaxis().set_tick_params(direction='out', width=1)
    ax.get_yaxis().set_tick_params(direction='out', width=1)
    for y in range(cov.shape[0]):
        for x in range(cov.shape[1]):
            plt.text(x + 0.5, y + 0.5, '%.4f' % cov[y, x],
                     horizontalalignment='center',
                     verticalalignment='center',
                     )

def correlation_matrix(samples, burnin=1000, thin=10):
    """ Compute correlation matrix """
    print('... computing covariance matrix ...')
    burnin = int(0.6 * len(samples))
    covariance = compute_covariance(samples, burnin=burnin, thin=thin)
    print('... computing correlation matrix ...')
    correlation = cov2corr(covariance)
    print('... done ...')
    return correlation

def hpd(data, level) :
    """ The Highest Posterior Density (credible) interval of data at "level" level.

    :param data: sequence of real values
    :param level: (0 < level < 1)
    : returns median, +ve, -ve uncertainty
    """
    median = np.median(data)
    # calculate hpd - reverse array so have positive uncertainty first
    hpds = pymc3.stats.hpd(data, alpha=1-level)
    diffs = abs(median - hpds[::-1])

    return (median, *diffs)

def plot_back(f, p, output_dir, model=[], samples=[], func=[], \
              kic=[], params=[], save=[], lnln=False, \
              ff=[], pp=[], posterior=[]):
    fig = plt.figure()
    if lnln:
        ax1 = fig.add_subplot(211)
    else:
        ax1 = fig.add_subplot(111)
    if len(ff) > 0:
        ax1.plot(ff, pp, 'k-')
    ax1.plot(f,p, 'b-')
    if len(model) > 0:
        ax1.plot(f, model, 'r-')
    if len(samples) > 0:
        for i in np.arange(0, len(samples[:,0]), 2000):
            func.set_parameter_vector(samples[i,:])
            tmp = func.compute_value()
            ax1.plot(f, tmp, 'g-', alpha=0.5)

    # Plot individual components
    if len(samples) > 0:
        median_vals = np.median(samples, axis=0)
        print(np.shape(median_vals))
        harvey1, harvey2, harvey3, gauss, white = func.compute_sep(median_vals)
        ax1.plot(f, harvey1, 'r--')
        ax1.plot(f, harvey2, 'b--')
        ax1.plot(f, harvey3, 'g--')
        ax1.plot(f, gauss, 'y-')
        ax1.plot(f, white*np.ones_like(f), 'c--')

    if len(posterior) > 0:
        ain = plt.axes([0.23,0.25,0.3,0.2])
        n, bins, patches = ain.hist(posterior, 25, normed=1, facecolor='green')
        start, end = ain.get_xlim()
        ain.set_xticks(np.arange(start, end, (end-start)/2))
        ain.set_xlabel(r"Frequency ($\mu$Hz)")
        ain.set_ylabel("PPD")

    if lnln:
        ax2 = fig.add_subplot(212)
        ax2.plot(f,p, 'k-')
        #    ax2.plot(f,GRDdata.smooth_power(p,100), 'g-')
        if len(model) > 0:
            ax2.plot(f, model, 'r-')
        if len(samples) > 0:
            for i in np.arange(0, len(samples[:,0]), 1000):
                func.set_parameter_vector(samples[i,:])
                tmp = func.compute_value()
                ax2.plot(f, tmp, 'b-', alpha=0.5)
        ax2.set_xlim([f.min(), f.max()])
        if len(params) >  0:
            best = func(params)
            ax2.plot(f, best, 'r--')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel(r'Frequency ($\mu$Hz)', \
                       {'fontsize' : 18})
        ax2.set_ylabel(r'PSD (ppm$^{2}$$\mu$Hz$^{-1}$)', \
                       {'fontsize' : 18})
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim([f.min(), f.max()])
    ax1.set_ylim([1e-3, np.max(pp)+0.1*np.max(pp)])
    ax1.set_xlabel(r'Frequency ($\mu$Hz)', \
                  {'fontsize' : 18})
    ax1.set_ylabel(r'PSD (ppm$^{2}$$\mu$Hz$^{-1}$)', \
                  {'fontsize' : 18})
    plt.tight_layout()
    if len(save) > 0:
        plt.savefig(str(output_dir)+save + "_" + kic + '.png')
    plt.close()


def plot_numax(samples, median, sigma, output_dir, kic=[]):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    n, bins, patches = ax1.hist(samples, np.sqrt(len(samples))/10, \
                                normed=1, facecolor='blue')
    bincenters = 0.5*(bins[1:]+bins[:-1])
    y = mlab.normpdf( bincenters, median, sigma)
    l = ax1.plot(bincenters, y, 'r--', linewidth=1)
    ax1.set_xlabel('Frequency')
    ax1.set_ylabel('Probability')
    ax1.grid(True)
    if len(kic) > 0:
        plt.savefig(str(output_dir)+'numax_' + kic + '.png')

def rebin(f, smoo):
    if smoo < 1.5:
        return f
    smoo = int(smoo)
    m = len(f) // smoo
    ff = f[:m*smoo].reshape((m,smoo)).mean(1)
    return ff

def compute_effective_sample_size(samples):
    """ Compute effective sample size and autocorrelation time
    """
    tau = np.mean(autocorr.integrated_time(np.mean(samples, axis=1), c=5))
    neff = len(samples)/tau
    return tau, neff

def MCMC(freq, power, kic, ff, pp, params, like, prior, output_dir, plot=False):
    """
    Run MCMC

    :param freq: array of frequency values
    :param power: array of power values
    :param kic: KIC number of current stars
    :param ff: rebinned frequency values
    :param pp: rebinned power values
    :param params: ?
    :param like: likelihood function
    :param prior: prior function
    :param plot: whether to plot or not

    """

    # Initialise MCMC parameters
    ntemps, nwalkers, niter, ndims = 6, 400, 1000, int(len(params))

    # Initiliase Dummy likelihood class
    dummy = RGM.Dummy()
    # Initialise fit class
    fit = RGM.MCMC(freq, power, kic, ff, pp, prior, like, dummy, ntemps, nwalkers, niter, ndims, output_dir)
    # Run fit
    fit.run()

    samples = fit.samples
    logprobability = fit.logprobability

    # Compute AIC and BIC
    n_par = np.shape(samples)[1]
    AIC = 2.0 * n_par - 2.0 * np.max(logprobability)
    BIC = np.log(len(ff))*n_par - 2.0 * np.max(logprobability)

    # level for hpd calculation
    level = 0.683
    results = np.array(([hpd(samples[:,i], level) for i in range(np.shape(samples)[1])]))

    important_params = dict(
                            ntemps=ntemps,
                            nwalkers=nwalkers,
                            niter=niter,
                            AIC=AIC,
                            BIC=BIC,
                            completed=fit.COMPLETED,
                            total_niter=fit.NITER_TOT,
                            GR=fit.GR,
                            initial_numax=fit.numax_est,
                            initial_numax_err=fit.numax_est_err,
                            dnu=fit.dnu_est,
                            dnu_err=fit.dnu_est_err,
                            bw=fit.bw,
                            )

    return fit.samples, results, important_params, fit.freq, fit.power


def run(ff, pp, kic, freeze_exp,\
        output_dir, plot=True, show=True, smoo=1, \
        save="back",\
        MLE=False):
    # Compute the bin width
    bw = np.median(np.diff(ff))
    # Compute Nyquist frequency
    nyq = ff.max()
    start_time = time.time()
    if np.isnan(pp).any() == True:
        return None
    fnumax_file = "fnumax" + kic + ".txt"
    kic = str(kic)
    # Set up model with random parameters - just for initialisation
    numax = 50.0
    nuwidth = 10.0
    height = (100.0 / nuwidth)**(1/0.35)
    #  Estimate white noise
    white = np.median(pp[-100:])/2.0
    # If long cadence
    if nyq < 300.0:
        # Take 5uHz for K2 and 1uHz for Kepler
        sel = np.where(ff > 1.0)
        f = rebin(ff[sel], smoo)
        p = rebin(pp[sel], smoo)
        mod = Model.Model(
                          hsig1 = np.log(1e6 / numax**1.6),
                          hfreq1 = 1.0,
                          hexp1 = 4.,
                          hsig2 = np.log(3.e6 / numax**1.7),
                          hfreq2 = numax,
                          hexp2 = 4.,
                          numax = numax,
                          denv = nuwidth,
                          Henv = np.log(height),
                          alpha = np.log(1.e6 / numax**1.7),
                          beta = 1.5*numax,
                          exp3 = 4.,
                          white = white,
                          bounds = dict(
                                       hsig1 = (-3.0, 20.0),
                                       hfreq1 = (0.0, 4.0),
                                       hexp1 = (2,6),
                                       hsig2 = (-3.0, 20.0),
                                       hfreq2 = (0.1, 400.0),
                                       hexp2 = (2,6),
                                       numax = (1.0, 350.0),
                                       denv = (0.1, 150),
                                       Henv = (-3, 20.0),
                                       alpha = (-3.0, 20.0),
                                       beta = (0.1, 500.0),
                                       exp3 = (2,6),
                                       white = (white/20.0, white*20.0)
                                      ),
                        )
    else:
        # Take 5uHz for K2 and 1uHz for Kepler
        sel = np.where(ff > 10.0)
        smoo = int(5.0/bw)
        f = rebin(ff[sel], smoo)
        p = rebin(pp[sel], smoo)
        mod = Model.Model(
                          hsig1 = np.log(1e6 / numax**1.6),
                          hfreq1 = 0.3*numax,
                          hexp1 = 4.,
                          hsig2 = np.log(3.e6 / numax**1.7),
                          hfreq2 = numax,
                          hexp2 = 4.,
                          numax = numax,
                          denv = nuwidth,
                          Henv = np.log(height),
                          alpha = np.log(1.e6 / numax**1.7),
                          beta = 1.5*numax,
                          exp3 = 4.,
                          white = np.median(p[-100:]),
                          bounds = dict(
                                       hsig1 = (-3.0, 20.0),
                                       hfreq1 = (0.1, 5000.0),
                                       hexp1 = (2,6),
                                       hsig2 = (-3.0, 20.0),
                                       hfreq2 = (0.1, 5000.0),
                                       hexp2 = (2,6),
                                       numax = (1.0, 5000.0),
                                       denv = (0.1, 2000),
                                       Henv = (-3, 20.0),
                                       alpha = (-3.0, 20.0),
                                       beta = (0.1, 5000.0),
                                       exp3 = (2,6),
                                       white = (white/20.0, white*4.0)
                                      ),
                        )
    # Decide whether to fix exponents or not
    if freeze_exp == 'True':
        print("Fixing exponents to 4!")
        mod.freeze_parameter('hexp1')
        mod.freeze_parameter('hexp2')
        mod.freeze_parameter('exp3')
    else:
        pass
    # Set up frequencies and Nyquist frequency
    mod.setup_freqs(f, ff.max())
    param_names = mod.get_parameter_names()
    params = mod.get_parameter_vector()
    like = RGM.Likelihood(f, p, mod, smoo=smoo)

    # Changed exponent priors from (2,6) to (1,10) and (0,6) to (0,10) to see what happens 10/1/17 - changed back 19/1/17
    # Added in ability to check between short cadence and long-cadence and change priors accordingly
    # Divide power by smoothed version to approximate background to estimate numax
    gaussian = [(0,0),(0,0),(0,0),\
                (0,0),(0,0),(0,0),\
                (0,0),(0,0),(0,0),\
                (0,0),(0,0),(0,0),(0,0)]

    prior = RGM.Prior(gaussian, mod)

    samples, results, extra_params, f, p = MCMC(f, p, kic, ff, pp, params, \
                                  like, prior, output_dir, plot=plot)

    print("... computing autocorrelation times and effective sample size ...")
    tau, neff = compute_effective_sample_size(samples)
    print("Autocorrelation time: {}".format(tau))
    print("Effective sample size: {}".format(neff))
    print("Length of chains: {}".format(len(samples)))

    # Compute all statistics using effective sample size
    samples = samples[::int(np.ceil(tau)),:]
    param_names = list(param_names)
    param_names = param_names + ["a1", "a2", "a3"]
    param_names = tuple(param_names)
    results = np.c_[param_names, results]

    fig = corner.corner(samples, labels=param_names)
    fig.savefig(str(output_dir)+str(kic)+'_corner.png')
    plt.close()

    # Compute correlation matrix
    correlation = correlation_matrix(samples)
    plot_covariance(correlation, param_names)
    plt.savefig(str(output_dir)+str(kic)+'_corr_matrix.png')
    plt.close()


    # Output JSON with important information!
    end_time = time.time()
    with open(str(output_dir)+str(kic)+"_diag.json", "w") as output_file:
        json.dump(dict(
                       ntemps=extra_params['ntemps'],
                       nwalkers=extra_params['nwalkers'],
                       niter=extra_params['niter'],
                       ndims=np.shape(samples)[1],
                       total_niter=extra_params['total_niter'],
                       completed=extra_params['completed'],
                       tau=tau,
                       Neff=neff,
                       AIC=extra_params['AIC'],
                       BIC=extra_params['BIC'],
                       c_fixed=freeze_exp,
                       total_time=end_time-start_time,
                       GR=extra_params['GR'],
                       initial_numax=extra_params['initial_numax'],
                       initial_numax_err=extra_params['initial_numax_err'],
                       dnu=extra_params['dnu'],
                       dnu_err=extra_params['dnu_err'],
                       bw=extra_params['bw'],
                       ), output_file)

    # Add in parameter names to results file

    print(np.shape(results))
    # Add header to results file
    results = np.vstack([["ParameterName", "Median", "UpperErr", "LowerErr"], results])
    np.savetxt(str(output_dir)+fnumax_file, results, fmt="%s")

    if plot:
        ind_numax = param_names.index("numax")
        plot_back(f,p, output_dir, samples=samples[:,:-3], func=mod, \
                  kic=kic, params=results,
                  save=save, ff=ff, pp=pp, posterior=samples[:,ind_numax])
        #figg = corner.corner(samples, labels=param_names)
    if show:
        plt.show()
    plt.close('all')
