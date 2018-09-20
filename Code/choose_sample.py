#!/usr/bin/python

from __future__ import division

import math
import os
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import sys
import glob

import os

def hpd(data, level) :
    """ The Highest Posterior Density (credible) interval of data at 		"level" level.

    :param data: sequence of real values
    :param level: (0 < level < 1)
    """

    d = list(data)
    d.sort()

    nData = len(data)
    nIn = int(round(level * nData))
    if nIn < 2 :
        raise RuntimeError("not enough data")

    i = 0
    r = d[i+nIn-1] - d[i]
    for k in range(len(d) - (nIn - 1)) :
        rk = d[k+nIn-1] - d[k]
        if rk < r :
            r = rk
            i = k

    assert 0 <= i <= i+nIn-1 < len(d)

    return (d[i], d[i+nIn-1])

def compute_values(x, level):
	"""
	Compute median, and credible interval using highest posterior density
	"""
	
	x_value = np.median(x)
	region = hpd(x, level)
	
	return [x_value, region[1]-x_value, x_value-region[0]]

def calc_heights(a, gamma):
    T = 365.25 * 86400.0 * 4.0
    height = 2.0 * a **2.0 * T / (np.pi * gamma * T + 2)
    return height / 1e6

def lorentzian(f, a, b, c, d):
    return d - a / (1 + (4.0/b**2)*(f - c)**2)

def mass_scaling(numax, dnu):
    Teff = 4994./5777.
    mass = (numax/3090)**3 * (dnu/135.1)**-4 * (Teff)**1.5
    return mass

if __name__ == "__main__":
    # Create folder to store data and plots in

    fname = sys.argv[1]
    list_dirs, numax, dnu = np.loadtxt(fname, usecols=(0,1,2), dtype=float, unpack=True)
    list_dirs = list_dirs.astype(int)
    os.getcwd()
    #rootdir = '/home/jsk389/Dropbox/Python/Peak-Bagging/Red_Giants/'
    rootdir = '/home/jsk389/Dropbox/Python/Angle_of_inc/'
    stars = []
    core = []
    core_err = []
    env_err = []
    mass = []
    env = []
    for i in range(len(list_dirs)):
        # Firstly identify the samples for each star
        print("Reading in star {0} of {1}".format(i+1, len(list_dirs)))
        dirs = glob.glob(rootdir+str(list_dirs[i])+'/*/samples.txt')
        param_dirs = [s.strip('samples.txt') for s in dirs]

        samples_tot = []   
        splitting = []
        s_err = []
        freq = []
        f_err = []

        for j in range(len(dirs)):
            backg, frequency, amp, width, split, inc = np.loadtxt(param_dirs[j]+'limit_parameters.txt', usecols=(0,), unpack=True)
            errs = np.loadtxt(param_dirs[j]+'limit_parameters.txt', usecols=(1,2), unpack=True)
            errs = np.mean(errs.T, axis=1)
            snr = calc_heights(amp, width*1e-6) / backg
            if (width >= 0.00787 * 1.0) and (snr > 18.9) and (inc > 30.): 
                splitting = np.append(splitting, split)
                s_err = np.append(s_err, errs[4])
                freq = np.append(freq, frequency)
                f_err = np.append(f_err, errs[1])

        print(list_dirs[i])

        eps = 0.634 + 0.546*np.log10(dnu[i])

        n_p = np.floor(freq / dnu[i])

        # Sort data
        unsorted_f = ((freq / dnu[i]) - (n_p - eps)) % 1
        sorted_idx = np.argsort(unsorted_f)
        sorted_f = unsorted_f[sorted_idx]
        sorted_split = splitting[sorted_idx]

        plt.errorbar(((freq / dnu[i]) - (n_p - eps)) % 1, splitting, yerr=s_err, fmt='o')
        plt.show()
   
    #np.savetxt('Andrea_stars.txt', stars)


