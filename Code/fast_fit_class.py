#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import glob
import sys
import scipy.interpolate as intrp
import pandas as pd
import pyfits

# This way home is the same across all users and on all computers!
from os.path import expanduser
home = expanduser("~")

class FFit:
    def __init__(self, f_, p_, kic_):
        self.f = f_
        self.p = p_
        # Give KIC
        self.kic = kic_
        # Compute bin width
        self.bw = self.f[1]-self.f[0]
        # Set nyquist frequency
        self.nyq = f_.max()
        # Set number of bins to smooth over
        self.smoo = int(1.0 / self.bw)

        # Import priors
        self.dfp = pd.read_csv('Background_fitting/back_fit.csv')

    def cut_down_data(self, cut):
        """ Cut data at appropriate point
        """
        return self.f[self.f > cut], self.p[self.f > cut]

    def harvey(self, hsig, htau, hexp):
        """
        Harvey profile
        """
        return hsig / (1.0 + (self.f * htau)**hexp)

    def gaussian(self, numax, width, height):
        """
        Gaussian to describe oscillations
        """
        width = width / (2.0 * np.sqrt(2.0*np.log(2)))
        m = np.exp(height) * np.exp(-(self.f - numax)**2 / (2.0 * width**2))
        return m

    def set_priors(self, dfp, numax):
        self.hsig1 = np.exp(dfp[dfp.param.str.contains('hsig1')].m.values[0] * np.log(numax) + dfp[dfp.param.str.contains('hsig1')].c.values[0])
        self.htau1 = np.exp(dfp[dfp.param.str.contains('htau1')].m.values[0] * np.log(numax) + dfp[dfp.param.str.contains('htau1')].c.values[0])
        self.hexp1 = 4.0
        self.hsig2 = np.exp(dfp[dfp.param.str.contains('hsig2')].m.values[0] * np.log(numax) + dfp[dfp.param.str.contains('hsig2')].c.values[0])
        self.htau2 = np.exp(dfp[dfp.param.str.contains('htau2')].m.values[0] * np.log(numax) + dfp[dfp.param.str.contains('htau2')].c.values[0])
        self.hexp2 = 4.0
        self.width = np.exp(dfp[dfp.param.str.contains('width')].m.values[0] * np.log(numax) + dfp[dfp.param.str.contains('width')].c.values[0])
        self.height = (dfp[dfp.param.str.contains('height')].m.values[0] * np.log(numax) + dfp[dfp.param.str.contains('height')].c.values[0])
        return 1

    def model(self, numax, white):
        a = self.set_priors(self.dfp, numax)
        model = self.harvey(self.hsig1, self.htau1, self.hexp1)
        model += self.harvey(self.hsig2, self.htau2, self.hexp2)
        model += self.gaussian(numax, self.width, self.height)
        model *= np.sinc(self.f / 2.0 / self.nyq)**2.0
        wh = white - np.mean(np.abs(model[-50:]))
        model += wh
        return np.abs(model)

    def rebin_quick(self, f, p, smoo):
        if smoo < 1:
            return f, p
        smoo = int(smoo)
        m = int(len(p) / smoo)
        ff = f[:m*smoo].reshape((m,smoo)).mean(1)
        pp = p[:m*smoo].reshape((m,smoo)).mean(1)
        return ff, pp

    def like(self, p):
        numax, white = p
        m = self.model(numax, white)
        L = (np.log(m) + \
                          self.p/m)
        L = 1.0*np.sum(L[L < L.max()*0.95])
        Lwo = (np.log(m*0.8) + \
                          self.p/m/0.8)
        Lwo = np.sum(Lwo[Lwo < Lwo.max()*0.8])
        return L, Lwo

    def run_fit(self, plot=False, save=False):
        """ Edit to improve -> add in iteration over length of power spectrum given and compare
            likelihood function values
        """
        # Estimate white noise
        white = np.median(self.p[-50:])*1.4
        LL = []
        numax = []
        # If white noise below a threshold then should be able to see granulation and oscillations
        if white < 1e4:
            # Set a large max likelihood threshold
            L_max = 1e12
            # Initialise numax estimate to zero
            nu_est = 0.0
            wo = False
            for cut in [0.0, 0.1, 0.5, 1, 5]:
                # Cute down data
                self.p = self.p[self.f > cut]
                self.f = self.f[self.f > cut]
                # Rebin power spectrum
                self.ff, self.pp = self.rebin_quick(self.f, self.p, self.smoo)
                for i in np.arange(0,500,2):
                    L, Lwo = self.like([1.0+i,white])
                    LL = np.append(LL, L)

                    if L < L_max:
                        nu_est = 1.0 + i
                        L_max = L
                        wo = False
                    if Lwo < L_max:
                        nu_est = 1.0 + i
                        L_max = Lwo
                        wo = True
                    numax = np.append(numax, 1.0+i)
            #print(nu_est, L_max)
            if plot:
                m = self.model(nu_est, white/2.0)
                fig, ax = plt.subplots()
                ax.plot(self.f, self.p, 'k-')
                ax.plot(self.ff, self.pp, 'b-')
                if wo:
                    ax.plot(self.ff, m*0.8, 'r-')
                    ax.plot(self.ff, m, 'r--')
                else:
                    ax.plot(self.ff, m, 'r-')
                    ax.plot(self.ff, m*0.8, 'r--')
                ax.set_xlim([0.1,288])
                ax.set_ylim([10,1e6])
                ax.set_xscale('log')
                ax.set_yscale('log')
                plt.show()
            if save:
                fig.savefig(self.kic + '_ffit.png')
            plt.close('all')

#            plt.plot(self.f, self.p/self.p.max(), 'k')
            #plt.plot(numax, LL, 'r')
            #plt.show()

            return nu_est

if __name__ == "__main__":
    dennis=False
    WG1 = False
    PSD = True
    if WG1:
        run_name = 'WG1_LC_Q'
        sdir='/home/davies/Dropbox/DATA_SETS/Kepler/WG1_LC/KASOC_DOWNLOAD/'
        dat_files = get_file_list(sdir)
        sloc = [-31, -22]
        noise = 10**np.arange(-1,8,1)
    APOKASC = True
    if APOKASC:
        run_name = 'APOKASC_LC_Q'
        sdir=home+'/Dropbox/Reggae/Data/APOKASC_DR1_unweighted/'
        dat_files = get_file_list(sdir, PSD=True)
        sloc = [-31, -22]
        noise = 10**np.arange(-1,5,1)
    K2_C1 = False
    if K2_C1:
        run_name = 'K2_C1'
        sdir='/home/davies/Dropbox/DATA_SETS/K2/C1/'
        PSD=True
        dat_files = get_file_list(sdir, PSD=True)
        sloc = [-30, -21]
        noise = 10**np.arange(-1,5,1)
    Thomas = False
    if Thomas:
        run_name = 'Thomas'
        sdir='/home/davies/Dropbox/Thomas/RetA/Extra_det_c8/'
        PSD=True
        dat_files = get_file_list(sdir, PSD=True)
        sloc = [-30, -21]
    K2_C1_Dennis = False
    if K2_C1_Dennis:
        run_name = 'K2_C1_Dennis'
        sdir ='/home/davies/Projects/Dennis/K2_GAP/C1/'
        dat_files = get_file_list(sdir, dennis=True)
        sloc = [-34, -25]
        dennis = True
        noise = [0]
        dff = []

    print('Number of files: ', len(dat_files))
    # Loop over number of files
    for sfile in dat_files:
        kic = sfile[sloc[0]:sloc[1]]
        # If PSD keyword is set then retrieve power spectrum, otherwise generate power spectrum from timeseries
        if PSD:
            freq, power, bw = get_psd_pow(sfile)
        else:
            time, flux = get_timeseries(sfile, dennis=dennis)
            time, flux = fix_llc(time, flux)
            # y = flux + np.random.randn(len(flux)) * n * np.random.uniform()
            freq, power = create_psd(time, flux)
        # Set number of bins to smooth over
        smoo = 1.0 / (freq[1] - freq[0])
        # Rebin power spectrum
        ff, pp = rebin_quick(freq, power, smoo)
        # Initialise FFit class
        fit = FFit(ff, pp)
        # Estimate white noise
        white = np.median(power[-50:])*1.4
        # If white noise below a threshold then should be able to see granulation and oscillations
        if white < 1e4:
            # Set a large max likelihood threshold
            L_max = 1e12
            # Initialise numax estimate to zero
            nu_est = 0.0
            wo = False
            for i in np.arange(0,500,2):
                L, Lwo = fit.like([1.0+i,white])
                if L < L_max:
                    nu_est = 1.0 + i
                    L_max = L
                    wo = False
                if Lwo < L_max:
                    nu_est = 1.0 + i
                    L_max = Lwo
                    wo = True
            print(nu_est, L_max)
            m = fit.model(nu_est, white)
            fig, ax = plt.subplots()
            ax.plot(freq, power, 'k-')
            ax.plot(ff, pp, 'b-')
            if wo:
                ax.plot(ff, m*0.8, 'r-')
                ax.plot(ff, m, 'r--')
            else:
                ax.plot(ff, m, 'r-')
                ax.plot(ff, m*0.8, 'r--')
            ax.set_xlim([0.1,288])
            ax.set_ylim([10,1e6])
            ax.set_xscale('log')
            ax.set_yscale('log')
            fig.savefig(kic + '_ffit.png')
        plt.close('all')
