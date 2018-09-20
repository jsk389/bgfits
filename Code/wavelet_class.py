import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as sig
from scipy import interpolate
import scipy.special as spec
import scipy.misc as misc
from scipy.optimize import curve_fit
import wavelet
from weighted_quantiles import *
#from . import Template
import matplotlib.gridspec as gridspec
#from .weighted_quantiles import *
import time
from sklearn import mixture
import scipy.ndimage as nd
from matplotlib.colors import LogNorm


class WAVELET():
    def __init__(self, _f, _p, _kic):
        self.freq = _f
        self.psd = _p
        self.kic = _kic

    def set_width(self, numax, a=0.66, b=0.88, factor=1.5):
        return a * numax**b * factor

    def get_snr(self, smoo=0, skips=50):

        print(len(self.freq), len(self.psd))
        med = [np.median(self.psd[np.abs(self.freq - d) < self.set_width(d, factor=1)]) for d in self.freq[::skips]]
        f = interpolate.interp1d(self.freq[::skips], med, bounds_error=False)
        med = f(self.freq)
        self.snr = self.psd / med
        self.snr[:skips] = 1.0
        self.snr[-skips:] = 1.0
        if smoo > 1:
            self.snr = nd.filters.uniform_filter1d(self.snr, int(smoo[0]))

    def get_wavelet(self):
        slevel  = 0.99
        self.dt = np.median(np.diff(self.freq)) * 1e-6
        std = np.nanstd(self.snr)
        std2 = std**2
        var = (self.snr[np.isfinite(self.snr)] - 1) / std

        N = len(self.snr)
        dj = 0.05 #0.1 # 0.01
        s0 = 2*self.dt#0.007e-6#dt
        # Adaptive such that y-axis reaches 200 no matter the frequency resolution
        J = np.log2(200e-6 / (4*self.dt))/dj #-1 #9.5/ dj

        #if self.mother == 'Morlet':
        #    mother=wavelet.Morlet(6.)
        #elif self.mother == 'DOG':
        #    mother = wavelet.DOG(2.)
        #else:
            # Defaults to Morlet
        mother=wavelet.Morlet(6.)


        wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(var, self.dt, dj, s0, J,
                                                          mother)
        #iwave = wavelet.icwt(wave, scales, dt, dj, mother)
        self.power = (abs(wave)) ** 2             # Normalized wavelet power spectrum
        #fft_power = std2 * abs(fft) ** 2     # FFT power spectrum
        self.period = scales * 1e6 * 2 #(1. / freqs) * 1e6 * 2
        #for k in range(len(self.period)):
        #    self.power[k,:] /= self.period[k]
        self.coi = coi * 2e6

    def compute_gamma_prior(self, k, theta):
        """
        Impose a gamma distribution prior on numax for K2-like data. This
        suppresses any very low frequency spikes that could throw off the
        determination of numax.
        """
        # Want to move mode from ~2 to ~35 in line with real data
        S = self.freq/17.5
        return 1.0 / (spec.gamma(k) * theta**2)*S[:,None].T**(k-1)*np.exp(-S[:,None].T/theta)

    def get_points(self):
        T, S = np.meshgrid(self.freq, self.period)
        x = y = z = []

        thresh = np.percentile(self.post.flatten(), 99.5)
        high = np.max(self.post)
        for i in np.arange(thresh, high, (high - thresh) / 100.):
            sel = np.where(self.post > i)
            x = np.append(x, T[sel].flat)
            y = np.append(y, S[sel].flat)
            z = np.append(z, self.post[sel].flat)
        return x, y, z

    def get_posterior(self, alpha = 0.276, beta=0.751):
        """
        Multiply wavelet power spectrum by the delta_nu-numax prior
        Also multiply by gamma prior in case of numax
        """
        # Meshgrid is slow, using numpy broadcasting is much faster!
        sigma = self.period[:,None] * 0.3
        self.alpha = alpha
        self.beta = beta
        self.prior = np.exp(-0.5 * (self.period[:,None] - alpha*self.freq[None,:]**beta)**2 / sigma**2)

        # Compute and apply priors
        # If data is K2-like in terms of length of timeseries
        if self.dt > 0.1e-6:
            # Mean of gamma dist = k*theta -> want k=2 and theta = 2
            k = 1.8 #2
            theta = 3 #2
            self.gamma_prob = self.compute_gamma_prior(k, theta)
            self.prior *= self.gamma_prob
        self.post = self.power * self.prior

    def run_GMM(self):
        """
        Run GMM to determine delta nu and numax
        """

        # Prepare data for GMM
        x, y, z = self.get_points()
        data = np.zeros([len(x), 3])
        data[:,0] = x
        data[:,1] = y
        data[:,2] = z

        n_comp = 1
        print("... running GMM ...")
        clf = mixture.GMM(n_components=n_comp, covariance_type='full')
        clf.fit(data)
        print("... done ...")
#
#        print("GMM means: ", clf.means_)
#        print("GMM covars: ", clf.covars_)
#        print("GMM weights: ", clf.weights_)

        xx = np.linspace(x.min(), x.max())
        yy = np.linspace(y.min(), y.max())
        zz = np.linspace(z.min(), z.max())
        X, Y, Z = np.meshgrid(xx, yy, zz)
        XX = np.array([X.ravel(), Y.ravel(), Z.ravel()]).T

        Z = -clf.score_samples(XX)[0]
        Z = Z.reshape(X.shape)
        # Only want 2D for plotting so average over third dimension
        Z = Z.mean(2)
        # Plotting
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        top = Z.max()
        if top > 50:
            top = 50
        levels = np.arange(0, top, 1)
        X, Y = np.meshgrid(xx, yy)
        CS = ax1.contourf(self.freq, self.period, self.power, 50)
        plt.plot(self.freq, self.alpha * self.freq ** self.beta, lw=2, color='r')
        CB = fig.colorbar(CS, shrink=0.8, label='Power', extend='both')
        # Compute numax and delta nu
        numax = median(clf.means_[:,0], clf.weights_)
        numax_err = np.sqrt(np.diag(clf.covars_[0]))[0]
        dnu =  median(clf.means_[:,1], clf.weights_)
        dnu_err = np.sqrt(np.diag(clf.covars_[0]))[1]
        ax1.errorbar(clf.means_[:,0], clf.means_[:,1],
                     xerr=numax_err, yerr=dnu_err, fmt='o', c='k')

        print("Numax_Est: {} +/- {}".format(numax, np.mean(numax_err)))
        print("Delta_nu_Est: {} +/- {}".format(dnu, np.mean(dnu_err)))


        plt.xlim(0, 5000)
        plt.ylim(0, 200)
        plt.xlabel(r'$\nu_{\mathrm{max}}$ ($\mu$Hz)', fontsize=18)
        plt.ylabel(r'$\Delta\nu$ ($\mu$Hz)', fontsize=18)
        plt.savefig(str(self.kic) + '_WAVELET.png')
        plt.close()

        return numax, numax_err, dnu, dnu_err

    def __call__(self):
        self.get_snr()
        self.get_wavelet()
        self.get_posterior()
        numax, numax_err, dnu, dnu_err = self.run_GMM()
        print("FINISHED WAVELETS")
        return numax, numax_err, dnu, dnu_err

if __name__ == "__main__":
    from K2config import *
    import K2data
    settings = readConfiguration()
    run = settings['run']
    pipeline = settings[run]
    data_dir = settings['work_dir'] + pipeline['data_dir']
    stars = pd.read_csv(data_dir + pipeline['csv_file'])
    row = stars.loc[9]
    s = time.time()
    for _, row in stars.iterrows():
        data_file = data_dir + 'kplr'+str(int(row['EPIC']))+'_llc_concat.dat'
        ds = K2data.Dataset(str(int(row['EPIC'])), data_file)
#        ds.read_timeseries()
        ds.power_spectrum()#length=48*80)
        #print((len(ds.freq) * 29.4 * 60) / 86400.0)
        wl_method = WAVELET(ds)
        res = wl_method()
        #wl_method.pprint()
        #sys.exit()
        fig = wl_method.pplot()
        fig.savefig('wl_' + str(row['EPIC']) + '.png')
        plt.show()
        #sys.exit()
plt.close('all')
