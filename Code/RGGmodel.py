#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import modeling
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class Model(modeling.Model):

    parameter_names = ('hsig1', 'hfreq1', 'hexp1',
                       'hsig2', 'hfreq2', 'hexp2',
                       'numax', 'denv', 'Henv',
                       'alpha', 'beta', 'exp3', 'white')
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)

    def setup_freqs(self, _f, _nyq):
        self.f = _f
        self.nyq = _nyq

    def harvey(self, hsig, htau, hexp):
        """
        Extended Harvey profile to deal with super nyquist cases
        """
        return hsig / (1.0 + (self.f/htau)**hexp)

    def gaussian(self, numax, width, height):
        """
        Extended Gaussian for super-nyquist cases
        """
        tmp = width / (2.0 * np.sqrt(2.0 * np.log(2)))
        return height * np.exp(-(self.f - numax)**2 / (2.0 * tmp**2))

    def compute_sep(self, params):
        """ Output individual components for ease of plotting """

        hsig1 = np.exp(self.hsig1)
        hsig2 = np.exp(self.hsig2)
        alpha = np.exp(self.alpha)
        height = np.exp(self.Henv)
        # Compute Harvey-like profiles
        harvey1 = self.harvey(hsig1, self.hfreq1, self.hexp1) * np.sinc(self.f / 2.0 / self.nyq)**2.0
        harvey2 = self.harvey(hsig2, self.hfreq2, self.hexp2) * np.sinc(self.f / 2.0 / self.nyq)**2.0
        harvey3 = self.harvey(alpha, self.beta, self.exp3) * np.sinc(self.f / 2.0 / self.nyq)**2.0
        # Compute gaussian envelope
        gauss = self.gaussian(self.numax, self.denv, height) * np.sinc(self.f / 2.0 / self.nyq)**2.0
        return harvey1, harvey2, harvey3, gauss, self.white

    def compute_value(self):
        """
        Create model
        """

        # Raise logarithmic parameters to exponential
        hsig1 = np.exp(self.hsig1)
        hsig2 = np.exp(self.hsig2)
        alpha = np.exp(self.alpha)
        height = np.exp(self.Henv)

        # Compute harvey profiles
        model = self.harvey(hsig1, self.hfreq1, self.hexp1)
        model += self.harvey(hsig2, self.hfreq2, self.hexp2)
        model += self.harvey(alpha, self.beta, self.exp3)

        # Compute gaussian envelope
        model += self.gaussian(self.numax, self.denv, height)

        # Multiply by sinc-squared
        model *= np.sinc(self.f / 2.0 / self.nyq)**2.0
        #new_model = model#[self.start] + model[self.end][::-1]
        model += self.white
        return model

    def log_prior(self):
        lp = super(Model, self).log_prior()
        if not np.isfinite(lp):
            return -np.inf
        return lp
