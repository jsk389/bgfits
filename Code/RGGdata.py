#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import glob
import numpy as np
import pandas as pd
import scipy.ndimage
import warnings
warnings.filterwarnings("ignore")


from astropy.table import Table


def get_psd(pfile):
    pfile = glob.glob(pfile)[0]
    # Add in try ... except
    if str(pfile).endswith('.fits'):
        data = Table.read(fits_file, format='fits')
        df = data.to_pandas()
        f = df['frequency']
        p = df['psd']
        bw = np.median(np.diff(f))
    else:
        f, p = np.loadtxt(txt_file, unpack=True)
        bw = np.median(np.diff(f))
    return f, p, bw

def rebin(f, smoo):
    if smoo < 1.5:
        return f
    smoo = int(smoo)
    m = len(f) // smoo
#    print("m : ", m, len(f), smoo)
    ff = f[:m*smoo].reshape((m,smoo)).mean(1)
    return ff

def median_snr(p, smoo):
    smoo = int(smoo)
    if smoo <= 1:
        return p
    if smoo % 2 == 0:
        smoo += 1
    return 1.4238*p/scipy.ndimage.filters.median_filter(p, size=smoo, mode='reflect')

def get_red_psd(f, p, fmin):
    return f[f > fmin], p[f > fmin]


if __name__ == "__main__":
    print("This is the Reggae data file")
