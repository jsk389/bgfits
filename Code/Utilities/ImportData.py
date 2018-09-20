#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import glob
import scipy.ndimage

def get_psd(pfile):
    pfile = glob.glob(pfile)[0]
    # Add in try ... except
    if str(pfile).endswith('.fits'):
        f, p, bw = get_psd_fits(pfile)
    if str(pfile).endswith('.txt'):
        f, p, bw = get_psd_txt(pfile)
    if str(pfile).endswith('.pow'):
        f, p, bw = get_psd_pow(pfile)
    return f, p, bw

def get_psd_fits(fits_file):
# Read in the fits file ...
    import pyfits
    if str(fits_file).endswith('.fits'):
        # Account for fact that PSD*.fits and kplr*.fits aren't formatted
        # in the same way
        try: 
            data = pyfits.getdata(fits_file, 0)
            if data[1,0] - data[0,0] > 1e-3:
                f = data[:,0]
                p = data[:,1]
            else:
                f = data[:,0] * 1e6
                p = data[:,1]
            bw = f[1] - f[0]
        except:
            data = pyfits.getdata(fits_file)
            f = data['frequency']
            p = data['psd']
            bw = f[1] - f[0]
        return f, p, bw
    else:
        return -1, -1, -1

def get_psd_txt(txt_file):
# Read in the txt file ...
    if str(txt_file).endswith('.txt'):
        f, p = np.loadtxt(txt_file, unpack=True)
#        f *= 1e6
        bw = f[1] - f[0]
        return f, p, bw
    else:
        return -1, -1, -1

def get_psd_pow(txt_file):
# Read in the txt file ...
    if str(txt_file).endswith('.pow'):
        f, p = np.loadtxt(txt_file, unpack=True)
        bw = f[1] - f[0]
        return f, p, bw
    else:
        return -1, -1, -1

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
