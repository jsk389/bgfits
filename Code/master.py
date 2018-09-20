#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append('../Utilities/')
import ImportData as RGGdata
#import RGGplots
import RGGnumax
from os.path import expanduser

def run(psd_file, freeze_exp):
    print("File: ", psd_file)
    # Extract kic number from file name
    kic = psd_file.split('/')[-1]
    kic = kic.split('_')[0]
    kic = str(int(kic.lstrip('kic')))
    f, p, bw = RGGdata.get_psd(psd_file)

    smoo = int(1.0/bw)
    RGGnumax.run(f, p, kic, freeze_exp, plot=True, show=False, smoo=smoo, \
                          MLE=False)

if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
