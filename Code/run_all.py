#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from os.path import expanduser
import sys
home = expanduser("~")
sys.path.append(home+'/Dropbox/Reggae/Code')
import master
import time
import glob
import subprocess
import os
import pwd
import master
from config import readConfigFile

def get_files(data_dir):

    print(data_dir)
    files = glob.glob(data_dir+'*/*.pow')
    #print(files)
    return files

if __name__ == "__main__":

    # Configuration file setup
    config_file = str(sys.argv[1])
    settings = readConfigFile(config_file)
    run = settings['run']
    pipeline = settings[run]
    output_dir = pipeline['output_dir']
    data_dir = pipeline['data_dir']
    # Get filenames of files to fit
    files = get_files(data_dir)
    s = time.time()
    for i in files:
        print(i)
        master.run(str(i), output_dir, 'True')
        #subprocess.call(['python3', 'master.py', str(i), sys.argv[2]])
        #sys.exit()
    print("Time taken: ", (time.time() - s)/3600.0, " hours")
