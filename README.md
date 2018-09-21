# bgfits
Code for fitting background of power spectrum of asteroseismic targets (Kuszlewicz et al. in prep.)

Example
============

An example of how to run the code is contained in `run_all.py`. The output directory and data directory paths should be in the configuration file `config.yml`.

Outputs
============
The code will output several plots as it runs:

1. Output of wavelet analysis showing the approximate values of numax a delta nu used for initial guesses.
2. Corner plot once the MCMC has finished.
3. Correlation matrix for all parameters.
4. `.json` file containing all diagnostics from the fit (integrated autocorrelation time, effective sample size, etc.)
5. `fnumax*.txt` which contains the summary statistics from the fit.

Dependencies
============

The code requires the use of [`ptemcee`](https://github.com/willvousden/ptemcee) for the parallel tempering version of `emcee`.
