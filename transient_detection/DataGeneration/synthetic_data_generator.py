# Copyright (c) 2023-present Samuele Colombo
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and 
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
syntetic_data_generator.py
================

This module provides functions to generate synthetic data with uniformly distributed events and Gaussian-distributed events.

Functions
---------
generate_uniform_events(num_samples)
    Generate uniformly distributed events in a hypercube with range [0, 1].

generate_gaussian_events(num_samples)
    Generate Gaussian-distributed events in a cube with range [0, 1]. The PI dimension is generated from a Planck distribution in the range 0.15 KeV to 15 KeV.

generate_data(seed, num_uniform_samples, num_gaussian_samples)
    Generate data with uniformly distributed events and Gaussian-distributed events.

save_data_to_fits(X, Y, TIME, PI, ISEVENT, filename)
    Save data to a FITS file.

Notes
-----
The generated data is suitable for testing and experimentation purposes. It can be used to evaluate algorithms or perform data analysis tasks.

The generated data consists of events in a 4-dimensional space, with dimensions labeled as 'X', 'Y', 'TIME', and 'PI'. The events are divided into two types: uniformly distributed events and Gaussian-distributed events. Uniformly distributed events are randomly generated within the hypercube range [0, 1] in each dimension. Gaussian-distributed events have random means and standard deviations (sigma) in each dimension and are also generated within the hypercube range [0, 1].

The functions support setting a random seed for replicability. By setting the same seed, the same data can be generated each time.

The generated data can be saved to a FITS (Flexible Image Transport System) file, a common format used in astronomy and astrophysics. The data is saved as a binary table with columns corresponding to each dimension of the data.

Command-Line Execution
----------------------
This module can be executed as a standalone script to generate and save data files. The command-line arguments are as follows:

    python syntetic_data_generator.py <num_files> <filename_pattern> [--num_uniform_samples <num_uniform_samples>] [--num_gaussian_samples <num_gaussian_samples>] [--seed <seed>]

    <num_files>                : Number of files to generate.
    <filename_pattern>         : Pattern for the filename. Use "{}" as a placeholder for file index.
    --num_uniform_samples      : Number of uniformly distributed data samples to generate in each file (default: 1000).
    --num_gaussian_samples     : Number of Gaussian distributed data samples to generate in each file (default: 1000).
    --seed                     : Random seed for replicability (default: 123).

Examples
--------
To generate and save a single data file:

    X, Y, TIME, PI, ISEVENT = generate_data(123, 1000, 1000)
    save_data_to_fits(X, Y, TIME, PI, ISEVENT, 'data.fits')

To generate and save multiple data files using command-line execution:

    python syntetic_data_generator.py 5 data_{}.fits --num_uniform_samples 500 --num_gaussian_samples 500 --seed 456

"""

import argparse
import random
import numpy as np

import numpy as np
from scipy.constants import pi
import scipy.constants as sc # h, c, k, sigma
from scipy.stats import rv_continuous
from astropy.io import fits

def generate_uniform_events(num_samples):
    """
    Generate uniformly distributed events in a hypercube with range [0, 1].

    Parameters
    ----------
    num_samples : int
        Number of data samples to generate.

    Returns
    -------
    tuple
        Tuple containing arrays for each dimension of the generated uniformly distributed events.

    Notes
    -----
    This function generates uniformly distributed events in a hypercube with range [0, 1] in each dimension.

    Example
    -------
    X_uniform, Y_uniform, TIME_uniform, PI_uniform, ISEVENT_uniform = generate_uniform_events(1000)
    """

    X_uniform = np.random.uniform(0, 1, num_samples)
    Y_uniform = np.random.uniform(0, 1, num_samples)
    TIME_uniform = np.random.uniform(0, 1, num_samples)
    PI_uniform = np.random.uniform(0, 1, num_samples)
    ISEVENT_uniform = np.zeros(num_samples)

    return X_uniform, Y_uniform, TIME_uniform, PI_uniform, ISEVENT_uniform

def generate_soft_xray_photon_energies(temperature, sample_number):
    """
    Generate random photon energies distributed as the soft X-ray section of the Planck distribution
    for a given temperature.

    Parameters:
        temperature (float): The temperature in Kelvin.
        sample_number (int): The number of random samples to generate.

    Returns:
        numpy.ndarray: Array of random photon energies in keV.

    Notes:
        The function generates random photon energies following the soft X-ray section of the Planck distribution
        for the specified temperature. The energy range is set from 0.15 keV to 15 keV.

    Examples:
        >>> temperature = 300.0
        >>> sample_number = 1000
        >>> energies_kev = generate_soft_xray_photon_energies(temperature, sample_number)
    """

    def soft_xray_pdf(x):
        """
        Probability density function (PDF) for the soft X-ray section of the Planck distribution.

        Parameters:
            x (float or numpy.ndarray): Input value(s) at which to evaluate the PDF.

        Returns:
            float or numpy.ndarray: Probability density function evaluated at x.
        """
        energy = sc.h * sc.c / x
        planck_factor = 1.0 / (np.exp(energy / (sc.k * temperature)) - 1)
        return energy ** 3 / (planck_factor * x ** 2)

    soft_xray_dist = rv_continuous(name='soft_xray')
    soft_xray_dist._pdf = soft_xray_pdf
    soft_xray_dist.a = 0.15 # soft X-ray bounds for XMM-Newton
    soft_xray_dist.b = 15

    energies_kev = soft_xray_dist.rvs(size=sample_number)
    return energies_kev

def generate_gaussian_events(temperature, num_samples):
    """
    Generate Gaussian-distributed events in a cube with range [0, 1]. The PI dimension is generated from a Planck distribution in the range 0.15 KeV to 15 KeV.

    Parameters
    ----------
    temperature : float
        Temperature of the black body that generates the photons.

    num_samples : int
        Number of data samples to generate.

    Returns
    -------
    tuple
        Tuple containing arrays for each dimension of the generated Gaussian-distributed events.

    Notes
    -----
    This function generates Gaussian-distributed events in a cube with range [0, 1] in each space-time dimension.
    The mean and standard deviation (sigma) for each dimension are randomly chosen.

    The PI data is distributed according to the Planck (black body) distribution, with temperature given as function argument.

    The function first generates the full array of events and then replaces the outliers outside the cube range [0, 1].

    Example
    -------
    X_gaussian, Y_gaussian, TIME_gaussian, PI_gaussian, ISEVENT_gaussian = generate_gaussian_events(1000)
    """

    means = [np.random.uniform(0, 1) for _ in range(3)]
    sigmas = [np.random.uniform(0.1, 0.5) for _ in range(3)]

    np.random.seed(123)  # Set a fixed seed for reproducibility

    # Generate the full array of events
    X_gaussian = np.random.normal(means[0], sigmas[0], num_samples)
    Y_gaussian = np.random.normal(means[1], sigmas[1], num_samples)
    TIME_gaussian = np.random.normal(means[2], sigmas[2], num_samples)
    PI_gaussian = generate_soft_xray_photon_energies(temperature, num_samples)
    ISEVENT_gaussian = np.ones(num_samples)

    # Replace outliers outside the hypercube range [0, 1]
    mask = (X_gaussian < 0) | (X_gaussian > 1) | \
           (Y_gaussian < 0) | (Y_gaussian > 1) | \
           (TIME_gaussian < 0) | (TIME_gaussian > 1)

    while np.any(mask):
        num_outliers = np.sum(mask)
        X_gaussian[mask] = np.random.normal(means[0], sigmas[0], num_outliers)
        Y_gaussian[mask] = np.random.normal(means[1], sigmas[1], num_outliers)
        TIME_gaussian[mask] = np.random.normal(means[2], sigmas[2], num_outliers)
        mask = (X_gaussian < 0) | (X_gaussian > 1) | \
               (Y_gaussian < 0) | (Y_gaussian > 1) | \
               (TIME_gaussian < 0) | (TIME_gaussian > 1)

    return X_gaussian, Y_gaussian, TIME_gaussian, PI_gaussian, ISEVENT_gaussian

class SalpeterIMF(rv_continuous):
    """
    Salpeter Initial Mass Function (IMF) probability distribution.

    The Salpeter IMF represents the distribution of stellar masses in a
    stellar population. It follows a power-law distribution with an index
    of -2.35 within the mass range of 0.1 to 120 solar masses.

    Parameters
    ----------
    a : float
        Lower bound of the mass range.
    b : float
        Upper bound of the mass range.
    name : str
        Name of the distribution.

    Returns
    -------
    pdf : float
        Probability density function value at the given x.

    Notes
    -----
    The probability density function (pdf) for the Salpeter IMF is given by:
        pdf(x) = x**(-2.35)    for x in [0.1, 120]
                0              otherwise
    """
    def _pdf(self, x):
        if x >= 0.1 and x <= 120:
            return x**(-2.35)
        else:
            return 0

def generate_stellar_temperatures(num_samples):
    """
    Generate a distribution of stellar temperatures based on the Salpeter IMF.

    This function generates a distribution of stellar temperatures by first
    sampling a distribution of stellar masses from the Salpeter IMF using the
    `rvs` method, and then converting those masses to temperatures using the
    mass-to-temperature conversion based on the mass-luminosity relation and
    the Stefan-Boltzmann law.

    Parameters
    ----------
    num_samples : int
        Number of stellar temperatures to generate.

    Returns
    -------
    temperatures : ndarray
        Array of generated stellar temperatures.

    Notes
    -----
    This function assumes that the stellar masses are given in solar masses.

    Examples
    --------
    >>> num_samples = 1000
    >>> temperatures = generate_stellar_temperatures(num_samples)
    >>> print(temperatures)
    """
    salpeter_imf = SalpeterIMF(a=0.1, b=120, name='salpeter_imf')
    stellar_masses = salpeter_imf.rvs(size=num_samples)
    
    luminosities = stellar_masses ** 3.5
    surface_temperatures = (luminosities / (4 * pi * sc.sigma)) ** 0.25
    
    return surface_temperatures

def generate_data(temperature, num_uniform_samples, num_gaussian_samples, seed):
    """
    Generate data with uniformly distributed events and Gaussian-distributed events.

    Parameters
    ----------
    temperature : float
        temperature associated with the emission of gaussian events
    num_uniform_samples : int
        Number of uniformly distributed data samples to generate.
    num_gaussian_samples : int
        Number of Gaussian distributed data samples to generate.
    seed : int
        Random seed for replicability.

    Returns
    -------
    tuple
        Tuple containing arrays for each dimension of the generated data.

    Notes
    -----
    This function generates data with uniformly distributed events and Gaussian-distributed events in a hypercube
    with range [0, 1]. The number of uniformly distributed events and Gaussian-distributed events will be the same,
    resulting in a total of 2 * num_samples data samples.

    The random seed is used to ensure replicability of the generated data. By setting the same seed, the same data
    will be generated each time.

    Example
    -------
    X, Y, TIME, PI, ISEVENT = generate_data(123, 1000)
    """

    random.seed(seed)
    np.random.seed(seed)

    X_uniform, Y_uniform, TIME_uniform, PI_uniform, ISEVENT_uniform = generate_uniform_events(num_uniform_samples)
    X_gaussian, Y_gaussian, TIME_gaussian, PI_gaussian, ISEVENT_gaussian = generate_gaussian_events(temperature, num_gaussian_samples)

    # Concatenate the uniformly distributed events and Gaussian-distributed events
    X = np.concatenate([X_uniform, X_gaussian])
    Y = np.concatenate([Y_uniform, Y_gaussian])
    TIME = np.concatenate([TIME_uniform, TIME_gaussian])
    PI = np.concatenate([PI_uniform, PI_gaussian])
    ISEVENT = np.concatenate([ISEVENT_uniform, ISEVENT_gaussian])

    # Shuffle the data
    indices = np.arange(num_uniform_samples + num_gaussian_samples)
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    TIME = TIME[indices]
    PI = PI[indices]
    ISEVENT = ISEVENT[indices]

    return X, Y, TIME, PI, ISEVENT


def save_data_to_fits(X, Y, TIME, PI, ISEVENT, filename):
    """
    Save data to a FITS file.

    Parameters
    ----------
    X : array-like
        Array containing the 'X' dimension data.
    Y : array-like
        Array containing the 'Y' dimension data.
    TIME : array-like
        Array containing the 'TIME' dimension data.
    PI : array-like
        Array containing the 'PI' dimension data.
    ISEVENT : array-like
        Array containing the 'ISEVENT' dimension data.
    filename : str
        Name of the FITS file to be saved.

    Returns
    -------
    None
        The function saves the data to the specified FITS file.

    Notes
    -----
    This function saves the given data arrays to a FITS file. The FITS file will contain a binary table with
    columns corresponding to each dimension of the data. The 'X', 'Y', 'TIME', and 'PI' dimensions are expected
    to be floating-point arrays, while the 'ISEVENT' dimension is expected to be an integer array.

    Example
    -------
    X = np.random.uniform(0, 1, 1000)
    Y = np.random.uniform(0, 1, 1000)
    TIME = np.random.uniform(0, 1, 1000)
    PI = np.random.uniform(0, 1, 1000)
    ISEVENT = np.zeros(1000)

    save_data_to_fits(X, Y, TIME, PI, ISEVENT, 'data.fits')
    """

    # Create a FITS table
    table = fits.BinTableHDU.from_columns([
        fits.Column(name='X', format='D', array=X),
        fits.Column(name='Y', format='D', array=Y),
        fits.Column(name='TIME', format='D', array=TIME),
        fits.Column(name='PI', format='D', array=PI),
        fits.Column(name='ISEVENT', format='I', array=ISEVENT)
    ])

    # Save the FITS file
    table.writeto(filename, overwrite=True)
    print("Data saved to", filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate data and save to FITS files.')
    parser.add_argument('num_files', type=int, help='Number of files to generate.')
    parser.add_argument('filename_pattern', type=str, help='Pattern for the filename. Use "{}" as a placeholder for file index.')
    parser.add_argument('--num_uniform_samples', type=int, default=1000, help='Number of uniformly distributed data samples to generate in each file.')
    parser.add_argument('--num_gaussian_samples', type=int, default=1000, help='Number of gaussian distributed data samples to generate in each file.')
    parser.add_argument('--seed', type=int, default=123, help='Random seed for replicability.')

    args = parser.parse_args()

    num_files = args.num_files
    filename_pattern = args.filename_pattern
    num_uniform_samples = args.num_uniform_samples
    num_gaussian_samples = args.num_gaussian_samples
    temperatures = generate_stellar_temperatures(num_files)
    seed = args.seed

    for i, temperature in enumerate(temperatures):
        X, Y, TIME, PI, ISEVENT = generate_data(temperature, num_uniform_samples, num_gaussian_samples, seed + i)
        filename = filename_pattern.format(i)
        save_data_to_fits(X, Y, TIME, PI, ISEVENT, filename)
