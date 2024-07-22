# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 12:08:19 2024

@author: Dennis Scheidt
"""
from _helper_functions import *

import numpy as np
import matplotlib.pyplot as plt
from math import factorial
from numpy.fft import fft2, fftshift
from scipy.linalg import hadamard
import spgl1
from numpy.linalg import pinv
from scipy.fftpack import idct
from scipy.io import loadmat
import os

from PIL import Image

import time



pi = np.pi

def prism_phase(X,Y,kx,ky):
    """
    Helper to create a prism phase
    
    X: mehsgrid of x coordinates
    Y: meshgrid of y coordinates
    kx: displacement in x
    ky: displacement in y
    """
    return (X*kx + Y *ky) % (2*pi)


def _is_odd(integer):
    """Helper for testing if an integer is odd by bitwise & with 1.
    TAKEN FROM: https://github.com/mperrin/poppy/blob/master/poppy/zernike.py
    """
    
    return integer & 1

def R(n, m, rho):
    """Compute R[n, m], the Zernike radial polynomial

    Parameters
    ----------
    n, m : int
        Zernike function degree
    rho : array
        Image plane radial coordinates. `rho` should be 1 at the desired pixel radius of the
        unit circle
        
    TAKEN FROM: https://github.com/mperrin/poppy/blob/master/poppy/zernike.py
    """

    m = int(np.abs(m))
    n = int(np.abs(n))
    output = np.zeros(rho.shape)
    if _is_odd(n - m):
        return 0
    else:
        for k in range(int((n - m) / 2) + 1):
            coef = ((-1) ** k * factorial(n - k) /
                    (factorial(int(k)) * factorial(int((n + m) / 2. - k)) * factorial(int((n - m) / 2. - k))))
            output += coef * rho ** (n - 2 * k)
        return output



def zernike(n, m, npix=100, rho=None, theta=None, outside=np.nan,
            noll_normalize=True, **kwargs):
    """Return the Zernike polynomial Z[m,n] for a given pupil.

    For this function the desired Zernike is specified by 2 indices m and n.
    See zernike1 for an equivalent function in which the polynomials are
    ordered by a single index.

    You may specify the pupil in one of two ways:

        zernike(n, m, npix)
            where npix specifies a pupil diameter in pixels.
            The returned pupil will be a circular aperture
            with this diameter, embedded in a square array
            of size npix*npix.

        zernike(n, m, rho=r, theta=theta)
            Which explicitly provides the desired pupil coordinates
            as arrays r and theta. These need not be regular or contiguous.

    The expressions for the Zernike terms follow the normalization convention
    of Noll et al. JOSA 1976 unless the `noll_normalize` argument is False.

    Parameters
    ----------
    n, m : int
        Zernike function degree
    npix : int
        Desired diameter for circular pupil. Only used if `rho` and
        `theta` are not provided.
    rho, theta : array_like
        Image plane coordinates. `rho` should be 0 at the origin
        and 1.0 at the edge of the circular pupil. `theta` should be
        the angle in radians.
    outside : float
        Value for pixels outside the circular aperture (rho > 1).
        Default is `np.nan`, but you may also find it useful for this to
        be 0.0 sometimes.
    noll_normalize : bool
        As defined in Noll et al. JOSA 1976, the Zernike definition is
        modified such that the integral of Z[n, m] * Z[n, m] over the
        unit disk is pi exactly. To omit the normalization constant,
        set this to False. Default is True.

    Returns
    -------
    zern : 2D numpy array
        Z(m,n) evaluated at each (rho, theta)
        
    TAKEN FROM: https://github.com/mperrin/poppy/blob/master/poppy/zernike.py
    """
    if not n >= m:
        raise ValueError("Zernike index m must be >= index n")
    if (n - m) % 2 != 0:
        print("Radial polynomial is zero for these inputs: m={}, n={} "
                  "(are you sure you wanted this Zernike?)".format(m, n))
    print("Zernike(n=%d, m=%d)" % (n, m))

    if theta is None and rho is None:
        x = (np.arange(npix, dtype=np.float64) - (npix - 1) / 2.) / ((npix - 1) / 2.)
        y = x
        xx, yy = np.meshgrid(x, y)

        rho = np.sqrt(xx ** 2 + yy ** 2)
        theta = np.arctan2(yy, xx)
    elif (theta is None and rho is not None) or (theta is not None and rho is None):
        raise ValueError("If you provide either the `theta` or `rho` input array, you must "
                         "provide both of them.")

    if not np.all(rho.shape == theta.shape):
        raise ValueError('The rho and theta arrays do not have consistent shape.')

    aperture = (rho <= 1)

    if m == 0:
        if n == 0:
            zernike_result = aperture
        else:
            norm_coeff = np.sqrt(n + 1) if noll_normalize else 1
            zernike_result = norm_coeff * R(n, m, rho) * aperture
    elif m > 0:
        norm_coeff = np.sqrt(2) * np.sqrt(n + 1) if noll_normalize else 1
        zernike_result = norm_coeff * R(n, m, rho) * np.cos(np.abs(m) * theta) * aperture
    else:
        norm_coeff = np.sqrt(2) * np.sqrt(n + 1) if noll_normalize else 1
        zernike_result = norm_coeff * R(n, m, rho) * np.sin(np.abs(m) * theta) * aperture

    zernike_result[(rho > 1)] = outside
    return zernike_result

def noll_indices(j):
    """Convert from 1-D to 2-D indexing for Zernikes or Hexikes.

    Parameters
    ----------
    j : int
        Zernike function ordinate, following the convention of Noll et al. JOSA 1976.
        Starts at 1.
        
    TAKEN FROM: https://github.com/mperrin/poppy/blob/master/poppy/zernike.py

    """

    if j < 1:
        raise ValueError("Zernike index j must be a positive integer.")

    # from i, compute m and n
    # I'm not sure if there is an easier/cleaner algorithm or not.
    # This seems semi-complicated to me...

    # figure out which row of the triangle we're in (easy):
    n = int(np.ceil((-1 + np.sqrt(1 + 8 * j)) / 2) - 1)
    if n == 0:
        m = 0
    else:
        nprev = (n + 1) * (n + 2) / 2  # figure out which entry in the row (harder)
        # The rule is that the even Z obtain even indices j, the odd Z odd indices j.
        # Within a given n, lower values of m obtain lower j.

        resid = int(j - nprev - 1)

        if _is_odd(j):
            sign = -1
        else:
            sign = 1

        if _is_odd(n):
            row_m = [1, 1]
        else:
            row_m = [0]

        for i in range(int(np.floor(n / 2.))):
            row_m.append(row_m[-1] + 2)
            row_m.append(row_m[-1])

        m = row_m[resid] * sign

    print("J=%d:\t(n=%d, m=%d)" % (j, n, m))
    return n, m

def vec_2_mask(vec,mask_shape,superpixel_shape):
    """
    vec: 1d vector 
    mask_shape: shape of the 2d grid (N,N)
    superpixel_shape: shape of the 2d superpixel (M,M)
    
    take a 1d vector and project it into a 2d distribution
    """
    mask = np.zeros(mask_shape)
    [n,m] = np.array(mask_shape) // np.array(superpixel_shape)
    
    arr_x = np.linspace(0,mask_shape[0], n+1, dtype = int)
    arr_y = np.linspace(0,mask_shape[1], m+1, dtype = int)
    
    arr_x1 = arr_x[0:-1]; arr_x2 = arr_x[1:]
    arr_y1 = arr_y[0:-1]; arr_y2 = arr_y[1:]
    iterator = 0
    for i1, i2 in zip(arr_x1,arr_x2):
        for j1,j2 in zip(arr_y1,arr_y2):
            mask[i1:i2,j1:j2] = vec[iterator]
            iterator += 1
    return mask

def cs_reconstruction(vec, dat, cr, lambd = 10000, typ = 'lasso'):
    """
    vec: measurement matrix
    dat: data vector 
    cr: compression ratio
    lambd: parameter for the basis pursuit algorithm -- unsused
    typ: type of the basis pursuit algorithm -- unused
    
    Reconstruct the amplitude and phase from the measurement, depending on the compression ratio cr
    if cr = 1: just solve the system of equations
    if cr <1: partiotionate the data and apply compressive sensing
    
    The reconstruction is donce for each phase offset 
    Afterwards, the complex phasor is reconstructed, which yields amplitude and phase 
    """
    start = time.time()
    N,m = dat.shape
    ms = np.arange(m)
    M = int(cr * N)
    x = np.zeros([N,m],complex)
    if cr == 1:
        for j in ms:
            x[:,j] = pinv(vec) @ dat[:,j]
    else:
        Phi = vec[:M,:]
        Theta = np.zeros([M,N])
        b = dat[:M,:]
        s = np.zeros([N,m],complex)
        for i in range(N):
            ek = np.zeros(N)
            ek[i] = 1
            psi = idct(ek)
            Theta[:,i] = Phi @ psi
            
        for j in ms:
            if typ == 'bp':
                 s[:,j], _ , _ , _ = spgl1.spg_bp(Theta,b[:,j],verbosity = 0)
            elif typ == 'bpdn':
                s[:,j], _ , _ , _ = spgl1.spg_bpdn(Theta,b[:,j],lambd,verbosity = 0)
            elif typ == 'lasso':
                s[:,j], _ , _ , _ = spgl1.spg_lasso(Theta,b[:,j],lambd,verbosity = 0)
            for i in range(N):
                ek = np.zeros(N)
                ek[i] = 1
                psi = idct(ek)
                x[:,j] += psi * s[i,j]
    if m == 3:           
        phasor = -1/3 * (x[:,1] + x[:,2] - 2*x[:,0]) + 1j/np.sqrt(3) * (x[:,2] - x[:,1])
    elif m == 4:
        phasor = 1/4 * (x[:,0] - x[:,2]) + 1j/4 *(x[:,1] - x[:,3])
    amp = abs(phasor)
    phas = np.angle(phasor)
    t = time.time() - start
    
    return amp, phas, t

def spiral_mask(X,Y):
    """
    function to create a binary star mask
    """
    n,m = X.shape
    angs = np.arctan2(X-m//2, Y-n//2)%(2*pi/4)
    angs = angs/angs.max() * 2*pi
    angs[angs <= pi] = 0
    angs[angs!=0] = 1
    return angs

def unpickle(file):
    """
    fuction to decript the data of CIFAR-10 database
    """
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
