from cmath import sqrt
import sys
from math import cos, sin
from typing import Tuple

import numpy as np

from scipy.linalg import norm

from .constants import *

from math import prod, floor
from itertools import product



def outgoing_kz(kz_squared, atol=1e-12):
    """Return the outgoing/decaying square-root branch used by Khepri.

    Khepri propagates the +z branch with exp(-1j*kz*z). Therefore an
    outgoing propagating wave uses Re(kz) >= 0, while a purely evanescent
    +z wave uses Im(kz) <= 0 so that it decays with increasing z.

    Only the SIGN of the square root is changed. The root is never complex
    conjugated, so kz**2 remains exactly equal to kz_squared for complex
    media as well.
    """
    scalar = np.ndim(kz_squared) == 0
    kz = np.sqrt(np.asarray(kz_squared, dtype=np.complex128))

    flip = np.real(kz) < -atol
    on_imag_axis = np.abs(np.real(kz)) <= atol
    flip |= on_imag_axis & (np.imag(kz) > 0.0)
    kz = np.where(flip, -kz, kz)

    if scalar:
        return kz.item()
    return kz



def compute_kplanar(eps_inc, wavelength,theta_deg=0.0, phi_deg=0.0):
    phi = np.deg2rad(phi_deg)
    theta = np.deg2rad(theta_deg)
    kp = np.array([np.cos(phi), np.sin(phi)], dtype=complex)
    return kp * sqrt(eps_inc) * 2 * np.pi / wavelength * np.sin(theta)

def rotation_matrix(theta):
    return np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])

def block2dense(block_matrix):
    blocks_shape = block_matrix.shape[0:2]
    submatrices_shape = block_matrix.shape[2:]
    dense = np.swapaxes(block_matrix.copy(), 1, 2)
    return dense.reshape([blocks_shape[i]*submatrices_shape[i] for i in [0,1]])


def convolution_matrix(structure, harmonics):
    fourier = np.fft.fftshift(np.fft.fft2(structure)) / prod(structure.shape)
    return convolution_matrix_fourier(fourier, harmonics)


def convolution_matrix_fourier(fourier_coefficients, harmonics, dtype=np.complex128, fourier=False):
    convmat_dim = prod(harmonics)
    convmat_shape = convmat_dim, convmat_dim
    convmat = np.zeros(convmat_shape, dtype=dtype)

    Nx, Ny = fourier_coefficients.shape
    P, Q = harmonics

    g0 = np.array([floor(n/2) for n in (Nx, Ny)])

    gparkour = list(product(range(Q), range(P)))
    for qrow, prow in gparkour:
        row = qrow*P + prow
        for qcol, pcol in gparkour:
            col = qcol*P + pcol
            gd = np.array([prow - pcol, qrow - qcol])
            g = g0 + gd
            convmat[row, col] = fourier_coefficients[g[0], g[1]]
    return convmat.astype("complex")



def unitcellarea(a1, a2):
    """ Returns the area given base vectors. """
    return abs(a1[0] * a2[1] - a1[1] * a2[0])

def reciproc(a1, a2):
    """ Compute reciproc lattice basis vectors. """
    coef = twopi / (a1[0] * a2[1] - a1[1] * a2[0])
    b1 = (  a2[1] * coef, -a2[0] * coef)
    b2 = ( -a1[1] * coef,  a1[0] * coef)
    return b1, b2
