# -*- coding: utf-8 -*-
"""
Conversion functions and other calculators relevant to the linear
susceptibility.

"""

__all__ = ["n_to_beta", "beta_to_n", "D_to_beta2", "beta2_to_D"]


# %% Imports

from scipy.constants import pi, c


#%% Converters

# TODO: forward and backward transformations, test with equivalents from medium.Mode

#---- Wavenumber and Linear Susceptibility chi1
def chi1_to_k(v_grid, chi1):
    k = 2*pi*v_grid/c * (1 + chi1)**0.5
    alpha = 2*k.imag
    beta = k.real
    return beta, alpha

def k_to_chi1(v_grid, beta, alpha=None):
    if alpha is None:
        k = beta
    else:
        k = beta + 1j/2 * alpha
    return (c/(2*pi*v_grid) * k)**2 - 1

#---- Phase Coefficient and Refractive Index
def n_to_beta(v_grid, n):
    """
    Refractive index to phase coefficient.

    Parameters
    ----------
    v_grid : array_like of float
        The frequency grid.
    n : array_like of float
        The refractive indices.

    Returns
    -------
    beta

    """
    return n * (2*pi*v_grid/c)

def beta_to_n(v_grid, beta):
    """
    Phase coefficient to refractive index

    Parameters
    ----------
    v_grid : array_like of float
        The frequency grid.
    beta : array_like of float
        The angular wavenumbers.

    Returns
    -------
    n

    """
    return beta / (2*pi*v_grid/c)

#---- GVD and Dispersion Parameter D
def D_to_beta2(v_grid, D):
    """
    Dispersion parameter D to group velocity dispersion beta_2 (GVD).

    Parameters
    ----------
    v_grid : array_like of float
        The frequency grid.
    D : array_like of float
        The dispersion parameter D, in units of ``s/m**2``.

    Returns
    -------
    The GVD, in units of ``s**2/m``.

    """
    return D / (-2*pi * v_grid**2/c)

def beta2_to_D(v_grid, beta2):
    """
    Group velocity dispersion beta_2 (GVD) to dispersion parameter D.

    Parameters
    ----------
    v_grid : array_like of float
        The frequency grid.
    beta2 : array_like of float
        The GVD parameter, in units of ``s**2/m``.

    Returns
    -------
    The dispersion parameter, in units of ``s/m**2``.

    """
    return beta2 * (-2*pi * v_grid**2/c)


# %% Calculator Functions

def linear_length():
    pass #TODO
