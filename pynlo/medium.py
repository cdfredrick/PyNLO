# -*- coding: utf-8 -*-
"""
Optical modes in waveguides or other media.

"""

__all__ = ["Mode"]


# %% Imports

import bisect
import collections
import copy

import numpy as np
from scipy.constants import c, pi


# %% Collections

_LinearOperator = collections.namedtuple("LinearOperator", ["u", "gain", "phase", "phase_raw"])
_LinearZ = collections.namedtuple("LinearZ", ["any", "alpha", "beta"])
_NonlinearZ = collections.namedtuple("NonlinearZ", ["any", "g2", "pol", "g3", "r3"])


# %% Single Mode

class Mode():
    """
    An optical mode.

    An optical mode is defined in the frequency domain. All properties must be
    input as effective values, i.e. those found by integrating out the
    transverse spatial dependence of the medium and mode. If a parameter is z
    dependent, it can be input as a function who's first argument is the
    propagation distance.

    Parameters
    ----------
    v_grid : array_like of float
        The frequency grid.
    beta : array_like of float or callable
        The phase coefficient, the real part of the complex wavenumber.
    alpha : array_like of float or callable, optional
        The gain coefficient, twice the imaginary part of the complex
        wavenumber.
    g2 : array_like of complex or callable, optional
        The effective 2nd-order nonlinearity.
    g2_inv : array_like of float, optional
        The location of all poled domain inversion boundaries.
    g3 : array_like of complex or callable, optional
        The effective 3rd-order nonlinearity.
    rv_grid : array_like of float, optional
        An origin-contiguous frequency grid associated with the 3rd-order
        nonlinear response function.
    r3 : array_like of complex or callable, optional
        The effective 3rd-order nonlinear response function containing both
        the Raman and instantaneous nonlinearities.
    z : float, optional
        The initial position within the mode. The default is 0.

    Notes
    -----
    Forward traveling waves of a mode are defined using the following
    conventions:

    .. math:: E, H \\sim a \\, e^{i(\\omega t - \\kappa z)} + \\text{c.c} \\\\
              \\kappa = \\beta + i \\frac{\\alpha}{2}, \\quad
              \\beta = n \\frac{\\omega}{c}

    """
    def __init__(self, v_grid, beta, alpha=None,
                 g2=None, g2_inv=None, g3=None, rv_grid=None, r3=None,
                 z=0.0):
        #---- Position
        self._z = z

        #---- Frequency Grid
        self._v_grid = np.asarray(v_grid, dtype=float)
        self._w_grid = 2*pi*self._v_grid

        #---- Refractive Index
        if callable(beta):
            assert (len(beta(z)) == len(v_grid)), "The length of beta must match v_grid."
            self._beta = beta
        else:
            assert (len(beta) == len(v_grid)), "The length of beta must match v_grid."
            self._beta = np.asarray(beta, dtype=float)

        #---- Gain
        if (alpha is None) or callable(alpha):
            self._alpha = alpha
        else:
            self._alpha = np.asarray(alpha, dtype=float)

        #---- 2nd-Order Nonlinearity
        if (g2 is None) or callable(g2):
            self._g2 = g2
        else:
            self._g2 = np.asarray(g2, dtype=complex)

        if g2_inv is None:
            self._g2_inv = None
            self._g2_inv_sorted = []
        else:
            assert (g2 is not None) and (g2_inv is not None), (
                "Poling can only be defined when g2 is defined")
            self._g2_inv_sorted = sorted(g2_inv)
            self._g2_inv = {z:(idx + 1) % 2 for idx, z in enumerate(self._g2_inv_sorted)}


        #---- 3rd-Order Nonlinearity
        if (g3 is None) or callable(g3):
            self._g3 = g3
        else:
            self._g3 = np.asarray(g3, dtype=complex)

        # Nonlinear Response Function
        if (rv_grid is not None) and (r3 is not None):
            assert (g3 is not None) and (r3 is not None), (
                "Raman nonlinearity can only be defined when g3 is defined")
            self._rv_grid = np.asarray(rv_grid, dtype=float)

            if callable(r3):
                self._r3 = r3
            else:
                assert (len(r3) == len(rv_grid)), "The length of r3 must match rv_grid."
                self._r3 = np.asarray(r3, dtype=complex)
        else:
            assert (rv_grid is None) and (r3 is None), (
                "rv_grid and r3 must both be defined at the same time or not at all.")
            self._rv_grid = None
            self._r3 = None

        #---- Z Dependence
        self._z_linear = _LinearZ(
            any=callable(alpha) or callable(beta),
            alpha=callable(alpha), beta=callable(beta))
        self._z_nonlinear = _NonlinearZ(
            any=callable(g2) or callable(g3) or callable(r3),
            g2=callable(g2), pol=g2_inv is not None,
            g3=callable(g3), r3=callable(r3))
        self._z_mode = self.z_linear.any or self.z_nonlinear.any or self.z_nonlinear.pol

    #---- General Properties
    @property
    def z(self):
        """
        The position within the mode, with units of ``m``.

        Returns
        -------
        float

        """
        return self._z
    @z.setter
    def z(self, z):
        self._z = z

    @property
    def v_grid(self):
        """
        The frequency grid, with units of ``Hz``.

        Returns
        -------
        ndarray of float

        """
        return self._v_grid

    @property
    def rv_grid(self):
        """
        The origin-contiguous frequency grid associated with the Raman
        response. Units are in ``Hz``.

        Returns
        -------
        None or ndarray of float

        """
        return self._rv_grid

    @property
    def z_mode(self):
        """
        The z dependence of the mode.

        Returns
        -------
        bool

        """
        return self._z_mode

    @property
    def z_linear(self):
        """
        The z dependence of the linear terms.

        Returns
        -------
        any : bool
            Whether there is any z dependence of the linearity.
        alpha : bool
            Z-dependent gain coefficient.
        beta : bool
            Z-dependent phase coefficient.

        """
        return self._z_linear

    @property
    def z_nonlinear(self):
        """
        The z dependence of the nonlinear terms.

        Returns
        -------
        any : bool
            Whether there is any z dependence of the nonlinearity (excluding
            poling).
        g2 : bool
            Z-dependent 2nd-order nonlinear parameter.
        pol : bool
            Poled 2nd-order nonlinearity.
        g3 : bool
            Z-dependent 3rd-order nonlinear parameter.
        r3 : bool
            Z-dependent Raman response.

        """
        return self._z_nonlinear

    #---- 1st-Order Properties
    @property
    def alpha(self):
        """
        The gain coefficient, with units of ``1/m``.

        Positive values correspond to gain and negative values to loss.

        Returns
        -------
        None or ndarray of float

        """
        return self._alpha(self.z) if callable(self._alpha) else self._alpha

    @property
    def beta(self):
        """
        The phase coefficient, or angular wavenumber, with units of ``1/m``.

        Returns
        -------
        ndarray of float

        """

        return self._beta(self.z) if callable(self._beta) else self._beta

    @property
    def n(self):
        """
        The refractive index.

        Returns
        -------
        ndarray of float

        """
        return self.beta*c/self._w_grid

    @property
    def beta1(self):
        """
        The group walk-off parameter, with units of ``s/m``.

        Returns
        -------
        ndarray of float

        """
        return np.gradient(self.beta, self._w_grid, edge_order=2)

    def d_12(self, v0=None):
        """
        The group velocity mismatch, with units of ``s/m``.

        Parameters
        ----------
        v0 : float, optional
            The target reference frequency. The central frequency is selected
            by default.

        Returns
        -------
        ndarray of float

        """
        if v0 is None:
            v0 = self.v_grid[self.v_grid.size//2]
        v0_idx = np.argmin(np.abs(v0 - self.v_grid))
        beta1 = self.beta1
        return beta1[v0_idx] - beta1

    @property
    def n_g(self):
        """
        The group index.

        Returns
        -------
        ndarray of float

        """
        return c*self.beta1

    @property
    def v_g(self):
        """
        The group velocity, with units of ``m/s``.

        Returns
        -------
        ndarray of float

        """
        return 1/self.beta1

    @property
    def beta2(self):
        """
        The group velocity dispersion (GVD), with units of ``s**2/m``.

        Returns
        -------
        ndarray of float

        """
        return np.gradient(self.beta1, self._w_grid, edge_order=2)

    @property
    def D(self):
        """
        The dispersion parameter D, with units of ``s/m**2``.

        Returns
        -------
        ndarray of float

        """
        return -2*pi/c * self.v_grid**2 * self.beta2 #TODO: test against chi1 helper functions

    def linear_operator(self, dz, v0=None):
        """
        The linear operator which advances a pulse over a distance `dz`.

        The linear operator acts on the analytic spectrum through
        multiplication.

        Parameters
        ----------
        dz : float
            The step size.
        v0 : float, optional
            The target reference frequency of the comoving frame. The default
            selects the central frequency.

        Returns
        -------
        u : ndarray of complex
            The forward evolution operator.
        gain : float or ndarray
            The accumulated gain or loss (multiplicative).
        phase : ndarray of float
            The accumulated phase in the comoving frame (additive).
        phase_raw : ndarray of float
            The raw accumulated phase.

        """
        #---- Gain
        alpha = self.alpha
        if alpha is None:
            alpha = 0.0
        gain = np.exp(alpha*dz)

        #---- Phase
        beta_raw = self.beta

        # Comoving frame
        if v0 is None:
            v0 = self.v_grid[self.v_grid.size//2]
        v0_idx = np.argmin(np.abs(v0 - self.v_grid))
        beta_cm = beta_raw - self.beta1[v0_idx]*self._w_grid

        #---- Propagation Constant
        kappa = beta_cm + 0.5j*alpha

        #---- Linear Operator
        operator = np.exp(-1j*kappa*dz)

        lin_operator = _LinearOperator(
            u=operator, gain=gain, phase=dz*beta_cm, phase_raw=dz*beta_raw)
        return lin_operator

    #---- 2nd-Order Properties
    @property
    def g2(self):
        """
        The magnitude of the effective 2nd-order nonlinear parameter, with
        units of ``1/(W**0.5*m*Hz)``.

        Returns
        -------
        None or ndarray of complex

        """
        return self._g2(self.z) if callable(self._g2) else self._g2

    @property
    def g2_inv(self):
        """
        The location of all 2nd-order domain inversion boundaries within the
        mode.

        A value of 1 indicates the start of an inverted domain. A value of
        0 indicates the start of an unpoled region.

        Returns
        -------
        None or dict of int

        """
        return self._g2_inv

    @property
    def g2_pol(self):
        """
        The poling status at the current z position.

        A value of 1 indicates that the current position is in a region with an
        inverted domain. A value of 0 indicates an unpoled region.

        Returns
        -------
        int

        """
        poled = bisect.bisect_right(self._g2_inv_sorted, self.z) % 2
        return poled

    #---- 3rd-Order Properties
    @property
    def g3(self):
        """
        The effective 3rd-order nonlinear parameter, with units of
        ``1/(W*m*Hz)``.

        Returns
        -------
        None or ndarray of complex

        """
        return self._g3(self.z) if callable(self._g3) else self._g3

    @property
    def gamma(self):
        """
        The nonlinear parameter :math:`\\gamma`, with units of ``1/(W*m)``.

        Returns
        -------
        None or ndarray of complex
        """
        g3 = self.g3
        if g3 is not None and len(g3.shape)>=2:
            g3 = g3[0] * np.sum(g3[1:]**3, axis=0)
        return 3/2*self._w_grid*g3 if g3 is not None else None #TODO: test against chi3 helper functions

    @property
    def r3(self):
        """
        The effective 3rd-order nonlinear response function containing both
        the Raman and instantaneous nonlinearities.

        Returns
        -------
        None or ndarray of complex
        """
        return self._r3(self.z) if callable(self._r3) else self._r3

    #---- Misc
    def copy(self):
        """A copy of the mode."""
        return copy.deepcopy(self)


# class GaussianMode():
#     """
#     Fundamental Gaussian modes for simulating single-mode free-space
#     propagation.
#     - effective area based on distance to nominal waist location
#     - could also include convenience functions for setting up the beam
#       through focusing, propagation, etc.
#     - check Boyd 2.10 "Nonlinear Optical Interactions with Focused Gaussian
#       Beams" for complicating factors.
#
#     """


# %% Multimode

# class Waveguide():
#     """
#     Collection of modes and the nonlinear interactions between them
#     """
#     def __init__(self, modes, coupling):
#         pass

# class FreeSpace(Waveguide):
#     """
#     Collection of Hermite–Gaussian or Laguerre–Gaussian modes for simulating
#     free space propagation of arbitrary distribution.
#     - effective area based on distance to nominal waist location
#     - could also include convenience functions for setting up the beam
#       through focusing, propagation, etc.
#
#     """
