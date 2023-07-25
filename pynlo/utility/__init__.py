# -*- coding: utf-8 -*-
"""
Time and frequency grid utilities and other miscellaneous helper functions.

The submodules contain calculator-type functions for converting between
physically relevant parameters related to the linear and nonlinear
susceptibilities, as well as an efficient interface to fast Fourier transforms.

"""

__all__ = ["chi1", "chi2", "chi3", "fft",
           "vacuum", "taylor_series",
           "shift", "resample_v", "resample_t",
           "TFGrid"]


# %% Imports

import collections
import copy

import numpy as np
from scipy.constants import pi, h

from pynlo.utility import chi1, chi2, chi3, fft


# %% Collections

_ResampledV = collections.namedtuple("ResampledV", ["v_grid", "f_v", "dv", "dt"])

_ResampledT = collections.namedtuple("ResampledT", ["t_grid", "f_t", "dt"])

_RTFGrid = collections.namedtuple("RTFGrid", ["n",
                                             "v_grid", "v_ref", "dv", "v_window",
                                             "t_grid", "t_ref", "dt", "t_window"])


# %% Routines

def taylor_series(x0, fn):
    """
    Calculate a Taylor series expansion given the derivatives of a function
    about a point.

    Parameters
    ----------
    x0 : float
        The center point of the Taylor series expansion.
    fn : array_like
        The value and derivatives of the function evaluated at `x0`. The
        coefficients must be given in order of increasing degree, i.e.
        ``[f(x0), f'(x0), f''(x0), ...]``.

    Returns
    -------
    pwr_series : numpy.polynomial.Polynomial
        A NumPy `Polynomial` object representing the Taylor series expansion.

    """
    window = np.array([-1, 1])
    domain = window + x0
    poly_coefs = [coef/np.math.factorial(n) for (n, coef) in enumerate(fn)]
    pwr_series = np.polynomial.Polynomial(poly_coefs, domain=domain, window=window)
    return pwr_series

def vacuum(v_grid, rng=None):
    """
    Generate a root-power spectrum due to quantum vacuum fluctuations.

    Parameters
    ----------
    v_grid : array_like of float
        The frequency grid.
    rng : np.random.Generator, optional
        A NumPy random number generator. The default initializes a new
        `Generator` on each function call.

    Notes
    -----
    The combined noise of a coherent state's amplitude and phase quadratures is
    equal to that of the vacuum. A coherent state :math:`|\\alpha\\rangle` is
    defined by the displacement :math:`\\alpha = x_1 + i \\, x_2`, where
    :math:`x_1` and :math:`x_2` are the "amplitude" and "phase" (real and
    imaginary) quadrature amplitudes. In the number state basis
    :math:`|n\\rangle`, a coherent state takes the form of a Poissonian
    distribution:

    ..  math::
        |\\alpha\\rangle = e^{-\\frac{|\\alpha|^2}{2}}
            \sum_{n=0}^\\infty \\frac{\\alpha^n}{\sqrt{n!}} |n\\rangle

    The probability :math:`P[\\alpha]` of measuring displacement
    :math:`\\alpha` from a coherent state with average displacement
    :math:`\\beta`, a simultaneous measurement of :math:`x_1` and :math:`x_2`,
    is as follows:

    ..  math::
        P[\\alpha] = \\frac{1}{\\pi} |\\langle \\alpha | \\beta\\rangle|^2
            = \\frac{1}{\\pi} e^{-|\\alpha - \\beta|^2}

    This probability distribution is Gaussian, and its noise can be completely
    described by calculating the variance of each quadrature component. Scaled
    to the number of photons (:math:`N=\\alpha^2`), the combined noise from
    both quadratures gives a total variance of one photon per measurement:

    ..  math:: \\sigma_{x_1}^2 = \\sigma_{x_2}^2 = \\frac{1}{2}

    ..  math:: \\sigma_\\alpha^2 = \\sigma_{x_1}^2 + \\sigma_{x_2}^2 = 1

    The width of the probability distribution is independent of the coherent
    state's average displacement, which can be zero. This means that the
    root-photon noise can be generated independent of the state by sampling a
    normal distribution centered about zero mean. Also, since the Fourier
    transform of Gaussian noise is also Gaussian noise, the root-photon noise
    can be equivalently generated in either the time or frequency domains.
    Normalizing to the number of photons per measurement interval, the root
    photon noise for both quadratures becomes ``1/(2 * dt)**0.5`` for the time
    domain and ``1/(2 * dv)**0.5`` for the frequency domain. The final
    root-power noise is found by multiplying the frequency domain root-photon
    noise by the square root of the photon energy associated with each bin's
    frequency.

    Returns
    -------
    a_v : ndarray of complex
        The randomly-generated vacuum state root-power spectrum.

    """
    if rng is None:
        rng = np.random.default_rng()

    v_grid = np.asarray(v_grid, dtype=float)
    dv = np.mean(np.diff(v_grid))
    n = v_grid.size

    a_v = ((h*v_grid)/(2*dv))**0.5 * (rng.standard_normal(n) + 1j*rng.standard_normal(n))
    return a_v

def shift(f_t, dt, t_shift):
    """
    Fourier shift.

    The output array is at `f(t - t_shift)`.

    Parameters
    ----------
    f_t : array_like
        The input array.
    dt : float
        The grid step size.
    shift_t : float
        The amount to shift.

    Returns
    -------
    ndarray
        The shifted array.

    """
    f_t = np.asarray(f_t)

    # Grid
    n = f_t.shape[-1]
    dv = 1/(n*dt)
    v_grid = dv*(np.arange(n) - n//2)
    # Shift
    f_v = fft.fftshift(fft.fft(fft.ifftshift(f_t), fsc=dt))
    shift_v = np.exp(-1j*2*pi * v_grid * t_shift)
    shift_f_t = fft.fftshift(fft.ifft(fft.ifftshift(f_v * shift_v), fsc=dt))
    #
    if np.isreal(f_t).all():
        shift_f_t = shift_f_t.real
    return shift_f_t

def resample_v(v_grid, f_v, n):
    """
    Resample frequency-domain data to the given number of points.

    The complementary time data is assumed to be of finite support, so the
    resampling is accomplished by adding or removing trailing and leading time
    bins. Discontinuities in the frequency-domain amplitude will manifest as
    ringing when resampled.

    Parameters
    ----------
    v_grid : array_like of float
        The frequency grid of the input data.
    f_v : array_like of complex
        The frequency-domain data to be resampled.
    n : int
        The number of points at which to resample the input data. When the
        input corresponds to a real-valued time domain representation, this
        number is the number of points in the time domain.

    Returns
    -------
    v_grid : ndarray of float
        The resampled frequency grid.
    f_v : ndarray of real or complex
        The resampled frequency-domain data.
    dv : float
        The spacing of the resampled frequency grid.
    dt : float
        The spacing of the resampled time grid.

    Notes
    -----
    If the number of points is odd, there are an equal number of points on
    the positive and negative side of the time grid. If even, there is one
    extra point on the negative side.

    This method checks if the origin is contained in `v_grid` to determine
    whether real or complex transformations should be performed. In both cases
    the resampling is accomplished by removing trailing and leading time bins.

    For analytic representations, the returned frequency grid is defined
    symmetrically about its reference, as in the `TFGrid` class, and for
    real-valued representations the grid is defined starting at the origin.

    """
    assert isinstance(n, (int, np.integer)), "The requested number of points must be an integer"
    assert (n > 0), "The requested number of points must be greater than 0."
    assert (len(v_grid) == len(f_v)), (
        "The frequency grid and frequency-domain data must be the same length.")
    #---- Inverse Transform
    dv_0 = np.diff(v_grid).mean()
    if v_grid[0] == 0:
        assert np.isreal(f_v[0]), (
            "When the input is in the real-valued representation, the amplitude at the origin must be real.")

        # Real-Valued Representation
        if np.isreal(f_v[-1]):
            n_0 = 2*(len(v_grid)-1)
        else:
            n_0 = 2*(len(v_grid)-1) + 1
        dt_0 = 1/(n_0*dv_0)
        f_t = fft.fftshift(fft.irfft(f_v, fsc=dt_0, n=n_0))
    else:
        # Analytic Representation
        n_0 = len(v_grid)
        dt_0 = 1/(n_0*dv_0)
        v_ref_0 = v_grid[n_0//2]
        f_t = fft.fftshift(fft.ifft(fft.ifftshift(f_v), fsc=dt_0, overwrite_x=True))

    #---- Resample
    dn_n = n//2 - n_0//2 # leading time bins
    dn_p = (n-1)//2 - (n_0-1)//2 # trailing time bins
    if n > n_0:
        f_t = np.pad(f_t, (dn_n, dn_p), mode="constant", constant_values=0)
    elif n < n_0:
        f_t = f_t[-dn_n:n_0+dn_p]

    #---- Transform
    dt = 1/(n_0*dv_0)
    dv = 1/(n*dt)
    if v_grid[0] == 0:
        # Real-Valued Representation
        f_v = fft.rfft(fft.ifftshift(f_t), fsc=dt)
        v_grid = dv*np.arange(len(f_v))
    else:
        # Analytic Representation
        f_v = fft.fftshift(fft.fft(fft.ifftshift(f_t), fsc=dt, overwrite_x=True))
        v_grid = dv*(np.arange(n) - (n//2))
        v_grid += v_ref_0

    #---- Construct ResampledV
    resampled = _ResampledV(v_grid=v_grid, f_v=f_v, dv=dv, dt=1/(n*dv))
    return resampled

def resample_t(t_grid, f_t, n):
    """
    Resample time-domain data to the given number of points.

    The complementary frequency data is assumed to be band-limited, so the
    resampling is accomplished by adding or removing high frequency bins.
    Discontinuities in the time-domain amplitude will manifest as ringing when
    resampled.

    Parameters
    ----------
    t_grid : array_like of float
        The time grid of the input data.
    f_t : array_like of real or complex
        The time-domain data to be resampled.
    n : int
        The number of points at which to resample the input data.

    Returns
    -------
    t_grid : ndarray of float
        The resampled time grid.
    f_t : ndarray of real or complex
        The resampled time-domain data.
    dt : float
        The spacing of the resampled time grid.

    Notes
    -----
    If real, the resampling is accomplished by adding or removing the largest
    magnitude frequency components (both positive and negative). If complex,
    the input data is assumed to be analytic, so the resampling is accomplished
    by adding or removing the largest positive frequencies. This method checks
    the input data's type, not the magnitude of its imaginary component, to
    determine if it is real or complex.

    The returned time axis is defined symmetrically about the input's
    reference, such as in the `TFGrid` class.

    """
    assert isinstance(n, (int, np.integer)), "The requested number of points must be an integer"
    assert (n > 0), "The requested number of points must be greater than 0."
    assert (len(t_grid) == len(f_t)), (
        "The time grid and time-domain data must be the same length.")
    #---- Define Time Grid
    n_0 = len(t_grid)
    dt_0 = np.diff(t_grid).mean()
    t_ref_0 = t_grid[n_0//2]
    dv = 1/(n_0*dt_0)
    dt = 1/(n*dv)
    t_grid = dt*(np.arange(n) - (n//2))
    t_grid += t_ref_0

    #---- Resample
    if np.isrealobj(f_t):
        # Real-Valued Representation
        f_v = fft.rfft(fft.ifftshift(f_t), fsc=dt_0)
        if (n > n_0) and not (n % 2):
            f_v[-1] /= 2 # renormalize aliased Nyquist component
        f_t = fft.fftshift(fft.irfft(f_v, fsc=dt, n=n))
    else:
        # Analytic Representation
        f_v = fft.fftshift(fft.fft(fft.ifftshift(f_t), fsc=dt_0, overwrite_x=True))
        if n > n_0:
            f_v = np.pad(f_v, (0, n-n_0), mode="constant", constant_values=0)
        elif n < n_0:
            f_v = f_v[:n]
        f_t = fft.fftshift(fft.ifft(fft.ifftshift(f_v), fsc=dt, overwrite_x=True))

    #---- Construct ResampledT
    resampled = _ResampledT(t_grid=t_grid, f_t=f_t, dt=dt)
    return resampled


# %% Time and Frequency Grids

class TFGrid():
    """
    Complementary time- and frequency-domain grids for the representation of
    analytic functions with complex-valued envelopes.

    The frequency grid is shifted and scaled such that it is aligned with the
    origin and contains only positive frequencies. The values given to the
    initializers are only targets and may be adjusted slightly. If necessary,
    the reference frequency will be increased so that the grids can be formed
    without any negative frequencies.

    Parameters
    ----------
    n : int
        The number of grid points.
    v_ref : float
        The target central frequency of the grid.
    dv : float
        The frequency step size. This is equal to the reciprocal of the total
        time window.
    alias : int, optional
        The number of harmonics supported by the real-valued time domain grid
        without aliasing. The default is 1, which only generates enough points
        for one alias-free Nyquist zone. A higher number may be useful when
        simulating nonlinear interactions.

    Notes
    -----
    For discrete Fourier transforms (DFT), the frequency step multiplied by
    the time step is always equal to the reciprocal of the total number of
    points::

        dt*dv == 1/n

    Each grid point represents the midpoint of a bin that extends 0.5 grid
    spacings in both directions.

    Aligning the frequency grid to the origin facilitates calculations using
    real Fourier transforms, which have grids that start at zero frequency. The
    `rtf_grids` method and the `rn_range` and `rn_slice` attributes are useful
    when transitioning between the analytic representation of this class to the
    real-valued representation.

    By definition of the DFT, the time and frequency grids must range
    symmetrically about the origin, with the time grid incrementing in unit
    steps and the frequency grid in steps of ``1/n``. The grids of the `TFGrid`
    class are scaled and shifted such that they represent absolute time or
    frequency values. The scaling is accomplished by setting the forward scale
    parameter of the Fourier transforms to ``dt``. The `v_ref` and `t_ref`
    variables describe the amount that the `TFGrid` grids need to be shifted
    to come into alignment with the origins of the grids implicitly defined by
    the DFT.

    """
    def __init__(self, n, v_ref, dv, alias=1):
        assert isinstance(n, (int, np.integer)), "The number of points must be an integer."
        assert (n > 1),  "The number of points must be greater than 1."
        assert (dv > 0), "The frequency grid step size must be greater than 0."
        assert (v_ref > 0), "The target central frequency must be greater than 0."

        self._n = n
        self._dv = dv

        #---- Align Frequency Grid
        ref_idx = round(v_ref/self.dv)
        if ref_idx < self.n//2 + 1:
            ref_idx = self.n//2 + 1
        self._v_ref = ref_idx * dv

        min_idx = ref_idx - self.n//2
        max_idx = ref_idx + ((self.n-1) - self.n//2)
        self._rn_range = np.array([min_idx, max_idx])
        self._rn_slice = slice(self.rn_range.min(), self.rn_range.max() + 1)

        #---- Define Frequency Grid
        self.__v_grid = self.dv*(np.arange(self.n) - self.n//2) + self.v_ref
        self._v_ref = self.v_grid[self.n//2]
        self._v_window = self.n*self.dv

        #---- Define Complex Time Grid
        self._dt = 1/(self.n*self.dv)
        self.__t_grid = self.dt*(np.arange(self.n) - self.n//2)
        self._t_ref = self.t_grid[self.n//2]
        self._t_window = self.n*self.dt

        #---- Define Real-Valued Time and Frequency Domain Grids
        assert (alias >= 1), "There must be atleast 1 alias-free Nyquist zone."
        self.rtf_grids(alias=alias, update=True)

    #---- Class Methods
    @classmethod
    def FromFreqRange(cls, n, v_min, v_max, **kwargs):
        """
        Initialize a set of time and frequency grids given the total number of
        grid points and a target minimum and maximum frequency.

        Parameters
        ----------
        n : int
            The number of grid points.
        v_min : float
            The target minimum frequency.
        v_max : float
            The target maximum frequency.

        """
        assert (v_max > v_min), (
            "The target maximum frequency must be greater than the target minimum frequency.")
        dv = (v_max - v_min)/(n-1)
        v_ref = 0.5*(v_min + v_max)
        self = cls(n, v_ref, dv, **kwargs)
        return self

    #---- General Properties
    @property
    def n(self):
        """
        The number of grid points of the analytic representation.

        This value is the same for both the time and frequency grids.

        Returns
        -------
        int

        """
        return self._n

    @property
    def rn(self):
        """
        The number of grid points of the real-valued time domain
        representation.

        Returns
        -------
        int

        """
        return self._rn

    @property
    def rn_range(self):
        """
        The minimum and maximum indices of the origin-contiguous frequency
        grid, associated with the real-valued time domain representation, that
        correspond to the first and last points of the analytic frequency grid.

        These values are useful for indexing and constructing frequency grids
        for applications with real DFTs.

        Returns
        -------
        ndarray of float

        """
        return self._rn_range

    @property
    def rn_slice(self):
        """
        A slice object that indexes the origin-contiguous frequency grid,
        associated with the real-valued time domain representation, onto the
        analytic frequency grid.

        This is useful for indexing and constructing frequency gridded arrays
        for applications with real DFTs. It is assumed that the arrays are
        arranged such that the frequency coordinates are monotonically ordered.

        Returns
        -------
        slice

        """
        return self._rn_slice

    #---- Frequency Grid Properties
    @property
    def v_grid(self):
        """
        The frequency grid of the analytic representation, with units of
        ``Hz``.

        The frequency grid is aligned to the origin and contains only positive
        frequencies.

        Returns
        -------
        ndarray of float

        """
        return self.__v_grid

    @property
    def _v_grid(self):
        """
        The frequency grid of the analytic representation, arranged in standard
        fft order.

        Returns
        -------
        ndarray of float

        """
        return fft.ifftshift(self.v_grid)

    @property
    def v_ref(self):
        """
        The grid reference frequency of the analytic representation, with units
        of ``Hz``.

        This is the offset between `v_grid` and the frequency grid of the
        complex-envelope representation implicitly defined by a DFT with `n`
        points.

        Returns
        -------
        float

        """
        return self._v_ref

    @property
    def dv(self):
        """
        The frequency grid step size of the analytic representation, with units
        of ``Hz``.

        Returns
        -------
        float

        """
        return self._dv

    @property
    def v_window(self):
        """
        The span of the frequency grid in the analytic representation, with
        units of ``Hz``.

        This is equal to the number of grid points times the frequency grid
        step size.

        Returns
        -------
        float

        """
        return self._v_window

    #---- Time Grid Properties
    @property
    def t_grid(self):
        """
        The time grid of the analytic representation, with units of ``s``.

        The time grid is aligned symmetrically about the origin.

        Returns
        -------
        ndarray of float

        """
        return self.__t_grid

    @property
    def _t_grid(self):
        """
        The time grid of the analytic representation arranged in standard fft
        order.

        Returns
        -------
        ndarray of float

        """
        return fft.ifftshift(self.t_grid)

    @property
    def t_ref(self):
        """
        The grid reference time of the analytic representation, with units of
        ``s``.

        This is the offset between `t_grid` and the time grid of the
        complex-envelope representation implicitly defined by a DFT with `n`
        points.

        Returns
        -------
        float

        """
        return self._t_ref

    @property
    def dt(self):
        """
        The time grid step size of the analytic representation, with units of
        ``s``.

        The time step is used as the differential of Fourier transforms.
        Multiplying the input of the transform by this factor will preserve the
        integrated absolute squared magnitude of the transformed result::

            a_v = fft.fft(a_t, fsc=dt)
            np.sum(np.abs(a_t)**2 * dt) == np.sum(np.abs(a_v)**2 * dv)

        Returns
        -------
        float

        """
        return self._dt

    @property
    def t_window(self):
        """
        The span of the time grid in the analytic representation, with units of
        ``s``.

        This is equal to the number of grid points times the time grid step
        size.

        Returns
        -------
        float

        """
        return self._t_window

    #---- Real Time/Frequency Grid Properties
    @property
    def rv_grid(self):
        """
        The origin-contiguous frequency grid of the real-valued time domain
        representation, with units of ``Hz``.

        Returns
        -------
        ndarray of float

        """
        return self.__rv_grid

    @property
    def rv_ref(self):
        """
        The grid reference frequency of the real-valued time domain
        representation, with units of ``Hz``.

        Returns
        -------
        float

        """
        return self._rv_ref

    @property
    def rdv(self):
        """
        The frequency grid step size of the real-valued time domain
        representation, with units of ``Hz``.

        This is equal to the frequency grid step size of the analytic
        representation.

        Returns
        -------
        float

        """
        return self._dv

    @property
    def rv_window(self):
        """
        The span of the frequency grid in the real-valued time domain
        representation, with units of ``Hz``.

        Returns
        -------
        float

        """
        return self._rv_window

    @property
    def rt_grid(self):
        """
        The time grid of the real-valued time domain representation, with
        units of ``s``.

        Returns
        -------
        ndarray of float

        """
        return self.__rt_grid

    @property
    def _rt_grid(self):
        """
        The time grid of the real-valued time domain representation, arranged
        in standard fft order.

        Returns
        -------
        ndarray of float

        """
        return fft.ifftshift(self.rt_grid)

    @property
    def rt_ref(self):
        """
        The grid reference time of the real-valued time domain representation,
        with units of ``s``.

        Returns
        -------
        float

        """
        return self._rt_ref

    @property
    def rdt(self):
        """
        The time grid step size of the real-valued time domain representation,
        with units of ``s``.

        Returns
        -------
        float

        """
        return self._rdt

    @property
    def rt_window(self):
        """
        The span of the time grid in the real-valued time domain
        representation, with units of ``s``.

        Returns
        -------
        float

        """
        return self._rt_window

    def rtf_grids(self, alias=1, fast_n=True, update=False):
        """
        Complementary time and frequency domain grids for the representation of
        analytic functions with real-valued amplitudes.

        The `alias` parameter determines the number of harmonics the time grid
        supports without aliasing. In order to maintain efficient DFT behavior,
        the number of points can be extended further based on the output of
        `scipy.fft.next_fast_len` for aliases greater than or equal to 1. An
        alias of 0 returns the set of time and frequency grids consistent with
        a real-valued function defined over the original, analytic `t_grid`.

        The resulting frequency grid contains the origin and positive
        frequencies and is suitable for use with real DFTs (see `fft.rfft` and
        `fft.irfft`).

        Parameters
        ----------
        alias : float, optional
            The harmonic support of the generated grids (the number of
            alias-free Nyquist zones). The default is 1, the fundamental
            harmonic.
        fast_n : bool, optional
            A flag that determines whether the length of the new array is
            extended up to the next fast fft length. The default is to extend.
            This parameter has no effect when the `alias` is 0.
        update : bool, optional
            A flag that determines whether to update the real-valued time and
            frequency domain grids of the parent object with the results of
            this method. The default is to return the calculated grids without
            updating the associated values stored in the class. This parameter
            is only valid when `alias` is greater than or equal to 1.

        Returns
        -------
        n : int
            The number of grid points.
        v_grid : array of float
            The origin-contiguous frequency grid.
        v_ref : float
            The grid reference frequency.
        dv : float
            The frequency grid step size.
        v_window : float
            The span of the frequency grid.
        t_grid : array of float
            The time grid.
        t_ref : float
            The grid reference time.
        dt : float
            The time grid step size.
        t_window : float
            The span of the time grid.

        Notes
        -----
        To avoid dealing with case-specific amplitude scale factors when
        transforming between analytic and real-valued representations the
        frequency grid for complex-valued functions must not contain the origin
        and there must be enough points in the real-valued representation to
        avoid aliasing the Nyquist frequency of the analytic representation.
        The initializer of the `TFGrid` class enforces the first condition, the
        frequency grid starts at minimum one step size away from the origin,
        and this method enforces the second by making the minimum number of
        points odd if the real grid only extends to the first harmonic.

        The transformation between representations is performed as in the
        following example, with `tf` an instance of the `TFGrid` class, `rtf`
        the output of this method, `a_v` the spectrum of a complex-valued
        envelope defined over `v_grid`, `ra_v` the spectrum of the real-valued
        function defined over `rtf.v_grid`, and `ra_t` the real-valued
        function defined over `rtf.t_grid`. The ``1/2**0.5`` scale factor
        between `a_v` and `ra_v` preserves the integrated squared magnitude in
        the time domain::

            rtf = tf.rtf_grids()
            ra_v = np.zeros_like(rtf.v_grid, dtype=complex)
            ra_v[tf.rn_slice] = 2**-0.5 * a_v
            ra_t = fft.irfft(ra_v, fsc=rtf.dt, n=rtf.n)
            np.sum(ra_t**2 * rtf.dt) == np.sum(np.abs(a_v)**2 * tf.dv)

        """
        #---- Number of Points
        if alias==0:
            n = self.n
        else:
            assert (alias >= 1), "The harmonic support must be atleast 1."
            target_n_v = round(self.rn_range.max()*alias)
            if alias == 1:
                n = 2*target_n_v - 1 # odd
            else:
                n = 2*(target_n_v - 1) # even
            if fast_n:
                n = fft.next_fast_len(n)
        n_v = n//2 + 1 # points in the frequency grid

        #---- Define Frequency Grid
        v_grid = self.dv*np.arange(n_v)
        v_ref = v_grid[0]

        #---- Define Time Grid
        dt = 1/(n*self.dv)
        t_grid = dt*(np.arange(n) - n//2)
        t_ref = t_grid[n//2] # 0 by definition

        #---- Construct RTFGrid
        rtf_grids = _RTFGrid(
            n=n,
            v_grid=v_grid, v_ref=v_ref, dv=self.dv, v_window=n_v*self.dv,
            t_grid=t_grid, t_ref=t_ref, dt=dt, t_window=n*dt)

        if update and alias!=0:
            self._rn = rtf_grids.n

            # Frequency Grid
            self.__rv_grid = rtf_grids.v_grid
            self._rv_ref = rtf_grids.v_ref
            self._rv_window = rtf_grids.v_window

            # Time Grid
            self.__rt_grid = rtf_grids.t_grid
            self._rt_ref = rtf_grids.t_ref
            self._rdt = rtf_grids.dt
            self._rt_window = rtf_grids.t_window
        return rtf_grids

    #---- Misc
    def copy(self):
        """A copy of the time and frequency grids."""
        return copy.deepcopy(self)
