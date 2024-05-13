# -*- coding: utf-8 -*-
"""
Conversion functions and other calculators relevant to the 3rd-order nonlinear
susceptibility.

"""

__all__ = ["gamma_to_g3", "g3_to_gamma", "g3_split", "g3_spm", "raman"]


# %% Imports

import numpy as np
from scipy.constants import pi, c, epsilon_0 as e0

from pynlo.utility import fft


#%% Converters

#---- Effective Nonlinearities g3 and gamma
def gamma_to_g3(v_grid, gamma, t_shock=None):
    """
    Convert from the gamma to the g3 nonlinear parameter.

    The `t_shock` parameter is used to determine the strength of the
    self-steepening effect when gamma is only given at a single point. If an
    array is given for gamma, it is assumed to already contain the
    self-steepening effect.

    Parameters
    ----------
    v_grid : array_like of float
        The frequency grid.
    gamma : array_like
        The effective nonlinear parameter.
    t_shock : float, optional
        The characteristic time scale of optical shock formation. This
        time scale is typically equal to ``1/(2*pi*v0)``, where `v0` is the
        frequency at which `gamma` is defined, but it may also contain
        corrections to `gamma` due to the frequency dependence of the optical
        mode. This parameter is only valid if `gamma` is given at a single
        point.

    Returns
    -------
    g3

    """
    gamma = np.asarray(gamma)
    if t_shock is not None:
        assert gamma.size == 1, "t_shock is only valid when a gamma is given at a single point."
        v0_eff = 1/(2*pi*t_shock)
        gamma = gamma * v_grid/v0_eff
    return gamma / (3/2 * 2*pi*v_grid)

def g3_to_gamma(v_grid, g3):
    """
    Convert from the g3 to the gamma nonlinear parameter.

    Parameters
    ----------
    v_grid : array_like of float
        The frequency grid.
    g3 : array_like
        The effective nonlinear parameter.

    Returns
    -------
    gamma

    """
    return g3 * (3/2 * 2*pi*v_grid)

#---- 3rd-Order Nonlinear Susceptibility chi3 and Kerr Parameter
def n2_to_chi3(n, n2):
    return n2 * (4/3 * e0*c*n**2)

def chi3_to_n2(n, chi3):
    return chi3 / (4/3 * e0*c*n**2)

#---- Effective Nonlinearity gamma and Kerr Parameter
def n2_to_gamma(v_grid, a_eff, n2):
    return n2 * (2*pi*v_grid/(c*a_eff))

def gamma_to_n2(v_grid, a_eff, gamma):
    return gamma / (2*pi*v_grid/(c*a_eff))


# %% Nonlinearity

def g3_split(n_eff, a_eff, chi3_eff):
    """
    The 3rd-order nonlinear parameter decomposed according to unit analysis.

    Parameters
    ----------
    n_eff : array_like of float
        The refractive indices.
    a_eff : array_like of float
        The effective areas.
    chi3_eff : array_like
        The effective 3rd-order susceptibilities.

    Returns
    -------
    ndarray (2, n)
        The unit decomposition of the 3rd-order nonlinear parameter. The first
        index contains the output factors while the second index contains the
        input factors.

    """
    return np.array([1/2 * ((e0*a_eff)/(c*n_eff))**0.5 * chi3_eff,
                     1/(e0*c*n_eff*a_eff)**0.5])

def g3_spm(n_eff, a_eff, chi3_eff):
    """
    The 3rd-order nonlinear parameter weighted for self-phase modulation.

    Parameters
    ----------
    n_eff : array_like of float
        The effective refractive indices.
    a_eff : array_like of float
        The effective areas.
    chi3_eff : array_like
        The effective 3rd-order susceptibilities.

    Returns
    -------
    g3 : ndarray

    """
    return 1/2 * chi3_eff/(e0*c**2 * n_eff**2 * a_eff)

def raman(n, dt, r_weights, b_weights=None, analytic=True):
    """
    Calculate the frequency-domain Raman and instantaneous nonlinear response
    function.

    The equations used are the approximated formulations as summarized in
    section 2.3.3 of Agrawal's Nonlinear Fiber Optics [1]_. More accurate
    simulations may be obtainable using digitized experimental measurements,
    such as those shown in figure 2.2 of [1]_. The coefficients listed in
    Agrawal for silica-based fibers are as follows::

        r_weights = [0.245*(1-0.21), 12.2e-15, 32e-15] # resonant contribution
        b_weights = [0.245*0.21, 96e-15] # boson contribution

    Parameters
    ----------
    n : int
        The number of points in the time domain.
    dt : float
        The time grid step size.
    r_weights : array_like of float
        The contributions due to resonant vibrations. Must be given as
        ``[fraction, tau_1, tau_2]``, where `fraction` is the fractional
        contribution of the resonance to the total nonlinear response function,
        `tau_1` is the period of the vibrational frequency, and `tau_2` is the
        resonance's characteristic decay time. More than one resonance may be
        entered using an (n, 3) shaped array.
    b_weights : array_like of float, optional
        The contributions due to boson peaks found in amorphous materials. Must
        be given as ``[fraction, tau_b]``, where `fraction` is the
        fractional contribution of the boson peak to the total nonlinear
        response function, and `tau_b` is the boson peak's characteristic
        decay time. More than one peak may be entered using an (n, 2) shaped
        array.
    analytic : bool, optional
        A flag that sets the proper normalization for use with the analytic or
        real-valued representation. The default normalizes for the analytic
        representation, which is the proper format for use with the `NLSE`
        model. Set this parameter to `False` if using the `UPE` model.

    Returns
    -------
    rv_grid : ndarray of float
        The origin-continuous frequency grid associated with the nonlinear
        response function.
    nonlinear_v : ndarray of complex
        The frequency-domain nonlinear response function. This is defined over
        the frequency grid given by ``dv=1/(n*dt)``.

    Notes
    -----
    For the carrier-resolved or real-valued representation, an additional
    factor of 3/2 is necessary to properly normalize the Raman response. The
    formulas used in this method have been fit to the analytic representation,
    which is normalized assuming that all three self-phase modulation pathways
    fold through baseband. In the real-valued domain however, only two pass
    through baseband. The third pathway is through the second harmonic. Thus,
    in the real-valued representation the Raman response must be renormalized
    to produce the same nonlinear response against 2/3 the spectral amplitude.

    References
    ----------
    .. [1] Agrawal GP. Nonlinear Fiber Optics. Sixth ed. London; San Diego,
        CA;: Academic Press; 2019.
        https://doi.org/10.1016/B978-0-12-817042-7.00009-9

    """
    #---- Grids
    t_grid = np.arange(n) * dt
    dv = 1/(n*dt)
    rv_grid = np.arange(n//2 + 1) * dv

    #---- Raman Response
    raman_frac = 0 # initialize

    def h_r(fraction, tau_1, tau_2):
        """
        Contributions due to vibrational resonances.

        Parameters
        ----------
        fraction : float
        tau_1 : float
        tau_2 : float

        Returns
        -------
        ndarray of float

        """
        h_r = tau_1*(tau_1**-2 + tau_2**-2)*np.exp(-t_grid/tau_2)*np.sin(t_grid/tau_1)
        h_r *= fraction
        return h_r

    def h_b(fraction, tau_b):
        """
        Contributions due to boson peaks.

        Parameters
        ----------
        fraction : float
        tau_b : float

        Returns
        -------
        ndarray of float

        """
        h_b = (2*tau_b - t_grid)/tau_b**2 * np.exp(-t_grid/tau_b)
        h_b *= fraction
        return h_b

    # Resonant Vibrational Modes
    raman_t = np.zeros_like(t_grid)
    r_weights = np.asarray(r_weights, dtype=float)
    if len(r_weights.shape) == 1:
        r_weights = r_weights[np.newaxis]
    for weight in r_weights:
        raman_frac += weight[0]
        raman_t += h_r(*weight)

    # Boson Peaks
    if b_weights is not None:
        b_weights = np.asarray(b_weights, dtype=float)
        if len(b_weights.shape) == 1:
            b_weights = b_weights[np.newaxis]
        for weight in b_weights:
            raman_frac += weight[0]
            raman_t += h_b(*weight)

    # Normalize
    assert raman_frac <= 1, "The total fractional Raman contribution must be less than 1."
    if raman_frac != 0:
        raman_t *= raman_frac/(np.sum(raman_t)*dt)
    if not analytic:
        raman_t *= 3/2

    #---- Instantaneous Response
    delta_t = np.zeros_like(t_grid)
    delta_t[0] = 1/dt
    delta_t *= 1 - raman_frac # remove raman fraction

    #---- Total Nonlinear Response
    nonlinear_t = raman_t + delta_t
    nonlinear_v = fft.rfft(nonlinear_t, fsc=dt) # already in standard fft order
    return rv_grid, nonlinear_v



# %% Solitons

def nonlinear_length():
    pass #TODO

def soliton_number():
    # copy over linear_length, and test against in chi1
    # (chi1.linear_length = soliton_number**2 * nonlinear_length)
    pass #TODO:

def soltion_period():
    pass #TODO

def soliton_fission():
    pass #TODO: estimate of soliton fission point, TD compression, and FD expansion

def soliton_DW_dk(v_grid, beta, v0=None):
    pass #TODO

def soliton_SFS(gamma, ):
    pass #TODO: https://doi.org/10.1364/JOSAB.409240
