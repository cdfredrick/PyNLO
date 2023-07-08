# -*- coding: utf-8 -*-
"""
Conversion functions and other calculators relevant to the 2nd-order nonlinear
susceptibility.

"""

__all__ = ["g2_split", "g2_shg", "domain_inversions"]


# %% Imports

import numpy as np
from scipy.constants import pi, c, epsilon_0 as e0


# %% Converters

#---- 2nd-Order Nonlinear Susceptibilities chi2 and d
def d_to_chi2(d):
    return d * 2

def chi2_to_d(chi2):
    return chi2 / 2

#---- Poling Period and Wavenumber Mismatch
def dk_to_period(dk):
    return 2*pi/dk

def period_to_dk(period):
    return 2*pi/period


# %% Nonlinearity

def g2_split(n_eff, a_eff, chi2_eff):
    """
    The 2nd-order nonlinear parameter decomposed according to unit analysis.

    Parameters
    ----------
    n_eff : array_like of float
        The refractive indices.
    a_eff : array_like of float
        The effective areas.
    chi2_eff : array_like
        The effective 2nd-order susceptibilities.

    Returns
    -------
    ndarray (2, n)
        The unit decomposition of the 2nd-order nonlinear parameter. The first
        index contains the output factors while the second index contains the
        input factors.

    """
    return np.array([1/2 * ((e0*a_eff)/(c*n_eff))**0.5 * chi2_eff,
                     1/(e0*c*n_eff*a_eff)**0.5])

def g2_path(n_eff, a_eff, chi2_eff, paths):
    """
    The 2nd-order nonlinear parameter weighted for the given nonlinear
    pathways.

    Parameters
    ----------
    n_eff : array_like of float
        The effective refractive indices.
    a_eff : array_like of float
        The effective areas.
    chi2_eff : array_like
        The effective 2nd-order susceptibilities.
    paths : list
        Pairs of indices for each output frequency that correspond to the
        input frequencies of the input paths. If no valid path exists for a
        particular output frequency its input indices should be given as
        ``[None, None]``.

    Returns
    -------
    g2 : ndarray

    See Also
    --------
    dominant_paths : Determine the dominant paths based on phase mismatch.

    """
    g2s = g2_split(n_eff, a_eff, chi2_eff)
    assert g2s.size != len(paths), "The number of paths must equal the number of frequencies."

    g2_p = []
    for idx, path in enumerate(paths):
        if None in path:
            g2_p.append(0.0)
        else:
            g2 = np.mean([g2s[0, idx] * g2s[1, cpl[0]]*g2s[1, cpl[0]] for cpl in path])
            g2_p.append(g2)
    return np.arary(g2_p)

def g2_shg(v0, v_grid, n_eff, a_eff, chi2_eff):
    """
    The 2nd-order nonlinear parameter weighted for second harmonic generation
    centered around the given input frequency.

    Parameters
    ----------
    v0 : float
        The target fundamental frequency to be doubled.
    v_grid : array_like of float
        The frequency grid.
    n_eff : array_like of float
        The effective refractive indices.
    a_eff : array_like of float
        The effective areas.
    chi2_eff : array_like
        The effective 2nd-order susceptibilities.

    Returns
    -------
    g2 : ndarray

    """
    g2_out, g2_in = g2_split(n_eff, a_eff, chi2_eff)

    v1_idx = np.argmin(np.abs(v_grid - v0))
    v2_idx = np.argmin(np.abs(v_grid - 2*v0))
    vc_idx = (v1_idx + v2_idx)//2 # crossover point

    g2_in_fh = np.interp(2*v_grid, v_grid, g2_in) # input to fundamental harmonic
    g2_in_sh = np.interp(0.5*v_grid, v_grid, g2_in) # input to second harmonic

    # Fundamental (DFG)
    g2_fh = g2_out * g2_in_fh * g2_in

    # Second Harmonic (SFG)
    g2_sh = g2_out * g2_in_sh**2

    # Crossover
    g2 = np.zeros_like(g2_out)
    g2[:vc_idx] = g2_fh[:vc_idx]
    g2[vc_idx:] = g2_sh[vc_idx:]
    return g2

def g2_sfg(v0, v_grid, n_eff, a_eff, chi2_eff):
    """
    The 2nd-order nonlinear parameter weighted for sum frequency generation
    driven by the given input frequency.

    Parameters
    ----------
    v0 : float
        The target pump frequency.
    v_grid : array_like of float
        The frequency grid.
    n_eff : array_like of float
        The effective refractive indices.
    a_eff : array_like of float
        The effective areas.
    chi2_eff : array_like
        The effective 2nd-order susceptibilities.

    Returns
    -------
    g2 : ndarray

    """
    g2_out, g2_in = g2_split(n_eff, a_eff, chi2_eff)

    vc_idx = np.argmin(np.abs(v_grid - (v_grid.min() + v0))) # crossover point

    g2_in_v0 = np.interp(v0, v_grid, g2_in)
    g2_in_sf = np.interp(v_grid - v0, v_grid, g2_in) # input to sum (out = in + v0)
    g2_in_df = np.interp(v_grid + v0, v_grid, g2_in) # input to diff (out = in - v0)

    # Fundamental (DFG)
    g2_sf = g2_out * g2_in_sf * g2_in_v0

    # Second Harmonic (SFG)
    g2_df = g2_out * g2_in_df * g2_in_v0

    # Crossover
    g2 = np.zeros_like(g2_out)
    g2[:vc_idx] = g2_df[:vc_idx]
    g2[vc_idx:] = g2_sf[vc_idx:]
    return g2


# %% Phase Matching

def domain_inversions(z, dk, intp_order=1):
    """
    Find the location of the domain inversion boundaries that quasi-phase match
    (QPM) the given wavenumber mismatch.

    Parameters
    ----------
    z : float or array_like of float
        The propagation distance.
    dk : float or array_like of float
        The wavenumber mismatch, or ``2*pi/polling period``.
    intp_order : int, optional
        The interpolation order used to calculate the inversion points.

    Returns
    -------
    z_invs : ndarray of float
        The location of all domain inversion boundaries.
    domains : ndarray of float
        The length of each domain.
    poled : ndarray of int
        The poling status of each domain. A value of 1 indicates an inverted
        domain. A value of 0 indicates an unpoled region.

    Notes
    -----
    For continuous QPM profiles, the domain boundaries can be calculated by
    inverting the integral equation that gives the accumulated phase mismatch:

    .. math::
        2 \\pi N[z] &= \\int_{z_0}^z \\Delta k[z^\\prime] dz^\\prime
        = \\int_{z_0}^z \\frac{2 \\pi}{\\Lambda[z^\\prime]} dz^\\prime \\\\
        \\text{z}_{inv}[n] &= N^{-1}[n/2]

    where :math:`\Delta k` is the wavenumber mismatch compensated by poling
    period :math:`\Lambda`, :math:`N` is the accumulated number of phase
    cycles, and :math:`n` is an integer.

    """
    from scipy.interpolate import InterpolatedUnivariateSpline

    #---- Process Input
    z = np.asarray(z)
    dk = np.abs(np.asarray(dk))
    if dk.size > 1:
        assert dk.size == z.size, "If `dk` is given as an array it must have the same number of points as `z`."
    if z.size==1:
        z = np.linspace(0, z, intp_order+1)
    if dk.size == 1:
        dk = np.ones_like(z) * dk

    #---- Inversion Points
    n_cycles = InterpolatedUnivariateSpline(z, dk/(2*pi), k=intp_order).antiderivative()
    z_n = InterpolatedUnivariateSpline(2*n_cycles(z), z, k=intp_order)

    n_invs = int(2*n_cycles(z[-1])) # round down
    n_grid = np.arange(n_invs + 1)
    z_invs = z_n(n_grid) # all inversion points

    #---- Domains
    poled = n_grid % 2
    domains = np.diff(z_invs)
    if np.isclose(z_invs.max(), z.max(), rtol=0, atol=10e-9):
        # Ignore final inversion if at the end
        poled = poled[:-1]
        z_invs = z_invs[:-1]
    else:
        # Include final domain
        domains = np.append(domains, z.max() - z_invs.max())
    return z_invs[1:], domains, poled

def dominant_paths(v_grid, beta, beta_qpm=None, full=False): #TODO: add extent (for imshow)
    """
    For each output frequency, find the input frequencies that are coupled
    with the least phase mismatch.

    This function checks both sum frequency generation (SFG) pathways and
    difference frequency generation (DFG) pathways.

    Parameters
    ----------
    v_grid : array_like of float
        The frequency grid.
    beta : array_like of float
        The wavenumber of each input frequency.
    beta_qpm : float, optional
        The effective wavenumber of an applied quasi-phase matching (QPM)
        structure. If ``None``, the dominant path is calculated without poling.
    full : bool, optional
        A flag determining the nature of the return value. When ``False`` (the
        default) just the path indices are returned, when ``True`` the
        calculated wavenumber mismatch arrays are also returned.

    Returns
    -------
    paths : list
        Pairs of indices for each output frequency that correspond to the
        input frequencies of the dominant input paths. If no valid path exists
        for a particular output frequency its input indices are given as
        ``[None, None]``.
    (dk, dk_sfg, dk_dfg) : tuple
        These values are only returned if ``full=True`` \n
        dk : list
            The wavenumber mismatch for each path in `paths`.
        v_sfg : ndarray of float
            The frequencies that correspond to all SFG combinations.
        dk_sfg : ndarray of float
            The wavenumber mismatch for all SFG combinations. The mismatch of
            invalid paths are given as ``np.nan``.
        v_dfg : ndarray of float
            The frequencies that correspond to all DFG combinations.
        dk_dfg : ndarray of float
            The wavenumber mismatch for all DFG combinations. The mismatch of
            invalid paths are given as NaN.

    """
    #---- Setup
    v_grid = np.asarray(v_grid, dtype=float)
    n = v_grid.size
    v_idx = np.arange(n)
    v_idx2 = np.dstack(np.indices((n,n)))
    beta = np.asarray(beta, dtype=float)

    #---- Sum Frequency
    v_sfg = np.add.outer(v_grid, v_grid)

    # Indexing
    v_idx_sfg = np.interp(v_sfg, v_grid, v_idx, left=np.nan, right=np.nan).round()
    sfg_idxs = np.arange(np.nanmin(v_idx_sfg), np.nanmax(v_idx_sfg)+1, dtype=int)
    sfg_diag_offset = (n-1) - np.arange(sfg_idxs.size)

    # Wavenumber Mismatch
    beta_sfg_12 = np.add.outer(beta, beta)
    beta_sfg_3 = np.interp(v_sfg, v_grid, beta, left=np.nan, right=np.nan)
    if beta_qpm is None:
        dk_sfg = np.abs(beta_sfg_12 - beta_sfg_3)
    else:
        dk_sfg = np.abs(np.abs(beta_sfg_12 - beta_sfg_3) - beta_qpm)

    #---- Difference Frequency
    v_dfg = np.subtract.outer(v_grid, v_grid)

    # Indexing
    v_idx_dfg = np.interp(v_dfg, v_grid, v_idx, left=np.nan, right=np.nan).round()
    dfg_idxs = np.arange(np.nanmin(v_idx_dfg), np.nanmax(v_idx_dfg)+1, dtype=int)
    dfg_diag_offset = -(n-1) + np.arange(dfg_idxs.size)[::-1]

    # Wavenumber Mismatch
    beta_dfg_31 = np.subtract.outer(beta, beta)
    beta_dfg_2 = np.interp(v_dfg, v_grid, beta, left=np.nan, right=np.nan)
    if beta_qpm is None:
        dk_dfg = np.abs(beta_dfg_31 - beta_dfg_2)
    else:
        dk_dfg = np.abs(np.abs(beta_dfg_31 - beta_dfg_2) - beta_qpm)

    #---- Paths of Minimum Phase Mismatch
    paths = []
    dk = []
    for idx in v_idx:
        valid_sfg = (idx in sfg_idxs)
        valid_dfg = (idx in dfg_idxs)

        # SFG
        if valid_sfg:
            diag_offset = sfg_diag_offset[idx==sfg_idxs][0]
            sfg_diag = np.fliplr(dk_sfg).diagonal(diag_offset)
            min_dk_sfg = np.nanmin(sfg_diag)
            min_dk_sfg_idx = np.nonzero(sfg_diag==min_dk_sfg)
            sfg_path = np.fliplr(v_idx2).diagonal(diag_offset).T[min_dk_sfg_idx]

        # DFG
        if valid_dfg:
            diag_offset = dfg_diag_offset[idx==dfg_idxs][0]
            dfg_diag = dk_dfg.diagonal(diag_offset)
            min_dk_dfg = np.nanmin(dfg_diag)
            min_dk_dfg_idx = np.nonzero(dfg_diag==min_dk_dfg)
            dfg_path = v_idx2.diagonal(diag_offset).T[min_dk_dfg_idx]

        if valid_sfg and valid_dfg:
            if min_dk_sfg < min_dk_dfg:
                # SFG smaller
                paths.append(sfg_path.tolist())
            elif min_dk_sfg > min_dk_dfg:
                # DFG smaller
                paths.append(dfg_path.tolist())
            else:
                # Equal
                paths.append(sfg_path.tolist() + dfg_path.tolist())
            dk.append(min_dk_sfg) if min_dk_sfg < min_dk_dfg else dk.append(min_dk_dfg)
        elif valid_sfg:
            # Only SFG
            paths.append(sfg_path.tolist())
            dk.append(min_dk_sfg)
        elif valid_dfg:
            # Only DFG
            paths.append(dfg_path.tolist())
            dk.append(min_dk_dfg)
        else:
            # No path
            paths.append([[None, None]])
            dk.append(None)

    if full:
        return paths, (dk, v_sfg, dk_sfg, v_dfg, dk_dfg)
    else:
        return paths


# %% Calculator Functions

def effective_chi3():
    pass #TODO: effective 3rd-order from cascaded 2nd

def shg_conversion_efficiency(v0, p0, n_v, n_2v, a_eff, d_eff, z, qpm_order=None):
    """
    Conversion efficiency of second-harmonic generation in the undepleted-pump
    approximation.

    Parameters
    ----------
    v0 : float
        The fundamental frequency.
    p0 : float
        The average power.
    n_v : float
        The refractive index at the fundamental frequency.
    n_2v : float
        The refractive index at the second-harmonic frequency.
    a_eff : float
        The effective area.
    d_eff : float
        The effective second order nonlinearity.
    z : float
        The propagation distance.
    qpm_order : int
        The order of the quasi-phase-matching. Only odd orders can be
        quasi-phase matched.

    References
    ----------
    Robert W. Boyd, Nonlinear Optics (Fourth Edition), Academic Press, 2020
     https://doi.org/10.1016/C2015-0-05510-1

    """
    w0 = 2*pi*v0
    if qpm_order is not None:
        d_eff = d_eff * np.abs(np.sinc(qpm_order/2))
    return (2*d_eff**2 * w0**2 * p0)/(n_v**2 * n_2v * a_eff * e0 * c**3) * z**2
