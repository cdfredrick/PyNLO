# -*- coding: utf-8 -*-
"""
Through the process of second-harmonic generation (SHG) with a continuous
wave (CW) laser, these examples demonstrate the concepts of coherence length,
quasi-phase matching, and pump depletion. These simulations use the
unidirectional propagation equation (UPE) and are roughly based on the
parameters of a thin-film lithium niobate waveguide.

"""

# %% Imports
import numpy as np
from scipy.constants import pi, c
from matplotlib import pyplot as plt

import pynlo
from pynlo import utility as ut


# %% CW Properties
"""
The CW laser is initialized using one of the built-in spectral shapes of the
`Pulse` class. The fundamental is set at 200 THz, which puts the second
harmonic at 400 THz (~1.5 um and ~750 nm respectively). The minimum and maximum
of the frequency grid are chosen to place the fundamental and second harmonic
exactly on the grid. Since the input is a single frequency, only a few grid
points are necessary for an accurate simulation.

"""
n_points = 4
v_min = 100e12  # 100 THz
v_max = 400e12  # 400 THz

v0 = 200e12     # 200 THz
p_avg = 100e-3  # 100 mW

pulse = pynlo.light.Pulse.CW(n_points, v_min, v_max, v0, p_avg, alias=2)

# Index of the fundamental
idx_fn = pulse.v0_idx
# Index of the second harmonic
idx_sh = np.abs(pulse.v_grid - 2*pulse.v0).argmin()


# %% Mode Properties
"""
The waveguide properties are roughly based on those possible in thin-film
lithium niobate. The refractive indices are derived from a low-order fit to the
indices given by Zelmon for bulk lithium niobate in the 1.5 um region.

Dispersion of the refractive index causes the nonlinear transfer of power to
oscillate with propagation distance. This oscillation is characterized by a
coherence length `L_C`, which is a function of the phase mismatch or difference
between the phase coefficients (`beta = n*w/c`) of the nonlinear interaction's
input and output frequencies. In general, the concept of coherence length
applies to all types of nonlinear interactions. For second-harmonic generation
the coherence length is calculated as follows:

    L_C = pi/dk = pi/(beta(2*w0) - 2*beta(w0))

In the following examples the coherence length is approximately 9 um long,
i.e., the direction of the nonlinear power transfer switches direction every
9 um in the absence of phase matching.

References
----------
David E. Zelmon, David L. Small, and Dieter Jundt, "Infrared corrected
 Sellmeier coefficients for congruently grown lithium niobate and 5 mol. %
 magnesium oxide–doped lithium niobate," J. Opt. Soc. Am. B 14, 3319-3322
 (1997)
 https://doi.org/10.1364/JOSAB.14.003319

"""
a_eff = 1e-6 * 1e-6 # 1 um**2

#---- Phase Coefficient
n_n = [0]*4
n_n[0] = 2.14
n_n[1] = .275 * 1e-15 # fs
n_n[2] = -2.33 * 1e-15**2 # fs**2
n_n[3] = 25.6 * 1e-15**3 # fs**3
n = ut.taylor_series(pulse.v0, n_n)(pulse.v_grid)
beta = n * 2*pi*pulse.v_grid/c

#---- 2nd-order nonlinearity
d_eff = 27e-12 # 27 pm / V
chi2_eff = 2 * d_eff
g2 = ut.chi2.g2_shg(pulse.v0, pulse.v_grid, n, a_eff, chi2_eff)

#---- Length Scale
delta_beta = beta[idx_sh] - 2*beta[idx_fn]
L_C = pi/delta_beta # coherence length
print("Coherence Length \t= {:.3g} m".format(L_C))


# %% Phase Matching
"""
This example demonstrates the initial evolution of a phase-mismatched, a
phase-matched, and a quasi-phase-matched SHG process. When phase mismatched,
the direction of power transfer alternates every coherence length `L_C`. This
severely limits the accumulative nonlinear effect as compared to the
phase-matched case, which grows quadratically with propagation distance. By
changing the sign of the nonlinear interaction (i.e. by alternating the
direction of the crystal axis) at the end of each coherence length, the
nonlinear interaction can be quasi-phase matched (QPM) and the SHG power can be
made to grow monotonically. The length of each poled domain `L_P` must be an
odd-integer multiple of the coherence length:

    L_P = m * L_C = m * pi/dk

where `m` is the quasi-phase-matching order. On aggregate, a
quasi-phase-matched process also grows quadratically, but compared to the
phase-matched case the effective nonlinearity is reduced by a factor of
`sinc(m*pi/2)`. While less efficient, higher-order quasi-phase matching
broadens the number of interactions that can be simultaneously phase matched
and allows for phase matching of interactions that would otherwise require
physically infeasible domains sizes.

The simulation extends over 3 coherence lengths. The blue trace in the plot
shows the oscillating phase-mismatched case, the orange shows the 1st-order
quasi-phase-matched case, and the green shows the ideal phase-matched case. The
ideal case is calculated using the `shg_conversion_efficiency` function of the
`utility.chi2` submodule. Underlaid the orange trace is the equivalent
quadratic approximation for the 1st-order quasi-phase-matched case. For more
details on phase matching and the SHG process, see chapter 2 of Boyd.

References
----------
Robert W. Boyd, Nonlinear Optics (Fourth Edition), Academic Press, 2020
 https://doi.org/10.1016/C2015-0-05510-1

"""
#---- Poling
length = L_C*3 # ~25 um
z_invs, domains, poled = ut.chi2.domain_inversions(length, delta_beta)

mode_qpm = pynlo.medium.Mode(pulse.v_grid, beta, g2=g2, g2_inv=z_invs) # Quasi-phase matched
mode_pmm = pynlo.medium.Mode(pulse.v_grid, beta, g2=g2) # Phase mismatched


#---- Model
model_qpm = pynlo.model.UPE(pulse, mode_qpm) # Quasi-phase matched
model_pmm = pynlo.model.UPE(pulse, mode_pmm) # Phase mismatched

# Estimate step size
local_error = 1e-9
dz = model_pmm.estimate_step_size(local_error=local_error)


#---- Simulate
res_qpm = model_qpm.simulate( # Quasi-phase matched
    length, dz=dz, local_error=local_error, n_records=100)

res_pmm = model_pmm.simulate( # Phase mismatched
    length, dz=dz, local_error=local_error, n_records=100)


#---- Plot Results
fig = plt.figure("Phase Matching", clear=True)

y_scale = ut.chi2.shg_conversion_efficiency(
    pulse.v0, pulse.p_t.mean(), n[idx_fn], n[idx_sh], a_eff, d_eff, L_C,
    qpm_order=1)

# Phase mismatched
plt.plot(
    res_pmm.z/L_C,
    np.abs(res_pmm.a_v[:,idx_sh])**2 * pulse.dv/pulse.t_window/p_avg/y_scale,
    label="Phase Mismatched")

# Quasi-phase matched
plt.plot(
    res_qpm.z/L_C,
    np.abs(res_qpm.a_v[:,idx_sh])**2 * pulse.dv/pulse.t_window/p_avg/y_scale,
    label="Quasi-Phase Matched")
plt.ylim(plt.ylim())

# Phase-matched conversion efficiency
pm_con_eff = ut.chi2.shg_conversion_efficiency(
    pulse.v0, pulse.p_t.mean(), n[idx_fn], n[idx_sh], a_eff, d_eff, res_qpm.z,
    qpm_order=0)
plt.plot(res_qpm.z/L_C, pm_con_eff/y_scale, label="Phase Matched", zorder=-1)

# Quasi-phase-matched conversion efficiency
qpm_con_eff = ut.chi2.shg_conversion_efficiency(
    pulse.v0, pulse.p_t.mean(), n[idx_fn], n[idx_sh], a_eff, d_eff, res_qpm.z,
    qpm_order=1)
plt.plot(res_qpm.z/L_C, qpm_con_eff/y_scale, c="lightgrey", zorder=-1)

plt.legend()
plt.ylabel("Power (arb. unit)")
plt.xlabel("Propagation Distance ($L_C$)")
plt.margins(x=0)
plt.grid(alpha=0.2)
fig.tight_layout()
fig.show()


# %% Phase-Matched Pump Depletion
"""
The quadratic scaling of a phase-matched or quasi-phase-matched interaction
breaks down if the interaction is maintained over a long enough propagation
distance. Due to conservation of energy, the power at the fundamental decreases
as the power at the second harmonic increases. The rate of second-harmonic
generation slows as the pump power is depleted. As long as the phase-matching
condition is upheld, SHG will continue asymptotically towards a conversion
efficiency of 100%.

The simulation extends over thousands of coherence lengths, at which point
nearly all of the power from the fundamental (black trace) has been transferred
to the second harmonic (blue trace). The breakdown of the quadratic, or
undepleted-pump approximation can be seen about one third of the way through
the simulation. Where the approximation predicts 100% power transfer, the
simulation only yields a ~60:40 ratio.

"""
#---- Poling
length = L_C*1750 # ~15 mm
z_invs, domains, poled = ut.chi2.domain_inversions(length, delta_beta)
mode = pynlo.medium.Mode(pulse.v_grid, beta, g2=g2, g2_inv=z_invs)


#---- Model
model = pynlo.model.UPE(pulse, mode)

# Estimate step size
local_error = 1e-9
dz = model.estimate_step_size(local_error=local_error)


#---- Simulate
res = model.simulate(
    length, dz=dz, local_error=local_error, n_records=100)


#---- Plot Results
fig = plt.figure("Phase-Matched Pump Depletion", clear=True)

# Second harmonic
plt.plot(
    res.z/length,
    100*np.abs(res.a_v[:,idx_sh])**2 * pulse.dv/pulse.t_window/p_avg,
    label="Second Harmonic")
plt.ylim(plt.ylim())

# Undepleted-pump approximation
con_eff = ut.chi2.shg_conversion_efficiency(
    pulse.v0, pulse.p_t.mean(), n[idx_fn], n[idx_sh], a_eff, d_eff, res.z,
    qpm_order=1)
plt.plot(
    res.z/length,
    100*con_eff,
    label="SHG Quadratic Approx.", c="C2", zorder=-1)

# Fundamental
plt.plot(
    res.z/length,
    100*np.abs(res.a_v[:,idx_fn])**2 * pulse.dv/pulse.t_window/p_avg,
    label="Fundamental", c="k", zorder=-2)

plt.legend()
plt.ylabel("Power (arb. unit)")
plt.xlabel("Propagation Distance (arb. unit)")
plt.margins(x=0)
plt.grid(alpha=0.2)
fig.tight_layout()
fig.show()
