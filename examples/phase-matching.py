# -*- coding: utf-8 -*-
"""
These examples demonstrate the concepts of phase matching, quasi-phase
matching, and the effects of pump depeletion through second harmonic
generation (SHG) of a continuous wave (CW) laser. The simulations use the
unidirectional propagation equation (UPE) and are roughly based on the
parameters of thin-film lithium niobate.

"""

# %% Imports
import numpy as np
from scipy.constants import pi, c
from matplotlib import pyplot as plt

import pynlo
from pynlo import utility as ut


# %% CW Properties
"""
We initialize a CW laser using one of the built-in spectral shapes of the
`Pulse` class. The fundamental is set at 200 THz, which gives a second harmonic
at 400 THz (~1.5 um and ~750 nm respectively). Since the input is a single
frequency, only a few points are nescessary for an accurate simulation.

"""
n_points = 4
v_min = 100e12  # 100 THz
v_max = 400e12  # 500 THz

v0 = 200e12     # 200 THz
p_avg = 100e-3  # 100 mW

pulse = pynlo.light.Pulse.CW(n_points, v_min, v_max, v0, p_avg, alias=2)

idx_fn = pulse.v0_idx
idx_sh = np.abs(pulse.v_grid - pulse.v_grid[idx_fn]*2).argmin()


# %% Mode Properties
"""
The modal properties are rougly based on those found in thin-film lithium
niobate waveguides. The refractive indices used are a low-order fit to the
indices given by Zelmon for bulk lithium niobate in the 1.5 um region.

Dispersion of the refractive index causes the nonlinear transfer of power to
oscilate with propagation disstance. This is characterized by the coherence
length (`L_C`), which is a function of the phase mismatch or the difference
between the phase coeficient (`beta = n*w/c`) of the input and output
frequencies of the nonlinear interaction. For second harmonic generation, the
coherence length is as follows:

    L_C = pi/dk = pi/(beta(2*w0) - 2*beta(w0))

References
----------
David E. Zelmon, David L. Small, and Dieter Jundt, "Infrared corrected
 Sellmeier coefficients for congruently grown lithium niobate and 5 mol. %
 magnesium oxide–doped lithium niobate," J. Opt. Soc. Am. B 14, 3319-3322
 (1997)
 https://doi.org/10.1364/JOSAB.14.003319

"""
a_eff = 1e-6 * 1e-6 # 1 um**2

#--- Phase Coefficient
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


# %% Phase Matching
"""
This example demonstrates the initial evolution of a phase-mismatched, a
phase-matched, and a quasi-phase-matched SHG process. When phase mismatched,
the direction of power transfer alternates every coherence length `L_C`. This
severely limits the accumulative nonlinear effect compared to the phase-matched
case, which grows quadratically with propagation distance. By changing the sign
of the nonlinear interaction (i.e alternating the direction of the crystal
axis) at the end of each coherence length, the nonlinear interaction can be
quasi-phase matched and the power can be made to grow monotonically. While on
aggregate a quasi-phase-matched process also grows quadratically, compared to
the phase-matched case the effective nonlinearity is reduced by a factor of
`sinc(m*pi/2)`, where `m` is the number of coherence lengths between the poled
domains.

The simulation extends over 3 coherence lengths. The blue trace in the plot
shows the oscillating phase-mismatched case, the orange the 1st-order
quasi-phase-matched case, and the green the ideal phase-matched case. The
phase-matched case is calculated using the `shg_conversion_efficiency` function
of the `chi2` utility module. For more details on phase matching and the SHG
process, see chapter 2 of Boyd.

References
----------
Robert W. Boyd, Nonlinear Optics (Fourth Edition), Academic Press, 2020
 https://doi.org/10.1016/C2015-0-05510-1

"""
#---- Poling
length = L_C*3 # ~26 um
z_invs, domains, poled = ut.chi2.domain_inversions(length, beta[idx_sh] - 2*beta[idx_fn])

mode_qpm = pynlo.media.Mode(pulse.v_grid, beta, g2=g2, g2_inv=z_invs) # Quasi-phase matched
mode_pmm = pynlo.media.Mode(pulse.v_grid, beta, g2=g2) # Phase mismatched


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
distance. This is due to depletion of power from the pump. For SHG, as long as
the phase matching condition is upheld, the nonlinear transfer of power
continues asymptotically at large propagation distances.

The simulation extends over about 2000 coherence lengths, at which point
nearly all of the power from the fundamental (black trace) has been transfered
to the second harmonic (blue trace). The breakdown of the quadratic, or
undepleted-pump approximation can be seen about a third of the way through the
simulation. Where the approximation predicts 100% power transfer the simulation
yields a roughly 60:40 ratio.

"""
#---- Poling
length = L_C*1750 # ~15 mm
z_invs, domains, poled = ut.chi2.domain_inversions(length, 2*beta[idx_fn] - beta[idx_sh])
mode = pynlo.media.Mode(pulse.v_grid, beta, g2=g2, g2_inv=z_invs)


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
    pulse.v0, pulse.p_t.mean(), n[idx_fn], n[idx_sh], a_eff, d_eff, res.z, qpm_order=1)
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
