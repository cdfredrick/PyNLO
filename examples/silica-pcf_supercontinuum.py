# -*- coding: utf-8 -*-
"""
This example demonstrates supercontinuum generation due to soliton effects in
silica-based photonic crystal fiber. The simulation uses the nonlinear
Schrödinger equation (NLSE) and is based on the parameters given in part V-A of
Dudley et al. (2006).

References
----------
Dudley JM, Genty G, Coen S. Supercontinuum generation in photonic crystal
 fiber. Reviews of modern physics. 2006;78(4):1135-84.
 https://doi.org/10.1103/RevModPhys.78.1135

"""


# %% Imports
import numpy as np
from scipy.constants import pi, c
from matplotlib import pyplot as plt

import pynlo
from pynlo import utility as ut


# %% Pulse Properties
"""
We start by initializing a hyperbolic secant pulse using one of the built-in
pulse shapes of the `Pulse` class. The first few parameters constrain the
frequency grid, the number of points and the frequency range, while the last
three set the initial pulse properties, its center frequency, pulse energy, and
pulse width.

"""
n_points = 2**12
v_min = c/1400e-9   # c / 1400 nm
v_max = c/475e-9    # c / 475 nm

v0 = c/835e-9       # c / 835 nm
e_p = 285e-12       # 285 pJ
t_fwhm = 50e-15     # 50 fs

pulse = pynlo.light.Pulse.Sech(n_points, v_min, v_max, v0, e_p, t_fwhm)


# %% Mode Properties
"""
We next need to define both the linear and nonlinear properties of the
waveguide. In this example, we are only given the waveguide properties at a
single frequency so we must extrapolate to the rest of the frequency grid. For
the beta parameter this is accomplished using a Taylor series, but for the
nonlinearity we use the `gamma_to_g3` conversion function from the
`utility.chi3` submodule. This function calculates the generalized 3rd-order
nonlinear parameter (required by the PyNLO propagation models) from the gamma
parameter and the optical shock time scale. If available, the nonlinear
parameter can also be generated from the refractive index, effective area, and
nonlinear susceptibility, see `utility.chi3` for more details. The Raman effect
is implemented using the Raman response function given in section 2.3.3 of
Agrawal.

References
----------
Agrawal GP. Nonlinear Fiber Optics. Sixth ed. London; San Diego,
 CA;: Academic Press; 2019.
 https://doi.org/10.1016/B978-0-12-817042-7.00009-9

"""
length = 15e-2 # 15 cm

#---- Phase Coefficient
beta_n = 11*[0]
beta_n[2] = -11.830 * 1e-12**2/1e3       # -11.830 ps**2 / km
beta_n[3] = 8.1038e-2 * 1e-12**3/1e3     # 8.1038e-2 ps**3 / km
beta_n[4] = -9.5205e-5 * 1e-12**4/1e3    # -9.5205e-5 ps**4 / km
beta_n[5] = 2.0737e-7 * 1e-12**5/1e3     # 2.0737e-7 ps**5 / km
beta_n[6] = -5.3943e-10 * 1e-12**6/1e3   # -5.3943e-10 ps**6 / km
beta_n[7] = 1.3486e-12 * 1e-12**7/1e3    # 1.3486e-12 ps**7 / km
beta_n[8] = -2.5495e-15 * 1e-12**8/1e3   # -2.5495e-15 ps**8 / km
beta_n[9] = 3.0524e-18 * 1e-12**9/1e3    # 3.0524e-18 ps**9 / km
beta_n[10] = -1.7140e-21 * 1e-12**10/1e3 # -1.7140e-21 ps**10 / km

beta = ut.taylor_series(2*pi*pulse.v0, beta_n)(2*pi*pulse.v_grid)

#---- 3rd-Order Nonlinearity
gamma = 0.11        # 0.11 / W * m
t_shock = 0.56e-15  # 0.56 fs
g3 = ut.chi3.gamma_to_g3(pulse.v_grid, gamma, t_shock)

# Raman effect
r_weights = [0.245*(1-0.21), 12.2e-15, 32e-15] # resonant contribution
b_weights = [0.245*0.21, 96e-15] # boson contribution
rv_grid, raman = ut.chi3.raman(pulse.n, pulse.dt, r_weights, b_weights)

mode = pynlo.medium.Mode(pulse.v_grid, beta, g3=g3, rv_grid=rv_grid, r3=raman)


# %% Model
"""
The NLSE model is initialized with the pulse and mode objects defined above. At
this stage we also estimate the optimal initial step size given a target local
error.

"""
model = pynlo.model.NLSE(pulse, mode)

#---- Estimate step size
local_error = 1e-6
dz = model.estimate_step_size(local_error=local_error)


# %% Simulate
"""
The model's `simulate` method runs the simulation. We input the total
propagation length, the initial step size, local error, and the number of
simulation steps we wish to record. We receive the output pulse and the
propagations distance, pulse spectrum, and complex envelope at each record
point. To view real-time simulation results (updated whenever the simulation
reaches a record point), set the `plot` keyword to "frq", "wvl", or "time".

"""
pulse_out, z, a_t, a_v = model.simulate(
    length, dz=dz, local_error=local_error, n_records=100, plot=None)


# %% Plot Results
"""
For comparison with Dudley, we plot the evolution in the time and wavelength
domains. For accurate representation of the density, plotting over wavelength
requires converting the power from a per Hz basis to a per m basis. This is
accomplished by multiplying the frequency domain power spectral density by the
ratio of the frequency and wavelength differentials. The power spectral density
is then converted to decibel (dB) scale to increase the visible dynamic range.

"""
fig = plt.figure("Simulation Results", clear=True)
ax0 = plt.subplot2grid((3,2), (0, 0), rowspan=1)
ax1 = plt.subplot2grid((3,2), (0, 1), rowspan=1)
ax2 = plt.subplot2grid((3,2), (1, 0), rowspan=2, sharex=ax0)
ax3 = plt.subplot2grid((3,2), (1, 1), rowspan=2, sharex=ax1)

p_l_dB = 10*np.log10(np.abs(a_v)**2 * model.dv_dl)
p_l_dB -= p_l_dB.max()
ax0.plot(1e9*c/pulse.v_grid, p_l_dB[0], color="b")
ax0.plot(1e9*c/pulse.v_grid, p_l_dB[-1], color="g")
ax2.pcolormesh(1e9*c/pulse.v_grid, 1e3*z, p_l_dB,
               vmin=-40.0, vmax=0, shading="auto")
ax0.set_ylim(bottom=-50, top=10)
ax2.set_xlabel('Wavelength (nm)')

p_t_dB = 10*np.log10(np.abs(a_t)**2)
p_t_dB -= p_t_dB.max()
ax1.plot(1e12*pulse.t_grid, p_t_dB[0], color="b")
ax1.plot(1e12*pulse.t_grid, p_t_dB[-1], color="g")
ax3.pcolormesh(1e12*pulse.t_grid, 1e3*z, p_t_dB,
               vmin=-40.0, vmax=0, shading="auto")
ax1.set_ylim(bottom=-50, top=10)
ax3.set_xlabel('Time (ps)')

ax0.set_ylabel('Power (dB)')
ax2.set_ylabel('Propagation Distance (mm)')
fig.tight_layout()
fig.show()
