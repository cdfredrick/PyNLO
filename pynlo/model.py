# -*- coding: utf-8 -*-
"""
Models for simulating the propagation of light through optical modes.

"""

__all__ = ["Model", "NLSE", "UPE"]


# %% Imports

import collections
import warnings

import numpy as np
from scipy.constants import c, pi
from numba import njit

from pynlo.light import Pulse
from pynlo.medium import Mode
from pynlo.utility import fft


# %% Collections

SimulationResult = collections.namedtuple("SimulationResult", ["pulse", "z", "a_t", "a_v"])


# %% Routines

@njit(parallel=True)
def linear_operator(k, dz):
    """JIT-compiled exponential function."""
    return np.exp(-1j*dz * k)

@njit(parallel=True)
def l2_error(a_RK4, a_RK3):
    """JIT-compiled l2 norm error."""
    l2_norm = np.sum(np.real(a_RK4)**2 + np.imag(a_RK4)**2)**0.5
    rel_error = (a_RK4 - a_RK3)/l2_norm
    return np.sum(np.real(rel_error)**2 + np.imag(rel_error)**2)**0.5

@njit
def fdd(f, dx, idx):
    """JIT-compiled 2nd-order finite difference derivative."""
    #---- Right Bound
    if idx==f.size-1:
        # Derivative on the right-most edge
        return (3*f[idx] - 4*f[idx-1] + f[idx-2])/(2*dx)

    #---- Left Bound
    if idx==0:
        # Derivative on the left-most edge
        return (-3*f[idx] + 4*f[idx+1] - f[idx+2])/(2*dx)

    #---- Central
    return (f[idx+1] - f[idx-1])/(2*dx)

@njit
def prod(a, b):
    """JIT-compiled product. Useful for speeding up complex multiplications."""
    return a * b

@njit
def abs2(a):
    """JIT-compiled squared absolute value."""
    return a.real**2 + a.imag**2

@njit
def rk1(a, k, dz):
    """JIT-compiled 1st-order Runge-Kutta."""
    return a + dz*k

@njit
def rk3(b, k4, k5, dz):
    """JIT-compiled 3rd-order Runge-Kutta."""
    return b + dz/30.0 * (2.0*k4 + 3.0*k5)

@njit
def rk4(ai, ki1, ki2, ki3, k4, ip, dz):
    """JIT-compiled 4th-order Runge-Kutta."""
    bi = ai + dz/6.0 * (ki1 + 2.0*(ki2 + ki3))
    b = ip * bi # out of interaction picture
    rk4 = b + dz/6.0 * k4
    return rk4, b


# %% Single-Mode Models

class Model():
    """
    A model for simulating single-mode pulse propagation.

    This model only implements linear effects.

    Parameters
    ----------
    pulse : :py:class:`~pynlo.light.Pulse`
        The input pulse.
    mode : :py:class:`~pynlo.medium.Mode`
        The optical mode in which the pulse propagates.

    See Also
    --------
    NLSE : A model that implements the 3rd-order Kerr and Raman effects.
    UPE : A model that implements both 2nd- and 3rd-order nonlinearities.

    """
    def __init__(self, pulse, mode):
        #---- Pulse Parameters
        assert isinstance(pulse, Pulse)
        self.pulse = pulse.copy()

        #---- Mode Parameters
        assert isinstance(mode, Mode)
        self.mode = mode.copy()

        #---- Grids
        assert (pulse.v_grid == mode.v_grid).all(), (
            "The pulse and mode must be defined over the same frequency grid.")
        # Frequency Grids
        self.n_points = self.pulse.n
        self.v_grid = self.pulse.v_grid
        self.w_grid = 2*pi*self.v_grid
        self.dw = 2*pi*self.pulse.dv

        # Time Grid
        self.t_grid = self.pulse.t_grid
        self.dt = self.pulse.dt

        # Wavelength Grid
        self.l_grid = c/self.v_grid
        self.dv_dl = self.v_grid**2/c # power density conversion factor

        #---- Implementation Details
        # Define Operators
        self._linear_operator = self.linear_operator
        self._nonlinear_operator = self.nonlinear_operator

        # Initialize Mode Parameters
        self.update_linearity(force_update=True)
        self.update_nonlinearity(force_update=True)
        self.update_poling(force_update=True)

    def estimate_step_size(self, local_error=1e-6, dz=10e-6, n=10, a_v=None, z=0, db=False):
        """
        Estimate the step size that yields the target local error.

        This method uses the same adaptive step size algorithm as the main
        simulation. For a more accurate estimate, increase `n` to iteratively
        approach the optimal step size.

        Parameters
        ----------
        local_error : float, optional
            The relative local error. The default is 1e-6.
        dz : float, optional
            An initial guess for the optimal step size. The default is 10e-6.
        n : int, optional
            The number of times the algorithm iteratively executes. The default
            is to iterate 10.
        a_v : array_like of complex, optional
            The spectral amplitude of the pulse. The default is to use the
            spectral amplitude of the input pulse.
        z : float, optional
            The z position in the mode. The default is 0.
        db : bool, optional
            Debugging flag which turns on printing of intermediate results.

        Returns
        -------
        dz : float
            The new step size.

        """
        if a_v is None:
            a_v = self.pulse.a_v

        for _ in range(n):
            #---- Integrate by dz
            a_RK4_v, a_RK3_v, _ = self.step(a_v, z, dz)

            #---- Estimate the Relative Local Error
            est_error = l2_error(a_RK4_v, a_RK3_v)
            error_ratio = (est_error/local_error)**0.25
            if db: print("dz={:.3g},\t error={:.3g}".format(dz, est_error))

            #---- Update Step Size
            dz = dz/min(2, max(error_ratio, 0.5))
        return dz

    def simulate(self, z_grid, dz=None, local_error=1e-6, n_records=None, plot=None):
        """
        Simulate propagation of the input pulse through the optical mode.

        Parameters
        ----------
        z_grid : float or array_like of floats
            The total propagation distance over which to simulate, or the z
            positions at which to solve for the pulse spectrum. An adaptive
            step-size algorithm is used to propagate between these points. If
            only the end point is given the starting point is assumed to be the
            origin.
        dz : float, optional
            The initial step size. If ``None``, one will be estimated.
        local_error : float, optional
            The target relative local error for the adaptive step size
            algorithm. The default is 1e-6.
        n_records : None or int, optional
            The number of simulation points to return. If set, the z positions
            will be linearly spaced between the first and last points of
            `z_grid`. If ``None``, the default is to return all points as
            defined in `z_grid`. The record always includes the starting and
            ending points.
        plot : None or string, optional
            A flag that activates real-time visualization of the simulation.
            The options are ``"frq"``, ``"time"``, or ``"wvl"``, corresponding
            to the frequency, time, and wavelength domains. If set, the plot is
            updated each time the simulation reaches one of the z positions
            returned at the output. If ``None``, the default is to run the
            simulation without real-time plotting.

        Returns
        -------
        pulse : :py:class:`~pynlo.light.Pulse`
            The output pulse. This object can be used as the input to another
            simulation.
        z : ndarray of float
            The z positions at which the pulse spectrum (`a_v`) and complex
            envelope (`a_t`) have been returned.
        a_t : ndarray of complex
            The root-power complex envelope of the pulse at each z position.
        a_v : ndarray of complex
            The root-power spectrum of the pulse at each z position.
        """
        #---- Z Grid
        z_grid = np.asarray(z_grid, dtype=float)
        if z_grid.size==1:
            # Since only the end point was given, the start point is the origin
            z_grid = np.append(0.0, z_grid)

        if n_records is None:
            n_records = z_grid.size
            z_record = z_grid
        else:
            assert (n_records >= 2), "The output must include atleast 2 points."
            z_record = np.linspace(z_grid.min(), z_grid.max(), n_records)
            z_grid = np.unique(np.append(z_grid, z_record))
        z_record = {z:idx for idx, z in enumerate(z_record)}

        if self.mode.z_nonlinear.pol: # support subclasses with poling
            # always simulate up to the edge of a poled domain
            z_grid = np.unique(np.append(z_grid, list(self.mode.g2_inv)))

        #---- Setup
        z = z_grid[0]
        pulse_out = self.pulse.copy()

        # Frequency Domain
        a_v_record = np.empty((n_records,pulse_out.n), dtype=complex)
        a_v_record[0,:] = pulse_out.a_v

        # Time Domain
        a_t_record = np.empty((n_records,pulse_out.n), dtype=complex)
        a_t_record[0,:] = pulse_out.a_t

        # Step Size
        if dz is None:
            dz = self.estimate_step_size(pulse_out.a_v, z, local_error)
            print("Initial Step Size:\t{:.3g}m".format(dz))

        # Plotting
        if plot is not None:
            assert (plot in ["frq", "time", "wvl"]), (
                "Plot choice '{:}' is unrecognized").format(plot)
            # Setup Plots
            self._setup_plots(plot, pulse_out, z)

        #---- Propagate
        k5_v = None
        cont = False
        for z_stop in z_grid[1:]:
            # Step
            (pulse_out.a_v, z, dz, k5_v, cont) = self.propagate(
                pulse_out.a_v,
                z, z_stop, dz,
                local_error,
                k5_v=k5_v,
                cont=cont)

            # Record
            if z in z_record:
                idx = z_record[z]
                a_t_record[idx,:] = pulse_out.a_t
                a_v_record[idx,:] = pulse_out.a_v

                # Plot
                if plot is not None:
                    # Update Plots
                    self._update_plots(plot, pulse_out, z)

                    if z==z_grid[-1]:
                        # End animation with the last step
                        for artist in self._artists:
                            artist.set_animated(False)

            sim_res = SimulationResult(
                pulse=pulse_out, z=np.fromiter(z_record.keys(), dtype=float),
                a_t=a_t_record, a_v=a_v_record)
        return sim_res

    def propagate(self, a_v, z, z_stop, dz, local_error, k5_v=None, cont=False):
        """
        Propagate the given pulse spectrum from `z` to `z_stop` using an
        adaptive step size algorithm.

        The step size algorithm utilizes an embedded Runge–Kutta scheme with
        orders 3 and 4 (ERK4(3)-IP) [1]_.

        Parameters
        ----------
        a_v : ndarray of complex
            The root-power spectrum of the pulse.
        z : float
            The starting point.
        z_stop : float
            The stopping point.
        dz : float
            The initial step size.
        local_error : float
            The relative local error of the adaptive step size algorithm.
        k5_v : ndarray of complex, optional
            The action of the nonlinear operator on the solution from the
            preceding step. The default is ``None``.
        cont : bool, optional
            A flag that indicates the current step is continuous with the
            previous, i.e. that it begins where the other ended. The default is
            ``False``.

        Returns
        -------
        a_v : ndarray of complex
            The root-power spectrum of the pulse.
        z : float
            The z position in the mode.
        dz : float
            The step size.
        k5_v : ndarray of complex
            The nonlinear action of the 4th-order result.
        cont : bool
            A flag indicating that the next step may be continuous.

        References
        ----------
        .. [1] S. Balac and F. Mahé, "Embedded Runge–Kutta scheme for
            step-size control in the interaction picture method," Computer
            Physics Communications, Volume 184, Issue 4, 2013, Pages 1211-1219
            https://doi.org/10.1016/j.cpc.2012.12.020

        """
        while z < z_stop:
            z_next = z + dz
            if z_next >= z_stop:
                final_step = True
                z_next = z_stop
                dz_adaptive = dz # save value of last step size
                dz = z_next - z # force smaller step size to hit z_stop
            else:
                final_step = False

            #---- Integrate by dz
            a_RK4_v, a_RK3_v, k5_v_next = self.step(a_v, z, z_next, k5_v=k5_v, cont=cont)

            #---- Estimate Relative Local Error
            est_error = l2_error(a_RK4_v, a_RK3_v)
            error_ratio = (est_error/local_error)**0.25

            #---- Propagate Solution
            if error_ratio > 2:
                # Reject this step and calculate with a smaller dz
                dz = dz/2
                cont = False
            else:
                # Update parameters for the next loop
                z = z_next
                a_v = a_RK4_v
                k5_v = k5_v_next
                if (not final_step) or (error_ratio > 1):
                    dz = dz / max(error_ratio, 0.5)
                else:
                    dz = dz_adaptive # if final step, use adaptive step size
                cont = True

        return a_v, z, dz, k5_v, cont

    def step(self, a_v, z, z_next, k5_v=None, cont=False):
        """
        Advance the given pulse spectrum from `z` to `z_next`.

        This method is based on the 4th-order interaction picture Runge-Kutta
        scheme (ERK4(3)-IP) from Balac and Mahé [1]_.

        Parameters
        ----------
        a_v : ndarray of complex
            The root-power spectrum of the pulse.
        z : float
            The starting point.
        z_next : float
            The next point.
        k5_v : ndarray of complex, optional
            The action of the nonlinear operator on the solution from the
            preceding step. When included, it allows advancing the pulse with
            one less call to the nonlinear operator. The default is ``None``.
        cont : bool, optional
            A flag that indicates the current step is continuous with the
            previous, i.e. that it begins where the other ended. If ``False``,
            the linear and nonlinear parameters will be updated before the
            first calls to the linear and nonlinear operators. If ``True``,
            previously calculated values will be used. The default is
            ``False``.

        Returns
        -------
        a_RK4_v : ndarray of complex
            The 4th-order result.
        a_RK3_v : ndarray of complex
            The 3rd-order result.
        k5_v : ndarray of complex
            The nonlinear action of the 4th-order result.

        References
        ----------
        .. [1] S. Balac and F. Mahé, "Embedded Runge–Kutta scheme for
            step-size control in the interaction picture method," Computer
            Physics Communications, Volume 184, Issue 4, 2013, Pages 1211-1219
            https://doi.org/10.1016/j.cpc.2012.12.020

        """
        dz = z_next - z

        #---- k1
        if self.mode.z_mode and not cont:
            self.mode.z = z
            if self.mode.z_linear.any: self.update_linearity()
        ip_v = self._linear_operator(0.5*dz)

        if k5_v is None:
            if self.mode.z_nonlinear.any and not cont: self.update_nonlinearity()
            k5_v = self._nonlinear_operator(a_v)

        ai_v = ip_v * a_v # into interaction picture
        ki1_v = ip_v * k5_v # into interaction picture

        #---- k2 and k3
        if self.mode.z_mode:
            self.mode.z = 0.5*(z + z_next)
            if self.mode.z_nonlinear.any: self.update_nonlinearity()

        ai2_v = rk1(ai_v, ki1_v, 0.5*dz)
        ki2_v = self._nonlinear_operator(ai2_v)

        ai3_v = rk1(ai_v, ki2_v, 0.5*dz)
        ki3_v = self._nonlinear_operator(ai3_v)

        #---- k4
        if self.mode.z_mode:
            self.mode.z = z_next
            if self.mode.z_linear.any:
                self.update_linearity()
                ip_v = self._linear_operator(0.5*dz)
            if self.mode.z_nonlinear.any: self.update_nonlinearity()

        ai4_v = rk1(ai_v, ki3_v, dz)
        a4_v = ip_v * ai4_v # out of interaction picture
        k4_v = self._nonlinear_operator(a4_v)

        #---- RK4
        a_RK4_v, b_v = rk4(ai_v, ki1_v, ki2_v, ki3_v, k4_v, ip_v, dz)

        #---- k5
        k5_v = self._nonlinear_operator(a_RK4_v)

        #---- RK3
        a_RK3_v = rk3(b_v, k4_v, k5_v, dz)
        return a_RK4_v, a_RK3_v, k5_v

    #---- Operators
    def linear_operator(self, dz):
        """
        The action of the linear operator integrated over the given step size.

        Parameters
        ----------
        dz : float
            The step size.

        Returns
        -------
        ndarray of complex

        """
        # Linear Operator
        return linear_operator(self.kappa_cm, dz)

    def nonlinear_operator(self, a_v):
        """
        The action of the nonlinear operator on the given pulse spectrum.

        This model does not implement any nonlinear effects.

        Parameters
        ----------
        a_v : array_like of complex
            The root-power spectrum of the pulse.

        Returns
        -------
        ndarray of complex

        See Also
        --------
        NLSE.nonlinear_operator :
            Implementation of the 3rd-order Kerr and Raman effects.
        UPE.nonlinear_operator :
            Implementation of both 2nd- and 3rd-order nonlinearities

        """
        return 0.0j

    #---- Z-Dependency
    def update_linearity(self, force_update=False):
        """
        Update all z-dependent linear parameters.

        Parameters
        ----------
        force_update : bool, optional
            Force an update of all linear parameters. The default will only
            update those that are z dependent.

        """
        #---- Gain self.mode.z_linear
        if self.mode.z_linear.alpha or force_update:
            self.alpha = self.mode.alpha

        #---- Phase
        if self.mode.z_linear.beta or force_update:
            self.beta = self.mode.beta
            beta1_v0 = fdd(self.beta, self.dw, self.pulse.v0_idx)
            # Beta in comoving frame
            self.beta_cm = self.beta - beta1_v0*self.w_grid

        #---- Propagation Constant
        if self.alpha is not None:
            self.kappa_cm = (self.beta_cm - self.alpha**2/(8*self.beta)) + 0.5j*self.alpha
        else:
            self.kappa_cm = self.beta_cm

    def update_nonlinearity(self, force_update=False):
        """
        Update all z-dependent nonlinear parameters.

        Parameters
        ----------
        force_update : bool, optional
            Force an update of all nonlinear parameters. The default will only
            update those that are z dependent.

        """
        if self.mode.g2 is not None:
            warnings.warn("2nd-order nonlinearity is not implemented in this model", stacklevel=2)
        if self.mode.g3 is not None:
            warnings.warn("3rd-order nonlinearity is not implemented in this model", stacklevel=2)

    def update_poling(self, force_update=False):
        """
        Update the poled sign of the 2nd-order nonlinearity.

        Parameters
        ----------
        force_update : bool, optional
            Force an update of the poling. The default will only update if the
            z position is at a domain inversion.

        """
        if self.mode.z_nonlinear.pol:
            warnings.warn("Poling is not implemented in this model", stacklevel=2)

    #---- Plotting
    def _setup_plots(self, plot, pulse_out, z):
        """
        Initialize a figure for real-time visualization of a simulation.

        Parameters
        ----------
        plot : string
            The type of plot. The options are "frq", "time", or "wvl" and
            correspond to the frequency, time, and wavelength domains.
        pulse_out : :py:class:`~pynlo.light.Pulse`
            The pulse to be plotted.
        z : float
            The z position in the mode.

        """
        # Import if needed
        try:
            self._plt
        except AttributeError:
            import matplotlib.pyplot as plt
            self._plt = plt

        #---- Figure and Axes
        self._rt_fig = self._plt.figure("Real-Time Simulation", clear=True)
        self._ax_0 = self._plt.subplot2grid((2,1), (0,0), fig=self._rt_fig)
        self._ax_1 = self._plt.subplot2grid((2,1), (1,0), sharex=self._ax_0, fig=self._rt_fig)
        self._rt_fig.show()

        #---- Time Domain
        if plot=="time":
            # Lines
            self._ln_pwr, = self._ax_0.semilogy(
                1e12*self.t_grid,
                pulse_out.p_t,
                '.',
                markersize=1,
                animated=True)
            self._ln_phs, = self._ax_1.plot(
                1e12*self.t_grid,
                1e-12*pulse_out.vg_t,
                '.',
                markersize=1,
                animated=True)

            # Labels
            self._ax_0.set_title("Instantaneous Power")
            self._ax_0.set_ylabel("J / s")
            self._ax_0.set_xlabel("Delay (ps)")
            self._ax_1.set_ylabel("Frequency (THz)")
            self._ax_1.set_xlabel("Delay (ps)")

            # Y Boundaries
            self._ax_0.set_ylim(
                top=max(self._ln_pwr.get_ydata())*1e1,
                bottom=max(self._ln_pwr.get_ydata())*1e-9)
            excess = 0.05*(self.v_grid.max()-self.v_grid.min())
            self._ax_1.set_ylim(
                top=1e-12*(self.v_grid.max() + excess),
                bottom=1e-12*(self.v_grid.min() - excess))

        #---- Frequency Domain
        if plot=="frq":
            # Lines
            self._ln_pwr, = self._ax_0.semilogy(
                1e-12*self.v_grid,
                pulse_out.p_v,
                '.',
                markersize=1,
                animated=True)
            self._ln_phs, = self._ax_1.plot(
                1e-12*self.v_grid,
                1e12*pulse_out.tg_v,
                '.',
                markersize=1,
                animated=True)

            # Labels
            self._ax_0.set_title("Power Spectrum")
            self._ax_0.set_ylabel("J / Hz")
            self._ax_0.set_xlabel("Frequency (THz)")
            self._ax_1.set_ylabel("Delay (ps)")
            self._ax_1.set_xlabel("Frequency (THz)")

            # Y Boundaries
            self._ax_0.set_ylim(
                top=max(self._ln_pwr.get_ydata())*1e1,
                bottom=max(self._ln_pwr.get_ydata())*1e-9)
            excess = 0.05*(self.t_grid.max()-self.t_grid.min())
            self._ax_1.set_ylim(
                top=1e12*(self.t_grid.max() + excess),
                bottom=1e12*(self.t_grid.min() - excess))

        #---- Wavelength Domain
        if plot=="wvl":
            # Lines
            self._ln_pwr, = self._ax_0.semilogy(
                1e9*self.l_grid,
                self.dv_dl * pulse_out.p_v,
                '.',
                markersize=1,
                animated=True)
            self._ln_phs, = self._ax_1.plot(
                1e9*self.l_grid,
                1e12*pulse_out.tg_v,
                '.',
                markersize=1,
                animated=True)

            # Labels
            self._ax_0.set_title("Power Spectrum")
            self._ax_0.set_ylabel("J / m")
            self._ax_0.set_xlabel("Wavelength (nm)")
            self._ax_1.set_ylabel("Delay (ps)")
            self._ax_1.set_xlabel("Wavelength (nm)")

            # Y Boundaries
            self._ax_0.set_ylim(
                top=max(self._ln_pwr.get_ydata())*1e1,
                bottom=max(self._ln_pwr.get_ydata())*1e-9)
            excess = 0.05*(self.t_grid.max()-self.t_grid.min())
            self._ax_1.set_ylim(
                top=1e12*(self.t_grid.max() + excess),
                bottom=1e12*(self.t_grid.min() - excess))

        #---- Z Label
        #TODO: change to plt.barh, progress bar
        self._z_label = self._ax_1.legend(
            [],[],
            title='z = {:.6g} m'.format(z),
            loc=9,
            labelspacing=0,
            framealpha=1,
            shadow=False)
        self._z_label.set_animated(True)

        #---- Layout
        self._rt_fig.tight_layout()
        self._rt_fig.canvas.draw()

        #---- Blit
        self._artists = (self._ln_pwr, self._ln_phs, self._z_label)

        self._rt_fig_bkg_0 = self._rt_fig.canvas.copy_from_bbox(self._ax_0.bbox)
        self._rt_fig_bkg_1 = self._rt_fig.canvas.copy_from_bbox(self._ax_1.bbox)

    def _update_plots(self, plot, pulse_out, z):
        """
        Update the figure used for real-time visualization of a simulation.

        Parameters
        ----------
        plot : string
            The type of plot. The options are "frq", "time", or "wvl" and
            correspond to the frequency, time, and wavelength domains.
        pulse_out : :py:class:`~pynlo.light.Pulse`
            The pulse to be plotted.
        z : float
            The z position in the mode.

        """
        #---- Restore Background
        self._rt_fig.canvas.restore_region(self._rt_fig_bkg_0)
        self._rt_fig.canvas.restore_region(self._rt_fig_bkg_1)

        #---- Update Data
        if plot=="time":
            self._ln_pwr.set_data(
                1e12*self.t_grid,
                pulse_out.p_t)
            self._ln_phs.set_data(
                1e12*self.t_grid,
                1e-12*pulse_out.vg_t)

        if plot=="frq":
            self._ln_pwr.set_data(
                1e-12*self.v_grid,
                pulse_out.p_v)
            self._ln_phs.set_data(
                1e-12*self.v_grid,
                1e12*pulse_out.tg_v)

        if plot=="wvl":
            self._ln_pwr.set_data(
                1e9*self.l_grid,
                self.dv_dl * pulse_out.p_v)
            self._ln_phs.set_data(
                1e9*self.l_grid,
                1e12*pulse_out.tg_v)

        #---- Update Z Label
        self._z_label.set_title('z = {:.6g} m'.format(z))

        #---- Blit
        for artist in self._artists:
            artist.axes.draw_artist(artist)

        self._rt_fig.canvas.blit(self._ax_0.bbox)
        self._rt_fig.canvas.blit(self._ax_1.bbox)
        self._rt_fig.canvas.flush_events()


class NLSE(Model):
    """
    A model for simulating single-mode pulse propagation with the nonlinear
    Schrödinger equation (NLSE).

    This model only implements the 3rd-order Kerr and Raman effects. In
    general, it does not support 3rd-order sum- and difference-frequency
    generation.

    Parameters
    ----------
    pulse : :py:class:`~pynlo.light.Pulse`
        The input pulse.
    mode : :py:class:`~pynlo.medium.Mode`
        The optical mode in which the pulse propagates.

    See Also
    --------
    Model :
        Documentation of :py:meth:`~pynlo.model.Model.simulate` and other
        inherited methods.

    """
    def __init__(self, pulse, mode):
        super().__init__(pulse, mode)

        if mode.rv_grid is not None:
            assert (mode.rv_grid.size == pulse.rtf_grids(alias=0).v_grid.size), (
                "The pulse and mode must be defined over the same frequency grid")

        #---- Implementation Details
        self._linear_operator = self._linear_operator_fft_order
        self._nonlinear_operator = self._nonlinear_operator_fft_order

    def propagate(self, a_v, z, z_stop, dz, local_error, k5_v=None, cont=False):
        #---- Standard FFT Order
        a_v = fft.ifftshift(a_v)

        #---- Propagate
        a_v, z, dz, k5_v, cont = super().propagate(
            a_v, z, z_stop, dz, local_error, k5_v=k5_v, cont=cont)

        #---- Monotonic Order
        a_v = fft.fftshift(a_v)
        return a_v, z, dz, k5_v, cont

    #---- Operators
    def _linear_operator_fft_order(self, dz):
        """
        The action of the linear operator integrated over the given step size,
        arranged in standard fft order.

        Parameters
        ----------
        dz : float
            The step size.

        Returns
        -------
        ndarray of complex

        """
        return linear_operator(self._kappa_cm, dz)

    def _nonlinear_operator_fft_order(self, a_v):
        """
        The action of the nonlinear operator on the given pulse spectrum,
        arranged in standard fft order.

        This model implements the 3rd-order Kerr and Raman effects.

        Parameters
        ----------
        a_v : array_like of complex
            The root-power spectrum of the pulse, in standard fft order.

        Returns
        -------
        ndarray of complex

        """
        #---- Setup
        a_t = fft.ifft(a_v, fsc=self.dt)
        a2_t = abs2(a_t)

        #---- Raman
        if self.r3 is not None:
            a2_rv = fft.rfft(a2_t, fsc=self.dt)
            a2r_rv = prod(self.r3, a2_rv)
            a2_t = fft.irfft(a2r_rv, fsc=self.dt, n=self.n_points)

        #---- Kerr
        a3_t = prod(a_t, a2_t)
        a3_v = fft.fft(a3_t, fsc=self.dt)
        return prod(self._1j_gamma, a3_v) # minus sign included in _1j_gamma

    def nonlinear_operator(self, a_v):
        """
        The action of the nonlinear operator on the given pulse spectrum.

        This model only implements the 3rd-order Kerr and Raman effects. In
        general, it does not support 3rd-order sum- and difference-frequency
        generation.

        Parameters
        ----------
        a_v : array_like of complex
            The root-power spectrum of the pulse.

        Returns
        -------
        ndarray of complex

        """
        #---- Standard FFT Order
        _a_v = fft.ifftshift(a_v)

        #---- Nonlinear Operator
        _nl_a_v = self._nonlinear_operator_fft_order(_a_v)

        #---- Monotonic Order
        return fft.fftshift(_nl_a_v)

    #---- Z-Dependency
    def update_linearity(self, force_update=False):
        super().update_linearity(force_update=force_update)
        self._kappa_cm = fft.ifftshift(self.kappa_cm)

    def update_nonlinearity(self, force_update=False):
        if self.mode.g2 is not None:
            warnings.warn("2nd-order nonlinearity is not implemented in this model", stacklevel=2)

        #---- Gamma
        if self.mode.z_nonlinear.g3 or force_update:
            self.gamma = self.mode.gamma
            self._1j_gamma = fft.ifftshift(-1j * self.gamma)

        #---- Raman
        if self.mode.z_nonlinear.r3 or force_update:
            self.r3 = self.mode.r3


class UPE(Model):
    """
    A model for simulating single-mode pulse propagation with the
    unidirectional propagation equation (UPE).

    This model simultaneously implements 2nd- and 3rd-order nonlinearities.

    Parameters
    ----------
    pulse : :py:class:`~pynlo.light.Pulse`
        The input pulse.
    mode : :py:class:`~pynlo.medium.Mode`
        The optical mode in which the pulse propagates.

    See Also
    --------
    Model :
        Documentation of :py:meth:`~pynlo.model.Model.simulate` and other
        inherited methods.

    Notes
    -----
    Multiplication of functions in the time domain, an operation intrinsic to
    nonlinear interactions, is equivalent to convolution in the frequency
    domain. The support of a convolution is the sum of the support of its
    parts. In general, 2nd- and 3rd-order processes in the time domain need 2x
    and 3x the number of points in the frequency domain to avoid aliasing.

    By default, `Pulse` objects only initialize the minimum number of points
    necessary to represent the real-valued time-domain pulse (i.e., 1x). While
    this minimizes the numerical complexity of individual nonlinear operations,
    aliasing introduces systematic error. More points can be generated for a
    specific `Pulse` object during initialization, or through its
    :py:meth:`~pynlo.utility.TFGrid.rtf_grids` method, by setting the `alias`
    parameter greater than 1. Anti-aliasing is not always necessary as phase
    matching can suppress aliased interactions, but it is best practice to
    verify that behavior on a case-by-case basis.

    """
    def __init__(self, pulse, mode):
        super().__init__(pulse, mode)

        if mode.rv_grid is not None:
            assert (pulse.rv_grid.size == mode.rv_grid.size), (
                "The pulse and mode must be defined over the same frequency grid")

        #---- Implementation Details
        # Frequency Grid
        self._1j_w_grid = 1j * self.w_grid

        # Real-Valued Time Domain Grid
        self.rn_points = self.pulse.rn
        self.rdt = self.pulse.rdt

        # Carrier-Resolved Slice
        self.rn_slice = self.pulse.rn_slice

        # Initialize Arrays
        self._0_v = np.zeros_like(self.pulse.v_grid, dtype=complex)
        self._nl_v = np.zeros_like(self.pulse.v_grid, dtype=complex)
        self._a_rv = np.zeros_like(self.pulse.rv_grid, dtype=complex)
        self._0_rt = np.zeros_like(self.pulse.rt_grid, dtype=float)
        self._a2_rt = np.zeros_like(self.pulse.rt_grid, dtype=float)
        self._a3_rt = np.zeros_like(self.pulse.rt_grid, dtype=float)

    def propagate(self, a_v, z, z_stop, dz, local_error, k5_v=None, cont=False):
        #---- Update Poling
        if self.mode.z_nonlinear.pol:
            if not cont: self.mode.z = z
            if self.mode.z in self.mode.g2_inv:
                self.update_poling()
                k5_v = None # reset k5_v, 2nd-order nonlinearity changed sign

        #---- Propagate
        return super().propagate(a_v, z, z_stop, dz, local_error, k5_v=k5_v, cont=cont)

    #---- Operators
    def nonlinear_operator(self, a_v):
        """
        The action of the nonlinear operator on the given pulse spectrum.

        This model implements the Kerr and Raman effects, as well as 2nd- and
        3rd-order sum- and difference-frequency generation.

        Parameters
        ----------
        a_v : array_like of complex
            The root-power spectrum of the pulse.

        Returns
        -------
        ndarray of complex

        """
        #---- Setup
        self._nl_v[...] = self._0_v # zero
        self._a_rv[self.rn_slice] = a_v
        a_rt = fft.irfft(self._a_rv, fsc=self.rdt * 2**0.5, n=self.rn_points) # 1/2**0.5 for analytic to real
        a2_rt = a_rt * a_rt

        #---- 2nd-Order Nonlinearity
        if self.g2 is not None:
            a2_rv = fft.rfft(a2_rt, fsc=self.rdt * 2**0.5) # 2**0.5 for real to analytic
            if self.g2_pol: # poled
                self._nl_v += prod(self.g2, a2_rv[self.rn_slice])
            else: # not poled
                self._nl_v -= prod(self.g2, a2_rv[self.rn_slice])

        #---- 3rd-Order Nonlinearity
        if self.g3 is not None:
            # Raman
            if self.r3 is not None:
                a2_rv = (fft.rfft(a2_rt, fsc=self.rdt) if self.g2 is None
                           else a2_rv * 2**-0.5) # 1/2**0.5 for analytic to real
                a2r_rv = prod(self.r3, a2_rv)
                a2_rt = fft.irfft(a2r_rv, fsc=self.rdt, n=self.rn_points)
            # Kerr
            a3_rt = a_rt * a2_rt
            a3_rv = fft.rfft(a3_rt, fsc=self.rdt * 2**0.5) # 2**0.5 for real to analytic
            self._nl_v -= prod(self.g3, a3_rv[self.rn_slice])

        #---- Nonlinear Response
        return prod(self._1j_w_grid, self._nl_v) # minus sign included in _nl_v

    def nonlinear_operator_separable(self, a_v):
        """
        The action of the nonlinear operator on the given pulse spectrum. This
        operator is active when the nonlinear parameters have been input in
        separable form.

        This model implements the Kerr and Raman effects, as well as 2nd- and
        3rd-order sum- and difference-frequency generation.

        Parameters
        ----------
        a_v : array_like of complex
            The root-power spectrum of the pulse.

        Returns
        -------
        ndarray of complex

        Notes
        -----
        Chromatic variations of the nonlinearity, present in the full 2nd- and
        3rd-rank form of the nonlinear parameters, can be incorporated into the
        propagation model by decomposing the tensors into separable form:

        .. math:: g^{(2)}[\\nu_1, \\nu_2] &= h^{(2)}\\!\\left[\\nu\\right]
                  \\sum_n \\eta_n^{(2)}[\\nu_1] \\, \\eta_n^{(2)}[\\nu_2] \\\\
                  g^{(3)}[\\nu_1, \\nu_2, \\nu_3] &= h^{(3)}[\\nu] \\sum_n
                  \\eta_n^{(3)}[\\nu_1] \\, \\eta_n^{(3)}[\\nu_2] \\,
                  \\eta_n^{(3)}[\\nu_3]

        where the outer and inner terms (:math:`h` and :math:`\\eta`) depend on
        the output and input frequencies respectively.

        To use this operator the nonlinear parameters must be input as `(m, n)`
        arrays, where `m` is the number of inner and outer terms and `n` is the
        number of points. The first index is interpreted as the outer term
        while all others are interpreted as inner terms. A simplified
        decomposition may be generated using the
        :py:func:`~pynlo.utility.chi2.g2_split` and
        :py:func:`~pynlo.utility.chi3.g3_split` functions.

        """
        #---- Setup
        self._nl_v[...] = self._0_v # zero

        #---- 2nd-Order Nonlinearity
        if self.g2 is not None:
            self._a2_rt[...] = self._0_rt # zero
            for g2_internal in self.g2[1:]:
                self._a_rv[self.rn_slice] = prod(a_v, g2_internal)
                a_rt = fft.irfft(self._a_rv, fsc=self.rdt * 2**0.5, n=self.rn_points) # 1/2**0.5 for analytic to real
                self._a2_rt += a_rt * a_rt
            a2_rv = fft.rfft(self._a2_rt, fsc=self.rdt * 2**0.5) # 2**0.5 for real to analytic
            if self.g2_pol: # poled
                self._nl_v += prod(self.g2[0], a2_rv[self.rn_slice])
            else: # not poled
                self._nl_v -= prod(self.g2[0], a2_rv[self.rn_slice])

        #---- 3rd-Order Nonlinearity
        if self.g3 is not None:
            self._a3_rt[...] = self._0_rt # zero
            for g3_internal in self.g3[1:]:
                self._a_rv[self.rn_slice] = prod(a_v, g3_internal)
                a_rt = fft.irfft(self._a_rv, fsc=self.rdt * 2**0.5, n=self.rn_points) # 1/2**0.5 for analytic to real
                a2_rt = a_rt * a_rt
                if self.r3 is not None:
                    a2_rv = fft.rfft(a2_rt, fsc=self.rdt)
                    a2r_rv = prod(self.r3, a2_rv)
                    a2_rt = fft.irfft(a2r_rv, fsc=self.rdt, n=self.rn_points)
                self._a3_rt += a_rt * a2_rt
            a3_rv = fft.rfft(self._a3_rt, fsc=self.rdt * 2**0.5) # 2**0.5 for real to analytic
            self._nl_v -= prod(self.g3[0], a3_rv[self.rn_slice])

        #---- Nonlinear Response
        return prod(self._1j_w_grid, self._nl_v) # minus sign included in _nl_v

    #---- Z-Dependency
    def update_nonlinearity(self, force_update=False):
        #---- 2nd Order
        if self.mode.z_nonlinear.g2 or force_update:
            self.g2 = self.mode.g2
            self._g2_dim = len(self.g2.shape) if self.g2 is not None else 0

        #---- 3rd Order
        if self.mode.z_nonlinear.g3 or force_update:
            self.g3 = self.mode.g3
            self._g3_dim = len(self.g3.shape) if self.g3 is not None else 0
        if self.mode.z_nonlinear.r3 or force_update:
            self.r3 = self.mode.r3

        #---- Select Nonlinear Operator
        if self.mode.z_nonlinear.g2 or self.mode.z_nonlinear.g3 or force_update:
            # Separable 2nd- and 3rd-order terms
            if (self._g2_dim > 1):
                if self.g3 is not None and (self._g3_dim < 2):
                    self.g3 = [self.g3, complex(1.0)]
                    self._g3_dim = 2
                self._nonlinear_operator = self.nonlinear_operator_separable
            elif (self._g3_dim > 1):
                if self.g2 is not None:
                    self.g2 = [self.g2, complex(1.0)]
                    self._g2_dim = 2
                self._nonlinear_operator = self.nonlinear_operator_separable
            # Standard nonlinear terms
            else:
                self._nonlinear_operator = self.nonlinear_operator

    def update_poling(self, force_update=False):
        if self.mode.z_nonlinear.pol or force_update:
            try:
                self.g2_pol = self.mode.g2_inv[self.mode.z]
            except (KeyError, TypeError):
                if force_update: self.g2_pol = self.mode.g2_pol


# %% Multi-Mode Models

# class MultiModel():
#     def __init__(pulses, modes, couplings):
#         pass # reserved for multimode simulations
