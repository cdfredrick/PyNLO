from scipy.special import erf
import numpy as np
import scipy.constants as sc
import pynlo


def LN_alpha(v_grid):
    """
    This is the absorption coefficient of ppln. When doing DFG and such, you
    deal with pretty low powers in the MIR, so it actually becomes problematic
    if you let the simulation generate arbitrarily long wavelengths. I ended
    uphaving to include the absorption coefficient, which is defined here.

    Args:
        v_grid (1D array):
            frequency grid

    Returns:
        1D array:
            absorption coefficient
    """
    w_grid = v_grid * 2 * np.pi
    w_grid /= 1e12

    # 300/(2*np.pi) THz is the cutoff frequency
    # the 10 sqrt(2) just softens the cutoff
    return 1e6 * (1 + erf(-(w_grid - 300.0) / (10 * np.sqrt(2))))


def gaussian_beam_area(z_to_focus, v0, a_eff):
    """
    A gaussian beam can be accounted for by scaling the chi2 and chi3 parameter

    Args:
        z_to_focus (float):
            the distance from the focus
        v0 (float):
            center frequency
        a_eff (float):
            effective area (pi * w_0^2)

    Returns:
        float:
            the ratio of areas^1/2:
                1 / sqrt[ (pi * w^2) / (pi * w_0^2) ]

    Notes:
        returns 1 / (current area / original area)
    """
    w_0 = np.sqrt(a_eff / np.pi)  # beam radius
    wl = sc.c / v0
    z_R = np.pi * w_0**2 / wl  # rayleigh length
    w = w_0 * np.sqrt(1 + (z_to_focus / z_R) ** 2)
    return np.pi * w**2


def guoy_phase_shift(z_to_focus, v0, a_eff):
    w_0 = np.sqrt(a_eff / np.pi)  # beam radius
    wl = sc.c / v0
    z_R = np.pi * w_0**2 / wl  # rayleigh length
    return -np.arctan(z_to_focus / z_R)


def n_MgLN_G(v, T=24.5, axis="e"):
    """
    Range of Validity:
        - 500 nm to 4000 nm
        - 20 C to 200 C
        - 48.5 mol % Li
        - 5 mol % Mg

    Gayer, O., Sacks, Z., Galun, E. et al. Temperature and wavelength
    dependent refractive index equations for MgO-doped congruent and
    stoichiometric LiNbO3 . Appl. Phys. B 91, 343–348 (2008).

    https://doi.org/10.1007/s00340-008-2998-2

    """
    if axis == "e":
        a1 = 5.756  # plasmons in the far UV
        a2 = 0.0983  # weight of UV pole
        a3 = 0.2020  # pole in UV
        a4 = 189.32  # weight of IR pole
        a5 = 12.52  # pole in IR
        a6 = 1.32e-2  # phonon absorption in IR
        b1 = 2.860e-6
        b2 = 4.700e-8
        b3 = 6.113e-8
        b4 = 1.516e-4
    elif axis == "o":
        a1 = 5.653  # plasmons in the far UV
        a2 = 0.1185  # weight of UV pole
        a3 = 0.2091  # pole in UV
        a4 = 89.61  # weight of IR pole
        a5 = 10.85  # pole in IR
        a6 = 1.97e-2  # phonon absorption in IR
        b1 = 7.941e-7
        b2 = 3.134e-8
        b3 = -4.641e-9
        b4 = -2.188e-6

    else:
        raise ValueError("axis needs to be o or e")

    wvl = sc.c / v * 1e6  # um
    f = (T - 24.5) * (T + 570.82)
    n2 = (
        (a1 + b1 * f)
        + (a2 + b2 * f) / (wvl**2 - (a3 + b3 * f) ** 2)
        + (a4 + b4 * f) / (wvl**2 - a5**2)
        - a6 * wvl**2
    )
    return n2**0.5


def n_cLN(v_grid, T=24.5, axis="e"):
    """
    Refractive index of congruent lithium niobate.

    References
    ----------
    Dieter H. Jundt, "Temperature-dependent Sellmeier equation for the index of
     refraction, ne, in congruent lithium niobate," Opt. Lett. 22, 1553-1555
     (1997). https://doi.org/10.1364/OL.22.001553

    """

    assert axis == "e"

    a1 = 5.35583
    a2 = 0.100473
    a3 = 0.20692
    a4 = 100.0
    a5 = 11.34927
    a6 = 1.5334e-2
    b1 = 4.629e-7
    b2 = 3.862e-8
    b3 = -0.89e-8
    b4 = 2.657e-5

    wvl = sc.c / v_grid * 1e6  # um
    f = (T - 24.5) * (T + 570.82)
    n2 = (
        a1
        + b1 * f
        + (a2 + b2 * f) / (wvl**2 - (a3 + b3 * f) ** 2)
        + (a4 + b4 * f) / (wvl**2 - a5**2)
        - a6 * wvl**2
    )
    return n2**0.5


def Ds_to_beta_n(D, dD_dwl, wl_0):
    """
    convert D, D' to beta2, beta3

    Args:
        D (float):
            D (s / m^2)
        dD_dwl (float):
            D' (s / m^3)
        wl_0 (float):
            center wavelength

    Returns:
        tuple: beta2 (s^2/m), beta3 (s^3/m)

    Notes:
        You can derive the terms below starting from
            D = (-2 pi c / wl^2) beta_2
    """
    # D / (- 2 pi c / wl^2)
    beta_2 = D / (-2 * np.pi * sc.c / wl_0**2)

    # (D' + 2D/wl) / (2 pi c / wl^2)^2
    beta_3 = (dD_dwl + 2 * D / wl_0) * (2 * np.pi * sc.c / wl_0**2) ** -2
    return beta_2, beta_3


def beta_n_to_beta(v0, beta_n):
    """
    get beta(v_grid) from beta_n's

    Args:
        v0 (float):
            center frequency
        beta_n (list of floats):
            list of beta derivatives starting from beta_2

    Returns:
        callable:
            beta(v_grid)

    Notes:
        realize that in literature, and if retrieved from D's that beta
        derivatives are given for beta(2 * np.pi * v_grid), this is taken care
        of here
    """
    beta_omega = pynlo.utility.taylor_series(v0 * 2 * np.pi, [0, 0, *beta_n])

    beta = lambda v_grid: beta_omega(v_grid * 2 * np.pi)
    return beta


class MgLN:
    """
    This class is useful for MgLN waveguides or bulk crystals. For bulk
    crystals, the is_gaussian_beam kwarg should be set to True when calling
    generate_model
    """

    def __init__(self, T=24.5, axis="e"):
        self._T = T
        self._axis = axis

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, val):
        """
        set the temperature in Celsius

        Args:
            val (float):
                the temperature in Celsius
        """
        assert isinstance(val, float)
        self._T = val

    @property
    def axis(self):
        return self._axis

    @axis.setter
    def axis(self, val):
        """
        set the axis to be either extraordinary or ordinary

        Args:
            val (string):
                either "e" or "o"
        """
        assert np.any([val == "e", val == "o"]), 'the axis must be either "e" or "o"'
        self._axis = val

    @property
    def n(self):
        """
        Returns:
            callable:
                a function that calculates the index of refraction as a
                function of frequency
        """
        return lambda v: n_MgLN_G(v, T=self.T, axis=self.axis)

    @property
    def beta(self):
        """
        Returns:
            callable:
                a function that calculates the angular wavenumber as a function
                of frequency
        """
        # n * omega * c
        return lambda v: self.n(v) * 2 * np.pi * v / sc.c

    @property
    def d_eff(self):
        """
        d_eff of magnesium doped lithium niobate

        Returns:
            float: d_eff
        """
        return 27e-12  # 27 pm / V
        # return 0  # turn off chi 2

    @property
    def chi2_eff(self):
        """
        effective chi2 of magnesium doped lithium niobate

        Returns:
            float: 2 * d_eff
        """
        return 2 * self.d_eff

    @property
    def chi3_eff(self):
        """
        3rd order nonlinearity of magnesium doped lithium niobate

        Returns:
            float
        """
        return 5200e-24  # 5200 pm ** 2 / V ** 2
        # return 0.0  # turn off chi 3

    def g2_shg(self, v_grid, v0, a_eff):
        """
        The 2nd order nonlinear parameter weighted for second harmonic
        generation driven by the given input frequency.

        Args:
            v_grid (1D array):
                frequency grid
            v0 (float):
                center frequency
            a_eff (float):
                effective area

        Returns:
            1D array
        """
        return pynlo.utility.chi2.g2_shg(
            v0, v_grid, self.n(v_grid), a_eff, self.chi2_eff
        )

    def g3(self, v_grid, a_eff):
        """
        The 3rd order nonlinear parameter weighted for self-phase modulation.

        Args:
            v_grid (1D array):
                frequency grid
            a_eff (float):
                effective area

        Returns:
            1D array
        """
        n_eff = self.n(v_grid)
        return pynlo.utility.chi3.g3_spm(n_eff, a_eff, self.chi3_eff)

    def generate_model(
        self,
        pulse,
        a_eff,
        length,
        g2_inv=None,
        beta=None,
        is_gaussian_beam=False,
    ):
        """
        generate PyNLO model instance

        Args:
            pulse (object):
                PyNLO pulse instance
            a_eff (float):
                effective area
            length (float):
                crystal or waveguide length
            g2_inv (1D array, optional):
                locations of all inversion points inside the crystal, default
                is no inversion points
            beta (1D array, optional):
                the beta curve calculated over the pulse's frequency grid. The
                default is None, in which case beta is calculated from Mg-LN's
                material dispersion.
            is_gaussian_beam (bool, optional):
                whether the mode is a gaussian beam, default is False

        Returns:
            model (object):
                a PyNLO model instance

        Notes:
            If "is_gaussian_beam" is set to True, then the chi2 parameter is
            scaled by the ratio of effective areas^1/2 as a function of z, and
            the chi3 parameter is scaled by the ratio of effective areas
            (without the square root)

            If is_gaussian_beam is not True, then it is assumed that
            propagation occurs inside a waveguide, in which case a warning
            statement checks that the beta curve was provided (to account for
            waveguide dispersion).
        """

        # ------ g2 and g3 ----------------------------------------------------
        a_eff_gaussian_beam = lambda z: gaussian_beam_area(
            z_to_focus=z - length / 2,
            v0=pulse.v_grid,
            a_eff=a_eff,
        )

        if beta is None:  # calculate bulk refractive index
            if is_gaussian_beam:
                g2 = lambda z: self.g2_shg(
                    pulse.v_grid, pulse.v0, a_eff_gaussian_beam(z)
                )
                g3 = lambda z: self.g3(pulse.v_grid, a_eff_gaussian_beam(z))

            else:
                g2 = self.g2_shg(pulse.v_grid, pulse.v0, a_eff)
                g3 = self.g3(pulse.v_grid, a_eff)

        else:  # beta curve is provided
            assert (
                isinstance(beta, np.ndarray) and beta.shape == pulse.v_grid.shape
            ), "beta must be None or an array, make your own PyNLO mode for callable beta"
            n_eff = beta / (2 * np.pi * pulse.v_grid / sc.c)

            if is_gaussian_beam:
                g2 = lambda z: pynlo.utility.chi2.g2_shg(
                    pulse.v0, pulse.v_grid, n_eff, a_eff_gaussian_beam(z), self.chi2_eff
                )
                g3 = lambda z: pynlo.utility.chi3.g3_spm(
                    n_eff, a_eff_gaussian_beam(z), self.chi3_eff
                )

            else:
                g2 = pynlo.utility.chi2.g2_shg(
                    pulse.v0, pulse.v_grid, n_eff, a_eff, self.chi2_eff
                )
                g3 = pynlo.utility.chi3.g3_spm(n_eff, a_eff, self.chi3_eff)

        # ----- beta ----------------------------------------------------------
        if beta is None:  # calculate bulk dispersion
            beta_raw = self.beta(pulse.v_grid)

        else:  # beta curve is provided
            assert (
                isinstance(beta, np.ndarray) and beta.shape == pulse.v_grid.shape
            ), "beta must be None or an array, make your own PyNLO mode for callable beta"
            beta_raw = beta.copy()

        if is_gaussian_beam:

            def beta_gaussian_beam(z):
                z_to_focus = z - length / 2
                phi = guoy_phase_shift(z_to_focus, pulse.v_grid, a_eff)
                # final phase pickup should be βz + φ
                if z == 0:
                    return beta_raw
                else:
                    return beta_raw + phi / z

            beta = beta_gaussian_beam
            # beta = beta_raw

        else:
            beta = beta_raw

        # ----- mode and model ------------------------------------------------
        mode = pynlo.medium.Mode(
            pulse.v_grid,
            beta,
            alpha=-LN_alpha(pulse.v_grid),
            g2=g2,  # callable if gaussian beam
            g2_inv=g2_inv,  # callable
            g3=g3,  # callable if gaussian beam
            z=0.0,
        )

        # print("USING UPE")
        model = pynlo.model.UPE(pulse, mode)
        return model


class cLN(MgLN):
    """
    This class is useful for cLN waveguides or bulk crystals. For bulk
    crystals, the is_gaussian_beam kwarg should be set to True when calling
    generate_model
    """

    def __init__(self, T=24.5, axis="e"):
        super().__init__(T=T, axis=axis)

    # overides n in MgLN
    @property
    def n(self):
        """
        Returns:
            callable:
                a function that calculates the index of refraction as a
                function of frequency
        """
        return lambda v: n_cLN(v, T=self.T, axis=self.axis)


class SilicaFiber:
    """
    This class can really be used for any waveguide propagation that only uses
    3rd order nonlinear processes.

    It is called SilicaFiber only because the raman coefficients are by default
    set to those of silica. This can be altered as needed.

    Beta still needs to be set in all cases. This can be done by directly
    setting the beta property to a callable, or by calling
    set_beta_from_beta_n which generates a callable beta function using a
    taylor expansion starting from beta coefficients, or by calling
    set_beta_from_D_n which also generates a beta function but using a taylor
    expansion starting from D coefficients.

    Gamma is also a property that needs to be set to a float or an array
    (see connor's documentation in chi3.py in utility/)

    Both beta and gamma can be set by calling load_fiber_from_dict which
    imports the beta and gamma coefficients from a dictionary containing
    default parameters provided by OFS (see below)

    The flexibility of this class described above is illustrated in a few of
    the examples (optical-solitons.py, silica-pcf_supercontinuum.py, and
    intra-pulse_DFG.py)
    """

    def __init__(self):
        # Q. Lin and G. P. Agrawal, Raman Response Function for Silica Fibers,
        # Opt. Lett. 31, 3086 (2006).
        self._r_weights = [0.245 * (1 - 0.21), 12.2e-15, 32e-15]
        self._b_weights = [0.245 * 0.21, 96e-15]

        self._beta = None
        self._gamma = None

    @property
    def beta(self):
        assert self._beta is not None, "no beta has been defined yet"
        return self._beta

    @beta.setter
    def beta(self, beta):
        self._beta = beta

    @property
    def gamma(self):
        assert self._gamma is not None, "no gamma has been defined yet"
        return self._gamma

    @gamma.setter
    def gamma(self, gamma):
        self._gamma = gamma

    @property
    def r_weights(self):
        return self._r_weights

    @r_weights.setter
    def r_weights(self, r_weights):
        """
        r_weights : array_like of float
            The contributions due to vibrational resonances in the material. Must
            be given as ``[fraction, tau_1, tau_2]``, where `fraction` is the
            fractional contribution of the resonance to the total nonlinear
            response function, `tau_1` is the period of the vibrational frequency,
            and `tau_2` is the resonance's characteristic decay time. Enter more
            than one resonance using an (n, 3) shaped input array.
        """
        assert len(self._r_weights) == 3
        self._r_weights = r_weights

    @property
    def b_weights(self):
        return self._b_weights

    @b_weights.setter
    def b_weights(self, b_weights):
        """
        b_weights : array_like of float, optional
            The contributions due to boson peaks found in amorphous materials.
            Must be given as ``[fraction, tau_b]``, where `fraction` is the
            fractional contribution of the boson peak to the total nonlinear
            response function, and `tau_b` is the boson peak's characteristic
            decay time. Enter more than one peak using an (n, 2) shaped input
            array. The default behavior is to ignore this term.
        """
        assert len(self._b_weights) == 2
        self._b_weights = b_weights

    def set_beta_from_beta_n(self, v0, beta_n):
        """
        Set the callable beta(v_grid) from a taylor expansion of beta
        coefficients

        Args:
            v0 (float):
                center frequency
            beta_n (list):
                a list of beta coefficients (s^n/m) STARTING FROM BETA_2
        """
        self.beta = beta_n_to_beta(v0, beta_n)

    def set_beta_from_D_n(self, wl_0, D, dD_dwl):
        """
        Set the calllabe beta(v_grid) from a taylor expansion of beta
        coefficients. The beta coefficients are generated from D and D'.
        Currently higher order D''... are not supported. If you need those
        then use set_beta_from_beta_n

        Args:
            wl_0 (float):
                center wavelength
            D (float):
                D parameter (s/m^2)
            dD_dwl (float):
                D' parameter (s/m^3)
        """
        beta_2, beta_3 = Ds_to_beta_n(D, dD_dwl, wl_0)
        v0 = sc.c / wl_0
        self.set_beta_from_beta_n(v0, [beta_2, beta_3])

    def load_fiber_from_dict(self, dict_fiber, axis="slow"):
        """
        load fiber parameters from the dictionaries below

        Args:
            dict_fiber (dict):
                dict containing fiber parameters, with keys following naming
                convention shown below
            axis (str, optional):
                "slow" or "fast"
        """
        assert np.any([axis == "slow", axis == "fast"])
        assert "center wavelength" in dict_fiber.keys()
        assert "nonlinear coefficient" in dict_fiber.keys()

        if axis == "slow":
            assert "D slow axis" in dict_fiber.keys()
            assert "D slope slow axis" in dict_fiber.keys()

            D = dict_fiber["D slow axis"]
            dD_dwl = dict_fiber["D slope slow axis"]
        if axis == "fast":
            assert "D fast axis" in dict_fiber.keys()
            assert "D slope fast axis" in dict_fiber.keys()

            D = dict_fiber["D fast axis"]
            dD_dwl = dict_fiber["D slope fast axis"]

        wl_0 = dict_fiber["center wavelength"]
        self.set_beta_from_D_n(wl_0, D, dD_dwl)
        self.gamma = dict_fiber["nonlinear coefficient"]

    def g3(self, v_grid, t_shock=None):
        """
        g3 nonlinear parameter

        Args:
            v_grid (1D array):
                frequency grid
            t_shock (float, optional):
                the characteristic time scale of optical shock formation, default is None
                in which case it is taken to be 1 / (2 pi v0)

        Returns:
            g3
        """
        return pynlo.utility.chi3.gamma_to_g3(v_grid, self.gamma, t_shock=t_shock)

    def raman(self, n, dt, analytic=True):
        """
        Calculate the frequency-domain Raman and instantaneous nonlinear response
        function.

        This calculates the normalized Raman response using approximated formulas
        in the time domain. The total Raman fraction from the resonant and boson
        contributions should be less than 1.

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
        The equations used are the approximated formulations as summarized in
        section 2.3.3 of Agrawal's Nonlinear Fiber Optics [1]_. More accurate
        simulations may be obtainable using digitized experimental measurements,
        such as those shown in figure 2.2 of [1]_. The coefficients listed in
        Agrawal for silica-based fibers are as follows::

            r_weights = [0.245*(1-0.21), 12.2e-15, 32e-15] # resonant contribution
            b_weights = [0.245*0.21, 96e-15] # boson contribution

        For the carrier-resolved or real-valued representation, an additional
        factor of 3/2 is necessary to properly normalize the Raman response. The
        formulas used in this method have been fit to the analytic representation,
        which is normalized assuming that all three self-phase modulation pathways
        fold through baseband. In the real-valued domain however, only two pass
        through baseband. The third pathway is through the second harmonic. Thus,
        in the real-valued representation the Raman response must be normalized to
        produce the same nonlinear response against 2/3 the spectral amplitude.

        References
        ----------
        .. [1] Agrawal GP. Nonlinear Fiber Optics. Sixth ed. London; San Diego,
            CA;: Academic Press; 2019.
            https://doi.org/10.1016/B978-0-12-817042-7.00009-9

        """
        return pynlo.utility.chi3.raman(
            n,
            dt,
            self.r_weights,
            b_weights=self.b_weights,
            analytic=analytic,
        )

    def generate_model(
        self,
        pulse,
        t_shock="auto",
        raman_on=True,
        alpha=None,
        method="nlse",
    ):
        """
        generate pynlo.model.UPE or NLSE instance

        Args:
            pulse (object):
                instance of pynlo.light.Pulse
            t_shock (float, optional):
                time for optical shock formation, defaults to 1 / (2 pi pulse.v0)
            raman_on (bool, optional):
                whether to include raman effects, default is True
            alpha (array or callable, optional):
                default is 0, otherwise is a callable alpha(z, e_p) that returns a
                float or array, or fixed alpha.
            method (string, optional):
                nlse or upe

        Returns:
            model
        """

        # time for optical shock formation
        if isinstance(t_shock, str):
            assert t_shock.lower() == "auto"
            t_shock = 1 / (2 * np.pi * pulse.v0)
        else:
            assert isinstance(t_shock, float) or t_shock is None

        # absorption
        if alpha is not None:
            # absorption curve provided
            if isinstance(alpha, (np.ndarray, pynlo.utility.misc.ArrayWrapper)):
                assert (
                    alpha.size == pulse.n
                ), "if alpha is an array its size must match the simulation grid"
            # absorption constant provided
            elif isinstance(alpha, (float, int)):
                pass
            # absorption is a callable
            else:
                assert callable(
                    alpha
                ), "alpha must be None, array, float, or a callable: alpha(z, p_v)"

        # dispersion and g3 are calculated on pulse.v_grid for both UPE and
        # NLSE models
        v_grid = pulse.v_grid
        beta = self.beta(v_grid)
        g3 = self.g3(v_grid, t_shock=t_shock)

        # raman is calculated on pulse.rv_grid for UPE and pulse.v_grid for
        # NLSE
        method = method.lower()
        assert method == "nlse" or method == "upe"
        analytic = True if method == "nlse" else False
        n = pulse.n if method == "nlse" else pulse.rn
        dt = pulse.dt if method == "nlse" else pulse.rdt

        if raman_on:
            rv_grid, raman = self.raman(n, dt, analytic=analytic)
        else:
            rv_grid = raman = None

        mode = pynlo.medium.Mode(
            v_grid,
            beta,
            alpha=alpha,
            g2=None,  # not applicable
            g2_inv=None,  # not applicable
            g3=g3,
            rv_grid=rv_grid,
            r3=raman,
            z=0.0,
        )

        if method == "nlse":
            # print("USING NLSE")
            model = pynlo.model.NLSE(pulse, mode)
        else:
            # print("USING UPE")
            model = pynlo.model.UPE(pulse, mode)
        return model


# --- unit conversions -----
ps = 1e-12
nm = 1e-9
um = 1e-6
km = 1e3
W = 1.0
dB_to_linear = lambda x: 10 ** (x / 10)

# ---------- OFS fibers ----

# fiber ID 15021110740001
hnlf_2p2 = {
    "D slow axis": 2.2 * ps / (nm * km),
    "D slope slow axis": 0.026 * ps / (nm**2 * km),
    "D fast axis": 1.0 * ps / (nm * km),
    "D slope fast axis": 0.024 * ps / (nm**2 * km),
    "nonlinear coefficient": 10.5 / (W * km),
    "center wavelength": 1550 * nm,
}

# fiber ID 15021110740002
hnlf_5p7 = {
    "D slow axis": 5.7 * ps / (nm * km),
    "D slope slow axis": 0.027 * ps / (nm**2 * km),
    "D fast axis": 5.1 * ps / (nm * km),
    "D slope fast axis": 0.026 * ps / (nm**2 * km),
    "nonlinear coefficient": 10.5 / (W * km),
    "center wavelength": 1550 * nm,
}

# fiber ID 15021110740002
hnlf_5p7_pooja = {
    "D slow axis": 4.88 * ps / (nm * km),
    "D slope slow axis": 0.0228 * ps / (nm**2 * km),
    "D fast axis": 5.1 * ps / (nm * km),
    "D slope fast axis": 0.026 * ps / (nm**2 * km),
    "nonlinear coefficient": 10.9 / (W * km),
    "center wavelength": 1550 * nm,
}

pm1550 = {
    "D slow axis": 18 * ps / (nm * km),
    "D slope slow axis": 0.0612 * ps / (nm**2 * km),
    "nonlinear coefficient": 1.0 / (W * km),
    "center wavelength": 1550 * nm,
}