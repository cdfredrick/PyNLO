import os
import sys

# This GUI lives beside the pynlo package rather than inside it, so resolve the
# repo root ahead of site-packages. Without this, an older copy of PyNLO
# installed in the environment shadows the working tree and the fiber presets
# below silently disappear.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import pynlo
from pynlo import utility as ut
from scipy.constants import c, pi

from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)

# Optional: named fiber presets from the repo's materials module, which specify
# fibers in the datasheet D / D-slope format. Imported defensively so the GUI
# still runs if it is unavailable.
try:
    from pynlo.utility import materials as _materials

    FIBER_PRESETS = {
        name: obj
        for name, obj in vars(_materials).items()
        if isinstance(obj, dict)
        and "nonlinear coefficient" in obj
        and "center wavelength" in obj
    }
except Exception:
    _materials = None
    FIBER_PRESETS = {}


def db_per_m_to_alpha(loss_db_m):
    """Convert power attenuation in dB/m to PyNLO's alpha coefficient (1/m).

    PyNLO's Mode uses the sign convention where positive alpha is *gain* and
    negative alpha is *loss*, so a loss of ``loss_db_m`` maps to a negative
    alpha. Power then evolves as P(z) = P(0) * exp(alpha * z).
    """
    return -np.log(10.0) / 10.0 * loss_db_m


def normalized_db(x, floor_db=-60.0):
    """Normalize a nonnegative array to its peak and return dB with a floor."""
    x = np.asarray(x, dtype=float)
    peak = np.max(x)
    if peak <= 0:
        return np.full_like(x, floor_db)
    return 10.0 * np.log10(np.maximum(x / peak, 10.0 ** (floor_db / 10.0)))


def d_to_beta(d_ps_nm_km, d_slope_ps_nm2_km, wl_nm):
    """Convert dispersion D and D-slope (at wl_nm) to beta2/beta3 in ps^n/km."""
    k = 2.0 * pi * (c * 1e9 / 1e12)  # 2*pi*c expressed in nm/ps
    beta2 = -(wl_nm ** 2) * d_ps_nm_km / k
    beta3 = (wl_nm / k) ** 2 * (
        2.0 * wl_nm * d_ps_nm_km + wl_nm ** 2 * d_slope_ps_nm2_km
    )
    return beta2, beta3


def beta_to_d(beta2_ps2_km, beta3_ps3_km, wl_nm):
    """Inverse of d_to_beta: beta2/beta3 -> D, D-slope (at wl_nm)."""
    k = 2.0 * pi * (c * 1e9 / 1e12)
    d = -k * beta2_ps2_km / (wl_nm ** 2)
    d_slope = (beta3_ps3_km * (k / wl_nm) ** 2 - 2.0 * wl_nm * d) / (wl_nm ** 2)
    return d, d_slope


def fiber_dict_to_params(fiber_dict, axis="slow"):
    """Convert a materials.py fiber dict (D / D-slope, SI units) to GUI values.

    Returns D/D-slope (ps/nm/km, ps/nm^2/km), the equivalent beta2/beta3
    (ps^2/km, ps^3/km), gamma (1/(W km)), and the center wavelength (nm), using
    the same D -> beta conversion as materials.SilicaFiber.load_fiber_from_dict.
    """
    key_d = f"D {axis} axis"
    key_slope = f"D slope {axis} axis"
    if key_d not in fiber_dict or key_slope not in fiber_dict:
        raise ValueError(f"This fiber has no '{axis} axis' dispersion data.")

    d_si = fiber_dict[key_d]
    d_slope_si = fiber_dict[key_slope]
    beta2_si, beta3_si = _materials.Ds_to_beta_n(
        d_si, d_slope_si, fiber_dict["center wavelength"]
    )
    return {
        "d_ps_nm_km": d_si * 1e6,           # s/m^2 -> ps/(nm km)
        "d_slope_ps_nm2_km": d_slope_si * 1e-3,  # s/m^3 -> ps/(nm^2 km)
        "beta2_ps2_km": beta2_si / 1e-24 * 1e3,
        "beta3_ps3_km": beta3_si / 1e-36 * 1e3,
        "gamma_w_km": fiber_dict["nonlinear coefficient"] * 1e3,
        "wl_nm": fiber_dict["center wavelength"] * 1e9,
    }


def build_fiber_mode(
    pulse,
    beta2_ps2_km,
    beta3_ps3_km,
    gamma_w_km,
    loss_db_m,
    raman=True,
    self_steepening=True,
):
    """Build a pynlo.medium.Mode for a simple fiber from Taylor dispersion + gamma.

    Inputs use the usual fiber-optics units (ps^n/km, 1/(W km), dB/m); they are
    converted to base SI units for PyNLO.
    """
    beta_n = [
        0.0,
        0.0,
        beta2_ps2_km * 1e-24 / 1e3,   # ps^2/km -> s^2/m
        beta3_ps3_km * 1e-36 / 1e3,   # ps^3/km -> s^3/m
    ]
    beta = ut.taylor_series(2.0 * pi * pulse.v0, beta_n)(2.0 * pi * pulse.v_grid)

    gamma_w_m = gamma_w_km / 1e3
    t_shock = 1.0 / (2.0 * pi * pulse.v0) if self_steepening else None
    g3 = ut.chi3.gamma_to_g3(pulse.v_grid, gamma_w_m, t_shock)

    if raman:
        # Standard silica Raman response (same weights as PyNLO's examples).
        r_weights = [0.245 * (1.0 - 0.21), 12.2e-15, 32e-15]
        b_weights = [0.245 * 0.21, 96e-15]
        rv_grid, r3 = ut.chi3.raman(pulse.n, pulse.dt, r_weights, b_weights)
    else:
        rv_grid, r3 = None, None

    alpha = db_per_m_to_alpha(loss_db_m)
    return pynlo.medium.Mode(
        pulse.v_grid,
        beta,
        alpha=alpha,
        g3=g3,
        rv_grid=rv_grid,
        r3=r3,
    )


def propagate_chain(input_pulse, stages, n_records=80, local_error=1e-6):
    """Propagate a pulse through a sequence of fibers with splice loss between them.

    Each stage is a dict with keys: name, length_m, beta2_ps2_km, beta3_ps3_km,
    gamma_w_km, loss_db_m, splice_loss_db, raman, self_steepening. A stage's
    ``splice_loss_db`` is a lumped loss applied at that stage's input junction.

    Returns a dict with one continuous set of records across the whole chain:
    cumulative ``z`` (m), stacked ``a_t`` / ``a_v`` fields, per-stage boundary
    info, the input and output pulses, and the total length.
    """
    z_all, a_t_all, a_v_all = [], [], []
    stage_bounds = []
    pulse = input_pulse.copy()
    z_offset = 0.0

    for i, stage in enumerate(stages):
        splice_db = stage.get("splice_loss_db", 0.0)
        if splice_db:
            pulse.e_p = pulse.e_p * 10.0 ** (-splice_db / 10.0)

        mode = build_fiber_mode(
            pulse,
            beta2_ps2_km=stage["beta2_ps2_km"],
            beta3_ps3_km=stage["beta3_ps3_km"],
            gamma_w_km=stage["gamma_w_km"],
            loss_db_m=stage["loss_db_m"],
            raman=stage["raman"],
            self_steepening=stage["self_steepening"],
        )
        model = pynlo.model.NLSE(pulse, mode)
        dz = model.estimate_step_size(local_error=local_error)
        pulse_out, z, a_t, a_v = model.simulate(
            stage["length_m"],
            dz=dz,
            local_error=local_error,
            n_records=n_records,
            plot=None,
        )

        # Offset z into chain coordinates; drop the duplicated join point
        # (local z=0) for stages after the first so z stays increasing.
        z_shift = z + z_offset
        sl = slice(None) if i == 0 else slice(1, None)
        z_all.append(z_shift[sl])
        a_t_all.append(a_t[sl])
        a_v_all.append(a_v[sl])

        stage_bounds.append(
            {
                "z_start": z_offset,
                "z_end": z_offset + stage["length_m"],
                "name": stage.get("name") or f"Fiber {i + 1}",
                "splice_loss_db": splice_db,
            }
        )
        z_offset += stage["length_m"]
        pulse = pulse_out

    return {
        "z": np.concatenate(z_all),
        "a_t": np.concatenate(a_t_all, axis=0),
        "a_v": np.concatenate(a_v_all, axis=0),
        "stage_bounds": stage_bounds,
        "input_pulse": input_pulse,
        "output_pulse": pulse,
        "total_length": z_offset,
    }


class SechPulseSection(ttk.LabelFrame):
    def __init__(self, parent, on_pulse_created=None):
        super().__init__(parent, text="Sech Pulse", padding=12)

        self.on_pulse_created = on_pulse_created
        self.pulse = None

        # Defaults match Ru_HNLF.py: 5 W / 20 GHz = 250 pJ, 120 fs at 1550 nm.
        self.average_power_mw = tk.DoubleVar(value=5000.0)
        self.repetition_rate_mhz = tk.DoubleVar(value=20000.0)
        self.center_wavelength_nm = tk.DoubleVar(value=1550.0)
        self.pulse_width_fs = tk.DoubleVar(value=120.0)

        self._build_widgets()

    def _build_widgets(self):
        ttk.Label(self, text="Average power:").grid(
            row=0, column=0, sticky="w", padx=5, pady=5
        )
        ttk.Entry(
            self,
            textvariable=self.average_power_mw,
            width=14,
        ).grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        ttk.Label(self, text="mW").grid(
            row=0, column=2, sticky="w", padx=5, pady=5
        )

        ttk.Label(self, text="Repetition rate:").grid(
            row=1, column=0, sticky="w", padx=5, pady=5
        )
        ttk.Entry(
            self,
            textvariable=self.repetition_rate_mhz,
            width=14,
        ).grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        ttk.Label(self, text="MHz").grid(
            row=1, column=2, sticky="w", padx=5, pady=5
        )

        ttk.Label(self, text="Center wavelength:").grid(
            row=2, column=0, sticky="w", padx=5, pady=5
        )
        ttk.Entry(
            self,
            textvariable=self.center_wavelength_nm,
            width=14,
        ).grid(row=2, column=1, sticky="ew", padx=5, pady=5)
        ttk.Label(self, text="nm").grid(
            row=2, column=2, sticky="w", padx=5, pady=5
        )

        ttk.Label(self, text="Pulse width, intensity FWHM:").grid(
            row=3, column=0, sticky="w", padx=5, pady=5
        )
        ttk.Entry(
            self,
            textvariable=self.pulse_width_fs,
            width=14,
        ).grid(row=3, column=1, sticky="ew", padx=5, pady=5)
        ttk.Label(self, text="fs").grid(
            row=3, column=2, sticky="w", padx=5, pady=5
        )

        ttk.Button(
            self,
            text="Create Pulse",
            command=self.create_pulse,
        ).grid(
            row=4,
            column=0,
            columnspan=3,
            sticky="ew",
            padx=5,
            pady=(12, 5),
        )

        self.status_label = ttk.Label(
            self,
            text="No pulse created",
            wraplength=500,
        )
        self.status_label.grid(
            row=5,
            column=0,
            columnspan=3,
            sticky="w",
            padx=5,
            pady=5,
        )

        self.columnconfigure(1, weight=1)

    def create_pulse(self):
        try:
            average_power_mw = self.average_power_mw.get()
            repetition_rate_mhz = self.repetition_rate_mhz.get()
            center_wavelength_nm = self.center_wavelength_nm.get()
            pulse_width_fs = self.pulse_width_fs.get()

            if average_power_mw <= 0:
                raise ValueError("Average power must be greater than zero.")

            if repetition_rate_mhz <= 0:
                raise ValueError("Repetition rate must be greater than zero.")

            if center_wavelength_nm <= 0:
                raise ValueError("Center wavelength must be greater than zero.")

            if pulse_width_fs <= 0:
                raise ValueError("Pulse width must be greater than zero.")

            average_power_w = average_power_mw * 1e-3
            repetition_rate_hz = repetition_rate_mhz * 1e6

            pulse_energy_j = average_power_w / repetition_rate_hz

            # Current PyNLO uses SI units and an absolute (positive) frequency
            # grid. Pick the frequency window as a fraction of the center
            # frequency so v_min stays comfortably positive, then choose the
            # number of points so the time window is much wider than the pulse.
            t_fwhm_s = pulse_width_fs * 1e-15
            v0 = c / (center_wavelength_nm * 1e-9)

            v_min = 0.25 * v0
            v_max = 1.75 * v0
            v_span = v_max - v_min

            # For an FFT grid the time window is npts / v_span. Grow npts (as a
            # power of two) until the window comfortably contains the pulse.
            time_window_s = max(20.0e-12, 40.0 * t_fwhm_s)
            npts = 1
            while npts < v_span * time_window_s:
                npts *= 2
            npts = min(npts, 2**18)

            self.pulse = pynlo.light.Pulse.Sech(
                npts,
                v_min,
                v_max,
                v0,
                pulse_energy_j,
                t_fwhm_s,
            )

            self.status_label.config(
                text=(
                    f"Pulse created\n"
                    f"Pulse energy: {pulse_energy_j * 1e12:.3f} pJ\n"
                    f"Center frequency: {v0 * 1e-12:.3f} THz"
                )
            )

            if self.on_pulse_created is not None:
                self.on_pulse_created(self.pulse)

            return self.pulse

        except (tk.TclError, ValueError, AttributeError, TypeError) as error:
            messagebox.showerror(
                "Pulse creation error",
                str(error),
            )
            return None


class FiberSection(ttk.LabelFrame):
    def __init__(
        self,
        parent,
        title="Fiber",
        on_propagate=None,
        on_change=None,
        show_splice=False,
        show_propagate=True,
        defaults=None,
    ):
        super().__init__(parent, text=title, padding=12)

        self.on_propagate = on_propagate
        self.on_change = on_change
        self.show_splice = show_splice
        self.show_propagate = show_propagate

        defaults = defaults or {}
        self.length_m = tk.DoubleVar(value=defaults.get("length_m", 2.0))
        # Dispersion can be entered either as D / D-slope or as beta2 / beta3.
        self.disp_mode = tk.StringVar(value=defaults.get("disp_mode", "D"))
        self.d_ps_nm_km = tk.DoubleVar(value=defaults.get("d_ps_nm_km", 17.0))
        self.d_slope_ps_nm2_km = tk.DoubleVar(
            value=defaults.get("d_slope_ps_nm2_km", 0.056)
        )
        self.ref_wl_nm = tk.DoubleVar(value=defaults.get("ref_wl_nm", 1550.0))
        self.beta2_ps2_km = tk.DoubleVar(value=defaults.get("beta2_ps2_km", -21.7))
        self.beta3_ps3_km = tk.DoubleVar(value=defaults.get("beta3_ps3_km", 0.12))
        self.gamma_w_km = tk.DoubleVar(value=defaults.get("gamma_w_km", 1.3))
        self.loss_db_m = tk.DoubleVar(value=defaults.get("loss_db_m", 0.0))
        self.splice_loss_db = tk.DoubleVar(value=defaults.get("splice_loss_db", 0.0))
        self.raman = tk.BooleanVar(value=defaults.get("raman", True))
        self.self_steepening = tk.BooleanVar(
            value=defaults.get("self_steepening", True)
        )

        self._build_widgets()

    def _build_widgets(self):
        row = 0

        # Preset loader: pull a named fiber (D / D-slope format) from materials.py.
        preset_frame = ttk.Frame(self)
        preset_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        row += 1

        ttk.Label(preset_frame, text="Preset:").pack(side="left")
        self.preset_var = tk.StringVar()
        preset_names = sorted(FIBER_PRESETS.keys())
        self.preset_combo = ttk.Combobox(
            preset_frame,
            textvariable=self.preset_var,
            values=preset_names,
            state="readonly" if preset_names else "disabled",
            width=14,
        )
        if preset_names:
            self.preset_combo.current(0)
        self.preset_combo.pack(side="left", padx=4)

        self.axis_var = tk.StringVar(value="slow")
        ttk.Combobox(
            preset_frame,
            textvariable=self.axis_var,
            values=["slow", "fast"],
            state="readonly",
            width=6,
        ).pack(side="left", padx=4)

        ttk.Button(
            preset_frame,
            text="Load fiber",
            command=self._load_preset,
            state="normal" if preset_names else "disabled",
        ).pack(side="left", padx=4)

        # Splice loss applies at this fiber's input junction (previous → this).
        if self.show_splice:
            self._add_field(self, row, "Splice loss (in):",
                            self.splice_loss_db, "dB")
            row += 1

        self._add_field(self, row, "Fiber length:", self.length_m, "m")
        row += 1

        # Dispersion input mode: D / D-slope  or  beta2 / beta3.
        mode_frame = ttk.Frame(self)
        mode_frame.grid(row=row, column=0, columnspan=3, sticky="w",
                        padx=5, pady=(6, 0))
        row += 1
        ttk.Label(mode_frame, text="Dispersion:").pack(side="left")
        ttk.Radiobutton(
            mode_frame, text="D / D-slope", variable=self.disp_mode, value="D",
            command=self._toggle_disp_mode,
        ).pack(side="left", padx=4)
        ttk.Radiobutton(
            mode_frame, text="β₂ / β₃", variable=self.disp_mode, value="beta",
            command=self._toggle_disp_mode,
        ).pack(side="left", padx=4)

        # The two dispersion field groups share one grid slot; only one shows.
        self.disp_frame_d = ttk.Frame(self)
        self.disp_frame_d.grid(row=row, column=0, columnspan=3, sticky="ew")
        self._add_field(self.disp_frame_d, 0, "D:", self.d_ps_nm_km,
                        "ps/(nm·km)")
        self._add_field(self.disp_frame_d, 1, "D slope:",
                        self.d_slope_ps_nm2_km, "ps/(nm²·km)")
        self._add_field(self.disp_frame_d, 2, "D at λ:", self.ref_wl_nm, "nm")

        self.disp_frame_b = ttk.Frame(self)
        self.disp_frame_b.grid(row=row, column=0, columnspan=3, sticky="ew")
        self._add_field(self.disp_frame_b, 0, "β₂ (GVD):",
                        self.beta2_ps2_km, "ps²/km")
        self._add_field(self.disp_frame_b, 1, "β₃ (TOD):",
                        self.beta3_ps3_km, "ps³/km")
        row += 1
        self._show_disp_frame()

        self._add_field(self, row, "γ (nonlinearity):", self.gamma_w_km,
                        "1/(W·km)")
        row += 1
        self._add_field(self, row, "Loss:", self.loss_db_m, "dB/m")
        row += 1

        ttk.Checkbutton(
            self, text="Raman", variable=self.raman,
            command=self._notify_change,
        ).grid(row=row, column=0, sticky="w", padx=5, pady=5)
        ttk.Checkbutton(
            self, text="Self-steepening", variable=self.self_steepening,
            command=self._notify_change,
        ).grid(row=row, column=1, columnspan=2, sticky="w", padx=5, pady=5)
        row += 1

        if self.show_propagate:
            ttk.Button(
                self, text="Propagate Pulse", command=self._propagate
            ).grid(
                row=row, column=0, columnspan=3, sticky="ew", padx=5, pady=(12, 5)
            )
            row += 1

        self.status_label = ttk.Label(self, text="", wraplength=360)
        self.status_label.grid(
            row=row, column=0, columnspan=3, sticky="w", padx=5, pady=5
        )

        self.columnconfigure(1, weight=1)

    def _add_field(self, parent, r, label, var, unit):
        """Add a label / entry / unit row to ``parent`` at grid row ``r``."""
        ttk.Label(parent, text=label).grid(
            row=r, column=0, sticky="w", padx=5, pady=5
        )
        entry = ttk.Entry(parent, textvariable=var, width=14)
        entry.grid(row=r, column=1, sticky="ew", padx=5, pady=5)
        ttk.Label(parent, text=unit).grid(
            row=r, column=2, sticky="w", padx=5, pady=5
        )
        # Refresh the diagram when the user commits an edited value.
        entry.bind("<Return>", self._notify_change)
        entry.bind("<FocusOut>", self._notify_change)
        parent.columnconfigure(1, weight=1)

    def _show_disp_frame(self):
        """Show the field group for the active dispersion input mode."""
        if self.disp_mode.get() == "beta":
            self.disp_frame_d.grid_remove()
            self.disp_frame_b.grid()
        else:
            self.disp_frame_b.grid_remove()
            self.disp_frame_d.grid()

    def _toggle_disp_mode(self):
        """Convert the current dispersion into the newly selected mode's fields."""
        try:
            wl = self.ref_wl_nm.get()
            if self.disp_mode.get() == "beta":
                b2, b3 = d_to_beta(
                    self.d_ps_nm_km.get(), self.d_slope_ps_nm2_km.get(), wl
                )
                self.beta2_ps2_km.set(round(b2, 6))
                self.beta3_ps3_km.set(round(b3, 6))
            else:
                d, s = beta_to_d(
                    self.beta2_ps2_km.get(), self.beta3_ps3_km.get(), wl
                )
                self.d_ps_nm_km.set(round(d, 6))
                self.d_slope_ps_nm2_km.set(round(s, 6))
        except tk.TclError:
            pass
        self._show_disp_frame()
        self._notify_change()

    def get_params(self):
        if self.disp_mode.get() == "beta":
            beta2 = self.beta2_ps2_km.get()
            beta3 = self.beta3_ps3_km.get()
        else:
            beta2, beta3 = d_to_beta(
                self.d_ps_nm_km.get(),
                self.d_slope_ps_nm2_km.get(),
                self.ref_wl_nm.get(),
            )
        return dict(
            length_m=self.length_m.get(),
            beta2_ps2_km=beta2,
            beta3_ps3_km=beta3,
            gamma_w_km=self.gamma_w_km.get(),
            loss_db_m=self.loss_db_m.get(),
            splice_loss_db=self.splice_loss_db.get() if self.show_splice else 0.0,
            raman=self.raman.get(),
            self_steepening=self.self_steepening.get(),
        )

    def _load_preset(self):
        name = self.preset_var.get()
        if not name or name not in FIBER_PRESETS:
            messagebox.showwarning("No fiber", "No fiber preset selected.")
            return

        axis = self.axis_var.get()
        try:
            params = fiber_dict_to_params(FIBER_PRESETS[name], axis=axis)
        except (ValueError, KeyError, TypeError) as error:
            messagebox.showerror("Fiber load error", str(error))
            return

        # Populate both dispersion representations so either mode is correct.
        self.d_ps_nm_km.set(round(params["d_ps_nm_km"], 6))
        self.d_slope_ps_nm2_km.set(round(params["d_slope_ps_nm2_km"], 6))
        self.ref_wl_nm.set(round(params["wl_nm"], 3))
        self.beta2_ps2_km.set(round(params["beta2_ps2_km"], 6))
        self.beta3_ps3_km.set(round(params["beta3_ps3_km"], 6))
        self.gamma_w_km.set(round(params["gamma_w_km"], 6))

        self.status_label.config(
            text=(
                f"Loaded '{name}' ({axis} axis)\n"
                f"D={params['d_ps_nm_km']:.3f} ps/(nm·km), "
                f"γ={params['gamma_w_km']:.3f} 1/(W·km)\n"
                f"Fiber center λ: {params['wl_nm']:.0f} nm "
                "(set the pulse to match)"
            )
        )
        self._notify_change()

    def _notify_change(self, event=None):
        if self.on_change is not None:
            self.on_change()

    def _propagate(self):
        if self.on_propagate is not None:
            self.on_propagate()


class PyNLOGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("PyNLO GUI")
        self.geometry("1220x900")
        self.minsize(1000, 760)

        self.pulse = None
        self.sim = None
        self.clicked_end = None  # which fiber end is shown in the bottom panel

        self._build_gui()

    def _build_gui(self):
        main_frame = ttk.Frame(self, padding=15)
        main_frame.pack(fill="both", expand=True)

        title_label = ttk.Label(
            main_frame,
            text="PyNLO Pulse Builder",
            font=("TkDefaultFont", 16, "bold"),
        )
        title_label.pack(anchor="w", pady=(0, 8))

        # Fiber illustration across the full width at the top.
        self._build_illustration(main_frame)

        body = ttk.Frame(main_frame)
        body.pack(fill="both", expand=True, pady=(8, 0))

        # Left column: input parameters (scrollable if taller than the window).
        left_container = ttk.Frame(body)
        left_container.pack(side="left", fill="y", padx=(0, 12))
        left = self._make_scrollable(left_container, width=400)

        # Right column: the plots.
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)

        self.pulse_section = SechPulseSection(
            left,
            on_pulse_created=self.handle_pulse_created,
        )
        self.pulse_section.pack(fill="x", pady=(0, 8))

        # Fiber 1 defaults = Ru_HNLF.py ND-HNLF (D = -2.6, γ = 10.5, 2.7 m).
        self.fiber_section = FiberSection(
            left,
            title="Fiber 1",
            on_propagate=self.handle_propagate,
            on_change=self.update_illustration,
            show_propagate=False,
            defaults={
                "length_m": 2.7,
                "d_ps_nm_km": -2.6,
                "d_slope_ps_nm2_km": 0.026,
                "ref_wl_nm": 1550.0,
                "beta2_ps2_km": 3.3162,
                "beta3_ps3_km": 0.03684,
                "gamma_w_km": 10.5,
                "loss_db_m": 0.0008,  # 0.8e-5 dB/cm
            },
        )
        self.fiber_section.pack(fill="x", pady=(0, 8))

        # Fiber 2 defaults = Ru_HNLF.py PM1550 (D = 18, γ = 0.78, splice 0.70 dB).
        self.fiber_section2 = FiberSection(
            left,
            title="Fiber 2",
            on_propagate=self.handle_propagate,
            on_change=self.update_illustration,
            show_splice=True,
            show_propagate=False,
            defaults={
                "length_m": 0.5,
                "d_ps_nm_km": 18.0,
                "d_slope_ps_nm2_km": 0.0612,
                "ref_wl_nm": 1550.0,
                "beta2_ps2_km": -22.9581,
                "beta3_ps3_km": 0.13734,
                "gamma_w_km": 0.78,
                "splice_loss_db": 0.70,
            },
        )
        self.fiber_section2.pack(fill="x", pady=(0, 8))

        ttk.Button(
            left,
            text="Propagate through fibers",
            command=self.handle_propagate,
        ).pack(fill="x", pady=(0, 8))

        self.main_status = ttk.Label(left, text="Ready", wraplength=360)
        self.main_status.pack(anchor="w", pady=(2, 6))

        self._build_plot(right)

        # Now that the fiber controls exist, draw the schematic for real.
        self.update_illustration()

    def _make_scrollable(self, parent, width=400):
        """Return an inner frame that scrolls vertically inside ``parent``."""
        canvas = tk.Canvas(parent, highlightthickness=0, width=width)
        vbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vbar.set)
        vbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = ttk.Frame(canvas)
        window = canvas.create_window((0, 0), window=inner, anchor="nw")

        # Keep the scroll region matched to the inner frame's size, and make the
        # inner frame span the canvas width so contents fill the column.
        inner.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.bind(
            "<Configure>",
            lambda e: canvas.itemconfigure(window, width=e.width),
        )

        # Mouse-wheel scrolling while the pointer is over this column.
        def _on_wheel(event):
            if event.delta:
                canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

        inner.bind(
            "<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_wheel)
        )
        inner.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        return inner

    def _build_illustration(self, parent):
        self.illus_figure = Figure(figsize=(11.0, 2.0), dpi=100)
        self.ax_illus = self.illus_figure.add_subplot(111)
        self.draw_illustration()
        self.illus_figure.tight_layout()

        self.illus_canvas = FigureCanvasTkAgg(self.illus_figure, master=parent)
        self.illus_canvas.get_tk_widget().pack(fill="x", expand=False)
        self.illus_canvas.mpl_connect(
            "button_press_event", self.on_illustration_click
        )

    def on_illustration_click(self, event):
        if event.inaxes is not self.ax_illus or event.xdata is None:
            return
        if self.sim is None:
            return

        bounds = self.sim.get("stage_bounds", [])
        if not bounds:
            return

        # Left of the splice selects Fiber 1, right selects Fiber 2.
        self.clicked_end = 0 if (event.xdata < 4.65 or len(bounds) < 2) else 1
        self.refresh_plots()

    def _build_plot(self, parent):
        plot_frame = ttk.Frame(parent)
        plot_frame.pack(fill="both", expand=True)

        self.figure = Figure(figsize=(7, 6), dpi=100)
        # Top row: input vs the selected location (set by clicking a fiber).
        self.ax_time = self.figure.add_subplot(2, 2, 1)
        self.ax_freq = self.figure.add_subplot(2, 2, 2)
        # Lower row: temporal / spectral propagation maps.
        self.ax_tmap = self.figure.add_subplot(2, 2, 3)
        self.ax_vmap = self.figure.add_subplot(2, 2, 4)
        self.figure.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()

    def update_illustration(self):
        """Redraw only the fiber schematic (e.g. after a settings change)."""
        if not hasattr(self, "illus_canvas"):
            return
        self.draw_illustration()
        self.illus_canvas.draw()

    def draw_illustration(self):
        """Draw a schematic of the pulse propagating through the single fiber."""
        ax = self.ax_illus
        ax.clear()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

        if not hasattr(self, "fiber_section2"):
            return
        try:
            f1 = self.fiber_section.get_params()
            f2 = self.fiber_section2.get_params()
            f1["ref_wl_nm"] = self.fiber_section.ref_wl_nm.get()
            f2["ref_wl_nm"] = self.fiber_section2.ref_wl_nm.get()
        except tk.TclError:
            return

        y0 = 5.0  # vertical center of the pulse-in / boxes / pulse-out flow

        def fiber_box(x0, x1, title, params):
            cx = 0.5 * (x0 + x1)
            beta2 = params["beta2_ps2_km"]
            # Show dispersion as D / D-slope (converted from beta at ref λ).
            d, d_slope = beta_to_d(
                beta2, params["beta3_ps3_km"], params["ref_wl_nm"]
            )
            color = "#cfe8ff" if beta2 < 0 else "#ffe0cc"
            ax.add_patch(FancyBboxPatch(
                (x0, 2.3), x1 - x0, 5.4,
                boxstyle="round,pad=0.02,rounding_size=0.12",
                linewidth=1.5, edgecolor="#1f4e79", facecolor=color,
            ))
            ax.text(cx, 6.7, f"{title} — {params['length_m']:g} m",
                    ha="center", va="center", fontsize=8.5, fontweight="bold")
            ax.text(cx, 5.4, f"D = {d:.3g} ps/(nm·km)",
                    ha="center", va="center", fontsize=7.5)
            ax.text(cx, 4.3, f"D slope = {d_slope:.3g} ps/(nm²·km)",
                    ha="center", va="center", fontsize=7.5)
            ax.text(cx, 3.2, f"γ = {params['gamma_w_km']:g} /(W·km)",
                    ha="center", va="center", fontsize=7.5)

        # Input pulse icon + label + arrow into fiber 1.
        x_in = np.linspace(0.1, 1.1, 60)
        ax.plot(x_in, y0 + 1.3 / np.cosh((x_in - 0.6) / 0.12),
                color="tab:blue", lw=2)
        ax.text(0.6, y0 + 2.6, "Input", ha="center", fontsize=9,
                color="tab:blue")
        ax.annotate("", xy=(1.7, y0), xytext=(1.2, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=2, color="0.3"))

        # Fiber 1 -> splice -> Fiber 2.
        fiber_box(1.75, 4.25, "Fiber 1", f1)
        ax.annotate("", xy=(5.05, y0), xytext=(4.25, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=2, color="0.3"))
        ax.text(4.65, y0 + 1.9, f"splice\n{f2['splice_loss_db']:g} dB",
                ha="center", va="center", fontsize=7.5, color="grey")
        fiber_box(5.05, 7.55, "Fiber 2", f2)

        # Marker at the fiber end whose spectrum is shown in the bottom panel.
        if self.clicked_end is not None:
            x_edge = 4.25 if self.clicked_end == 0 else 7.55
            ax.plot([x_edge, x_edge], [2.3, 7.7],
                    color="tab:green", ls="--", lw=1.4)
            ax.plot([x_edge], [8.4], marker="v", color="tab:green", ms=10)

        # Arrow out of fiber 2 + output pulse icon + label.
        ax.annotate("", xy=(8.55, y0), xytext=(7.75, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=2, color="0.3"))
        x_out = np.linspace(8.65, 9.9, 60)
        if self.sim is not None:
            ax.plot(x_out, y0 + 1.5 / np.cosh((x_out - 9.25) / 0.09),
                    color="tab:orange", lw=2)
            e_out = self.sim["pulse_out"].e_p * 1e12
            ax.text(9.25, y0 + 2.6, f"Output · {e_out:.0f} pJ",
                    ha="center", fontsize=9, color="tab:orange")
        else:
            ax.plot(x_out, y0 + 1.3 / np.cosh((x_out - 9.25) / 0.12),
                    color="grey", lw=2, ls="--")
            ax.text(9.25, y0 + 2.6, "Output", ha="center", fontsize=9,
                    color="grey")

    def refresh_plots(self):
        self.update_illustration()

        pulse = self.pulse
        if pulse is None:
            self.figure.tight_layout()
            self.canvas.draw()
            return
        sim = self.sim

        wl_nm = c / pulse.v_grid * 1e9
        order = np.argsort(wl_nm)
        wl_sorted = wl_nm[order]
        t_ps = pulse.t_grid * 1e12

        # Time window centered on the input peak, a few widths wide.
        p_t_in = pulse.p_t
        peak_t = t_ps[int(np.argmax(p_t_in))]
        half = max(2.0, 6.0 * pulse.t_width().fwhm * 1e12)

        # Selected location for the top-row overlay (defaults to the end of the
        # last fiber = the chain output; clicking a fiber changes it).
        sel_idx = None
        sel_label = "Output"
        if sim is not None:
            bounds = sim.get("stage_bounds", [])
            stage_idx = (
                self.clicked_end
                if self.clicked_end is not None
                else len(bounds) - 1
            )
            stage_idx = max(0, min(stage_idx, len(bounds) - 1))
            sel_idx = int(
                np.argmin(np.abs(sim["z"] - bounds[stage_idx]["z_end"]))
            )
            sel_label = (
                f"{bounds[stage_idx]['name']} (z={sim['z'][sel_idx]:.2f} m)"
            )

        # ---- Time domain (input vs selected location) ----
        # Both curves are normalized to the INPUT peak, so the selected pulse
        # shows its true size relative to the input (it can be above or below 1).
        self.ax_time.clear()
        t_norm = p_t_in.max()
        self.ax_time.plot(
            t_ps, p_t_in / t_norm, color="tab:blue", label="Input"
        )
        if sim is not None:
            p_t_sel = np.abs(sim["a_t"][sel_idx]) ** 2
            self.ax_time.plot(
                t_ps, p_t_sel / t_norm,
                color="tab:orange", label=sel_label,
            )
        self.ax_time.set_xlim(peak_t - half, peak_t + half)
        self.ax_time.set_title("Time domain")
        self.ax_time.set_xlabel("Time (ps)")
        self.ax_time.set_ylabel("Power (norm. to input peak)")
        self.ax_time.grid(True, alpha=0.3)
        self.ax_time.legend(fontsize=8)

        # ---- Spectrum (input vs selected location), in dB vs the INPUT peak ----
        # 0 dB is the input spectral peak, so the selected spectrum shows its
        # true level relative to the input (not re-normalized to its own peak).
        self.ax_freq.clear()
        floor_db = -60.0
        v_ref = pulse.p_v.max()

        def spec_db(power):
            return 10.0 * np.log10(
                np.maximum(power / v_ref, 10.0 ** (floor_db / 10.0))
            )

        db_in = spec_db(pulse.p_v[order])
        self.ax_freq.plot(wl_sorted, db_in, color="tab:blue", label="Input")
        signal = db_in > -50
        if sim is not None:
            db_sel = spec_db(np.abs(sim["a_v"][sel_idx]) ** 2)[order]
            self.ax_freq.plot(
                wl_sorted, db_sel, color="tab:orange", label=sel_label
            )
            signal = signal | (db_sel > -50)
        if np.any(signal):
            self.ax_freq.set_xlim(wl_sorted[signal].min(), wl_sorted[signal].max())
        self.ax_freq.set_ylim(-60, 5)
        self.ax_freq.set_title("Spectrum")
        self.ax_freq.set_xlabel("Wavelength (nm)")
        self.ax_freq.set_ylabel("PSD (dB, norm. to input peak)")
        self.ax_freq.grid(True, alpha=0.3)
        self.ax_freq.legend(fontsize=8)

        # ---- Propagation maps (only after propagation) ----
        self.ax_tmap.clear()
        self.ax_vmap.clear()
        if sim is None:
            for ax, txt in (
                (self.ax_tmap, "Propagate to see\ntemporal evolution"),
                (self.ax_vmap, "Propagate to see\nspectral evolution"),
            ):
                ax.text(
                    0.5, 0.5, txt, ha="center", va="center",
                    transform=ax.transAxes, color="grey",
                )
                ax.set_xticks([])
                ax.set_yticks([])
        else:
            z_cm = sim["z"] * 100.0

            p_t_map = np.abs(sim["a_t"]) ** 2
            p_t_map = p_t_map / p_t_map.max()
            self.ax_tmap.pcolormesh(
                t_ps, z_cm, p_t_map, shading="auto", cmap="magma"
            )
            self.ax_tmap.set_xlim(peak_t - half, peak_t + half)
            self.ax_tmap.set_title("Temporal evolution")
            self.ax_tmap.set_xlabel("Time (ps)")
            self.ax_tmap.set_ylabel("Distance (cm)")

            p_v_map = normalized_db(np.abs(sim["a_v"]) ** 2, floor_db=-40.0)
            self.ax_vmap.pcolormesh(
                wl_sorted, z_cm, p_v_map[:, order],
                shading="auto", cmap="magma", vmin=-40, vmax=0,
            )
            if np.any(signal):
                self.ax_vmap.set_xlim(
                    wl_sorted[signal].min(), wl_sorted[signal].max()
                )
            self.ax_vmap.set_title("Spectral evolution")
            self.ax_vmap.set_xlabel("Wavelength (nm)")
            self.ax_vmap.set_ylabel("Distance (cm)")

            # Cyan dashes = fiber interfaces; green line = selected location.
            for bound in bounds[:-1]:
                for ax in (self.ax_tmap, self.ax_vmap):
                    ax.axhline(bound["z_end"] * 100.0,
                               color="cyan", ls="--", lw=1, alpha=0.7)
            for ax in (self.ax_tmap, self.ax_vmap):
                ax.axhline(sim["z"][sel_idx] * 100.0,
                           color="tab:green", lw=1.4, alpha=0.9)

        self.figure.tight_layout()
        self.canvas.draw()

    def handle_pulse_created(self, pulse):
        self.pulse = pulse
        self.sim = None
        self.clicked_end = None

        self.main_status.config(
            text="The pulse is now available as self.pulse."
        )

        self.refresh_plots()

        print("\nSech pulse successfully created.")
        print(f"Pulse object: {pulse}")

        average_power_w = self.pulse_section.average_power_mw.get() * 1e-3
        repetition_rate_hz = self.pulse_section.repetition_rate_mhz.get() * 1e6
        pulse_energy_pj = average_power_w / repetition_rate_hz * 1e12
        print(f"Pulse energy: {pulse_energy_pj:.3f} pJ")

    def handle_propagate(self):
        if self.pulse is None:
            messagebox.showwarning(
                "No pulse", "Create a pulse before propagating."
            )
            return

        try:
            stage1 = dict(self.fiber_section.get_params(), name="Fiber 1")
            stage2 = dict(self.fiber_section2.get_params(), name="Fiber 2")

            for stage in (stage1, stage2):
                if stage["length_m"] <= 0:
                    raise ValueError(
                        f"{stage['name']} length must be greater than zero."
                    )

            self.main_status.config(text="Propagating…")
            self.update_idletasks()

            result = propagate_chain(
                self.pulse, [stage1, stage2], n_records=100, local_error=1e-6
            )

            self.sim = {
                "z": result["z"],
                "a_t": result["a_t"],
                "a_v": result["a_v"],
                "stage_bounds": result["stage_bounds"],
                "pulse_out": result["output_pulse"],
            }
            # Default the lower panels to the end of the last fiber (output).
            self.clicked_end = len(result["stage_bounds"]) - 1
            self.refresh_plots()

            total_len = result["total_length"]
            e_out = result["output_pulse"].e_p * 1e12
            self.main_status.config(
                text=(
                    f"Propagated {total_len:.3f} m total "
                    f"(Fiber 1: {stage1['length_m']:g} m, "
                    f"Fiber 2: {stage2['length_m']:g} m).  "
                    f"Output energy: {e_out:.3f} pJ"
                )
            )

            print("\nPropagation complete.")
            print(f"Output energy: {e_out:.3f} pJ")

        except (ValueError, AttributeError, TypeError, tk.TclError) as error:
            messagebox.showerror("Propagation error", str(error))
            self.main_status.config(text="Propagation failed")


def main():
    app = PyNLOGUI()
    app.mainloop()


if __name__ == "__main__":
    main()