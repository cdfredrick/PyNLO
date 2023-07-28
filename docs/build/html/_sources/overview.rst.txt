Package Overview
================
In PyNLO, object-oriented programming is used to emulate physical objects. Whenever possible each physical entity with intrinsic properties is mapped to a Python class, e.g. an optical pulse or nonlinear fiber. These classes keep track of the objects’ properties and provide simple calculator-type helper functions.


Time and Frequency Grids
------------------------
The core functionality of this package is built off of the time and frequency grids given by the :py:class:`pynlo.utility.TFGrid` class. These grids represent physical coordinate spaces. Critically, the frequency grid is only defined for positive frequencies and is aligned to the origin. This facilitates inverse Fourier transforms using both real-valued representations and analytic, or complex-envelope representations. See the notes on :doc:`Fourier transforms </notes/fourier_transforms>` for more details.


Pulses
------
Optical pulses are represented in PyNLO with :py:class:`pynlo.light.Pulse` objects. These contain methods for accessing the time and frequency domain properties of the pulse and the underlying time and frequency grids. There are several convenience methods for generating pulses from pre-defined shapes (:py:meth:`~pynlo.light.Pulse.Sech`, :py:meth:`~pynlo.light.Pulse.Gaussian`, etc.) and from existing spectral data (:py:meth:`~pynlo.light.Pulse.FromPowerSpectrum`).


Modes
-----
PyNLO represents the modes within an optical medium as :py:class:`pynlo.medium.Mode` objects. In addition to the linear properties (i.e. the phase and gain coefficients :math:`\beta` and :math:`\alpha`), they also contain an optical mode's effective 2nd- and 3rd-order nonlinearity. If a material has properties that are :math:`z` dependent, those properties can be input as functions who's first argument is the propagation distance.


Models
------
The propagation models implemented in PyNLO are found in :py:mod:`pynlo.model`. While derived through modal expansion of Maxwell's equations (see the notes on :doc:`nonlinear optics </notes/nonlinear_optics>` for more details), only single-mode simulations are currently supported. Models are initialized with :py:class:`~pynlo.light.Pulse` and :py:class:`~pynlo.medium.Mode` objects. Simulation specific parameters (propagation distance, local error, etc.) should be entered when running a model's :py:meth:`~pynlo.model.Model.simulate` method. Real-time visualizations of a simulation are available by setting the `simulate` method's `plot` keyword.


Utilities
---------
The :py:mod:`pynlo.utility` module includes helper and calculator-type functions as well as other miscellaneous items such as the :py:class:`~pynlo.utility.TFGrid` class, functions for evaluating :py:func:`~pynlo.utility.taylor_series`, calculating :py:func:`~pynlo.utility.vacuum` noise, etc. The utilities are organized into submodules based on their relationship to the linear (:py:mod:`~pynlo.utility.chi1`) or nonlinear (:py:mod:`~pynlo.utility.chi2` and :py:mod:`~pynlo.utility.chi3`) susceptibilities. The fast Fourier transforms used in PyNLO are defined in the :py:mod:`~pynlo.utility.fft` submodule.
