PyNLO: Python Nonlinear Optics
==============================
This is a fork of the original PyNLO, a package for modeling the nonlinear interaction of light with matter. It started as an attempt to add 2nd-order nonlinearities to the pulse propagation model and grew into a large-scale rewrite. It is not yet at feature parity with the original, but it is getting close! Contributions and suggestions are welcome.


Introduction
------------
The PyNLO package provides an easy-to-use, object-oriented set of tools for modeling the nonlinear interaction of light with matter. It provides many functionalities for representing pulses of light and nonlinear materials.

Features:
	- A solver for the propagation of light through materials with both 2nd- and 3rd-order nonlinearities.

	- A highly-efficient adaptive step size algorithm based on the ERK4(3)-IP method from `Balac and Mahé (2013) <https://doi.org/10.1016/j.cpc.2012.12.020>`_.

	- A flexible object-oriented system for treating laser pulses and optical modes.

	- ...and much more!


Installation
------------
PyNLO requires Python 3. If you do not already have Python, the Anaconda distribution is a good all-in-one collection of Python packages. PyNLO depends on the NumPy, SciPy, Numba, and mkl_fft packages. Matplotlib is necessary if viewing real-time simulation updates or if running the example code.

The easiest way to install this fork is to download or clone the repository into a directory of your choice, and then insert that directory into your `sys.path <https://docs.python.org/3/library/sys.html#sys.path>`_ variable before importing the package. Test out the installation with the scripts in the examples folder.


Contributing
------------
Open a new issue or discussion on the GitHub repository to add suggestions for improvement, ask questions, or make other comments. Contributions to the documentation, tests, and examples are highly appreciated. New additions should be based off the `develop` branch.


License
-------
PyNLO is licensed under the `GNU LGPLv3 license <https://choosealicense.com/licenses/lgpl-3.0/>`_. This means that you are free to use PyNLO for any project, but all modifications to it must be kept open source. PyNLO is provided "as is" with absolutely no warranty.
