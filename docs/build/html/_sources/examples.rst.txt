Examples
========
The PyNLO workflow starts by defining a :py:class:`~pynlo.light.Pulse`. This also provides a set of time and frequency grids you can use throughout the rest of your project. The next step is to define a :py:class:`~pynlo.medium.Mode`, which holds the effective linear and nonlinear properties of your fiber, waveguide, or bulk medium. With a `Pulse` and `Mode` object you can initialize a propagation model. Both 2nd-order and 3rd-order nonlinearities can be simulated using the :py:class:`~pynlo.model.UPE` model. The simpler :py:class:`~pynlo.model.NLSE` model can be used if you only need the 3rd-order Kerr and Raman nonlinearities.

The examples below explore various nonlinear effects and are organized by the nonlinear propagation model used in the simulation.

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :caption: NLSE

   examples/optical-solitons
   examples/silica-pcf_supercontinuum


.. toctree::
   :maxdepth: 1
   :titlesonly:
   :caption: UPE

   examples/phase-matching
   examples/ppln_cascaded-chi2
