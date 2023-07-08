Examples
========
The PyNLO workflow typically starts with a :py:class:`~pynlo.light.Pulse`, as this defines a set of time and frequency grids that you can use throughout the rest of your project. The next step is to define a :py:class:`~pynlo.media.Mode`, which holds the effective linear and nonlinear properties of your fiber, waveguide, or bulk medium. With a `Pulse` and `Mode` object you can initialize a propagation model. Use the :py:class:`~pynlo.model.NLSE` if you only need the Kerr and Raman nonlinearities. If you want to simulate 2nd-order nonlinearities, or combine 2nd- and 3rd-order effects, use the :py:class:`~pynlo.model.UPE`.

The examples below explore various nonlinear effects and are organized by nonlinear propagation model used in the simulation.

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :caption: NLSE

   examples/silica-pcf_supercontinuum
   examples/optical-solitons


.. toctree::
   :maxdepth: 1
   :titlesonly:
   :caption: UPE

   examples/ppln_cascaded-chi2
   examples/phase-matching
