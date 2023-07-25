# -*- coding: utf-8 -*-
"""
Miscellaneous helper classes and functions.

"""

__all__ = []


# %% Imports

import numpy as np


# %% Helper Functions

def replace(array, values, key):
    """Copy `array` and replace the elements at `key` with `values`."""
    array = array.copy()
    array[key] = values
    return array


# %% Array Properties for Classes

class NDArrayDescriptor(np.lib.mixins.NDArrayOperatorsMixin):
    """Emulates an NDArray using custom item getters and item setters."""
    def __init__(self, getter=None, setter=None):
        self._getter = getter
        self._setter = setter

    def __getitem__(self, key):
        # Return a view of the array descriptor
        def item_getter(new_key):
            return self._getter(key)[new_key]
        def item_setter(new_key, value):
            self._setter(key, replace(self._getter(key), value, new_key))
        return NDArrayDescriptor(getter=item_getter, setter=item_setter)

    def __setitem__(self, key, value):
        self._setter(key, value)

    def __array__(self, dtype=None):
        # Return the NumPy array
        array = self._getter(...)
        if dtype is None:
            return array
        else:
            return array.astype(dtype=dtype)

    def __repr__(self):
        return repr(self.__array__())

    def __len__(self):
        return len(self.__array__())

    def __copy__(self):
        return self.__array__()

    def __deepcopy__(self, memo):
        return self.__array__().copy()

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        """
        Implemented to support use of the `out` ufunc keyword and seamless
        conversion to NumPy arrays.

        Modified from NumPy docs, "__array_ufunc__ for ufuncs"

        """
        #---- Convert Input to NumPy Arrays
        inputs = tuple(x.__array__() if isinstance(x, NDArrayDescriptor) else x
                       for x in inputs)

        #---- Apply Ufunc
        if out:
            # Convert Output to NumPy Arrays
            outputs = []
            out_args = []
            for idx, output in enumerate(out):
                if isinstance(output, NDArrayDescriptor):
                    outputs.append([idx, output])
                    out_args.append(output.__array__())
                else:
                    out_args.append(output)
            kwargs['out'] = tuple(out_args)

            # Apply Ufunc
            result = getattr(ufunc, method)(*inputs, **kwargs)

            # Write In-Place Output to NDArrayDescriptor
            for idx, output in outputs:
                output[...] = out_args[idx] # "in place" equivalent
        else:
            result = getattr(ufunc, method)(*inputs, **kwargs)

        #---- Return Result
        if method == 'at':
            return None # no return value
        else:
            return result

    def __getattr__(self, attr):
        """Catch-all for other NumPy functions"""
        return getattr(self.__array__(), attr)


class ndproperty(property):
    """
    A subclass of `property` that allows extending the getter and setter
    formalism to NumPy array elements.

    Notes
    -----
    To allow usage of both `__get__`/`__getitem__` and `__set__`/`__setitem__`,
    the methods fed into `ndproperty` must contain a keyword argument and logic
    for processing the keys used by `__getitem__` and `__setitem__`. In the
    `setter` method, the `value` parameter must precede the `key` parameter. In
    the following example, the default key is an open slice (ellipsis), the
    entire array is retrieved when individual elements are not requested.::

        class C(object):
            def __init__(self):
                self.x = np.array([1,2,3,4])

            @ndproperty
            def y(self, key=...):
                return self.x[key]**2

            @y.setter
            def y(self, value, key=...):
                self.x[key] = value**0.5

    See the documentation of `property` for other implementation details.

    """
    def __get__(self, obj, objtype):
        # Return self if not instantiated
        if obj is None:
            return self

        # Define Item Getter and Setter
        def item_getter(key):
            return self.fget(obj, key)

        def item_setter(key, value):
            if self.fset is None:
                self.__set__(obj, value) # raise AttributeError if fset is None
            self.fset(obj, value, key)

        # Return ndarray with custom item getters and item setters
        array = NDArrayDescriptor(getter=item_getter, setter=item_setter)
        return array
