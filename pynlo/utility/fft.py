# -*- coding: utf-8 -*-
"""
Aliases to fast FFT implementations and associated helper functions.

"""

__all__ = ["fft", "ifft", "rfft", "irfft",
           "fftshift", "ifftshift"]


# %% Imports

from scipy.fft import next_fast_len, fftshift as _fftshift, ifftshift as _ifftshift
import mkl_fft


# %% Helper Functions

#---- FFT Shifts
def fftshift(x, axis=-1):
    """
    Shift the origin from the beginning to the center of the array.

    This function is used after an `fft` operation to shift from fft to monotic
    ordering.

    Parameters
    ----------
    x : array_like
        Input array
    axis : int, optional
        The axis over which to shift. The default is the last axis.

    Returns
    -------
    ndarray
        The shifted array.

    """
    return _fftshift(x, axes=axis)

def ifftshift(x, axis=-1):
    """
    Shift the origin from the center to the beginning of the array.

    The inverse of fftshift. This function is used before an `fft` operation to
    shift from monotonic to fft ordering. Although identical for even-length
    `x`, `ifftshift` differs from `fftshift` by one sample for odd-length `x`.

    Parameters
    ----------
    x : array_like
        Input array.
    axis : int, optional
        The axis over which to shift. The default is the last axis.

    Returns
    -------
    ndarray
        The shifted array.

    """
    return _ifftshift(x, axes=axis)


# %% Transforms

#---- FFTs
def fft(x, fsc=1.0, n=None, axis=-1, overwrite=None):
    """
    Use MKL to perform a 1D FFT of the input array along the given axis.

    Parameters
    ----------
    x : array_like
        Input array, can be complex.
    fsc : float, optional
        The forward-transform scale factor. The default is 1.0.
    n : int, optional
        Length of the transformed axis of the output. If `n` is smaller than
        the length of the input, the input is cropped. If it is larger, the
        input is padded with zeros.
    axis : int, optional
        Axis over which to compute the FFT. The default is the last axis.
    overwrite : array_like or bool, optional
        The output array used for in-place computation. If True, the contents
        of x will be overwritten. The default is None.

    Returns
    -------
    complex ndarray
        The transformed array.

    """
    if overwrite is True: overwrite = x
    return mkl_fft._pydfti._c2c_fft1d_impl(x, n=n, axis=axis, out=overwrite, direction=1, fsc=fsc)

def ifft(x, fsc=1.0, n=None, axis=-1, overwrite=None):
    """
    Use MKL to perform a 1D IFFT of the input array along the given axis.

    Parameters
    ----------
    x : array_like
        Input array, can be complex.
    fsc : float, optional
        The forward-transform scale factor. Internally, this function sets the
        reverse-transform scale factor as ``1/(n*fsc)``. The default is 1.0.
    n : int, optional
        Length of the transformed axis of the output. If `n` is smaller than
        the length of the input, the input is cropped. If it is larger, the
        input is padded with zeros.
    axis : int, optional
        Axis over which to compute the inverse FFT. The default is the last
        axis.
    overwrite : array_like or bool, optional
        The output array used for in-place computation. If True, the contents
        of x will be overwritten. The default is None.

    Returns
    -------
    complex ndarray
        The transformed array.

    """
    if overwrite is True: overwrite = x
    return mkl_fft._pydfti._c2c_fft1d_impl(x, n=n, axis=axis, out=overwrite, direction=-1, fsc=fsc)

#---- Real FFTs
def rfft(x, fsc=1.0, n=None, axis=-1, overwrite=None):
    """
    Use MKL to perform a 1D FFT of the real input array along the given axis.
    The output array is complex and only contains positive frequencies.

    The length of the transformed axis is ``n//2 + 1``.

    Parameters
    ----------
    x : array_like
        Input array, must be real.
    fsc : float, optional
        The forward-transform scale factor. The default is 1.0.
    n : int, optional
        Number of points to use along the transformed axis of the input. If
        `n` is smaller than the length of the input, the input is cropped. If
        it is larger, the input is padded with zeros.
    axis : int, optional
        Axis over which to compute the FFT. The default is the last axis.
    overwrite : array_like, optional
        The output array used for in-place computation. The default is None.

    Returns
    -------
    complex ndarray
        The transformed array.

    """
    return mkl_fft._pydfti._r2c_fft1d_impl(x, n=n, axis=axis, out=overwrite, fsc=fsc)

def irfft(x, fsc=1.0, n=None, axis=-1, overwrite=None):
    """
    Use MKL to perform a 1D IFFT of the input array along the given axis. The
    input is assumed to contain only positive frequencies, and the output is
    always real.

    If `n` is not given the length of the transformed axis is ``2*(m-1)``,
    where `m` is the length of the transformed axis of the input. To get an odd
    number of output points, `n` must be specified.

    Parameters
    ----------
    x : array_like
        Input array, can be complex.
    fsc : float, optional
        The forward-transform scale factor. Internally, this function sets the
        reverse-transform scale factor as ``1/(n*fsc)``. The default is 1.0.
    n : int, optional
        Length of the transformed axis of the output. For `n` output points,
        ``n//2+1`` input points are necessary. If the input is longer than
        this, it is cropped. If it is shorter than this, it is padded with
        zeros.
    axis : int, optional
        Axis over which to compute the inverse FFT. The default is the last
        axis.
    overwrite : array_like, optional
        The output array used for in-place computation. The default is None.

    Returns
    -------
    ndarray
        The transformed array.

    """
    return mkl_fft._pydfti._c2r_fft1d_impl(x, n=n, axis=axis, out=overwrite, fsc=fsc)
