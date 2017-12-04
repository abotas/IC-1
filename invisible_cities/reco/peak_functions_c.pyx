"""
code: peak_functions_c.pyx
description: Cython peak finding functions.

credits: see ic_authors_and_legal.rst in /doc

last revised: JJGC, July-2017
"""
cimport numpy as np
import  numpy as np
from scipy import signal

from .. core.system_of_units_c import units

cpdef rebin_responses(np.ndarray[np.float32_t, ndim=1] times,
                      np.ndarray[np.float32_t, ndim=2] waveforms,
                      int                              rebin_stride):

    if rebin_stride < 2: return times, waveforms

    cdef int n_bins    = np.ceil(len(times) / rebin_stride).astype(int)
    cdef int n_sensors = waveforms.shape[0]

    cdef np.ndarray[np.float32_t, ndim=1] rebinned_times = np.zeros(            n_bins , dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] rebinned_wfs   = np.zeros((n_sensors, n_bins), dtype=np.float32)

    cdef double [:] t
    cdef double [:] e

    for i in range(n_bins):
        s  = slice(rebin_stride * i, rebin_stride * (i + 1))
        t  = times    [   s]
        e  = waveforms[:, s]
        rebinned_times[   i] = np.average(t, weights=e)
        rebinned_wfs  [:, i] = np.sum    (e,    axis=1)
    return np.asarray(rebinned_times), np.asarray(rebinned_wfs)
