"""
Cython version of some peak functions
JJGC December, 2016

"""
cimport numpy as np
import numpy as np
from scipy import signal


cpdef rebin_responses(np.ndarray[np.float32_t, ndim=1] times,
                      np.ndarray[np.float32_t, ndim=2] waveforms,
                      int                              rebin_stride)
