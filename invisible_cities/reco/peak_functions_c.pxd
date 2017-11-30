"""
Cython version of some peak functions
JJGC December, 2016

"""
cimport numpy as np
import numpy as np
from scipy import signal


"""
Given a dictionary, pbounds, mapping potential peak number to potential peak, return a
dictionary, bounds, mapping peak numbers (consecutive and starting from 0) to those peaks in
pbounds of allowed length.
"""
cpdef  _select_peaks_of_allowed_length(dict peak_bounds_temp, length)


"""
find_peaks finds the start and stop indices of all the peaks within the time boundaries prescribed
by time.

Note: for now find_peaks cannot be used to find s2si peaks as time associated with indices is
assumed to be index*25ns in time_from_index function
"""
cpdef find_peaks(int [:] index, time, length, int stride=*)



"""
computes the ZS calibrated sum of the PMTs
after correcting the baseline with a MAU to suppress low frequency noise.
input:
CWF:    Corrected waveform (passed by BLR)
adc_to_pes: a vector with calibration constants
pmt_active: a list of active PMTs
n_MAU:  length of the MAU window
thr_MAU: treshold above MAU to select sample
"""
cpdef calibrated_pmt_sum(double [:, :] CWF,
                         double [:] adc_to_pes,
                         list       pmt_active = *,
                         int        n_MAU      = *,
                         double thr_MAU        = *)


"""
Return a vector of calibrated PMTs
after correcting the baseline with a MAU to suppress low frequency noise.
input:
CWF:    Corrected waveform (passed by BLR)
adc_to_pes: a vector with calibration constants
pmt_active: a list of active PMTs
n_MAU:  length of the MAU window
thr_MAU: treshold above MAU to select sample
"""
cpdef calibrated_pmt_mau(double [:, :]  CWF,
                         double [:] adc_to_pes,
                         list       pmt_active = *,
                         int        n_MAU      = *,
                         double     thr_MAU    = *)


"""
takes a waveform wf and returns the values of the wf above threshold:
if the input waveform is of the form [e1,e2,...en],
where ei is the energy of sample i,
then then the algorithm returns a vector [e1,e2...ek],
where k <=n and ei > threshold and
a vector of indexes [i1,i2...ik] which label the position
of the zswf of [e1,e2...ek]
For example if the input waveform is:
[1,2,3,5,7,8,9,9,10,9,8,5,7,5,6,4,1] and the trhesold is 5
then the algoritm returns
a vector of amplitudes [7,8,9,9,10,9,8,7,6] and a vector of indexes
[4,5,6,7,8,9,10,12,14]
"""
cpdef wfzs(double [:] wf, double threshold=*)


"""
returns the times (in ns) corresponding to the indexes in indx
"""
cpdef _time_from_index(int [:] indx)


"""
subtracts the baseline
Uses a MAU to set the signal threshold (thr, in PES)
returns ZS waveforms for all SiPMs
"""
cpdef signal_sipm(np.ndarray[np.int16_t, ndim=2] SIPM,
                  double [:] adc_to_pes, thr,
                  int n_MAU=*)


cpdef rebin_responses(np.ndarray[np.float32_t, ndim=1] times,
                      np.ndarray[np.float32_t, ndim=2] waveforms,
                      int                              rebin_stride)
