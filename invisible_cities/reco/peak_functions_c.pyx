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


cpdef calibrated_pmt_sum(double [:, :]  CWF,
                         double [:]     adc_to_pes,
                         list           pmt_active = [],
                         int            n_MAU = 100,
                         double         thr_MAU =   3):
    """
    Computes the ZS calibrated sum of the PMTs
    after correcting the baseline with a MAU to suppress low frequency noise.
    input:
    CWF:    Corrected waveform (passed by BLR)
    adc_to_pes: a vector with calibration constants
    pmt_active: a list of active PMTs
    n_MAU:  length of the MAU window
    thr_MAU: treshold above MAU to select sample
    """

    cdef int j, k
    cdef int NPMT = CWF.shape[0]
    cdef int NWF  = CWF.shape[1]
    cdef double [:] MAU = np.array(np.ones(n_MAU),
                                   dtype = np.double) * (1 / n_MAU)


    # CWF if above MAU threshold
    cdef double [:, :] pmt_thr  = np.zeros((NPMT,NWF), dtype=np.double)
    cdef double [:]    csum     = np.zeros(      NWF , dtype=np.double)
    cdef double [:]    csum_mau = np.zeros(      NWF , dtype=np.double)
    cdef double [:]    MAU_pmt  = np.zeros(      NWF , dtype=np.double)

    cdef list PMT = list(range(NPMT))
    if len(pmt_active) > 0:
        PMT = pmt_active

    for j in PMT:
        # MAU for each of the PMTs, following the waveform
        MAU_pmt = signal.lfilter(MAU, 1, CWF[j,:])

        for k in range(NWF):
            if CWF[j,k] >= MAU_pmt[k] + thr_MAU: # >= not >: found testing!
                pmt_thr[j,k] = CWF[j,k]

    for j in PMT:
        for k in range(NWF):
            csum_mau[k] += pmt_thr[j, k] * 1 / adc_to_pes[j]
            csum[k] += CWF[j, k] * 1 / adc_to_pes[j]

    return np.asarray(csum), np.asarray(csum_mau)


cpdef calibrated_pmt_mau(double [:, :]  CWF,
                         double [:]     adc_to_pes,
                         list           pmt_active = [],
                         int            n_MAU = 200,
                         double         thr_MAU =   5):
    """
    Returns the calibrated waveforms for PMTs correcting by MAU.
    input:
    CWF:    Corrected waveform (passed by BLR)
    adc_to_pes: a vector with calibration constants
    list: list of active PMTs
    n_MAU:  length of the MAU window
    thr_MAU: treshold above MAU to select sample
    """

    cdef int j, k
    cdef int NPMT = CWF.shape[0]
    cdef int NWF  = CWF.shape[1]
    cdef list PMT = list(range(NPMT))
    if len(pmt_active) > 0:
        PMT = pmt_active


    cdef double [:] MAU = np.array(np.ones(n_MAU),
                                   dtype = np.double) * (1 / n_MAU)

    # CWF if above MAU threshold
    cdef double [:, :] pmt_thr  = np.zeros((NPMT, NWF), dtype=np.double)
    cdef double [:, :] pmt_thr_mau  = np.zeros((NPMT, NWF), dtype=np.double)
    cdef double [:]    MAU_pmt  = np.zeros(      NWF, dtype=np.double)

    for j in PMT:
        # MAU for each of the PMTs, following the waveform
        MAU_pmt = signal.lfilter(MAU, 1, CWF[j,:])

        for k in range(NWF):
            pmt_thr[j,k] = CWF[j,k] * 1 / adc_to_pes[j]
            if CWF[j,k] >= MAU_pmt[k] + thr_MAU: # >= not >: found testing!
                pmt_thr_mau[j,k] = CWF[j,k] * 1 / adc_to_pes[j]

    return np.asarray(pmt_thr), np.asarray(pmt_thr_mau)


cpdef find_peak_bounds(np.ndarray[np.int32_t, ndim=1] index,
                       time, length, int stride=4):
    cdef double tmin, tmax
    cdef double [:] T = _time_from_index(index)
    cdef list peak_bounds  = []
    cdef int i, j, i_i, i_min
    cdef np.ndarray where_after_tmin
    tmin, tmax = time
    lmin, lmax = length

    i_min = tmin / (25 * units.ns)                 # index in csum of tmin
    where_after_tmin = np.where(index >= i_min)[0] # where, in index, time is >= tmin
    if len(where_after_tmin) == 0 : return []      # find no peaks in this case
    i_i = where_after_tmin.min()                   # index in index of tmin (or first
                                                   # time not threshold suppressed)
    peak_bounds.append(np.array([index[i_i], index[i_i] + 1], dtype=np.int32))


    j = 0
    for i in range(i_i + 1, len(index)):
        assert T[i] > tmin
        if T[i] > tmax: break
        # New peak_bounds, create new start and end index
        elif index[i] - stride > index[i-1]:
            j += 1
            peak_bounds.append(np.array([index[i], index[i] + 1], dtype=np.int32))
        # Update end index in current peak_bounds
        else: peak_bounds[j][1] = index[i] + 1

    return list(filter(lambda b: length.contains(np.diff(b)[0]), peak_bounds))


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


cpdef signal_sipm(np.ndarray[np.int16_t, ndim=2] SIPM,
                  double [:] adc_to_pes, thr,
                  int n_MAU=100):
    """
    subtracts the baseline
    Uses a MAU to set the signal threshold (thr, in PES)
    returns ZS waveforms for all SiPMs

    """

    cdef int j, k
    cdef double [:, :] SiWF = SIPM.astype(np.double)
    cdef int NSiPM = SiWF.shape[0]
    cdef int NSiWF = SiWF.shape[1]
    cdef double [:] MAU = np.array(np.ones(n_MAU),
                                   dtype = np.double) * (1 / n_MAU)

    cdef double [:, :] siwf = np.zeros((NSiPM, NSiWF), dtype=np.double)
    cdef double [:]    MAU_ = np.zeros(        NSiWF , dtype=np.double)
    cdef double [:]    thrs = np.full ( NSiPM, thr)
    cdef double pmean

    # loop over all SiPMs. Skip any SiPM with adc_to_pes constant = 0
    # since this means SiPM is dead
    for j in range(NSiPM):
        if adc_to_pes[j] == 0:
            #print('adc_to_pes[{}] = 0, setting sipm waveform to zero'.format(j))
            continue

        # compute and subtract the baseline
        pmean = 0
        for k in range(NSiWF):
            pmean += SiWF[j,k]
        pmean /= NSiWF

        for k in range(NSiWF):
            SiWF[j,k] = SiWF[j,k] - pmean

        # MAU for each of the SiPMs, following the ZS waveform
        MAU_ = signal.lfilter(MAU, 1, SiWF[j,:])

        # threshold using the MAU
        for k in range(NSiWF):
            if SiWF[j,k]  > MAU_[k] + thrs[j] * adc_to_pes[j]:
                siwf[j,k] = SiWF[j,k] / adc_to_pes[j]

    return np.asarray(siwf)
