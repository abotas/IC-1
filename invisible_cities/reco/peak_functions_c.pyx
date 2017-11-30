"""
code: peak_functions_c.pyx
description: Cython peak finding functions.

credits: see ic_authors_and_legal.rst in /doc

last revised: JJGC, July-2017
"""
cimport numpy as np
import  numpy as np
from scipy import signal

from .. evm.pmaps import S1
from .. evm.pmaps import S2
from .. evm.pmaps import S2Si
from .. evm.pmaps import S1Pmt
from .. evm.pmaps import S2Pmt

from .. core.exceptions        import InitializedEmptyPmapObject
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


cpdef wfzs(double [:] wf, double threshold=0):
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
    cdef int len_wf = wf.shape[0]
    cdef double [:] wfzs_e = np.zeros(len_wf, dtype=np.double)
    cdef int    [:] wfzs_i = np.zeros(len_wf, dtype=np.int32)

    cdef int i,j
    j = 0
    for i in range(len_wf):
        if wf[i] > threshold:
            wfzs_e[j] = wf[i]
            wfzs_i[j] =    i
            j += 1

    cdef double [:] wfzs_ene  = np.zeros(j, dtype=np.double)
    cdef int    [:] wfzs_indx = np.zeros(j, dtype=np.int32)

    for i in range(j):
        wfzs_ene [i] = wfzs_e[i]
        wfzs_indx[i] = wfzs_i[i]

    return np.asarray(wfzs_ene), np.asarray(wfzs_indx)


cpdef _time_from_index(int [:] indx):
    """
    returns the times (in ns) corresponding to the indexes in indx
    """
    cdef int len_indx = indx.shape[0]
    cdef double [:] tzs = np.zeros(len_indx, dtype=np.double)

    cdef int i
    cdef double step = 25 #ns
    for i in range(len_indx):
        tzs[i] = step * float(indx[i])

    return np.asarray(tzs)


cpdef _select_peaks_of_allowed_length(dict peak_bounds_temp, length):
    """
    Given a dictionary, pbounds, mapping potential peak number to potential peak, return a
    dictionary, bounds, mapping peak numbers (consecutive and starting from 0) to those peaks in
    pbounds of allowed length.
    """

    cdef int j = 0
    cdef dict peak_bounds = {}
    cdef int [:] bound_temp
    for bound_temp in peak_bounds_temp.values():
        if length.min <= bound_temp[1] - bound_temp[0] < length.max:
            peak_bounds[j] = bound_temp
            j+=1
    return peak_bounds


cpdef find_peaks(int [:] index, time, length, int stride=4):
    cdef double tmin, tmax
    cdef double [:] T = _time_from_index(index)
    cdef dict peak_bounds  = {}
    cdef int i, j, i_i, i_min
    cdef np.ndarray where_after_tmin
    tmin, tmax = time
    lmin, lmax = length

    i_min = tmin / (25*units.ns)                                # index in csum of tmin
    where_after_tmin = np.where(np.asarray(index) >= i_min)[0]  # where, in index, time is >= tmin
    if len(where_after_tmin) == 0 : return {}                   # find no peaks in this case
    i_i = where_after_tmin.min()                                # index in index of tmin (or first
                                                                # time not threshold suppressed)
    peak_bounds[0] = np.array([index[i_i], index[i_i] + 1], dtype=np.int32)

    j = 0
    for i in range(i_i + 1, len(index)):
        assert T[i] > tmin
        if T[i] > tmax: break
        # New peak_bounds, create new start and end index
        elif index[i] - stride > index[i-1]:
            j += 1
            peak_bounds[j] = np.array([index[i], index[i] + 1], dtype=np.int32)
        # Update end index in current peak_bounds
        else: peak_bounds[j][1] = index[i] + 1

    return _select_peaks_of_allowed_length(peak_bounds, length)


cpdef rebin_responses(double[:] times, double[:, :] pmt_wfs, int rebin_stride):
    int number_of_bins = np.ceil(len(times) // rebin_stride).astype(int)

    double [:]    rebinned_times
    double [:, :] rebinned_wfs
    double [:]    t
    double [:]    e

    for i in range(number_of_bins):
        s  = slice(rebin_stride * i, rebin_stride * (i + 1))
        t  = times[s]
        e  = pmt_wfs[:, s]
        rebinned_times[i]    = np.average(t, weights=e)
        rebinned_wfs  [:, i] = np.sum(e, axis=1)
    return rebinned_times, rebinned_wfs


cpdef filter_out_empty_sipms(double[:, :] sipm_wfs, double thr):
    cdef int   [:]    selected_ids
    cdef double[:, :] selected_wfs

    selected_ids = np.where(sipm_wfs.sum(axis=1) >= thr)[0]
    selected_wfs = sipm_wfs[selected_ids]
    return selected_ids, selected_wfs


cpdef find_s1s(int   [:]    index,
               double[:]    times,
               double[:, :] ccwf,
               time, length,
               int stride):
    cdef double [:]    pk_times
    cdef double [:, :] pmt_wfs
    cdef list          peaks   = []
    cdef int    [:]    pmt_ids = np.arange(ccwf.shape[0])

    s1_peaks = find_peaks(index, time, length, stride)
    for peak_no, pmt_indices in s1_peaks.items():
        pk_times = times[   slice(*pmt_indices)]
        pmt_wfs  = ccwf [:, slice(*pmt_indices)]
        pmt_r    = PMTResponse(pmt_ids, pmt_wfs)
        pk       = S1(pk_times, pmt_r, empty_sipm_response)
        peaks.append(pk)
    return peaks


cpdef find_s2s(int   [:]    index,
               double[:]    times,
               double[:, :] ccwf,
               double[:, :] sipmzs,
               time, length,
               int stride,
               int rebin_stride):
    cdef double [:]    pk_times
    cdef double [:, :] pmt_wfs
    cdef list          peaks   = []
    cdef int    [:]    pmt_ids = np.arange(ccwf.shape[0])

    s2_peaks = find_peaks(index, time, length, stride)
    for peak_no, pmt_indices in s2_peaks.items():
        pk_times = times[   slice(*pmt_indices)]
        pmt_wfs  = ccwf [:, slice(*pmt_indices)]

        (pk_times,
         pmt_wfs) = rebin_responses(pk_times, pmt_wfs, rebin_stride)

        sipm_indices = tuple(i // rebin_stride for i in pmt_indices)
        sipm_wfs     = sipmzs[:, slice(*sipm_indices)]
        (sipm_ids,
         sipm_wfs)   = filter_out_empty_sipms(sipm_wfs, thr_sipm_s2)

        pmt_r  =  PMTResponses( pmt_ids,  pmt_wfs)
        sipm_r = SiPMResponses(sipm_ids, sipm_wfs)
        pk     = S2(pk_times, pmt_r, sipm_r)
        peaks.append(pk)
    return peaks


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


cpdef select_sipm(double [:, :] sipmzs, double eps=1e-3):
    """
    Selects the SiPMs with signal
    and returns a dictionary:
    input: sipmzs[i,k], where:
           i: runs over number of SiPms (with signal)
           k: runs over waveform.

           sipmzs[i,k] only includes samples above
           threshold (e.g, dark current threshold)

    returns {j: [i, sipmzs[i]]}, where:
           j: enumerates sipms with psum >0
           i: sipm ID
    """
    cdef int NSIPM = sipmzs.shape[0]
    cdef int NWFM = sipmzs.shape[1]
    cdef dict SIPM = {}
    cdef int i, j, k
    cdef double psum

    selected_ids, selected_wfs = filter_out_empty_sipms(sipmzs, eps)
    return dict(zip(selected_ids, selected_wfs))
