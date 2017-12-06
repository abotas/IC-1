"""
code: peak_functions.py

description: functions related to the pmap creation.

credits: see ic_authors_and_legal.rst in /doc

last revised: @abotas & @gonzaponte. Dec 1st 2017
"""

import numpy        as np
import scipy.signal as signal

from .. core.system_of_units_c import units
from .. evm .new_pmaps         import S1
from .. evm .new_pmaps         import S2
from .. evm .new_pmaps         import PMap
from .. evm .new_pmaps         import PMTResponses
from .. evm .new_pmaps         import SiPMResponses


def calibrate_pmts(cwfs, adc_to_pes, active,
                   n_MAU = 100, thr_MAU = 3):
    adc_to_pes = adc_to_pes.reshape(adc_to_pes.size, 1)
    active     = active    .reshape(    active.size, 1)
    MAU        = np.full(n_MAU, 1 / n_MAU)
    mau        = signal.lfilter(MAU, 1, CWF, axis=1)

    # ccwf stands for calibrated corrected waveform
    ccwf       = np.where(                     active, cwf/adc_to_pes, 0)
    ccwf_zs    = np.where((cwf >= mau + thr) & active, cwf/adc_to_pes, 0)

    cwf_sum    = np.sum(ccwf   , axis=0)
    cwf_sum_zs = np.sum(ccwf_zs, axis=0)
    return ccwf, ccwf_zs, cwf_sum, cwf_sum_zs


def calibrate_sipms(rwfs, adc_to_pes, thr, n_MAU=100):
    unifm = np.full(rwfs.shape[1], 1/n_MAU)
    blswf = rwfs.T - np.mean(rwfs ,           axis=1)
    mau   = signal.lfilter  (unifm, 1, blswf, axis=1)
    calwf = np.where(blswf > mau + thr * adc_to_pes, blswf / adc_to_pes, 0)
    return calwf.T


def indices_and_wf_above_threshold(wf, thr):
    indices_above_thr = np.where(wf >= thr)[0]
    wf_above_thr      = wf[indices_above_thr]
    return indices_above_thr, wf_above_thr


def select_sipms_above_time_integrated_thr(sipm_wfs, thr):
    selected_ids = np.where(np.sum(sipm_wfs, axis=1) >= thr)[0]
    selected_wfs = sipm_wfs[selected_ids]
    return selected_ids, selected_wfs


def split_in_peaks(indices, stride):
    where = np.where(np.diff(indices) > stride)[0]
    return np.split(indices, where + 1)


def select_peaks(peaks, time, length):
    def is_valid(indices):
        return (time  .contains(indices[ 0] * 25 * units.ns) and
                time  .contains(indices[-1] * 25 * units.ns) and
                length.contains(indices.size))
    return list(filter(is_valid, peaks))


def extract_peak_from_wfs(indices, times, wfs, rebin_stride=1):
    slice_           = slice(indices)
    pk_times_        = times[   slice_] if times is not None else None
    wfs_             = wfs  [:, slice_]
    pk_times, sr_wfs = rebin_times_and_waveforms(pk_times_, wfs_, rebin_stride)
    return pk_times, sr_wfs


def get_sipm_responses(indices, sipm_wfs, thr_sipm_s2):
    if sipm_wfs is not None:
        _       , sipm_r_wfs_ = extract_peak_from_wfs(si_indices, None, sipm_wfs)
        sipm_ids, sipm_r_wfs  = select_sipms_above_time_integrated_thr(sipm_r_wfs_, thr_sipm_s2)
        return sipm_r         = SiPMResponses(sipm_ids, sipm_r_wfs)
    return None


def find_peaks(ccwf, index,
               time, length,
               stride, rebin_stride,
               Pk, pmt_ids=None,
               sipm_wfs=None, thr_sipm_s2=0):
    if ccwf.ndim ==  1:    ccwf = ccwf[np.newaxis]
    if pmt_ids is None: pmt_ids = np.arange(ccwf.shape[0])

    peaks           = []
    times           = np.arange(ccwf.shape[1]) * 25 * units.ns
    indices_split   = split_in_peaks(index, stride)
    selected_splits = select_peaks  (indices_split, time, length)

    for peak_no, indices in selected_splits.items():
        pk_times, pmt_wfs = extract_peak_from_wfs(indices, times, ccwf, rebin_stride)
        pmt_r  = PMTResponses(pmt_ids, pmt_wfs)

        si_indices  = tuple(index // rebin_stride for index in indices)
        sipm_r = get_sipm_responses(si_indices, ipm_wfs, thr_sipm_s2)


        pk = Pk(pk_times, pmt_r, sipm_r)
        peaks.append(pk)
    return peaks


def get_pmap(ccwf, s1_indx, s2_indx, sipm_zs_wf, s1_params, s2_params):
    return PMap(find_peaks(ccwf, s1_indx, Pk=S1,                        **s1_params),
                find_peaks(ccwf, s2_indx, Pk=S2, sipm_zs_wf=sipm_zs_wf, **s2_params))


def rebin_times_and_waveforms(times, waveforms, rebin_stride):
    if rebin_stride < 2: return times, waveforms

    n_bins    = int(np.ceil(len(times) / rebin_stride))
    n_sensors = waveforms.shape[0]

    rebinned_times = np.zeros(            n_bins , dtype=np.float32)
    rebinned_wfs   = np.zeros((n_sensors, n_bins), dtype=np.float32)

    for i in range(n_bins):
        s  = slice(rebin_stride * i, rebin_stride * (i + 1))
        t  = times    [   s]
        e  = waveforms[:, s]
        w  = np.sum(e, axis=0) if np.any(e) else None
        rebinned_times[   i] = np.average(t, weights=w)
        rebinned_wfs  [:, i] = np.sum    (e,    axis=1)
    return rebinned_times, rebinned_wfs
