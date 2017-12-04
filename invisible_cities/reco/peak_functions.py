"""
code: peak_functions.py

description: functions related to the pmap creation.

credits: see ic_authors_and_legal.rst in /doc

last revised: @abotas & @gonzaponte. Dec 1st 2017
"""


import numpy  as np

from .. core.system_of_units_c import units
from .. evm .new_pmaps         import S1
from .. evm .new_pmaps         import S2
from .. evm .new_pmaps         import PMap
from .. evm .new_pmaps         import PMTResponses
from .. evm .new_pmaps         import SiPMResponses
from .                         import peak_functions_c as cpf


def calibrated_pmt_mau(cwfs, adc_to_pes, active,
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


def indices_and_wf_above_threshold(wf, thr):
    indices_above_thr = np.where(wf >= thr)[0]
    wf_above_thr      = wf[indices_above_thr]
    return indices_above_thr, wf_above_thr


def select_sipms_above_time_integrated_thr(sipm_wfs, thr):
    selected_ids = np.where(sipm_wfs.sum(axis=1) >= thr)[0]
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


def find_peak_bounds(indices, time, length, stride):
    potential_peaks = split_in_peaks(indices, stride)
    return select_peaks(potential_peaks, time, length)


def find_peaks(ccwf, index,
               time, length,
               stride, rebin_stride,
               Pk, sipm_zs_wf=None):
    peaks   = []
    pmt_ids = np.arange(ccwf.shape[0])
    times   = np.arange(ccwf.shape[1]) * 25 * units.ns
    sipm_r  = None

    peak_bounds = cpf.find_peak_bounds(index, time, length, stride)
    for peak_no, pmt_bounds in peak_bounds.items():
        pk_times_  = times[   slice(*pmt_bounds)]
        pmt_wfs_   = ccwf [:, slice(*pmt_bounds)]
        (pk_times,
         pmt_wfs)  = rebin_responses(pk_times_, pmt_wfs_, rebin_stride)
        pmt_r      = PMTResponses(pmt_ids, pmt_wfs)

        if sipm_zs_wf is not None:
            sipm_bounds = tuple(i // rebin_stride for i in pmt_bounds)
            sipm_wfs    = sipm_zs_wf[:, slice(*sipm_bounds)]
            (sipm_ids,
             sipm_wfs)  = select_sipms_above_time_integrated_thr(sipm_wfs, thr_sipm_s2)
            sipm_r     = SiPMResponses(sipm_ids, sipm_wfs)

        pk = Pk(pk_times, pmt_r, sipm_r)
        peaks.append(pk)
    return peaks


def get_pmap(ccwf, s1_indx, s2_indx, sipm_zs_wf, s1_params, s2_params):
    return PMap(find_peaks(ccwf, s1_indx, Pk=S1,                        **s1_params),
                find_peaks(ccwf, s2_indx, Pk=S2, sipm_zs_wf=sipm_zs_wf, **s2_params))


def signal_sipm(rwfs, adc_to_pes, thr, n_MAU=100):
    unifm = np.full(rwfs.shape[1], 1/n_MAU)
    blswf = rwfs.T - np.mean(rwfs ,           axis=1)
    mau   = signal.lfilter  (unifm, 1, blswf, axis=1)
    calwf = np.where(blswf > mau + thr * adc_to_pes, blswf / adc_to_pes, 0)
    return calwf.T



"""
data = np.random.rand(5, 100)
MAU  = np.ones(100)/100.

def with_loop():
    zs  = np.zeros_like(data)
    for i in range(5):
        bls  = np.zeros_like(data[i])
        mean = 0
        for j in range(100):
            mean += data[i,j]
        bls = data[i] - mean/100
        mau = lfilter(MAU, 1, bls)
        for j in range(100):
            if bls[j]  > mau[j] + 0.1:
                zs[i,j] = bls[j]
    return zs


def without_loop():
    bls = data - np.mean(data, axis=1)[:, np.newaxis]
    mau = lfilter(MAU, 1, bls, axis=1)
    zs  = np.where(bls > mau + 0.1, bls, 0)
    return zs


wl  = with_loop()
wol = without_loop()
"""
