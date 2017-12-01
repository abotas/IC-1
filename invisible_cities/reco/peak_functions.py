"""
code: peak_functions.py

description: functions related to the pmap creation.

credits: see ic_authors_and_legal.rst in /doc

last revised: @abotas & @gonzaponte. Dec 1st 2017
"""


import numpy  as np

from .. core.system_of_units import units
from .. evm .new_pmaps       import S1
from .. evm .new_pmaps       import S2
from .. evm .new_pmaps       import PMap
from .. evm .new_pmaps       import PMTResponses
from .. evm .new_pmaps       import SiPMResponses
from .                       import peak_functions_c as cpf


def indices_and_wf_above_threshold(wf, thr):
    indices_above_thr = np.where(wf >= thr)[0]
    wf_above_thr      = wf[indices_above_thr]
    return indices_above_thr, wf_above_thr


def select_sipms_above_time_integrated_thr(sipm_wfs, thr):
    selected_ids = np.where(sipm_wfs.sum(axis=1) >= thr)[0]
    selected_wfs = sipm_wfs[selected_ids]
    return selected_ids, selected_wfs


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

        if Pk is S2:
            sipm_bounds = tuple(i // rebin_stride for i in pmt_bounds)
            sipm_wfs    = sipm_zs_wf[:, slice(*sipm_bounds)]
            (sipm_ids,
             sipm_wfs)  = select_sipms_above_time_integrated_thr(sipm_wfs, thr_sipm_s2)
            sipm_r     = SiPMResponses(sipm_ids, sipm_wfs)

        pk = Pk(pk_times, pmt_r, sipm_r)
        peaks.append(pk)
    return peaks


def get_pmap(ccwf, s1_indx, s2_indx, sipm_zs_wf, s1_params, s2_params):
    return PMap(find_peaks(ccwf, s1_indx, Pk=S1,                **s1_params),
                find_peaks(ccwf, s2_indx, Pk=S2, sipm_zs_wf=sipm_zs_wf, **s2_params))
