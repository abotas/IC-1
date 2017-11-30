"""
code: peak_functions.py
description: peak finding functions. Most of the functions
are private and used only for testing (e.g, comparison with the
corresponding cython functions). Private functions are labelled _function_name
the corresponding public function (to be found in the cython module)
is labelled function_name.

credits: see ic_authors_and_legal.rst in /doc

last revised: JJGC, 10-July-2017
"""


import numpy  as np

from . import peak_functions_c as cpf

from .. evm.new_pmaps import S1
from .. evm.new_pmaps import S2
from .. evm.new_pmaps import PMap
from .. evm.new_pmaps import PMTResponses
from .. evm.new_pmaps import SiPMResponses


def select_sipms_above_time_integrated_thr(sipm_wfs, thr):
    selected_ids = np.where(sipm_wfs.sum(axis=1) >= thr)[0]
    selected_wfs = sipm_wfs[selected_ids]
    return selected_ids, selected_wfs


def find_peaks(index, times, ccwf, time, length, stride, rebin_stride, Pk, sipmzs=None):
    peaks   = []
    pmt_ids = np.arange(ccwf.shape[0])
    sipm_r  = None

    peak_bounds = find_peaks(index, time, length, stride)
    for peak_no, pmt_bounds in peak_bounds.items():
        pk_times_  = times[   slice(*pmt_bounds)]
        pmt_wfs_   = ccwf [:, slice(*pmt_bounds)]
        (pk_times,
         pmt_wfs)  = rebin_responses(pk_times_, pmt_wfs_, rebin_stride)
        pmt_r      = PMTResponses(pmt_ids, pmt_wfs)

        if Pk is S2:
            sipm_bounds = tuple(i // rebin_stride for i in pmt_bounds)
            sipm_wfs    = sipmzs[:, slice(*sipm_bounds)]
            (sipm_ids,
             sipm_wfs)  = select_sipms_above_time_integrated_thr(sipm_wfs, thr_sipm_s2)
            sipm_r     = SiPMResponses(sipm_ids, sipm_wfs)

        pk = Pk(pk_times, pmt_r, sipm_r)
        peaks.append(pk)
    return peaks
