"""Python (private) functions used for testing.

1. _integrate_sipm_charges_in_peak(s2si, peak_number)
Returns (np.array[[nsipm_1 ,     nsipm_2, ...]],
         np.array[[sum(q_1), sum(nsipm_2), ...]])

Last revised, JJGC, July, 2017.

"""
import copy
import numpy  as    np
from .. reco import peak_functions_c  as  pf
from .. reco import peak_functions_c  as cpf
from .. reco import pmaps_functions_c as pmpc
from .. core import core_functions_c  as ccf
from .. core.system_of_units_c      import units
from .. core.exceptions             import NegativeThresholdNotAllowed
from .. evm.new_pmaps               import S1
from .. evm.new_pmaps               import S2
from .. evm.new_pmaps               import PMap

from typing import Dict


def get_pmap(s1_indx, s2_indx, ccwf, csum, sipmzs, s1_params, s2_params, thr_sipm_s2):
    """Computes s1, s2, s2si, s1pmt and s2pmt objects"""
    s1s = pf.find_peaks(ccwf, csum, s1_indx, S1,                **s1_params._asdict())
    s2s = pf.find_peaks(ccwf, csum, s2_indx, S2, sipmzs=sipmzs, **s2_params._asdict())
    return PMap(s1s, s2s)


def rebin_s2si(s2, s2si, rf):
    """given an s2 and a corresponding s2si, rebin them by a factor rf"""
    assert rf >= 1 and rf % 1 == 0
    s2d_rebin   = {}
    s2sid_rebin = {}
    for pn in s2.peaks:
        if pn in s2si.peaks:
            t, e, sipms = rebin_s2si_peak(s2.peaks[pn].t, s2.peaks[pn].E, s2si.s2sid[pn], rf)
            s2sid_rebin[pn] = sipms
        else:
            t, e, _ = rebin_s2si_peak(s2.peaks[pn].t, s2.peaks[pn].E, {}, rf)

        s2d_rebin[pn] = [t, e]

    return S2(s2d_rebin), S2Si(s2d_rebin, s2sid_rebin)


def rebin_s2si_peak(t, e, sipms, stride):
    """rebin: s2 times (taking mean), s2 energies, and s2 sipm qs, by stride"""
    # cython rebin_array is returning memoryview so we need to cast as np array
    return   ccf.rebin_array(t , stride, remainder=True, mean=True), \
             ccf.rebin_array(e , stride, remainder=True)           , \
      {sipm: ccf.rebin_array(qs, stride, remainder=True) for sipm, qs in sipms.items()}


def copy_s2si(s2si_original : S2Si) -> S2Si:
    """ return an identical copy of an s2si. ** note these must be deepcopies, and a deepcopy of
    the s2si itself does not seem to work. """
    return S2Si(copy.deepcopy(s2si_original.s2d),
                copy.deepcopy(s2si_original.s2sid))


def copy_s2si_dict(s2si_dict_original: Dict[int, S2Si]) -> Dict[int, S2Si]:
    """ returns an identical copy of the input s2si_dict """
    return {ev: copy_s2si(s2si) for ev, s2si in s2si_dict_original.items()}


def raise_s2si_thresholds(s2si_dict_original: Dict[int, S2Si],
                         thr_sipm           : float,
                         thr_sipm_s2        : float) -> Dict[int, S2Si]:
    """
    returns s2si_dict after imposing more thr_sipm and/or thr_sipm_s2 thresholds.
    ** NOTE:
        1) thr_sipm IS IMPOSED BEFORE thr_sipm_s2
        2) thresholds cannot be lowered. this function will do nothing if thresholds are set below
           previous values.
    """
    # Ensure thresholds are acceptable values
    if thr_sipm     is None: thr_sipm    = 0
    if thr_sipm_s2  is None: thr_sipm_s2 = 0
    if thr_sipm < 0 or thr_sipm_s2 < 0:
        raise NegativeThresholdNotAllowed('Threshold can be 0 or None, but not negative')
    elif thr_sipm == 0 and thr_sipm_s2 == 0: return s2si_dict_original
    else: s2si_dict = copy_s2si_dict(s2si_dict_original)

    # Impose thresholds
    if thr_sipm    > 0:
        s2si_dict  = pmpc._impose_thr_sipm_destructive   (s2si_dict, thr_sipm   )
    if thr_sipm_s2 > 0:
        s2si_dict  = pmpc._impose_thr_sipm_s2_destructive(s2si_dict, thr_sipm_s2)
    # Get rid of any empty dictionaries
    if thr_sipm > 0 or thr_sipm_s2 > 0:
        s2si_dict  = pmpc._delete_empty_s2si_peaks      (s2si_dict)
        s2si_dict  = pmpc._delete_empty_s2si_dict_events(s2si_dict)
    return s2si_dict
