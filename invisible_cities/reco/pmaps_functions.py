"""
Last revised, @abotas and @gonzaponte. Nov 30th 2017
"""

from .. reco           import peak_functions_c as pf
from .. evm .new_pmaps import S1
from .. evm .new_pmaps import S2
from .. evm .new_pmaps import PMap


def get_pmap(s1_indx, s2_indx, ccwf, csum, sipmzs, s1_params, s2_params, thr_sipm_s2):
    """Computes s1, s2, s2si, s1pmt and s2pmt objects"""
    s1s = pf.find_peaks(ccwf, csum, s1_indx, S1,                **s1_params._asdict())
    s2s = pf.find_peaks(ccwf, csum, s2_indx, S2, sipmzs=sipmzs, **s2_params._asdict())
    return PMap(s1s, s2s)
