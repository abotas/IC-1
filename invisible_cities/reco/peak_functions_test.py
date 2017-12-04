"""
code: peak_functions_test.py
description: tests for peak functions.

credits: see ic_authors_and_legal.rst in /doc

last revised: JJGC, 10-July-2017
"""

from collections import namedtuple

import tables        as tb
import numpy         as np
import numpy.testing as npt

from pytest import fixture
from pytest import approx

from hypothesis            import given
from hypothesis            import composite
from hypothesis.strategies import arrays
from hypothesis.strategies import floats

from .. core                   import core_functions   as cf
from .. core.system_of_units_c import units

from .. database               import load_db

from .. sierpe                 import blr

from .                         import peak_functions   as pf
from .                         import peak_functions_c as cpf
from .                         import tbl_functions    as tbl
from .. evm.ic_containers      import S12Params
from .. evm.ic_containers      import ThresholdParams
from .. evm.ic_containers      import DeconvParams
from .. evm.ic_containers      import CalibVectors
from .. types.ic_types         import minmax


# TODO: rethink this test (list(6) could stop working anytime if DataPMT is changed)
@fixture(scope='module')
def csum_zs_blr_cwf(electron_RWF_file):
    """Test that:
     1) the calibrated sum (csum) of the BLR and the CWF is the same
    within tolerance.
     2) csum and zeros-supressed sum (zs) are the same
    within tolerance
    """

    run_number = 0

    with tb.open_file(electron_RWF_file, 'r') as h5rwf:
        pmtrwf, pmtblr, sipmrwf = tbl.get_vectors(h5rwf)
        DataPMT = load_db.DataPMT(run_number)
        pmt_active = np.nonzero(DataPMT.Active.values)[0].tolist()
        coeff_c    = abs(DataPMT.coeff_c.values)
        coeff_blr  = abs(DataPMT.coeff_blr.values)
        adc_to_pes = abs(DataPMT.adc_to_pes.values)

        event = 0
        CWF  = blr.deconv_pmt(pmtrwf[event], coeff_c, coeff_blr, pmt_active)
        CWF6 = blr.deconv_pmt(pmtrwf[event], coeff_c, coeff_blr, list(range(6)))
        csum_cwf, _ =      cpf.calibrated_pmt_sum(
                               CWF,
                               adc_to_pes,
                               pmt_active = pmt_active,
                               n_MAU = 100,
                               thr_MAU =   3)

        csum_blr, _ =      cpf.calibrated_pmt_sum(
                               pmtblr[event].astype(np.float64),
                               adc_to_pes,
                               pmt_active = pmt_active,
                               n_MAU = 100,
                               thr_MAU =   3)

        csum_cwf_pmt6, _ = cpf.calibrated_pmt_sum(
                               CWF,
                               adc_to_pes,
                               pmt_active = list(range(6)),
                               n_MAU = 100,
                               thr_MAU =   3)

        csum_blr_pmt6, _ = cpf.calibrated_pmt_sum(
                               pmtblr[event].astype(np.float64),
                               adc_to_pes,
                               pmt_active = list(range(6)),
                               n_MAU = 100,
                               thr_MAU =   3)

        CAL_PMT, CAL_PMT_MAU  =  cpf.calibrated_pmt_mau(
                                     CWF,
                                     adc_to_pes,
                                     pmt_active = pmt_active,
                                     n_MAU = 100,
                                     thr_MAU =   3)


        wfzs_indx, wfzs_ene   = pf.indices_and_wf_above_threshold(csum_blr, 0.5)

        return (namedtuple('Csum',
                        """cwf cwf6
                           csum_cwf csum_blr
                           csum_cwf_pmt6 csum_blr_pmt6
                           CAL_PMT CAL_PMT_MAU""")
        (cwf               = CWF,
         cwf6              = CWF6,
         csum_cwf          = csum_cwf,
         csum_blr          = csum_blr,
         csum_cwf_pmt6     = csum_cwf_pmt6,
         csum_blr_pmt6     = csum_blr_pmt6,
         CAL_PMT           = CAL_PMT,
         CAL_PMT_MAU       = CAL_PMT_MAU))


@fixture(scope="module")
def toy_S1_wf():
    s1      = {}
    indices = [np.arange(125, 130), np.arange(412, 417), np.arange(113, 115)]
    for i, index in enumerate(indices):
        s1[i] = index * 25., np.random.rand(index.size)

    wf = np.random.rand(1000)
    return s1, wf, indices


def test_csum_cwf_close_to_csum_of_calibrated_pmts(csum_zs_blr_cwf):
    p = csum_zs_blr_cwf

    csum = 0
    for pmt in p.CAL_PMT:
        csum += np.sum(pmt)

    assert np.isclose(np.sum(p.csum_cwf), np.sum(csum), rtol=0.0001)


def test_csum_cwf_close_to_csum_blr(csum_zs_blr_cwf):
    p = csum_zs_blr_cwf
    assert np.isclose(np.sum(p.csum_cwf), np.sum(p.csum_blr), rtol=0.01)


def test_csum_cwf_pmt_close_to_csum_blr_pmt(csum_zs_blr_cwf):
    p = csum_zs_blr_cwf
    assert np.isclose(np.sum(p.csum_cwf_pmt6), np.sum(p.csum_blr_pmt6),
                      rtol=0.01)


def test_csum_cwf_close_to_wfzs_ene(csum_zs_blr_cwf):
    p = csum_zs_blr_cwf
    assert np.isclose(np.sum(p.csum_cwf), np.sum(p.wfzs_ene), rtol=0.1)


def test_csum_blr_close_to_csum_blr_py(csum_zs_blr_cwf):
    p = csum_zs_blr_cwf
    assert np.isclose(np.sum(p.csum_blr), np.sum(p.csum_blr_py), rtol=1e-4)


def test_csum_blr_pmt_close_to_csum_blr_py_pmt(csum_zs_blr_cwf):
    p = csum_zs_blr_cwf
    assert np.isclose(np.sum(p.csum_blr_pmt6), np.sum(p.csum_blr_py_pmt6),
                      rtol=1e-3)


def test_wfzs_ene_close_to_wfzs_ene_py(csum_zs_blr_cwf):
    p = csum_zs_blr_cwf
    assert np.isclose(np.sum(p.wfzs_ene), np.sum(p.wfzs_ene_py), atol=1e-4)


def test_wfzs_indx_close_to_wfzs_indx_py(csum_zs_blr_cwf):
    p = csum_zs_blr_cwf
    npt.assert_array_equal(p.wfzs_indx, p.wfzs_indx_py)


@fixture(scope='module')
def pmaps_electrons(electron_RWF_file):
    """Compute PMAPS for ele of 40 keV. Check that results are consistent."""

    event = 0
    run_number = 0

    s1par = S12Params(time = minmax(min   =  99 * units.mus,
                                    max   = 101 * units.mus),
                      length = minmax(min =   4,
                                      max =  20),
                      stride              =   4,
                      rebin_stride        =   1)

    s2par = S12Params(time = minmax(min   =    101 * units.mus,
                                    max   =   1199 * units.mus),
                      length = minmax(min =     80,
                                      max = 200000),
                      stride              =     40,
                      rebin_stride        =     40)

    thr = ThresholdParams(thr_s1   =  0.2 * units.pes,
                          thr_s2   =  1   * units.pes,
                          thr_MAU  =  3   * units.adc,
                          thr_sipm =  5   * units.pes,
                          thr_SIPM = 30   * units.pes )


    with tb.open_file(electron_RWF_file,'r') as h5rwf:
        pmtrwf, pmtblr, sipmrwf = tbl.get_vectors(h5rwf)
        DataPMT = load_db.DataPMT(run_number)
        DataSiPM = load_db.DataSiPM(run_number)

        calib = CalibVectors(channel_id = DataPMT.ChannelID.values,
                             coeff_blr = abs(DataPMT.coeff_blr   .values),
                             coeff_c = abs(DataPMT.coeff_c   .values),
                             adc_to_pes = DataPMT.adc_to_pes.values,
                             adc_to_pes_sipm = DataSiPM.adc_to_pes.values,
                             pmt_active = np.nonzero(DataPMT.Active.values)[0].tolist())

        deconv = DeconvParams(n_baseline = 28000,
                              thr_trigger = 5)

        #TODO: LOAD THIS SHIT

        # csum, pmp = pf._compute_csum_and_pmaps(event,
        #                                       pmtrwf,
        #                                       sipmrwf,
        #                                       s1par,
        #                                       s2par,
        #                                       thr,
        #                                       calib,
        #                                       deconv)
        #
        # _, pmp2 = pf._compute_csum_and_pmaps(event,
        #                                     pmtrwf,
        #                                     sipmrwf,
        #                                     s1par,
        #                                     s2par._replace(rebin_stride=1),
        #                                     thr,
        #                                     calib,
        #                                     deconv)

    return pmp, pmp2, csum


def test_rebinning_does_not_affect_the_sum_of_S2(pmaps_electrons):
    pmp, pmp2, _ = pmaps_electrons
    np.isclose(np.sum(pmp.S2[0][1]), np.sum(pmp2.S2[0][1]), rtol=1e-05)


def test_sum_of_S2_and_sum_of_calibrated_sum_vector_must_be_close(pmaps_electrons):
    pmp, _, csum = pmaps_electrons
    np.isclose(np.sum(pmp.S2[0][1]), np.sum(csum.csum), rtol=1e-02)


def test_length_of_S1_time_array_must_match_energy_array(pmaps_electrons):
    pmp, _, _ = pmaps_electrons
    if pmp.S1:
        assert len(pmp.S1[0][0]) == len(pmp.S1[0][1])


def test_length_of_S2_time_array_must_match_energy_array(pmaps_electrons):
    pmp, _, _ = pmaps_electrons
    if pmp.S2:
        assert len(pmp.S2[0][0]) == len(pmp.S2[0][1])


def toy_pmt_signal():
    """ Mimick a PMT waveform."""
    v0 = cf.np_constant(200, 1)
    v1 = cf.np_range(1.1, 2.1, 0.1)
    v2 = cf.np_constant(10, 2)
    v3 = cf.np_reverse_range(1.1, 2.1, 0.1)

    v   = np.concatenate((v0, v1, v2, v3, v0))
    pmt = np.concatenate((v, v, v))
    return pmt


def toy_cwf_and_adc(v, npmt=10):
    """Return CWF and adc_to_pes for toy example"""
    CWF = [v] * npmt
    adc_to_pes = np.ones(v.shape[0])
    return np.array(CWF), adc_to_pes


def vsum_zsum(vsum, threshold=10):
    """Compute ZS over vsum"""
    return vsum[vsum > threshold]


def test_find_s12_finds_first_correct_candidate_peak():
    """
    Checks that find_s12 initializes S12[0] (the array defining the boundaries of the
    0th candidate peak) to the correct values
    """
    wf  = np.array([0,0,1,0,0], dtype=np.float64)
    ene = np.array([2], dtype=np.int32)
    S12L = cpf.find_s12(wf, ene,
                 time   = minmax(0, 1e+6),
                 length = minmax(0, 1000000),
                 stride=10, rebin_stride=10)
    assert len(S12L) == 1
    assert np.allclose(S12L[0][0], np.array([2*25*units.ns + 25/2 *units.ns]))
    assert np.allclose(S12L[0][1], np.array([1]))


def test_cwf_are_empty_for_masked_pmts(csum_zs_blr_cwf):
    assert np.all(csum_zs_blr_cwf.cwf6[6:] == 0.)


def test_select_peaks_of_allowed_length():
    pbounds = {}
    length = minmax(5,10)
    for k in range(15):
        i_start = np.random.randint(999, dtype=np.int32)
        i_stop  = i_start + np.random.randint(length.min, dtype=np.int32)
        pbounds[k] = np.array([i_start, i_stop], dtype=np.int32)
    for k in range(15, 30):
        i_start = np.random.randint(999, dtype=np.int32)
        i_stop  = i_start + np.random.randint(length.min, length.max, dtype=np.int32)
        pbounds[k] = np.array([i_start, i_stop], dtype=np.int32)
    for k in range(30, 45):
        i_start = np.random.randint(999, dtype=np.int32)
        i_stop  = i_start + np.random.randint(length.max, 999, dtype=np.int32)
        pbounds[k] = np.array([i_start, i_stop], dtype=np.int32)
    bounds = cpf._select_peaks_of_allowed_length(pbounds, length)
    for i, k in zip(range(15), bounds):
        assert k == i
        l = bounds[k][1] - bounds[k][0]
        assert l >= length.min
        assert l <  length.max


def test_find_peak_bounds_finds_peaks_when_index_spaced_by_less_than_or_equal_to_stride():
    # explore a range of strides
    for stride in range(2,8):
        # for each stride create a index array with ints spaced by (1 to stride)
        for s in range(1, stride + 1):
            # the concatenated np.array checks that find peaks will find separated peaks
            index  = np.concatenate((np.arange(  0, 500, s, dtype=np.int32),
                                     np.arange(600, 605, 1, dtype=np.int32)))
            bounds = cpf.find_peak_bounds(index, time   = minmax(0, 1e+6),
                                           length = minmax(5, 9999),
                                           stride = stride)
            assert len(bounds)  ==    2            # found both peaks
            assert bounds[0][0] ==    0            # find correct start i for first  p
            assert bounds[0][1] == (499//s)*s + 1  # find correct end   i for first  p
            assert bounds[1][0] ==  600            # find correct start i for second p
            assert bounds[1][1] ==  605            # find correct end   i for second p


def test_find_peak_bounds_finds_no_peaks_when_index_spaced_by_more_than_stride():
    for stride in range(2,8):
        index  = np.concatenate((np.arange(  0, 500, stride + 1, dtype=np.int32),
                                 np.arange(600, 605,          1, dtype=np.int32)))
        bounds = cpf.find_peak_bounds(index, time   = minmax(0, 1e+6),
                                       length = minmax(2, 9999),
                                       stride = stride)
        assert len(bounds)  ==    1
        assert bounds[0][0] ==  600
        assert bounds[0][1] ==  605


def test_find_peak_bounds_when_no_index_after_tmin():
    stride = 2
    index = np.concatenate((np.arange(  0, 500, stride, dtype=np.int32),
                            np.arange(600, 605,      1, dtype=np.int32)))
    assert cpf.find_peak_bounds(index, time   = minmax(9e9, 9e10),
                                 length = minmax(2, 9999),
                                 stride = stride) == {}


@fixture(scope='module')
def toy_ccwfs_and_csum():
    ccwf  = np.zeros((3, 52000), dtype=np.float64)
    psize = 200
    peak  = np.random.normal(loc=10, size=(ccwf.shape[0], psize))
    indx  = np.arange(650, 650+psize, dtype=np.int32)
    ccwf[:, indx[0]: indx[-1] + 1] += peak
    csum  = ccwf.sum(axis=0)
    return indx, ccwf, csum


@fixture(scope="session")
def toy_sipm_signal():
    NSIPM = 1792
    WL    = 100

    common_threshold      = np.random.uniform(0.3, 0.7)
    individual_thresholds = np.random.uniform(0.3, 0.7, size=NSIPM)

    adc_to_pes  = np.full(NSIPM, 100, dtype=np.double)
    signal_adc  = np.random.randint(0, 100, size=(NSIPM, WL), dtype=np.int16)

    # subtract baseline and convert to pes
    signal_pes  = signal_adc - np.mean(signal_adc, axis=1)[:, np.newaxis]
    signal_pes /= adc_to_pes[:, np.newaxis]

    signal_zs_common_threshold = np.array(signal_pes)
    signal_zs_common_threshold[signal_pes < common_threshold] = 0

    # thresholds must be reshaped to allow broadcasting
    individual_thresholds_reshaped = individual_thresholds[:, np.newaxis]

    signal_zs_individual_thresholds = np.array(signal_pes)
    signal_zs_individual_thresholds[signal_pes < individual_thresholds_reshaped] = 0

    return (signal_adc, adc_to_pes,
            signal_zs_common_threshold,
            signal_zs_individual_thresholds,
            common_threshold,
            individual_thresholds)


def test_signal_sipm_common_threshold(toy_sipm_signal):
    (signal_adc, adc_to_pes,
     signal_zs_common_threshold, _,
     common_threshold, _) = toy_sipm_signal

    zs_wf = cpf.signal_sipm(signal_adc, adc_to_pes, common_threshold)
    assert np.allclose(zs_wf, signal_zs_common_threshold)


def test_signal_sipm_individual_thresholds(toy_sipm_signal):
    (signal_adc, adc_to_pes,
     _, signal_zs_individual_thresholds,
     _, individual_thresholds) = toy_sipm_signal

    zs_wf = cpf.signal_sipm(signal_adc, adc_to_pes, individual_thresholds)
    assert np.allclose(zs_wf, signal_zs_individual_thresholds)



wf_min = 0
wf_max = 1e3

@composite
def waveforms(draw):
    nwfs     = draw(integers(1,  5))
    nsamples = draw(integers(1, 10))
    shape    = nsensors, nsamples
    return draw(arrays(float, shape, floats(wf_min, wf_max)))


@given(waveforms(), floats())
def test_select_sipms_above_time_integrated_thr(sipm_wfs, thr):
    selected_ids, selected_wfs = pf.select_sipms_above_time_integrated_thr(sipm_wfs, thr)
    where_above_thr = np.where(sipm_wfs.sum(axis=1) >= thr)[0]
    assert (selected_ids == where_above_thr).all()
    assert  selected_wfs == approx(sipm_wfs[where_above_thr])




















#
