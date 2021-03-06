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

        csum_blr_py, _, _ = pf._calibrated_pmt_sum(
                               pmtblr[event].astype(np.float64),
                               adc_to_pes,
                               pmt_active = pmt_active,
                               n_MAU=100, thr_MAU=3)

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

        csum_blr_py_pmt6, _, _ = pf._calibrated_pmt_sum(
                                    pmtblr[event].astype(np.float64),
                                    adc_to_pes,
                                    pmt_active = list(range(6)),
                                    n_MAU=100, thr_MAU=3)

        CAL_PMT, CAL_PMT_MAU  =  cpf.calibrated_pmt_mau(
                                     CWF,
                                     adc_to_pes,
                                     pmt_active = pmt_active,
                                     n_MAU = 100,
                                     thr_MAU =   3)


        wfzs_ene,    wfzs_indx    = cpf.wfzs(csum_blr,    threshold=0.5)
        wfzs_ene_py, wfzs_indx_py =  pf._wfzs(csum_blr_py, threshold=0.5)

        return (namedtuple('Csum',
                        """cwf cwf6
                           csum_cwf csum_blr csum_blr_py
                           csum_cwf_pmt6 csum_blr_pmt6 csum_blr_py_pmt6
                           CAL_PMT, CAL_PMT_MAU,
                           wfzs_ene wfzs_ene_py
                           wfzs_indx wfzs_indx_py""")
        (cwf               = CWF,
         cwf6              = CWF6,
         csum_cwf          = csum_cwf,
         csum_blr          = csum_blr,
         csum_blr_py       = csum_blr_py,
         csum_cwf_pmt6     = csum_cwf_pmt6,
         csum_blr_pmt6     = csum_blr_pmt6,
         CAL_PMT           = CAL_PMT,
         CAL_PMT_MAU       = CAL_PMT_MAU,
         csum_blr_py_pmt6  = csum_blr_py_pmt6,
         wfzs_ene          = wfzs_ene,
         wfzs_ene_py       = wfzs_ene_py,
         wfzs_indx         = wfzs_indx,
         wfzs_indx_py      = wfzs_indx_py))


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

        csum, pmp = pf._compute_csum_and_pmaps(event,
                                              pmtrwf,
                                              sipmrwf,
                                              s1par,
                                              s2par,
                                              thr,
                                              calib,
                                              deconv)

        _, pmp2 = pf._compute_csum_and_pmaps(event,
                                            pmtrwf,
                                            sipmrwf,
                                            s1par,
                                            s2par._replace(rebin_stride=1),
                                            thr,
                                            calib,
                                            deconv)

    return pmp, pmp2, csum


def test_rebin_waveform():
    """
    Uses a toy wf and and all possible combinations of S12 start and stop times to assure that
    rebin_waveform performs across a wide parameter space with particular focus on edge cases.
    Specifically it checks that:
    1) time bins are properly aligned such that there is an obvious S2-S2Si time bin mapping
        (one to one, onto when stride=40) by computing the expected mean time for each time bin.
    2) the correct energy is distributed to each time bin.
    """

    nmus   =  3
    stride = 40
    bs     = 25*units.ns
    rbs    = stride * bs
    wf     = np.ones (int(nmus*units.mus / (bs))) * units.pes
    times  = np.arange(0, nmus*units.mus,   bs)
    # Test for every possible combination of start and stop time over an nmus microsecond wf
    for s in times:
        for f in times[times > s]:
            # compute the rebinned waveform
            [T, E] = cpf.rebin_waveform(s, f, wf[int(s/bs): int(f/bs)], stride=stride)
            # check the waveforms values...
            for i, (t, e) in enumerate(zip(T, E)):
                # ...in the first time bin
                if i==0:
                    assert np.isclose(t, min(np.mean((s, (s // (rbs) + 1)*rbs)), np.mean((f, s))))
                    assert e == min(((s // (rbs) + 1)*rbs - s) / bs, (f - s) / (bs))
                # ...in the middle time bins
                elif i < len(T) - 1:
                    assert np.isclose(t,
                        np.ceil(T[i - 1] / (rbs))*rbs + stride * bs / 2)
                    assert e == stride
                # ...in the remainder time bin
                else:
                    assert i == len(T) - 1
                    assert np.isclose(t, np.mean((np.ceil(T[i-1] / (rbs))*rbs, f)))
                    assert e == (f - np.ceil(T[i-1] / (rbs)) * rbs) / (bs)


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


def test_length_of_S2_time_array_and_length_of_S2Si_energy_array_must_be_the_same(pmaps_electrons):
    pmp, _, _ = pmaps_electrons

    if pmp.S2 and pmp.S2Si:
        for nsipm in pmp.S2Si[0]:
            assert len(pmp.S2Si[0][nsipm]) == len(pmp.S2[0][0])


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


def test_csum_zs_s12():
    """Several sequencial tests:
    1) Test that csum (the object of the test) and vsum (sum of toy pmt
    waveforms) yield the same result.
    2) Same for ZS sum
    3) Test that time_from_index is the same in python and cython functions.
    4) test that rebin is the same in python and cython functions.
    5) test that find_S12 is the same in python and cython functions.
    """
    v = toy_pmt_signal()
    npmt = 10
    vsum = v * npmt
    CWF, adc_to_pes = toy_cwf_and_adc(v, npmt=npmt)
    csum, _ = cpf.calibrated_pmt_sum(CWF, adc_to_pes, n_MAU=1, thr_MAU=0)
    npt.assert_allclose(vsum, csum)

    vsum_zs = vsum_zsum(vsum, threshold=10)
    wfzs_ene, wfzs_indx = cpf.wfzs(csum, threshold=10)
    npt.assert_allclose(vsum_zs, wfzs_ene)

    t1 = cpf._time_from_index(wfzs_indx)
    t2 = cpf._time_from_index(wfzs_indx)
    i0 = wfzs_indx[0]
    i1 = wfzs_indx[-1] + 1
    npt.assert_allclose(t1, t2)

    t = pf._time_from_index(wfzs_indx)
    e = wfzs_ene


    pt1, pe1 = pf._rebin_waveform(t1[0], t1[-1] + 25*units.ns, csum[i0:i1], stride=40)
    ct2, ce2 = cpf.rebin_waveform(t1[0], t1[-1] + 25*units.ns, csum[i0:i1], stride=40)

    npt.assert_allclose(pt1, ct2)
    npt.assert_allclose(pe1, ce2)

    S12L1 = pf._find_s12(csum, wfzs_indx,
             time   = minmax(0, 1e+6),
             length = minmax(0, 1000000),
             stride=4, rebin_stride=1)

    S12L2 = cpf.find_s12(csum, wfzs_indx,
             time   = minmax(0, 1e+6),
             length = minmax(0, 1000000),
             stride=4, rebin_stride=1)

    #pbs   = cpf.find_peaks(wfzs_indx, time=minmax(0, 1e+6), length=minmax(0, 1000000), stride=4)
    #S12L3 = cpf.extract_peaks_from_waveform(csum, pbs, rebin_stride=1)

    for i in S12L1:
        t1 = S12L1[i][0]
        e1 = S12L1[i][1]
        t2 = S12L2[i][0]
        e2 = S12L2[i][1]
        #t3 = S12L3[i][0]
        #e3 = S12L3[i][1]

        npt.assert_allclose(t1, t2)
        npt.assert_allclose(e1, e2)
        #npt.assert_allclose(t2, t3)
        #npt.assert_allclose(e2, e3)

    # toy yields 3 idential vectors of energy
    E = np.array([ 11,  12,  13,  14,  15,  16,  17,  18,  19,  20,
                   20,  20,  20,  20,  20,  20,  20,  20,  20,  20,
                   20,  19,  18,  17,  16,  15,  14,  13,  12,  11])
    for i in S12L2.keys():
        e = S12L2[i][1]
        npt.assert_allclose(e,E)

    # rebin
    S12L2 = cpf.find_s12(csum, wfzs_indx,
             time   = minmax(0, 1e+6),
             length = minmax(0, 1000000),
             stride=10, rebin_stride=10)

    E = np.array([155,  200,  155])

    for i in S12L2.keys():
        e = S12L2[i][1]
        npt.assert_allclose(e, E)


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


def test_correct_S1_ene_returns_correct_energies(toy_S1_wf):
    S1, wf, indices = toy_S1_wf
    corrS1 = cpf.correct_s1_ene(S1, wf)
    for peak_no, (t, E) in corrS1.s1d.items():
        assert np.all(E == wf[indices[peak_no]])


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


def test_find_peaks_finds_peaks_when_index_spaced_by_less_than_or_equal_to_stride():
    # explore a range of strides
    for stride in range(2,8):
        # for each stride create a index array with ints spaced by (1 to stride)
        for s in range(1, stride + 1):
            # the concatenated np.array checks that find peaks will find separated peaks
            index  = np.concatenate((np.arange(  0, 500, s, dtype=np.int32),
                                     np.arange(600, 605, 1, dtype=np.int32)))
            bounds = cpf.find_peaks(index, time   = minmax(0, 1e+6),
                                           length = minmax(5, 9999),
                                           stride = stride)
            assert len(bounds)  ==    2            # found both peaks
            assert bounds[0][0] ==    0            # find correct start i for first  p
            assert bounds[0][1] == (499//s)*s + 1  # find correct end   i for first  p
            assert bounds[1][0] ==  600            # find correct start i for second p
            assert bounds[1][1] ==  605            # find correct end   i for second p


def test_find_peaks_finds_no_peaks_when_index_spaced_by_more_than_stride():
    for stride in range(2,8):
        index  = np.concatenate((np.arange(  0, 500, stride + 1, dtype=np.int32),
                                 np.arange(600, 605,          1, dtype=np.int32)))
        bounds = cpf.find_peaks(index, time   = minmax(0, 1e+6),
                                       length = minmax(2, 9999),
                                       stride = stride)
        assert len(bounds)  ==    1
        assert bounds[0][0] ==  600
        assert bounds[0][1] ==  605


def test_find_peaks_when_no_index_after_tmin():
    stride = 2
    index = np.concatenate((np.arange(  0, 500, stride, dtype=np.int32),
                            np.arange(600, 605,      1, dtype=np.int32)))
    assert cpf.find_peaks(index, time   = minmax(9e9, 9e10),
                                 length = minmax(2, 9999),
                                 stride = stride) == {}
    assert pf._find_peaks(index, time   = minmax(9e9, 9e10),
                                 length = minmax(2, 9999),
                                 stride = stride) == {}


def test_extract_peaks_from_waveform():
    wf = np.random.uniform(size=52000)
    # Generate peak_bounds
    peak_bounds = {}
    for k in range(100):
        i_start = np.random.randint(52000)
        i_stop  = np.random.randint(i_start + 1, 52001)
        peak_bounds[k] = np.array([i_start, i_stop], dtype=np.int32)
    # Extract peaks
    S12L = cpf.extract_peaks_from_waveform(wf, peak_bounds, rebin_stride=1)
    for k in peak_bounds:
        T = cpf._time_from_index(np.arange(peak_bounds[k][0], peak_bounds[k][1], dtype=np.int32))
        assert np.allclose(S12L[k][0], T)                                         # Check times
        assert np.allclose(S12L[k][1], wf[peak_bounds[k][0]: peak_bounds[k][1]])  # Check energies
    # Check that _extract... did not return extra peaks
    assert len(peak_bounds) == len(S12L)


def test_get_ipmtd():
    npmts  = 4
    ntbins = 52000
    cCWF   = np.random.random(size=(npmts, ntbins)) # generate wfs for pmts
    csum   = cCWF.sum(axis=0)                 # and their sum
    npeaks = 20
    peak_bounds  = {}
    rebin_strides = [1,7,40]

    for pn in range(npeaks):
        start = np.random.randint(      0, ntbins  )
        stop  = np.random.randint(1+start, ntbins+1)
        peak_bounds[pn] = np.array([start, stop], dtype=np.int32) # generate some peak_bounds

    for rebin_stride in rebin_strides:
        # extract the peaks from the csum, and for each pmt extract the peaks from the cCWF
        S12L   = cpf.extract_peaks_from_waveform(csum, peak_bounds, rebin_stride=rebin_stride)
        s12pmtd = cpf.get_ipmtd(cCWF, peak_bounds, rebin_stride=rebin_stride)
        for s12l, s12_pmts, i_peak in zip(S12L.values(), s12pmtd.values(), peak_bounds.values()):
            # check that the sum of the individual pmt s12s equals the total s12, at each time bin
            assert np.allclose(s12l[1], s12_pmts.sum(axis=0))
            # check that the correct energy is in each pmt
            assert np.allclose(s12_pmts.sum(axis=1), cCWF[:, i_peak[0]: i_peak[1]].sum(axis=1))


def test_sum_waveforms():
    wfs = np.random.random((12,1300*40))
    assert np.allclose(pf.sum_waveforms(wfs), np.sum(wfs, axis=0))


@fixture(scope='module')
def toy_ccwfs_and_csum():
    ccwf  = np.zeros((3, 52000), dtype=np.float64)
    psize = 200
    peak  = np.random.normal(loc=10, size=(ccwf.shape[0], psize))
    indx  = np.arange(650, 650+psize, dtype=np.int32)
    ccwf[:, indx[0]: indx[-1] + 1] += peak
    csum  = ccwf.sum(axis=0)
    return indx, ccwf, csum


def test_find_s12_ipmt_find_same_s12_as_find_s12(toy_ccwfs_and_csum):
    indx, ccwf, csum = toy_ccwfs_and_csum
    s10    = cpf.find_s1(csum, indx,
                     time   = minmax(0, 1e+6),
                     length = minmax(0, 1000000),
                     stride = 4,
                     rebin_stride = 1)
    s11, _ = cpf.find_s1_ipmt(ccwf, csum, indx,
                     time   = minmax(0, 1e+6),
                     length = minmax(0, 1000000),
                     stride = 4,
                     rebin_stride = 1)
    s20    = cpf.find_s2(csum, indx,
                     time   = minmax(0, 1e+6),
                     length = minmax(0, 1000000),
                     stride = 4,
                     rebin_stride = 40)
    s21, _ = cpf.find_s2_ipmt(ccwf, csum, indx,
                     time   = minmax(0, 1e+6),
                     length = minmax(0, 1000000),
                     stride = 4,
                     rebin_stride = 0)
    # Check same s1s found
    assert s10.s1d.keys() == s11.s1d.keys()
    for peak0, peak1 in zip(s10.s1d.values(), s11.s1d.values()):
        assert np.allclose(peak0, peak1)
    # Check same s2s found
    assert s20.s2d.keys() == s21.s2d.keys()
    for peak0, peak1 in zip(s10.s1d.values(), s11.s1d.values()):
        assert np.allclose(peak0, peak1)


def test_find_s12_ipmt_return_none_when_empty_index(toy_ccwfs_and_csum):
    indx, ccwf, csum = toy_ccwfs_and_csum
    # Check no s1 found
    for pmap_class in cpf.find_s1_ipmt(ccwf, csum, np.array([], dtype=np.int32),
                         time   = minmax(0, 1e+6),
                         length = minmax(0, 1000000),
                         stride = 4,
                         rebin_stride = 1):
        assert pmap_class is None
    # Check no s2 found
    for pmap_class in cpf.find_s2_ipmt(ccwf, csum, np.array([], dtype=np.int32),
                         time   = minmax(0, 1e+6),
                         length = minmax(0, 1000000),
                         stride = 4,
                         rebin_stride = 1):
        assert pmap_class is None


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
