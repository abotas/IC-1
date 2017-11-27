
import numpy as np

from pytest import approx
from pytest import raises

from hypothesis             import given
from hypothesis.strategies  import integers
from hypothesis.strategies  import floats
from hypothesis.strategies  import sampled_from
from hypothesis.strategies  import composite
from hypothesis.extra.numpy import arrays

from .. core.core_functions import weighted_mean_and_std

from  . new_pmaps import  PMTResponses
from  . new_pmaps import SiPMResponses
from  . new_pmaps import S1
from  . new_pmaps import S2
from  . new_pmaps import PMap


wf_min =   0
wf_max = 100


@composite
def sensor_responses(draw, nsamples=None, type_=None):
    nsensors    = draw(integers(1,  100))
    nsamples    = draw(integers(1, 1000)) if nsamples is None else nsamples
    shape       = nsensors, nsamples
    ids         = draw(arrays(  int, nsensors, integers(0, 1e3), unique=True))
    all_wfs     = draw(arrays(float,    shape, floats  (wf_min, wf_max)))
    PMT_or_SiPM =(draw(sampled_from((PMTResponses, SiPMResponses)))
                  if type_ is None else type_)
    args        = ids, all_wfs
    return args, PMT_or_SiPM(*args)


@composite
def peaks(draw, type_=None):
    nsamples  = draw(integers(1, 1000))
    _, pmt_r  = draw(sensor_responses(nsamples,  PMTResponses))
    _, sipm_r = draw(sensor_responses(nsamples, SiPMResponses))
    times     = draw(arrays(float, nsamples,
                            floats(min_value=0, max_value=1e3),
                            unique = True))
    S1_or_S2  = draw(sampled_from((S1, S2))) if type_ is None else type_
    args      = times, pmt_r, sipm_r
    return args, S1_or_S2(*args)


@composite
def pmaps(draw):
    n_s1 = draw(integers(0, 5))
    n_s2 = draw(integers(0, 5))
    s1s  = tuple(draw(peaks(S1)) for i in range(n_s1))
    s2s  = tuple(draw(peaks(S2)) for i in range(n_s2))
    args = s1s, s2s
    return args, PMap(*args)


@given(sensor_responses())
def test_SensorResponses_all_waveforms(srs):
    (_, all_waveforms), sr = srs
    assert (all_waveforms == sr.all_waveforms).all()


@given(sensor_responses())
def test_SensorResponses_ids(srs):
    (ids, _), sr = srs
    assert (ids == sr.ids).all()


@given(sensor_responses())
def test_SensorResponses_waveform(srs):
    (ids, all_waveforms), sr = srs
    for sensor_id, waveform in zip(ids, all_waveforms):
        assert (waveform == sr.waveform(sensor_id)).all()


@given(sensor_responses())
def test_SensorResponses_time_slice(srs):
    (_, all_waveforms), sr = srs
    for i, time_slice in enumerate(all_waveforms.T):
        assert (time_slice == sr.time_slice(i)).all()


@given(sensor_responses())
def test_SensorResponses_sum_over_times(srs):
    (_, all_waveforms), sr = srs
    assert np.sum(all_waveforms, axis=1) == approx(sr.sum_over_times)


@given(sensor_responses())
def test_SensorResponses_sum_over_sensors(srs):
    (_, all_waveforms), sr = srs
    assert np.sum(all_waveforms, axis=0) == approx(sr.sum_over_sensors)


@given(sampled_from((PMTResponses, SiPMResponses)), integers(1, 10))
def test_SensorResponses_raises_exception_when_shapes_dont_match(SR, a):
    with raises(ValueError):
        sr = SR(np.empty(a),
                np.empty((a + 1, 1)))


def assert_SensorResponses_equality(sr0, sr1):
    # This is sufficient to assert equality since all of SensorResponses other
    # properties depend solely on .all_waveforms, and all of those properties
    # are tested.
    assert (sr0.all_waveforms == sr1.all_waveforms).all()


@given(peaks())
def test_Peak_sipms(pks):
    (_, _, sipm_r), peak = pks
    assert_SensorResponses_equality(sipm_r, peak.sipms)


@given(peaks())
def test_Peak_pmts(pks):
    (_, pmt_r, _), peak = pks
    assert_SensorResponses_equality(pmt_r, peak.pmts)


@given(peaks())
def test_Peak_times(pks):
    (times, _, _), peak = pks
    assert (times == peak.times).all()


@given(peaks())
def test_Peak_time_at_max_energy(pks):
    _, peak = pks
    index_at_max_energy = np.argmax(peak.pmts.sum_over_sensors)
    assert peak.time_at_max_energy == peak.times[index_at_max_energy]


@given(peaks())
def test_Peak_total_energy(pks):
    _, peak = pks
    assert peak.total_energy == approx(peak.pmts.all_waveforms.sum())


@given(peaks())
def test_Peak_total_charge(pks):
    _, peak = pks
    assert peak.total_charge == approx(peak.sipms.all_waveforms.sum())


@given(peaks())
def test_Peak_height(pks):
    _, peak = pks
    assert peak.height == approx(peak.pmts.sum_over_sensors.max())


@given(peaks())
def test_Peak_width(pks):
    _, peak = pks
    assert peak.width == approx(peak.times[-1] - peak.times[0])


def get_indices_above_thr(sr, thr):
    return np.where(sr.sum_over_sensors >= thr)[0]


@given(peaks())
def test_Peak_energy_above_threshold_less_than_equal_to_wf_min(pks):
    _, peak = pks
    sum_wf_min = peak.pmts.sum_over_sensors.min()
    assert peak.energy_above_threshold(sum_wf_min) == approx(peak.total_energy)


@given(peaks())
def test_Peak_energy_above_threshold_greater_than_wf_max(pks):
    _, peak = pks
    assert peak.energy_above_threshold(peak.height + 1) == 0


@given(peaks(), floats(wf_min, wf_max))
def test_Peak_energy_above_threshold(pks, thr):
    _, peak = pks
    i_above_thr = get_indices_above_thr(peak.pmts, thr)
    assert (peak.pmts.sum_over_sensors[i_above_thr].sum() ==
            approx(peak.energy_above_threshold(thr)))


@given(peaks())
def test_Peak_charge_above_threshold_less_than_equal_to_wf_min(pks):
    _, peak = pks
    sum_wf_min = peak.sipms.sum_over_sensors.min()
    assert peak.charge_above_threshold(sum_wf_min) == approx(peak.total_charge)


@given(peaks())
def test_Peak_charge_above_threshold_greater_than_wf_max(pks):
    _, peak = pks
    sipms_max = peak.sipms.sum_over_sensors.max()
    assert peak.charge_above_threshold(sipms_max + 1) == 0


@given(peaks(), floats(wf_min, wf_max))
def test_Peak_charge_above_threshold(pks, thr):
    _, peak = pks
    i_above_thr = get_indices_above_thr(peak.sipms, thr)
    assert (peak.sipms.sum_over_sensors[i_above_thr].sum() ==
            approx(peak.charge_above_threshold(thr)))


@given(peaks())
def test_Peak_width_above_threshold_equal_to_wf_min(pks):
    _, peak = pks
    sum_wf_min = peak.pmts.sum_over_sensors.min()
    assert peak.width == approx(peak.width_above_threshold(sum_wf_min))


@given(peaks())
def test_Peak_width_above_threshold_greater_than_wf_max(pks):
    _, peak = pks
    assert peak.width_above_threshold(peak.height + 1) == 0


@given(peaks(), floats(wf_min, wf_max))
def test_Peak_width_above_threshold(pks, thr):
    _, peak = pks
    i_above_thr     = get_indices_above_thr(peak.pmts, thr)
    times_above_thr = peak.times[i_above_thr]
    expected        = (times_above_thr[-1] - times_above_thr[0]
                       if np.size(i_above_thr) > 0
                       else 0)
    assert peak.width_above_threshold(thr) == approx(expected)


@given(peaks(), floats(wf_min, wf_max))
def test_Peak_rms_above_threshold(pks, thr):
    _, peak = pks
    i_above_thr     = get_indices_above_thr(peak.pmts, thr)
    times_above_thr = peak.times[i_above_thr]
    wf_above_thr    = peak.pmts.sum_over_sensors[i_above_thr]
    expected        = (weighted_mean_and_std(times_above_thr, wf_above_thr)[1]
                       if np.size(i_above_thr) > 1 and np.sum(wf_above_thr) > 0
                       else 0)
    assert peak.rms_above_threshold(thr) == approx(expected)


@given(peaks())
def test_Peak_rms_above_threshold_equal_to_wf_min(pks):
    _, peak = pks
    sum_wf_min = peak.pmts.sum_over_sensors.min()
    assert peak.rms == approx(peak.rms_above_threshold(sum_wf_min))


@given(peaks())
def test_Peak_rms_above_threshold_greater_than_wf_max(pks):
    _, peak = pks
    assert peak.rms_above_threshold(peak.height + 1) == 0


@given(sampled_from((S1, S2)), sensor_responses(), sensor_responses())
def test_Peak_raises_exception_when_shapes_dont_match(PK, sr1, sr2):
    with raises(ValueError):
        (ids, wfs), sr1 = sr1
        _         , sr2 = sr2
        n_samples       = wfs.shape[1]
        pk = PK(np.empty(n_samples + 1), sr1, sr2)


@given(pmaps())
def test_PMap_s1s(pmps):
    (s1s, _), pmp = pmps
    assert pmp.s1s == s1s


@given(pmaps())
def test_PMap_s2s(pmps):
    (_, s2s), pmp = pmps
    assert pmp.s2s == s2s


@given(pmaps())
def test_PMap_number_of_s1s(pmps):
    (s1s, _), pmp = pmps
    assert pmp.number_of_s1s == len(s1s)


@given(pmaps())
def test_PMap_number_of_s2s(pmps):
    (_, s2s), pmp = pmps
    assert pmp.number_of_s2s == len(s2s)
