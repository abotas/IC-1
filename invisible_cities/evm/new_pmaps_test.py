
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
