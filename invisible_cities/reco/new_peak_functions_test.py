import numpy as np

from pytest import approx

from hypothesis             import given
from hypothesis.strategies  import composite
from hypothesis.strategies  import floats
from hypothesis.strategies  import integers
from hypothesis.extra.numpy import arrays

from ..core.system_of_units_c import units
from ..types.ic_types_c       import minmax
from .                        import new_peak_functions as pf


wf_min =   0
wf_max = 100


@composite
def waveforms(draw):
    n_samples = draw(integers(1, 50))
    return draw(arrays(float, n_samples, floats(wf_min, wf_max)))


@composite
def multiple_waveforms(draw):
    n_sensors = draw(integers(1, 10))
    n_samples = draw(integers(1, 50))
    return draw(arrays(float, (n_sensors, n_samples), floats(wf_min, wf_max)))


@composite
def times_and_waveforms(draw):
    waveforms = draw(multiple_waveforms())
    n_samples = waveforms.shape[1]
    times     = draw(arrays(float, n_samples, floats(0, 10*n_samples), unique=True))
    return times, waveforms


@composite
def peak_indices(draw):
    size    = draw(integers(10, 50))
    indices = draw(arrays(int, size, integers(0, 5 * size), unique=True))
    indices = np.sort(indices)
    stride  = draw(integers(1, 5))
    peaks   = np.split(indices, 1 + np.where(np.diff(indices) > stride)[0])
    return indices, peaks, stride


def test_calibrate_pmts():
    pass


def test_calibrate_sipms():
    pass


@given(waveforms())
def test_indices_and_wf_above_threshold_minus_inf(wf):
    thr = -np.inf
    indices, wf_above_thr = pf.indices_and_wf_above_threshold(wf, thr)
    assert np.all(indices == np.arange(wf.size))
    assert wf_above_thr == approx(wf)


@given(waveforms())
def test_indices_and_wf_above_threshold_min(wf):
    thr = np.min(wf)
    indices, wf_above_thr = pf.indices_and_wf_above_threshold(wf, thr)
    assert np.all(indices == np.arange(wf.size))
    assert wf_above_thr == approx(wf)


@given(waveforms())
def test_indices_and_wf_above_threshold_plus_inf(wf):
    thr = +np.inf
    indices, wf_above_thr = pf.indices_and_wf_above_threshold(wf, thr)
    assert np.size(indices)      == 0
    assert np.size(wf_above_thr) == 0


@given(waveforms())
def test_indices_and_wf_above_threshold_max(wf):
    thr = np.nextafter(np.max(wf), np.inf)
    indices, wf_above_thr = pf.indices_and_wf_above_threshold(wf, thr)

    assert indices     .size == 0
    assert wf_above_thr.size == 0


@given(waveforms(), floats(wf_min, wf_max))
def test_indices_and_wf_above_threshold(wf, thr):
    indices, wf_above_thr = pf.indices_and_wf_above_threshold(wf, thr)
    expected_indices      = np.where(wf >= thr)
    expected_wf           = wf[expected_indices]
    assert np.all(indices == expected_indices)
    assert wf_above_thr == approx(expected_wf)


@given(multiple_waveforms())
def test_select_sipms_above_time_integrated_thr_minus_inf(sipm_wfs):
    thr      = -np.inf
    ids, wfs = pf.select_sipms_above_time_integrated_thr(sipm_wfs, thr)

    assert np.all(ids == np.arange(sipm_wfs.shape[0]))
    assert np.all(wfs == sipm_wfs)


@given(multiple_waveforms())
def test_select_sipms_above_time_integrated_thr_plus_inf(sipm_wfs):
    thr      = +np.inf
    ids, wfs = pf.select_sipms_above_time_integrated_thr(sipm_wfs, thr)

    assert ids.size == 0
    assert wfs.size == 0


@given(multiple_waveforms(), floats(wf_min, wf_max))
def test_select_sipms_above_time_integrated_thr(sipm_wfs, thr):
    ids, wfs     = pf.select_sipms_above_time_integrated_thr(sipm_wfs, thr)
    expected_ids = np.where(np.sum(sipm_wfs, axis=1) >= thr)[0]
    expected_wfs = sipm_wfs[expected_ids]

    assert np.all(ids == expected_ids)
    assert wfs == approx(expected_wfs)


@given(peak_indices())
def test_split_in_peaks(peak_data):
    indices, expected_peaks, stride = peak_data
    peaks = pf.split_in_peaks(indices, stride)

    assert len(peaks) == len(expected_peaks)
    for got, expected in zip(peaks, expected_peaks):
        assert np.all(got == expected)


@given(peak_indices())
def test_select_peaks_without_bounds(peak_data):
    _, peaks , _ = peak_data
    t_limits = minmax(-np.inf, np.inf)
    l_limits = minmax(-np.inf, np.inf)
    selected = pf.select_peaks(peaks, t_limits, l_limits)
    assert len(selected) == len(peaks)
    for got, expected in zip(selected, peaks):
        assert np.all(got == expected)


@given(peak_indices(),
       floats(0,  5), floats( 6, 10),
       floats(0, 10), floats(11, 20))
def test_select_peaks_filtered_out(peak_data, i0, i1, l0, l1):
    _, peaks , _ = peak_data
    i_limits = minmax(i0, i1)
    l_limits = minmax(l0, l1)
    selected = pf.select_peaks(peaks, i_limits * 25 * units.ns, l_limits)
    for peak in selected:
        assert i0 <= peak[0]   <= i1
        assert l0 <= peak.size <= l1


@given(peak_indices(),
       floats(0,  5), floats( 6, 10),
       floats(0, 10), floats(11, 20))
def test_select_peaks(peak_data, t0, t1, l0, l1):
    _, peaks , _ = peak_data
    i_limits = minmax(t0, t1)
    t_limits = i_limits * 25 * units.ns
    l_limits = minmax(l0, l1)
    selected = pf.select_peaks(peaks, t_limits, l_limits)

    select = lambda ids: (i_limits.contains(ids[ 0]) and
                          i_limits.contains(ids[-1]) and
                          l_limits.contains(ids.size)             )
    expected_peaks = list(filter(select, peaks))

    assert len(selected) == len(expected_peaks)
    for got, expected in zip(selected, expected_peaks):
        assert np.all(got == expected)


def test_find_peaks():
    pass


@given(times_and_waveforms(), integers(2, 10))
def test_rebin_times_and_waveforms_sum_axis_1_does_not_change(t_and_wf, stride):
    times, wfs = t_and_wf
    _, rb_wfs  = pf.rebin_times_and_waveforms(times, wfs, stride)
    assert np.sum(wfs, axis=1) == approx(np.sum(rb_wfs, axis=1))


@given(times_and_waveforms(), integers(2, 10))
def test_rebin_times_and_waveforms_sum_axis_0_does_not_change(t_and_wf, stride):
    times, wfs = t_and_wf
    sum_wf     = np.stack([np.sum(wfs, axis=0)])
    _, rb_wfs  = pf.rebin_times_and_waveforms(times,     wfs, stride)
    _, rb_sum  = pf.rebin_times_and_waveforms(times, sum_wf , stride)
    assert rb_sum[0] == approx(np.sum(rb_wfs, axis=0))


@given(times_and_waveforms(), integers(2, 10))
def test_rebin_times_and_waveforms_number_of_wfs_does_not_change(t_and_wf, stride):
    times, wfs  = t_and_wf
    _, rb_wfs = pf.rebin_times_and_waveforms(times, wfs, stride)
    assert len(wfs) == len(rb_wfs)


@given(times_and_waveforms(), integers(2, 10))
def test_rebin_times_and_waveforms_number_of_bins_is_correct(t_and_wf, stride):
    times, wfs       = t_and_wf
    rb_times, rb_wfs = pf.rebin_times_and_waveforms(times, wfs, stride)
    expected_n_bins  = times.size // stride
    if times.size % stride != 0:
        expected_n_bins += 1

    assert rb_times.size     == expected_n_bins
    assert rb_wfs  .shape[1] == expected_n_bins


@given(times_and_waveforms())
def test_rebin_times_and_waveforms_stride_1_does_not_rebin(t_and_wf):
    times, wfs       = t_and_wf
    rb_times, rb_wfs = pf.rebin_times_and_waveforms(times, wfs, 1)

    assert np.all(times == rb_times)
    assert wfs == approx(rb_wfs)


@given(times_and_waveforms(), integers(2, 10))
def test_rebin_times_and_waveforms_times_are_consistent(t_and_wf, stride):
    times, wfs  = t_and_wf

    # The samples falling in the last bin cannot be so easily
    # compared as the other ones so I remove them.
    remain = times.size - times.size % stride
    times  = times[:remain]
    wfs    = wfs  [:remain]
    rb_times, _ = pf.rebin_times_and_waveforms(times, np.ones_like(wfs), stride)

    assert np.sum(rb_times) * stride == approx(np.sum(times))
