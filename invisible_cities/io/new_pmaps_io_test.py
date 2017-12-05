import os

import tables as tb
import numpy  as np

from ..core.testing_utils import assert_pmaps_equal
from ..core.testing_utils import assert_dataframes_equal
from .                    import new_pmaps_io       as pmpio


def test_make_tables(output_tmpdir):
    output_filename = os.path.join(output_tmpdir, "make_tables.h5")

    with tb.open_file(output_filename, "w") as h5f:
        tables = pmpio._make_tables(h5f)

        assert "PMAPS" in h5f.root
        for tablename in ("S1", "S2", "S2Si", "S1Pmt", "S2Pmt"):
            assert tablename in h5f.root.PMAPS

            table = getattr(h5f.root.PMAPS, tablename)
            assert "columns_to_index" in table.attrs
            assert table.attrs.columns_to_index == ["event"]


def test_store_peak_s1(output_tmpdir, toy_pmaps):
    output_filename = os.path.join(output_tmpdir, "store_peak_s1.h5")
    toy_pmap        = toy_pmaps.pmaps[0]
    evt_number      = np.random.randint(100, 200)
    peak_number     = 0
    with tb.open_file(output_filename, "w") as h5f:
        s1_table, _, _, s1i_table, _ = pmpio._make_tables(h5f)

        peak = toy_pmap.s1s[peak_number]
        store_peak(s1_table, s1i_table, None,
                   peak, peak_number, evt_number)

        table = h5f.root.PMAPS.S1
        assert table.cols.event[:] == evt_number
        assert table.cols.peak [:] == peak_number
        assert table.cols.time [:] == peak.times
        assert table.cols.ene  [:] == peak.pmts.sum_over_sensors

        table = h5f.root.PMAPS.S1Pmt
        assert table.cols.event[:] == evt_number
        assert table.cols.peak [:] == peak_number
        assert table.cols.npmt [:] == np.tile(peak.pmts.ids, peak.times.size)
        assert table.cols.ene  [:] == peak.pmts.all_waveforms.T.flatten()


def test_store_peak_s2(output_tmpdir, toy_pmaps):
    output_filename = os.path.join(output_tmpdir, "store_peak_s2.h5")
    toy_pmap        = toy_pmaps.pmaps[0]
    evt_number      = np.random.randint(100, 200)
    peak_number     = 0
    with tb.open_file(output_filename, "w") as h5f:
        _, s2_table, si_table, _, s2i_table = pmpio._make_tables(h5f)

        peak = toy_pmap.s2s[peak_number]
        store_peak(s2_table, s2i_table, si_table,
                   peak, peak_number, evt_number)

        table = h5f.root.PMAPS.S2
        assert table.cols.event[:] == evt_number
        assert table.cols.peak [:] == peak_number
        assert table.cols.time [:] == peak.times
        assert table.cols.ene  [:] == peak.pmts.sum_over_sensors

        table = h5f.root.PMAPS.S2Pmt
        assert table.cols.event[:] == evt_number
        assert table.cols.peak [:] == peak_number
        assert table.cols.npmt [:] == np.repeat(peak.pmts.ids, peak.times.size)
        assert table.cols.ene  [:] == peak.pmts.all_waveforms.T.flatten()

        table = h5f.root.PMAPS.S2Si
        assert table.cols.event[:] == evt_number
        assert table.cols.peak [:] == peak_number
        assert table.cols.nsipm[:] == np.repeat(peak.sipms.ids, peak.times.size)
        assert table.cols.ene  [:] == peak.sipms.all_waveforms.T.flatten()


def test_store_pmap(output_tmpdir, toy_pmaps):
    output_filename = os.path.join(output_tmpdir, "store_pmap.h5")
    evt_numbers_set = np.random.randint(100, 200, size=len(toy_pmaps.pmaps))
    with tb.open_file(output_filename, "w") as h5f:
        tables = pmpio._make_tables(h5f)
        for evt_number, pmap in zip(evt_numbers_set, toy_pmaps.pmaps):
            store_pmap(tables, pmap, evt_number)

        evt_numbers      = []
        peak_numbers     = []
        evt_numbers_pmt  = []
        peak_numbers_pmt = []
        times            = []
        npmts            = []
        enes             = []
        enes_pmt         = []
        for evt_number, pmap in zip(evt_numbers, toy_pmaps.pmaps):
            for peak_number, peak in enumerate(pmap.s1s):
                size = peak.times   .size
                npmt = peak.pmts.ids.size
                evt_numbers     .extend([ evt_number] * size)
                peak_numbers    .extend([peak_number] * size)
                evt_numbers_pmt .extend([ evt_number] * size * npmt)
                peak_numbers_pmt.extend([peak_number] * size * npmt)
                times           .extend(peak.times)
                npmts           .extend(np.repeat(peak.pmts.ids, size))
                enes            .extend(peak.pmts.sum_over_sensors)
                enes_pmt        .extend(peak.pmts.all_waveforms.T.flatten())

        table = h5f.root.PMAPS.S1
        assert table.cols.event[:] == np.array(evt_numbers)
        assert table.cols.peak [:] == np.array(peak_numbers)
        assert table.cols.time [:] == np.array(times)
        assert table.cols.ene  [:] == np.array(enes)


        table = h5f.root.PMAPS.S1Pmt
        assert table.cols.event[:] == np.array(evt_numbers_pmt)
        assert table.cols.peak [:] == np.array(peak_numbers_pmt)
        assert table.cols.npmt [:] == np.array(npmts)
        assert table.cols.ene  [:] == np.array(enes_pmt)

        evt_numbers       = []
        peak_numbers      = []
        evt_numbers_pmt   = []
        peak_numbers_pmt  = []
        evt_numbers_sipm  = []
        peak_numbers_sipm = []
        times             = []
        npmts             = []
        nsipms            = []
        enes              = []
        enes_pmt          = []
        enes_sipm         = []
        for evt_number, pmap in zip(evt_numbers, toy_pmaps.pmaps):
            for peak_number, peak in enumerate(pmap.s1s):
                size  = peak.times    .size
                npmt  = peak.pmts .ids.size
                nsipm = peak.sipms.ids.size
                evt_numbers      .extend([ evt_number] * size)
                peak_numbers     .extend([peak_number] * size)
                evt_numbers_pmt  .extend([ evt_number] * size * npmt )
                peak_numbers_pmt .extend([peak_number] * size * npmt )
                evt_numbers_sipm .extend([ evt_number] * size * nsipm)
                peak_numbers_sipm.extend([peak_number] * size * nsipm)
                times       .extend(peak.times)
                npmts       .extend(np.repeat(peak. pmts.ids, size))
                nsipms      .extend(np.repeat(peak.sipms.ids, size))
                enes        .extend(peak.pmts.sum_over_sensors)
                enes_pmt    .extend(peak.pmts .all_waveforms.T.flatten())
                enes_sipm   .extend(peak.sipms.all_waveforms.T.flatten())

        table = h5f.root.PMAPS.S2
        assert table.cols.event[:] == np.array(evt_numbers)
        assert table.cols.peak [:] == np.array(peak_numbers)
        assert table.cols.time [:] == np.array(times)
        assert table.cols.ene  [:] == np.array(enes)

        table = h5f.root.PMAPS.S2Pmt
        assert table.cols.event[:] == np.array(evt_numbers_pmt)
        assert table.cols.peak [:] == np.array(peak_numbers_pmt)
        assert table.cols.npmt [:] == np.array(npmts)
        assert table.cols.ene  [:] == np.array(enes_pmt)

        table = h5f.root.PMAPS.S2Si
        assert table.cols.event[:] == np.array(evt_numbers_sipm)
        assert table.cols.peak [:] == np.array(peak_numbers_sipm)
        assert table.cols.nsipm[:] == np.array(nsipms)
        assert table.cols.ene  [:] == np.array(enes_sipm)


def test_load_pmaps_as_df(toy_pmaps):
    read_dfs = pmpio.load_pmaps_as_df(toy_pmaps.filename)
    true_dfs = toy_pmaps.dfs
    for read_df, true_df in zip(read_dfs, true_dfs):
        assert_dataframes_equal(read_df, true_df)


def test_load_pmaps_as_df_without_ipmt(toy_pmaps):
    dfs = pmpio.load_pmaps_as_df(toy_pmaps.filename)
    assert isinstance(dfs[0], pd.DataFrame)
    assert isinstance(dfs[1], pd.DataFrame)
    assert isinstance(dfs[2], pd.DataFrame)
    assert dfs[3] is None
    assert dfs[4] is None


def test_load_pmaps(toy_pmaps):
    read_pmaps = pmpio.load_pmaps(toy_pmaps.filename)
    true_pmaps = toy_pmaps.pmaps

    assert len(read_pmaps)   == len(true_pmaps)
    assert read_pmaps.keys() == true_pmaps.keys()
    for evt_number in true_pmaps:
        read_pmap = read_pmaps[evt_number]
        true_pmap = true_pmaps[evt_number]
        assert_pmaps_equal(read_pmap, true_pmap)


@mark.parametrize("signal_type", ("S1", "S2"))
def test_build_pmt_responses(toy_pmaps, signal_type):
    if signal_type == "S1":
        df, _, _, pmt_df, _ = toy_pmaps.dfs
    else:
        _, df, _, _, pmt_df = toy_pmaps.dfs
    for evt_number in set(df.event):
        df_event     = df    [    df.event == evt_number]
        pmt_df_event = pmt_df[pmt_df.event == evt_number]
        for peak_number in set(df_event.peak):
            df_peak      = df_event    [    df_event.peak == peak_number]
            pmt_df_peak  = pmt_df_event[pmt_df_event.peak == peak_number]
            times, pmt_r = pmpio.build_pmt_responses(df_peak, pmt_df_peak)

            assert times                           ==     df_peak.time
            assert pmt_r.sum_over_sensors          ==     df_peak.ene
            assert pmt_r.all_waveforms.T.flatten() == pmt_df_peak.ene


# Variant using pd.groupby
# @mark.parametrize("signal_type", ("S1", "S2"))
# def test_build_pmt_responses(toy_pmaps, signal_type):
#     if signal_type == "S1":
#         df, _, _, pmt_df, _ = toy_pmaps.dfs
#     else:
#         _, df, _, _, pmt_df = toy_pmaps.dfs
#
#     df_groupby     =     df.groupby(("event", "peak"))
#     pmt_df_groupby = pmt_df.groupby(("event", "peak"))
#     for (_, df_peak), (_, pmt_df_peak) in zip(df_groupby, pmt_df_groupby):
#         times, pmt_r = pmpio.build_pmt_responses(df_peak, pmt_df_peak)
#
#         assert times                           ==     df_peak.time
#         assert pmt_r.sum_over_sensors          ==     df_peak.ene
#         assert pmt_r.all_waveforms.T.flatten() == pmt_df_peak.ene




def test_build_sipm_responses(toy_pmaps):
    _, _, df, _, _ = toy_pmaps.dfs
    for evt_number in set(df.event):
        df_event = df[df.event == evt_number]
        for peak_number in set(df_event.peak):
            df_peak = df_event[df_event.peak == peak_number]
            sipm_r  = pmpio.build_sipm_responses(df_peak)

            assert sipm_r.all_waveforms.T.flatten() == sipm_df_peak.ene


# Variant using pd.groupby
# def test_build_sipm_responses(toy_pmaps):
#     for _, df_peak in df_event.groupby(("event", "peak")):
#         sipm_r = pmpio.build_sipm_responses(df_peak)
#
#         assert sipm_r.all_waveforms.T.flatten() == df_peak.ene
