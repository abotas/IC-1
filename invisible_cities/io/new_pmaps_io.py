from functools import partial

import numpy  as np
import tables as tb
import pandas as pd

from .. evm .new_pmaps     import S1
from .. evm .new_pmaps     import S2
from .. evm .new_pmaps     import  PMTResponses
from .. evm .new_pmaps     import SiPMResponses
from .. evm                import nh5     as table_formats
from .. reco.tbl_functions import filters as tbl_filters


def store_peak(pmt_table, pmti_table, si_table,
               peak, peak_number, event_number):
    pmt_row  =  pmt_table.row
    pmti_row = pmti_table.row

    for i, t in enumerate(peak.times):
        pmt_row['event'] = event_number
        pmt_row['peak' ] =  peak_number
        pmt_row['time' ] = t
        pmt_row['ene'  ] = peak.pmts.sum_over_sensors[i]
        pmt_row.append()

    for pmt_id in zip(peak.pmts.ids):
        for e in peak.pmts.waveform(pmt_id):
            pmti_row['event'] = event_number
            pmti_row['peak' ] =  peak_number
            pmti_row['npmt' ] = pmt_id
            pmti_row['ene'  ] = e
            pmti_row.append()

    if si_table is None: return

    si_row = si_table.row
    for sipm_id in zip(peak.sipms.ids):
        for q in peak.sipms.waveform(sipm_id):
            si_row['event'] = event_number
            si_row['peak' ] =  peak_number
            si_row['nsipm'] = sipm_id
            si_row['ene'  ] = q
            si_row.append()


def store_pmap(tables, pmap, event_number):
    s1_table, s2_table, si_table, s1i_table, s2i_table = tables
    for s1 in pmap.s1s:
        store_peak(s1_table, s1i_table,     None, s1, event_number)
    for s2 in pmap.s2s:
        store_peak(s2_table, s2i_table, si_table, s2, event_number)


def pmap_writer(file, *, compression="ZLIB4"):
    tables = _make_tables(file, compression)
    return partial(store_pmap, tables)


def _make_tables(hdf5_file, *, compression="ZLIB4"):
    compr       = tbl_filters(compression)
    pmaps_group = hdf5_file.create_group(hdf5_file.root, 'PMAPS')
    make_table  = partial(hdf5_file.create_table, pmaps_group, filters=compr)

    s1    = make_table('S1'   , table_formats.S12   ,    "S1 Table")
    s2    = make_table('S2'   , table_formats.S12   ,    "S2 Table")
    s2si  = make_table('S2Si' , table_formats.S2Si  ,  "S2Si Table")
    s1pmt = make_table('S1Pmt', table_formats.S12Pmt, "S1Pmt Table")
    s2pmt = make_table('S2Pmt', table_formats.S12Pmt, "S2Pmt Table")

    pmp_tables = s1, s2, s2si, s1pmt, s2pmt
    for table in pmp_tables:
        # Mark column to be indexed
        table.set_attr('columns_to_index', ['event'])

    return pmp_tables


def load_pmaps_as_df(filename):
    with tb.open_file(filename, 'r') as h5f:
        pmap  = h5f.root.PMAPS
        to_df = pd.DataFrame.from_records
        return (to_df(pmap.S1   .read()),
                to_df(pmap.S2   .read()),
                to_df(pmap.S2Si .read()),
                to_df(pmap.S1Pmt.read()) if 'S1Pmt' in pmap else None,
                to_df(pmap.S2Pmt.read()) if 'S2Pmt' in pmap else None)


def build_ipmtdf_from_sumdf(sumdf):
    ipmtdf = sumdf.copy()
    ipmtdf = ipmtdf.rename(index=str, columns={'time': 'npmt'})
    ipmtdf['npmt'] = -1
    return ipmtdf


def load_pmaps(filename):
    pmap_dict = {}
    s1df, s2df, sidf, s1pmtdf, s2pmtdf = load_pmaps_as_df(filename)

    # Hack fix to allow loading pmaps without individual pmts
    if s1pmtdf is None: s1pmtdf = build_ipmtdf_from_sumdf(s1df)
    if s2pmtdf is None: s2pmtdf = build_ipmtdf_from_sumdf(s2df)

    event_numbers = set.union(set(s1df.event), set(s2df.event))
    for event_number in event_numbers:
        s1s = s1s_from_df(s1df   [s1df   .event == event_number],
                          s1pmtdf[s1pmtdf.event == event_number])
        s2s = s2s_from_df(s2df   [s2df   .event == event_number],
                          s2pmtdf[s2pmtdf.event == event_number],
                          sidf   [sidf   .event == event_number])

        pmap_dict[event_number] = PMap(s1s, s2s)

    return pmap_dict


def build_pmt_responses(pmtdf, ipmtdf):
    times   =            pmtdf.time.values
    pmt_ids = np.unique(ipmtdf.npmt.values)
    enes    =           ipmtdf.ene .values.reshape(pmt_ids.size,
                                                     times.size)
    return times, PMTResponses(pmt_ids, enes)


def build_sipm_responses(sidf):
    sipm_ids = np.unique(sidf.nsipm.values)
    enes     =           sidf.ene  .values
    n_times  = enes.size // sipm_ids.size
    enes     = enes.reshape(sipm_ids.size, n_times)
    return SiPMResponses(sipm_ids, enes)


def s1s_from_df(s1df, s1pmtdf):
    s1s = []
    peak_numbers = set(s1df.peak)
    for peak_number in peak_numbers:
        times, pmt_r = build_pmt_responses(s1df   [s1df   .peak == peak_number],
                                           s1pmtdf[s1pmtdf.peak == peak_number])
        s1s.append(S1(times, pmt_r, None))

    return s1s


def s2s_from_df(s2df, s2pmtdf, sidf):
    s2s = []
    peak_numbers = set(s2df.peak)
    for peak_number in peak_numbers:
        times, pmt_r = build_pmt_responses(s1df   [s1df   .peak == peak_number],
                                           s1pmtdf[s1pmtdf.peak == peak_number])
        sipm_r       = build_sipm_responses(sidf  [sidf   .peak == peak_number])
        s2s.append(S2(times, pmt_r, sipm_r))

    return s2s


def read_run_and_event_from_pmaps_file(filename):
    with tb.open_file(filename, 'r') as h5f:
        event_t = h5f.root.Run.events
        run_t   = h5f.root.Run.runInfo
        to_df   = pd.DataFrame.from_records
        return (to_df(run_t  .read()),
                to_df(event_t.read()))
