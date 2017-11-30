from functools import partial

import tables as tb
import pandas as pd

from .. evm import nh5           as table_formats
from .. reco import tbl_functions as tbl
#from .. reco.pmaps_functions_c      import df_to_pmaps_dict
from .. reco.pmaps_functions_c      import df_to_s1_dict
from .. reco.pmaps_functions_c      import df_to_s2_dict
from .. reco.pmaps_functions_c      import df_to_s2si_dict
from .. reco.pmaps_functions_c      import df_to_s1pmt_dict
from .. reco.pmaps_functions_c      import df_to_s2pmt_dict


def store_peak(pmt_table, pmti_table, si_table,
               peak, peak_number,
               event_number, timestamp):
    pmt_row  = pmt_table.row
    pmti_row = pmti_table.row
    si_row   = si_table.row

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

    if si_table is None: continue
    for sipm_id in zip(peak.sipms.ids):
        for q in peak.sipms.waveform(sipm_id):
            si_row['event'] = event_number
            si_row['peak' ] =  peak_number
            si_row['nsipm'] = sipm_id
            si_row['ene'  ] = q
            si_row.append()


def store_pmap(tables, pmap, event_number, timestamp):
    s1_table, s2_table, si_table, s1i_table, s2i_table = tables
    for s1 in pmap.s1s: store_peak(s1_table, s1i_table,     None, s1, event_number, timestamp)
    for s2 in pmap.s2s: store_peak(s2_table, s2i_table, si_table, s2, event_number, timestamp)


def pmap_writer(file, *, compression='ZLIB4'):
    tables = _make_tables(file, compression)
    return partial(store_pmap, tables)


def _make_tables(hdf5_file, compression):
    c = tbl.filters(compression)
    pmaps_group = hdf5_file.create_group(hdf5_file.root, 'PMAPS')
    MKT         = partial(hdf5_file.create_table, pmaps_group)

    s1    = MKT('S1'   , table_formats.S12   ,    "S1 Table", c)
    s2    = MKT('S2'   , table_formats.S12   ,    "S2 Table", c)
    s2si  = MKT('S2Si' , table_formats.S2Si  ,  "S2Si Table", c)
    s1pmt = MKT('S1Pmt', table_formats.S12Pmt, "S1Pmt Table", c)
    s2pmt = MKT('S2Pmt', table_formats.S12Pmt, "S2Pmt Table", c)

    pmp_tables = s1, s2, s2si, s1pmt, s2pmt
    for table in pmp_tables:
        # Mark column to be indexed
        table.set_attr('columns_to_index', ['event'])

    return pmp_tables


def load_pmaps_as_df(filename):
    """Return the PMAPS as PD DataFrames."""
    with tb.open_file(filename, 'r') as h5f:
        s1t    = h5f.root.PMAPS.S1
        s2t    = h5f.root.PMAPS.S2
        s2sit  = h5f.root.PMAPS.S2Si
        s1pmtt = h5f.root.PMAPS.S1Pmt
        s2pmtt = h5f.root.PMAPS.S2Pmt

        return (pd.DataFrame.from_records(s1t   .read()),
                pd.DataFrame.from_records(s2t   .read()),
                pd.DataFrame.from_records(s2sit .read()),
                pd.DataFrame.from_records(s1pmtt.read()),
                pd.DataFrame.from_records(s2pmtt.read()))


def load_pmaps(filename):
    """Read the PMAP file and return transient PMAP rep."""
    pmap_dict = {}
    s1df, s2df, sidf, s1pmtdf, s2pmtdf = load_pmaps_as_df(filename)

    event_numbers = set.union(set(s1t.event), set(s2t.event))
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
    pmt_ids = np.unique( pmtdf.npmt.values)
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
    """Return the PMAPS as PD DataFrames."""
    with tb.open_file(filename, 'r') as h5f:
        event_t = h5f.root.Run.events
        run_t   = h5f.root.Run.runInfo

        return (pd.DataFrame.from_records(run_t  .read()),
                pd.DataFrame.from_records(event_t.read()))
