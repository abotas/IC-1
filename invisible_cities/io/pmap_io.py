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



def store_pmap(tables, pmap, event_number, timestamp):
    s1_table, s2_table, si_table, s1i_table, s2i_table = tables
    for s1 in pmap.s1s: store_peak(s1_table, s1i_table,     None, s1, event_number, timestamp)
    for s2 in pmap.s2s: store_peak(s2_table, s2i_table, si_table, s2, event_number, timestamp)


def store_peak(pmt_table, pmti_table, si_table, peak, peak_number, event_number, timestamp):
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
            pmti_row['event'] = event_number
            pmti_row['peak' ] =  peak_number
            pmti_row['npmt' ] = pmt_id
            pmti_row['ene'  ] = peak.pmts.waveform(pmt_id)[i]
            pmti_row.append()

        if si_table is None: continue
        for sipm_id in zip(peak.sipms.ids):
            si_row['event'] = event_number
            si_row['peak' ] =  peak_number
            si_row['nsipm'] = sipm_id
            si_row['ene'  ] = peak.sipms.waveform(sipm_id)[i]
            si_row.append()


def load_pmaps(PMP_file_name):
    """Read the PMAP file and return transient PMAP rep."""
    s1t, s2t, s2sit = read_pmaps(PMP_file_name)
    s1_dict              = df_to_s1_dict(s1t)
    s2_dict              = df_to_s2_dict(s2t)
    s2si_dict            = df_to_s2si_dict(s2t, s2sit)
    return s1_dict, s2_dict, s2si_dict


def load_ipmt_pmaps(PMP_file_name):
    """Read the PMAP file and return dicts containing s12pmts"""
    s1t, s2t        = read_s1_and_s2_pmaps(PMP_file_name)
    s1pmtt, s2pmtt  =      read_ipmt_pmaps(PMP_file_name)
    s1_dict         =    df_to_s1_dict(s1t         )
    s2_dict         =    df_to_s2_dict(s2t         )
    s1pmt_dict      = df_to_s1pmt_dict(s1t,  s1pmtt)
    s2pmt_dict      = df_to_s2pmt_dict(s2t,  s2pmtt)
    return s1pmt_dict, s2pmt_dict


def load_pmaps_with_ipmt(PMP_file_name):
    """Read the PMAP file and return dicts the s1, s2, s2si, s1pmt, s2pmt dicts"""
    s1t, s2t, s2sit =      read_pmaps(PMP_file_name)
    s1pmtt, s2pmtt  = read_ipmt_pmaps(PMP_file_name)
    s1_dict         =    df_to_s1_dict(s1t        )
    s2_dict         =    df_to_s2_dict(s2t        )
    s2si_dict       =  df_to_s2si_dict(s2t,  s2sit)
    s1pmt_dict      = df_to_s1pmt_dict(s1t, s1pmtt)
    s2pmt_dict      = df_to_s2pmt_dict(s2t, s2pmtt)
    return s1_dict, s2_dict, s2si_dict, s1pmt_dict, s2pmt_dict


def read_s1_and_s2_pmaps(PMP_file_name):
    """Read only the s1 and s2 PMAPS ad PD DataFrames"""
    with tb.open_file(PMP_file_name, 'r') as h5f:
        s1t   = h5f.root.PMAPS.S1
        s2t   = h5f.root.PMAPS.S2
        s2sit = h5f.root.PMAPS.S2Si

        return (pd.DataFrame.from_records(s1t  .read()),
                pd.DataFrame.from_records(s2t  .read()))


def read_pmaps(PMP_file_name):
    """Return the PMAPS as PD DataFrames."""
    with tb.open_file(PMP_file_name, 'r') as h5f:
        s1t   = h5f.root.PMAPS.S1
        s2t   = h5f.root.PMAPS.S2
        s2sit = h5f.root.PMAPS.S2Si

        return (pd.DataFrame.from_records(s1t  .read()),
                pd.DataFrame.from_records(s2t  .read()),
                pd.DataFrame.from_records(s2sit.read()))


def read_ipmt_pmaps(PMP_file_name):
    """Return the ipmt pmaps as PD DataFrames"""
    with tb.open_file(PMP_file_name, 'r') as h5f:
        s1pmtt = h5f.root.PMAPS.S1Pmt
        s2pmtt = h5f.root.PMAPS.S2Pmt

        return (pd.DataFrame.from_records(s1pmtt.read()),
                pd.DataFrame.from_records(s2pmtt.read()))


def read_run_and_event_from_pmaps_file(PMP_file_name):
    """Return the PMAPS as PD DataFrames."""
    with tb.open_file(PMP_file_name, 'r') as h5f:
        event_t = h5f.root.Run.events
        run_t   = h5f.root.Run.runInfo

        return (pd.DataFrame.from_records(run_t  .read()),
                pd.DataFrame.from_records(event_t.read()))


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
