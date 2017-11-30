"""
Cython functions providing pmaps io and some extra functionality

1. df_to_s1_dict --> transforms pandas df into {event:s1}
2. df_to_s2_dict --> transforms pandas df into {event:s2}
3. df_to_s2si_dict --> transforms pandas df into {event:s2si}

4. integrate_sipm_charges_in_peak(s2si, peak_number)
Returns (np.array[[nsipm_1 ,     nsipm_2, ...]],
         np.array[[sum(q_1), sum(nsipm_2), ...]])

Last revised, Alejandro Botas, August, 2017.
"""

cimport numpy as np
import  numpy as np

from .. evm.new_pmaps import S1
from .. evm.new_pmaps import S2

cpdef df_to_s1_dict(df, int max_events=-1):
    """Takes a table with the persistent representation of pmaps
    (in the form of a pandas data frame) and returns a dict {event:S1}

    """
    cdef dict s1_dict = {}
    cdef dict s12d_dict, s12d
    cdef int event_no

    s12d_dict = df_to_pmaps_dict(df, max_events)  # {event:s12d}

    for event_no, s12d in s12d_dict.items():
        try: s1_dict[event_no] = S1(s12d)
        except InitializedEmptyPmapObject: pass

    return s1_dict


cpdef df_to_s2_dict(df, int max_events=-1):
    """Takes a table with the persistent representation of pmaps
    (in the form of a pandas data frame) and returns a dict {event:S1}

    """
    cdef dict s2_dict = {}
    cdef dict s12d_dict, s12d
    cdef int event_no

    s12d_dict = df_to_pmaps_dict(df, max_events)  # {event:s12d}

    for event_no, s12d in s12d_dict.items():
        try: s2_dict[event_no] = S2(s12d)
        except InitializedEmptyPmapObject: pass

    return s2_dict


cpdef df_to_s1pmt_dict(dfs1, dfpmts, int max_events=-1):
    """Takes a table with the persistent representation of pmaps
    (in the form of a pandas data frame) and returns a dict {event:s1pmt}
    """
    cdef dict s1pmt_dict = {}
    cdef s1pmtd_dict, s1_dict, ipmtd, s1d
    cdef int event_no

    s1_dict     = df_to_s1_dict     (dfs1  , max_events)
    s1pmtd_dict = df_to_s12pmtd_dict(dfpmts, max_events)
    for event_no, ipmtd in s1pmtd_dict.items():
        s1d = s1_dict[event_no].s1d
        try: s1pmt_dict[event_no] = S1Pmt(s1d, ipmtd)
        except InitializedEmptyPmapObject: pass
    return s1pmt_dict


cpdef df_to_s2pmt_dict(dfs2, dfpmts, int max_events=-1):
    """Takes a table with the persistent representation of pmaps
    (in the form of a pandas data frame) and returns a dict {event:s2pmt}
    """
    cdef dict s2pmt_dict = {}
    cdef s2pmtd_dict, s2_dict, ipmtd, s2d
    cdef int event_no

    s2_dict     = df_to_s2_dict     (dfs2  , max_events)
    s2pmtd_dict = df_to_s12pmtd_dict(dfpmts, max_events)
    for event_no, ipmtd in s2pmtd_dict.items():
        s2d = s2_dict[event_no].s2d
        try: s2pmt_dict[event_no] = S2Pmt(s2d, ipmtd)
        except InitializedEmptyPmapObject: pass
    return s2pmt_dict


cpdef df_to_s2si_dict(dfs2, dfsi, int max_events=-1):
    """Takes a table with the persistent representation of pmaps
    (in the form of a pandas data frame) and returns a dict {event:S2Si}

    """
    cdef dict s2si_dict = {}
    cdef s2sid_dict, s2_dict, s2sid, s2d
    cdef int event_no

    s2sid_dict = df_to_s2sid_dict(dfsi, max_events)
    s2_dict    = df_to_s2_dict(dfs2, max_events)

    for event_no, s2sid in s2sid_dict.items():
        s2d = s2_dict[event_no].s2d
        try: s2si_dict[event_no] = S2Si(s2d, s2sid)
        except InitializedEmptyPmapObject: pass

    return s2si_dict


cdef df_to_pmaps_dict(df, int max_events):
    """Takes a table with the persistent representation of pmaps
    (in the form of a pandas data frame) and returns a dict {event:s12d}

    """

    cdef dict all_events = {}
    cdef dict current_event
    cdef tuple current_peak

    cdef int   [:] event = df.event.values
    cdef char  [:] peak  = df.peak .values
    cdef float [:] time  = df.time .values
    cdef float [:] ene   = df.ene  .values

    cdef int df_size = len(df.index)
    cdef int i
    cdef long limit = np.iinfo(int).max if max_events < 0 else max_events
    cdef int peak_number
    cdef list t, E

    for i in range(df_size):
        if event[i] >= limit: break

        current_event = all_events   .setdefault(event[i], {}      )
        current_peak  = current_event.setdefault( peak[i], ([], []))
        current_peak[0].append(time[i])
        current_peak[1].append( ene[i])

    # Postprocessing: Turn lists to numpy arrays before returning
    for current_event in all_events.values():
        for peak_number, (t, E) in current_event.items():
            current_event[peak_number] = [np.array(t), np.array(E)]

    return all_events


cdef df_to_s12pmtd_dict(df, int max_events):
    """ Transform S2Si from DF format to dict format."""
    cdef dict all_events = {}
    cdef dict current_event
    cdef list current_peak
    cdef int   [:] event = df.event.values
    cdef char  [:] peak  = df.peak .values
    cdef char  [:] npmt  = df.npmt .values
    cdef float [:] ene   = df.ene  .values
    cdef int  df_size        = len(df.index)
    cdef int  number_of_pmts = len(set(npmt))
    cdef long limit = np.iinfo(int).max if max_events < 0 else max_events

    cdef int  i, j
    for i in range(df_size):
        if event[i] >= limit: break
        current_event = all_events   .setdefault(event[i], {} )
        current_peak  = current_event.setdefault( peak[i], [[] for j in range(number_of_pmts)])
        current_peak[npmt[i]].append(ene[i])

    # Postprocessing: Turn lists to numpy arrays before returning and fill
    # empty slices with zeros
    cdef int  pn
    cdef list energy
    for current_event in all_events.values():
        for pn, energy in current_event.items():
            current_event[pn] = np.array(energy)

    return all_events


cdef df_to_s2sid_dict(df, int max_events):
    """ Transform S2Si from DF format to dict format."""
    cdef dict all_events = {}
    cdef dict current_event
    cdef dict current_peak
    cdef list current_sipm

    cdef int   [:] event = df.event.values
    cdef char  [:] peak  = df.peak .values
    cdef short [:] nsipm = df.nsipm.values
    cdef float [:] ene   = df.ene  .values

    cdef int df_size = len(df.index)
    cdef long limit = np.iinfo(int).max if max_events < 0 else max_events

    cdef int i
    for i in range(df_size):
        if event[i] >= limit: break

        current_event = all_events   .setdefault(event[i], {} )
        current_peak  = current_event.setdefault( peak[i], {} )
        current_sipm  = current_peak .setdefault(nsipm[i], [] )
        current_sipm.append(ene[i])

    cdef int  ID
    cdef list energy
    # Postprocessing: Turn lists to numpy arrays before returning and fill
    # empty slices with zeros
    for current_event in all_events.values():
        for current_peak in current_event.values():
            for ID, energy in current_peak.items():
                current_peak[ID] = np.array(energy)

    return all_events
