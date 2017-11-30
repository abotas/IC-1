import numpy as np

from .. core.system_of_units_c import units
from .. core.core_functions    import weighted_mean_and_std


class PMap:
    def __init__(self, s1s, s2s):
        self.s1s = tuple(s1s)
        self.s2s = tuple(s2s)

    def __str__(self):
        s  =  "---------------------\n"
        s += f"PMap instance\n"
        s +=  "---------------------\n"
        s += f"Number of S1s: {len(self.s1s)}\n"
        s += f"Number of S2s: {len(self.s2s)}\n"
        return s + "\n"


    __repr__ = __str__

class _Peak:
    _type = "Undefined"

    def __init__(self, times, pmts, sipms):
        length_times = len(times)
        length_pmts  = pmts .all_waveforms.shape[1]
        length_sipms = sipms.all_waveforms.shape[1]
        if not (length_times == length_pmts ==length_sipms):
            msg  =  "Shapes don't match!\n"
            msg += f"times has length {length_times}\n"
            msg += f"pmts  has length {length_pmts} \n"
            msg += f"sipms has length {length_sipms}\n"
            raise ValueError(msg)

        self.times = np.asarray(times)
        self.pmts  = pmts
        self.sipms = sipms

        i_max                   = np.argmax(self.pmts.sum_over_sensors)
        self.time_at_max_energy = self.times[i_max]
        self.height             = np.max(self.pmts.sum_over_sensors)
        self.total_energy       = self.energy_above_threshold(0)
        self.total_charge       = self.charge_above_threshold(0)
        self.width              = self. width_above_threshold(0)
        self.rms                = self.   rms_above_threshold(0)

    def energy_above_threshold(self, thr):
        i_above_thr  = self.pmts.where_above_threshold(thr)
        wf_above_thr = self.pmts.sum_over_sensors[i_above_thr]
        return np.sum(wf_above_thr)

    def charge_above_threshold(self, thr):
        i_above_thr  = self.sipms.where_above_threshold(thr)
        wf_above_thr = self.sipms.sum_over_sensors[i_above_thr]
        return np.sum(wf_above_thr)

    def  width_above_threshold(self, thr):
        i_above_thr  = self.pmts.where_above_threshold(thr)
        if np.size(i_above_thr) < 1:
            return 0

        times_above_thr = self.times[i_above_thr]
        return times_above_thr[-1] - times_above_thr[0]

    def    rms_above_threshold(self, thr):
        i_above_thr     = self.pmts.where_above_threshold(thr)
        times_above_thr = self.times[i_above_thr]
        wf_above_thr    = self.pmts.sum_over_sensors[i_above_thr]
        if np.size(i_above_thr) < 2 or np.sum(wf_above_thr) == 0:
            return 0

        return weighted_mean_and_std(times_above_thr, wf_above_thr)[1]

    def __str__(self):
        n_samples = len(self.times)
        s  =  "---------------------\n"
        s += f"{self._type} instance\n"
        s +=  "---------------------\n"
        s += f"Number of samples: {n_samples}\n"
        s += f"Times: {self.times / units.mus} µs\n"
        s += f"Time @ max energy: {self.time_at_max_energy / units.mus}\n"
        s += f"Width: {self.width / units.mus} µs\n"
        s += f"Height: {self.height} pes\n"
        s += f"Energy: {self.total_energy} pes\n"
        s += f"Charge: {self.total_charge} pes\n"
        s += f"RMS: {self.rms / units.mus} µs\n"
        return s + "\n"

    __repr__ = __str__

class S1(_Peak):
    _type = "S1"


class S2(_Peak):
    _type = "S2"


class _SensorResponses:
    _type = "Undefined"

    def __init__(self, ids, wfs):
        if len(ids) != len(wfs):
            msg  =  "Shapes do not match\n"
            msg += f"ids has length {len(ids)}\n"
            msg += f"wfs has length {len(wfs)}\n"
            raise ValueError(msg)

        self.ids              = np.asarray(ids)
        self.all_waveforms    = np.asarray(wfs)
        self.sum_over_times   = np.sum(self.all_waveforms, axis=1)
        self.sum_over_sensors = np.sum(self.all_waveforms, axis=0)

        self._wfs_dict        = dict(zip(self.ids, self.all_waveforms))

    def waveform(self, sensor_id):
        return self._wfs_dict[sensor_id]

    def time_slice(self, slice_number):
        return self.all_waveforms[:, slice_number]

    def where_above_threshold(self, thr):
        return np.where(self.sum_over_sensors >= thr)[0]

    def __str__(self):
        n_sensors = len(self.ids)
        s  =  "------------------------\n"
        s += f"{self._type} instance\n"
        s +=  "------------------------\n"
        s += f"Number of sensors: {n_sensors}\n"
        for ID, wf in self._wfs_dict.items():
            s += f"| ID: {ID}\n"
            s += f"| WF: {wf}\n"
            s +=  "| \n"
        return s + "\n"

    __repr__ = __str__


class PMTResponses(_SensorResponses):
    _type = "PMTResponses"


class SiPMResponses(_SensorResponses):
    _type = "SiPMResponses"
