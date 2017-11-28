import numpy as np

from .. core.system_of_units_c import units
from .. core.core_functions    import weighted_mean_and_std


class PMap:
    def __init__(self, s1s, s2s):
        self._s1s = tuple(s1s)
        self._s2s = tuple(s2s)

    @property
    def s1s(self):
        return self._s1s

    @property
    def s2s(self):
        return self._s2s

    # Optionally:
    @property
    def number_of_s1s(self):
        return len(self.s1s)

    @property
    def number_of_s2s(self):
        return len(self.s2s)

    def __str__(self):
        s  =  "---------------------\n"
        s += f"PMap instance\n"
        s +=  "---------------------\n"
        s += f"Number of S1s: {self.number_of_s1s)}\n"
        s += f"Number of S2s: {self.number_of_s2s)}\n"
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

        self._times = np.asarray(times)
        self._pmts  = pmts
        self._sipms = sipms

    @property
    def times(self):
        return self._times

    @property
    def pmts(self):
        return self._pmts

    @property
    def sipms(self):
        return self._sipms

    @property
    def time_at_max_energy(self):
        return self.times[np.argmax(self.pmts.sum_over_sensors)]

    @property
    def total_energy(self):
        return self.energy_above_threshold(0)

    @property
    def total_charge(self):
        return self.charge_above_threshold(0)

    @property
    def height(self):
        return np.max(self.pmts.sum_over_sensors)

    @property
    def width(self):
        return self.width_above_threshold(0)

    @property
    def rms(self):
        return self.rms_above_threshold(0)

    def  _pmt_indices_above_threshold(self, thr):
        return np.where(self.pmts.sum_over_sensors >= thr)[0]

    def _sipm_indices_above_threshold(self, thr):
        return np.where(self.sipms.sum_over_sensors >= thr)[0]

    def        energy_above_threshold(self, thr):
        i_above_thr  = self._pmt_indices_above_threshold(thr)
        wf_above_thr = self.pmts.sum_over_sensors[i_above_thr]
        return np.sum(wf_above_thr)

    def        charge_above_threshold(self, thr):
        i_above_thr  = self._sipm_indices_above_threshold(thr)
        wf_above_thr = self.sipms.sum_over_sensors[i_above_thr]
        return np.sum(wf_above_thr)

    def         width_above_threshold(self, thr):
        i_above_thr     = self._pmt_indices_above_threshold(thr)
        if np.size(i_above_thr) < 1:
            return 0

        times_above_thr = self.times[i_above_thr]
        return times_above_thr[-1] - times_above_thr[0]

    def           rms_above_threshold(self, thr):
        i_above_thr     = self._pmt_indices_above_threshold(thr)
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

        self._ids              = np.asarray(ids)
        self._all_waveforms    = np.asarray(wfs)
        self._wfs_dict         = dict(zip(self._ids, self._all_waveforms))
        self._sum_over_times   = np.sum(self._all_waveforms, axis=1)
        self._sum_over_sensors = np.sum(self._all_waveforms, axis=0)

    @property
    def all_waveforms(self):
        return self._all_waveforms

    def waveform(self, sensor_id):
        return self._wfs_dict[sensor_id]

    def time_slice(self, slice_number):
        return self._all_waveforms[:, slice_number]

    @property
    def ids(self):
        return self._ids

    @property
    def sum_over_times(self):
        return self._sum_over_times

    @property
    def sum_over_sensors(self):
        return self._sum_over_sensors

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
