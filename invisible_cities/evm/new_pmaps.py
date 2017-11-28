class PMap:

    @property
    def s1s(self): pass

    @property
    def s2s(self): pass

    # Optionally:
    @property
    def number_of_s1s(self): pass

    @property
    def number_of_s2s(self): pass


class _Peak:

    @property
    def sipms(self): pass

    @property
    def pmts(self): pass

    @property
    def times(self): pass

    @property
    def time_at_max_energy(self): pass

    @property
    def total_energy(self): pass

    @property
    def height(self): pass

    @property
    def width(self): pass

    @property
    def rms(self): pass

    def energy_above_threshold(self, thr): pass

    def  width_above_threshold(self, thr): pass

    def    rms_above_threshold(self, thr): pass


class S1(_Peak): pass
class S2(_Peak): pass


class _SensorResponses:
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
        s += f"{self.__class__.__name__} instance\n"
        s +=  "------------------------\n"
        s += f"Number of sensors: {n_sensors}\n"
        for ID, wf in self._wfs_dict.items():
            s += f"| ID: {ID}\n"
            s += f"| WF: {wf}\n"
            s +=  "| \n"
        return s + "\n"

    __repr__ = __str__


class PMTResponses (_SensorResponses): pass
class SiPMResponses(_SensorResponses): pass
