
from __future__ import print_function

from glob       import glob
from time       import time

import sys
import numpy  as np
import tables as tb
import pandas as pd

from invisible_cities.cities.base_cities import DetectorResponseCity
from invisible_cities.core.configure  import configure, print_configuration, \
    read_config_file
from invisible_cities.core.detector_response_functions import HPXeEL,  \
     generate_ionization_electrons, diffuse_electrons, sliding_window, \
     bin_EL, SiPM_response
from invisible_cities.core.detector_geometry_functions import TrackingPlaneBox, TrackingPlaneResponseBox

from invisible_cities.core.system_of_units_c import units


class Anastasia(DetectorResponseCity):
    """
    The city of ANASTASIA
    """
    def __init__(self,
                 run_number = 0,
                 files_in   = None,
                 file_out   = None,
                 nprint     = 10000,

                 # Parameters added at this level
                 NEVENTS = 0,
                 window_energy_threshold = 0.05,
                 d_cut = 15 * units.mm,
                 max_energy = 2.6 * units.MeV, # mm
                 ev_window = None
                ):

        DetectorResponseCity.__init__(self,
                                      run_number = run_number,
                                      files_in   = files_in,
                                      file_out   = file_out,
                                      nprint     = nprint,
                                      tplane_box = TrackingPlaneBox(),
                                      hpxe       = HPXeEL())

        self.NEVENTS = NEVENTS
        self.window_energy_threshold = window_energy_threshold
        self.d_cut = d_cut
        self.max_energy = max_energy


        filters = tb.Filters(complib='blosc', complevel=9, shuffle=False)
        atom    = tb.Atom.from_dtype(np.dtype('Float32'))

        self.f_out   = tb.open_file(self.output_file, 'w')
        self.tmaps   = f_out.create_earray(f_out.root, 'maps', atom,
                                     (0, self.xydim, self.xydim, self.zdim),
                                     filters=filters)

    def generate_s2(self):
        """
        -Loops over all the signal/background events in all the (desired) files

        -Calls drift_electrons for e ach event to determine drifting electrons'
        coordinates when electrons reach the EL.

        -Calls sliding_window, to find 3d 'window' of sipms central SiPMs
        for each event.

        -Calls EL_smear to compute SiPM_response if applying EL smear.
        Otherwise calls SiPM_responses directly to compute SiPM response
        within the window for each event.

        -Saves data for event by event because pytables EArray is fast, we
        do not know number of events ahead of time, np.concatenate can be
        slow, and this way we don't need to hold maps memory.

        arguments:
        tmaps is the earray in file where SiPM maps are to be saved
        """

        accepted_events  = 0
        discarded_events = 0

        # Loop over each desired file in filespath
        for fn in self.input_files:

            if accepted_events == self.NEVENTS: break

            #print('Processing file ' + str(fn))
            f_mc = tb.open_file(fn, 'r')
            ptab = f_mc.root.MC.MCTracks

            file_explored = False
            nrow = 0 # track current row in pytable since events are not separated

            # Iterates over each event
            while not file_explored:

                # Use pytable to generate ionization electrons from hit
                (electrons, nrow, file_explored) = generate_ionization_electrons(
                            ptab, nrow, self.max_energy, hpxe)

                # Call diffuse_electrons
                electrons = diffuse_electrons(electrons, hpxe)

                # Find sliding 3d window of anode centered around event
                EPOS = sliding_window(electrons, self.tplane_box, self.hpxe, self.d_cut,
                                      self.window_energy_threshold)

                # Unpack sliding window
                try: (electrons, xpos, ypos, zpos) =  EPOS

                # Or discard
                except ValueError:
                    if EPOS == 'Window Cut':
                        discarded_events += 1
                        continue
                    else: raise

                # Get SiPM maps
                ev_maps = bin_EL(electrons, xpos, ypos, zpos, self.tplane_box, self.hpxe)

                # Add noise in detection probability
                if self.photon_detection_noise: ev_maps += np.random.poisson(ev_maps)

                # Save the event's maps
                self.tmaps.append([ev_maps])

                accepted_events += 1
                if accepted_events == self.NEVENTS: break

            f_mc.close()

        print('Discarded ' + str(discarded_events) + ' events')
        print(self.f_out)
        self.f_out.close()

def ANASTASIA(argv=sys.argv):

    conf = configure(argv)

    A = Anastasia(
        NEVENTS  =      conf['NEVENTS'],
        files_in = glob(conf['FILE_IN']),
        file_out =      conf['FILE_OUT'])

    t0 = time(); A.generate_s2(); t1 = time();
    dt = t1 - t0

    print("run {} evts in {} s, time/event = {}".format(
        A.NEVENTS, dt, dt / A.NEVENTS))

if __name__ == "__main__":
    ANASTASIA(sys.argv)


#conf = read_config_file('/home/abotas/IC-1/invisible_cities/config/anastasia.conf')
