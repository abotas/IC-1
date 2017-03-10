
from __future__ import print_function

from glob       import glob
from time       import time

import sys
import numpy  as np
import tables as tb
import pandas as pd
import invisible_cities.reco.tbl_functions as tbl

from invisible_cities.cities.base_cities import DetectorResponseCity
from invisible_cities.core.configure  import configure, print_configuration, \
    read_config_file
from invisible_cities.core.detector_response_functions import HPXeEL,  \
     generate_ionization_electrons, diffuse_electrons, \
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
                 NEVENTS = 0):

        DetectorResponseCity.__init__(self,
                                      run_number = run_number,
                                      files_in   = files_in,
                                      file_out   = file_out,
                                      nprint     = nprint,
                                      tplane_box = TrackingPlaneBox(),
                                      hpxe       = HPXeEL())

        self.NEVENTS = NEVENTS

    def run(self):
        """genereate the SiPM maps for each event"""

        f_out = tb.open_file(self.output_file, 'w')
        f_out.create_earray(SiPM_resp,
                    atom  = tb.Float32Atom(),
                    shape = (0, self.tplane_box.x_dim,
                                self.tplane_box.y_dim,
                                self.tplane_box.z_dim),
                    expectedrows = self.NEVENTS,
                    filters = tbl.filters(self.compression))

        processed_events = 0
        # Loop over each desired file in filespath
        for fn in self.input_files:

            # Events is a dictionary mapping event index to dataframe of hits
            Events = gather_montecarlo_hits(fn)

            for ev in Events:
                ev_tp = np.zeros((self.tpb.x_dim, self.tpb.y_dim, self.tpb.z_dim),
                                 dtype=np.float32)

                ##TODO decide if this is DF or np array
                # Hits is a dictionary mapping hit index to dataframe of electrons
                Hits = generate_ionization_electrons(ev.values, hpxe.Wi, hpxe.ie_fano)

                for h in Hits:
                    # Call diffuse_electrons
                    electrons = diffuse_electrons(h, hpxe.dV, hpxe.xy_diff, hpxe.z_diff)

                    # Find TrackingPlaneResponseBox within TrackingPlaneBox
                    tprb  = TrackingPlaneResponseBox(h[0], h[1], h[2])

                    # Determine where elecetrons will produce photons in EL
                    F, IB = bin_EL(E, self.hpxe, tprb)
                    
                    # Get TrackingPlaneResponseBox response
                    for e, e_f, e_ib   in zip(E, F, IB): # electrons
                        for i, (f, ib) in enumerate(zip(e_f, e_ib)): # time bins
                             tprb.R += SiPM_response(tprb, e, ib, f) # or mod tprb??

                    # Integrate response into larger tracking plane
                    xs, xf, ys, yf, zs, zf = tbrp.situate(self.tpb)
                    ev_tp[xs: xf, ys: yf, zs: zf] += tbrp.R

                # Add poisson noise to SiPM responses
                ev_tp += np.random.poisson(lam=ev_tp)

                # Write SiPM map to file
                SiPM_resp.append([ev_tp])

                processed_events    += 1
                if processed_events == self.NEVENTS: break

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
