
from __future__ import print_function

from glob       import glob
from time       import time

import sys
import numpy  as np
import tables as tb
import pandas as pd
import invisible_cities.reco.tbl_functions as tbl

from invisible_cities.cities.base_cities               import DetectorResponseCity
from invisible_cities.core.configure                   import configure, \
     print_configuration, \
      read_config_file
from invisible_cities.core.detector_response_functions import HPXeEL,  \
     gather_montecarlo_hits, \
     generate_ionization_electrons, \
     diffuse_electrons, \
     bin_EL, \
     SiPM_response
from invisible_cities.core.detector_geometry_functions import TrackingPlaneBox, \
     TrackingPlaneResponseBox
from invisible_cities.core.system_of_units_c           import units


class Anastasia(DetectorResponseCity):
    """
    The city of ANASTASIA
    """
    def __init__(self, hpxe, tpb,
                 run_number = 0,
                 files_in   = None,
                 file_out   = None,
                 nprint     = 10000,

                 # Parameters added at this level
                 NEVENTS = 0,
                 wx_dim  = 8,
                 wy_dim  = 8,
                 wz_dim  = 4):

        DetectorResponseCity.__init__(self,
                                      run_number = run_number,
                                      files_in   = files_in,
                                      file_out   = file_out,
                                      nprint     = nprint,
                                      tpb        = tpb,
                                      hpxe       = hpxe)
        self.NEVENTS = NEVENTS
        self.wx_dim  = wx_dim
        self.wy_dim  = wy_dim
        self.wz_dim  = wz_dim

    def run(self):
        """genereate the SiPM maps for each event"""
        f_out     = tb.open_file(self.output_file, 'w')
        SiPM_resp = f_out.create_earray(f_out.root,
                    atom  = tb.Float32Atom(),
                    name  = 'SiPM_resp',
                    shape = (0, self.tpb.x_dim,
                                self.tpb.y_dim,
                                self.tpb.z_dim),
                    expectedrows = self.NEVENTS,
                    filters = tbl.filters(self.compression))

        processed_events = 0
        # Loop over each desired file in filespath
        for fn in self.input_files:
            if processed_events == self.NEVENTS: break

            # Events is a dictionary mapping event index to dataframe of hits
            Events = gather_montecarlo_hits(fn)

            # SLOWER IN PYTHON 2
            for ev in Events.values():
                if processed_events == self.NEVENTS: break

                ev_tp = np.zeros((self.tpb.x_dim,
                                  self.tpb.y_dim,
                                  self.tpb.z_dim),
                                 dtype=np.float32)

                # Hits is a dict mapping hit ind to a np.array of ionized e-
                Hits = generate_ionization_electrons(ev.values, self.hpxe)

                # SLOWER IN PYTHON 2
                for i, h in Hits.items():

                    # Call diffuse_electrons
                    E = diffuse_electrons(h, self.hpxe)

                    # Find TrackingPlaneResponseBox within TrackingPlaneBox
                    tprb = TrackingPlaneResponseBox(ev.values[i, 0],
                                                    ev.values[i, 1],
                                                    ev.values[i, 2],
                                                    x_dim = self.wx_dim,
                                                    y_dim = self.wy_dim,
                                                    z_dim = self.wz_dim)

                    # Determine where elecetrons will produce photons in EL
                    F, IB = bin_EL(E, self.hpxe, tprb)

                    # Get TrackingPlaneResponseBox response
                    for e, e_f, e_ib   in zip(E, F, IB): # electrons
                        for i, (f, ib) in enumerate(zip(e_f, e_ib)): # time bins
                            if f > 0: tprb.R[:,:, i] += SiPM_response(tprb, e, ib, f)

                    # Integrate response into larger tracking plane
                    xs, xf, ys, yf, zs, zf = tprb.situate(self.tpb)
                    ev_tp[xs: xf, ys: yf, zs: zf] += tprb.R

                # Make a flag to turn this off?
                ev_tp += np.random.poisson(ev_tp)

                # Write SiPM map to file
                SiPM_resp.append([ev_tp])
                processed_events += 1

        print(f_out)
        f_out.close()

def ANASTASIA(argv=sys.argv):

    conf = configure(argv)

    A = Anastasia(HPXeEL(dV      = conf['dV']   * units.mm/units.mus,
                         d       = conf['d']    * units.mm,
                         t       = conf['t']    * units.mm,
                         t_el    = conf['t_el'] * units.mus,
                         Wi      = conf['Wi']   * units.eV,
                         rf      = conf['rf']     ,
                         ie_fano = conf['ie_fano'],
                         g_fano  = conf['g_fano'] ,
                         diff_xy = conf['diff_xy'] * units.mm/np.sqrt(units.m) ,
                         diff_z  = conf['diff_z']  * units.mm/np.sqrt(units.m)),
        TrackingPlaneBox(x_min   = conf['x_min']   * units.mm  ,
                         x_max   = conf['x_max']   * units.mm  ,
                         y_min   = conf['y_min']   * units.mm  ,
                         y_max   = conf['y_max']   * units.mm  ,
                         z_min   = conf['z_min']   * units.mus ,
                         z_max   = conf['z_max']   * units.mus ,
                         x_pitch = conf['x_pitch'] * units.mm  ,
                         y_pitch = conf['y_pitch'] * units.mm  ,
                         z_pitch = conf['z_pitch'] * units.mus),

        files_in = glob(conf['FILE_IN']),
        NEVENTS  = conf['NEVENTS'],
        file_out = conf['FILE_OUT'],
        wx_dim   = conf['wx_dim'],
        wy_dim   = conf['wy_dim'],
        wz_dim   = conf['wz_dim'])

    t0 = time(); A.run(); t1 = time();
    dt = t1 - t0
    if A.NEVENTS > 0:
        print("run {} evts in {} s, time/event = {}".format(
            A.NEVENTS, dt, dt / A.NEVENTS))

if __name__ == "__main__":
    ANASTASIA(sys.argv)


#conf = read_config_file('/Users/alej/Desktop/Valencia/nextic/IC-1/invisible_cities/config/anastasia.conf')
