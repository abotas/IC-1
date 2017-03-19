
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
     distribute_gain, \
     compute_photon_emmission_boundaries, \
     SiPM_response
from invisible_cities.core.detector_geometry_functions import TrackingPlaneBox, \
     MiniTrackingPlaneBox
from invisible_cities.core.system_of_units_c           import units


class Anastasia(DetectorResponseCity):
    """
    The city of ANASTASIA
    """
    def __init__(self, files_in, file_out,
                 nprint     = 10000,
                 hpxe       = None,
                 tpbox      = None,
                 run_number = 0, #TODO what do we do about run_number? and nprint..
                 # Parameters added at this level
                 NEVENTS = 0,
                 w_dim   = (8, 8, 4)):

        DetectorResponseCity.__init__(self,
                                      run_number = run_number,
                                      files_in   = files_in,
                                      file_out   = file_out,
                                      nprint     = nprint,
                                      tpbox      = tpbox,
                                      hpxe       = hpxe)
        self.NEVENTS = NEVENTS
        self.w_dim   = w_dim

    # TODO: SHORTEN LINES
    # TODO: ADD MORE COMMENTS
    def run(self):
        """generate the SiPM maps for each event"""
        with tb.open_file(self.output_file, 'w') as f_out:
            #f_out     = tb.open_file(self.output_file, 'w')
            SiPM_resp = f_out.create_earray(f_out.root,
                        atom  = tb.Float32Atom(),
                        name  = 'SiPM_resp',
                        shape = (0, *self.tpbox.shape),
                        expectedrows = self.NEVENTS,
                        filters = tbl.filters(self.compression))

            processed_events = 0
            # Loop over each desired file in filespath
            for fn in self.input_files:
                if processed_events == self.NEVENTS: break

                # hits_f is a dictionary mapping event_indx to a np.array of hits
                hits_f = gather_montecarlo_hits(fn)

                # SLOWER IN PYTHON 2.7
                for hits_ev in hits_f.values():
                    if processed_events == self.NEVENTS: break

                    # Hits is a dict mapping hit ind to a np.array of ionized e-
                    electrons_ev = generate_ionization_electrons(hits_ev, self.hpxe)

                    # SLOWER IN PYTHON 2.7
                    for i, electrons_h in electrons_ev.items():

                        # Diffuse a hit's electrons
                        electrons_h = diffuse_electrons(electrons_h, self.hpxe)

                        # Find TrackingPlaneResponseBox within TrackingPlaneBox
                        tprb = MiniTrackingPlaneBox(hits_ev[i], self.tpbox, shape=self.w_dim)

                        # Determine fraction of gain from each electron that will
                        # be recieved by each time bin as e- crossing EL
                        FG = distribute_gain(electrons_h, self.hpxe, tprb)

                        # Compute
                        IB = compute_photon_emmission_boundaries(FG, self.hpxe, tprb.shape[2])

                        photons  = FG * self.hpxe.Ng / self.hpxe.rf
                        photons += np.random.normal(
                            scale=np.sqrt(photons * self.hpxe.g_fano))

                        # TODO: Make separate function
                        # Get TrackingPlaneResponseBox response
                        for e, e_f, e_ib   in zip(electrons_h, photons, IB): # electrons
                            for i, (f, ib) in enumerate(zip(e_f, e_ib)): # time bins
                                if f > 0: tprb.resp[:,:, i] += SiPM_response(tprb, e, ib, f)

                        # TODO: Make one line
                        # Integrate response into larger tracking plane
                        xs, xf, ys, yf, zs, zf = tprb.situate(self.tpbox)
                        self.tpbox.resp[xs: xf, ys: yf, zs: zf] += tprb.resp

                    # TODO: Make separate function
                    # Make a flag to turn this off?
                    self.tpbox.resp += np.random.poisson(self.tpbox.resp)

                    # Write SiPM map to file
                    SiPM_resp.append([self.tpbox.resp])
                    self.tpbox.clear_response()
                    # TODO FLUSH EAarray

                    processed_events += 1

            print(f_out)

def ANASTASIA(argv=sys.argv):

    conf = configure(argv)
    A = Anastasia(glob(conf['FILE_IN']), file_out = conf['FILE_OUT'],
        hpxe = HPXeEL(dV      = conf['dV']     * units.mm/units.mus,
                      d       = conf['d']      * units.mm,
                      t       = conf['t']      * units.mm,
                      t_el    = conf['t_el']   * units.mus,
                      Wi      = conf['Wi']     * units.eV,
                      rf      = conf['rf']     ,
                      ie_fano = conf['ie_fano'],
                      g_fano  = conf['g_fano'] ,
                      diff_xy = conf['diff_xy'] * units.mm/np.sqrt(units.m) ,
                      diff_z  = conf['diff_z']  * units.mm/np.sqrt(units.m)),
        tpbox = TrackingPlaneBox(
                      x_min   = conf['x_min']   * units.mm  ,
                      x_max   = conf['x_max']   * units.mm  ,
                      y_min   = conf['y_min']   * units.mm  ,
                      y_max   = conf['y_max']   * units.mm  ,
                      z_min   = conf['z_min']   * units.mus ,
                      z_max   = conf['z_max']   * units.mus ,
                      x_pitch = conf['x_pitch'] * units.mm  ,
                      y_pitch = conf['y_pitch'] * units.mm  ,
                      z_pitch = conf['z_pitch'] * units.mus),

        NEVENTS  =  conf['NEVENTS'],
        w_dim    = (conf['wx_dim'], conf['wy_dim'], conf['wz_dim']))

    t0 = time(); A.run(); t1 = time();
    dt = t1 - t0
    if A.NEVENTS > 0:
        print("run {} evts in {} s, time/event = {}".format(
            A.NEVENTS, dt, dt / A.NEVENTS))

if __name__ == "__main__":
    ANASTASIA(sys.argv)


#conf = read_config_file('/Users/alej/Desktop/Valencia/nextic/IC-1/invisible_cities/config/anastasia.conf')
