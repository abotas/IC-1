
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
     distribute_photons, \
     compute_photon_emmission_boundaries, \
     SiPM_response
from invisible_cities.core.detector_geometry_functions import TrackingPlaneBox, \
     MiniTrackingPlaneBox, determine_hrb_size
from invisible_cities.core.system_of_units_c           import units


class Anastasia(DetectorResponseCity):
    """
    The city of ANASTASIA
    """
    def __init__(self, files_in, file_out,
                 nprint     = 10000,
                 hpxe       = None,
                 tpbox      = None,
                 run_number = 0,
                 # Parameters added at this level
                 NEVENTS = 0):

        DetectorResponseCity.__init__(self,
                                      run_number = run_number,
                                      files_in   = files_in,
                                      file_out   = file_out,
                                      nprint     = nprint,
                                      tpbox      = tpbox,
                                      hpxe       = hpxe)
        self.NEVENTS = NEVENTS
        self.hrb     = MiniTrackingPlaneBox(tpbox)

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

            # Loop over the input files
            for fn in self.input_files:
                if processed_events == self.NEVENTS: break

                # get all the hits from a file into a dict mapping event
                # index to a numpy array containing all the hits for that event
                # ex_hit = [xcoord, ycoord, zcoord, energy]
                hits_f = gather_montecarlo_hits(fn)

                # SLOWER IN PYTHON 2.7
                for hits_ev in hits_f.values():
                    if processed_events == self.NEVENTS: break

                    # Generate all the (undiffused) ionization electrons created
                    # by an event. electrons_ev is a dict mapping hit index to a
                    # np array of the e- created by that hit.
                    # electron_ev[hit indx][e-] = [x, y, z] (same pos as hit)
                    electrons_ev = generate_ionization_electrons(
                        hits_ev, self.hpxe)

                    # SLOWER IN PYTHON 2.7
                    for hit, electrons_h in electrons_ev.items():

                        # Diffuse a hit's electrons. electrons_h is a np array
                        # of diffused ionization e- created by one hit
                        # electrons_h[e-] = [x, y, EL arrival time] (after diff)
                        electrons_h = diffuse_electrons(electrons_h, self.hpxe)


                        # Center a hrb of size hrb_shape around the center of
                        # where the hit should reach the EL (z=time).
                        hrb_shape = determine_hrb_size(hits_ev[hit, 2], self.hpxe, self.tpbox, nsig=3)

                        # TODO make this less ugly
                        self.hrb.center((*hits_ev[hit,:2], hits_ev[hit, 2] / self.hpxe.dV),
                            hrb_shape)
                        # TODO: Change size of window from hit to hit depending
                        # on z position of hit? (amount of diffusion possible)

                        # Determine fraction of gain from each ionization e-
                        # to be recieved by each time bin as e- crossing EL
                        # Ex: FG[e-, time bin] = 0.2
                        FG = distribute_gain(electrons_h, self.hpxe, self.hrb)

                        # Compute number of photons produced by ionization e-
                        # in each time bin (same shape as FG)
                        # photons[e-, time bin] = 7083
                        photons = distribute_photons(FG, self.hpxe)

                        # Compute z-distance of electron to SiPMs during each
                        # time bin. Ex:
                        # IB[e-, time bin] = [zd to SiPMs start, zd to SiPMs end]
                        IB = compute_photon_emmission_boundaries(FG, self.hpxe)

                        # Get cumulative response of all SiPMs within hrb to all
                        # the photons produced by electrons_h traveling thru EL
                        self.hrb.resp_h = SiPM_response(
                            electrons_h, photons, IB, self.hrb)

                        # Add SiPM response to this hit, to the SiPM response
                        # of the entire event
                        self.hrb.add_hit_resp_to_event_resp()

                    # Add poisson noise to num photons detected to simulate the
                    # fluctuation in number of photons detected
                    # Make a flag to turn this (and other posson noises off) off?
                    self.hrb.resp_ev += np.random.poisson(self.hrb.resp_ev)

                    # Write SiPM map to file
                    SiPM_resp.append([self.hrb.resp_ev])
                    self.hrb.clear_event_response()
                    processed_events += 1

            print(f_out)

def ANASTASIA(argv=sys.argv):

    conf = configure(argv)


    A = Anastasia(glob(conf.FILE_IN), file_out = conf.FILE_OUT,
        hpxe = HPXeEL(dV      = conf.dV     * units.mm/units.mus,
                      d       = conf.d      * units.mm,
                      t       = conf.t      * units.mm,
                      t_el    = conf.t_el   * units.mus,
                      Wi      = conf.Wi     * units.eV,
                      rf      = conf.rf     ,
                      ie_fano = conf.ie_fano,
                      g_fano  = conf.g_fano ,
                      diff_xy = conf.diff_xy * units.mm/np.sqrt(units.m) ,
                      diff_z  = conf.diff_z  * units.mm/np.sqrt(units.m)),
        tpbox = TrackingPlaneBox(
                      x_min   = conf.x_min   * units.mm  ,
                      x_max   = conf.x_max   * units.mm  ,
                      y_min   = conf.y_min   * units.mm  ,
                      y_max   = conf.y_max   * units.mm  ,
                      z_min   = conf.z_min   * units.mus ,
                      z_max   = conf.z_max   * units.mus ,
                      x_pitch = conf.x_pitch * units.mm  ,
                      y_pitch = conf.y_pitch * units.mm  ,
                      z_pitch = conf.z_pitch * units.mus),
        NEVENTS  =  conf.NEVENTS)

    t0 = time(); A.run(); t1 = time();
    dt = t1 - t0
    if A.NEVENTS > 0:
        print("run {} evts in {} s, time/event = {}".format(
            A.NEVENTS, dt, dt / A.NEVENTS))

if __name__ == "__main__":
    ANASTASIA(sys.argv)


#conf = read_config_file('/Users/alej/Desktop/Valencia/nextic/IC-1/invisible_cities/config/anastasia.conf')
