
from __future__ import print_function

from glob       import glob
from time       import time

import sys
import numpy  as np
import tables as tb

sys.path.append("/home/abotas/IC-1")

from invisible_cities.cities.base_cities import DetectorResponseCity
from invisible_cities.core.configure  import configure, print_configuration, \
    read_config_file  
from invisible_cities.core.detector_response_functions import generate_ionization_electrons, \
     diffuse_electrons, sliding_window, bin_EL, SiPM_response
    
class Anastasia(DetectorResponseCity):
    """
    The city of ANASTASIA
    
    1) use config file to set parameters
    self.set_geometry(...)
    self.set_drifting_params(...)
    self.set_sensor_response_params(...)
    
    2) everything else
 
    """
    def __init__(self,
                 run_number = 0,
                 files_in   = None,
                 file_out   = None,
                 nprint     = 10000,
                 
                 # Parameters added at this level
                 NEVENTS = 0,
                 window_energy_threshold = 0.05,
                 d_cut = 15 # mm
                ):
                                   
        DetectorResponseCity.__init__(self,
                                      run_number = run_number,
                                      files_in   = files_in,
                                      file_out   = file_out,
                                      nprint     = nprint)
        
        self.NEVENTS = NEVENTS
        self.window_energy_threshold = window_energy_threshold
        self.d_cut = d_cut
        
    def set_output_earray(self):
        """
        Define location for appending SiPM maps for each event
        """

        f_out   = tb.open_file(self.output_file, 'w')
        filters = tb.Filters(complib='blosc', complevel=9, shuffle=False)
        atom    = tb.Atom.from_dtype(np.dtype('Float32'))
        tmaps   = f_out.create_earray(f_out.root, 'maps', atom, 
                                     (0, self.xydim, self.xydim, self.zdim), 
                                     filters=filters) 
        self.tmaps = tmaps
        self.f_out = f_out
        

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

                # Call drift_electrons
                (electrons, nrow, file_explored) = generate_ionization_electrons(
                    ptab, nrow, self.max_energy, self.w_val, 
                    self.electrons_prod_F)
                
                # Call diffuse_electrons
                electrons = diffuse_electrons(electrons, 
                                              self.drift_speed, 
                                              self.transverse_diffusion, 
                                              self.longitudinal_diffusion)
                
                # Find sliding 3d window of anode centered around event
                EPOS = sliding_window(electrons, self.xydim, self.zdim, 
                                      self.xypitch, self.zpitch, self.min_xp, 
                                      self.max_xp, self.min_yp, self.max_yp, 
                                      self.min_zp, self.max_zp, self.d_cut, 
                                      self.el_traverse_time, self.drift_speed,   
                                      self.window_energy_threshold) 
                
                # Unpack sliding window
                try: (electrons, xpos, ypos, zpos) =  EPOS
                
                # Or discard
                except ValueError:  
                    if EPOS == 'Window Cut':
                        discarded_events += 1
                        continue
                    else: raise    

                # Use EL_smear to get SiPM maps
                if self.zmear: 
                    ev_maps = bin_EL(electrons, xpos, ypos, zpos,
                                     self.xydim, 
                                     self.zdim, 
                                     self.zpitch,
                                     self.el_traverse_time,
                                     self.el_width, 
                                     self.el_sipm_d, 
                                     self.t_gain, 
                                     self.gain_nf)

                # Call SiPM_response directly
                else:
                    
                    # z in units of time bin
                    TS = (electrons[:, 2] - zpos[0]) / self.zpitch 
                    
                    # SiPM maps for an event
                    ev_maps = np.zeros((self.xydim, self.xydim, self.zdim), 
                                       dtype=np.float32)

                    if self.gain_nf != 0:
                        G = np.random.normal(self.t_gain, 
                            scale=np.sqrt(self.t_gain * self.gain_nf), 
                            size=(len(electrons),))
                    else:
                        G = np.ones((len(electrons),), dtype=np.float32) * self.t_gain

                    # Sum SiPM response for each electron
                    for f_ts, e, g in zip(np.array(np.floor(TS), dtype=np.int8), electrons, G):

                        if f_ts < 0: continue # Only relevant for EL photon smearing

                        ev_maps[:, :, f_ts] += SiPM_response(e, xpos, ypos, self.xydim,
                            np.array([self.el_sipm_d + self.el_width, self.el_sipm_d], 
                            dtype=np.float32), g)

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

    A.set_geometry(conf['min_xp'], conf['max_xp']   , conf['min_yp'], 
                   conf['max_yp'], conf['min_zp']   , conf['max_zp'],
                   conf['xydim'] , conf['zdim']     , conf['xypitch'],
                   conf['zpitch'], conf['el_sipm_d'], conf['el_width'],
                   conf['el_traverse_time'])

    A.set_drifting_params(conf['max_energy'],       
                          conf['electrons_prod_F'], 
                          conf['reduce_electrons'], 
                          conf['w_val'] * conf['reduce_electrons'], 
                          conf['drift_speed'],      
                          conf['transverse_diffusion'],
                          conf['longitudinal_diffusion'])

    A.set_sensor_response_params(conf['t_gain'] * conf['reduce_electrons'], 
                                 conf['gain_nf'], 
                                 conf['zmear'], 
                                 conf['photon_detection_noise'])

    A.set_output_earray()

    t0 = time(); A.generate_s2(); t1 = time();
    dt = t1 - t0
    
    print("run {} evts in {} s, time/event = {}".format(
        A.NEVENTS, dt, dt / A.NEVENTS))

if __name__ == "__main__":
    ANASTASIA(sys.argv)

    
#conf = read_config_file('/home/abotas/IC-1/invisible_cities/config/anastasia.conf')