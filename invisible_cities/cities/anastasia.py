
from __future__ import print_function
from glob       import glob

import sys
import numpy  as np
import tables as tb

sys.path.append("/home/abotas/IC-1")

from invisible_cities.cities.base_cities import DetectorResponseCity
from invisible_cities.core.configure  import configure, print_configuration, \
    read_config_file
    
from invisible_cities.core.detector_response_functions import drift_electrons, \
     diffuse_electrons, EL_smear, SiPM_response
    
class Anastasia(DetectorResponseCity):
    """
    The city of ANASTASIA
    
    1) use config file to set parameters
    self.set_geometry(...)
    self.set_drifting_params(...)
    self.set_sensor_response_params(...)
    
    2) do everything else
 
    """
    def __init__(self,
                 run_number = 0,
                 files_in   = None,
                 file_out   = None,
                 nprint     = 10000,
                 
                 # Parameters added at this level
                 NEVENTS = 0,
                 stuff_NEW = False,
                 window_energy_threshold = 0.05,
                 d_cut = 15 # mm
                ):
                                   
        DetectorResponseCity.__init__(self,
                                      run_number = run_number,
                                      files_in   = files_in,
                                      file_out   = file_out,
                                      nprint     = nprint)
        
        self.NEVENTS = NEVENTS
        self.stuff_NEW = stuff_NEW
        self.window_energy_threshold = window_energy_threshold
        self.d_cut = d_cut
        
    def set_output_earray(self):
        """
        Define location for appending SiPM maps for each event
        """

        f_out      = tb.open_file(self.output_file, 'w')
        filters    = tb.Filters(complib='blosc', complevel=9, shuffle=False)
        atom       = tb.Atom.from_dtype(np.dtype('Float32'))
        tmaps      = f_out.create_earray(f_out.root, 'maps', atom, 
                                     (0, 20, 20, 60), filters=filters) 
        self.tmaps = tmaps
        self.f_out = f_out
        
    def sliding_window(self, E):
        """
        sliding window finds a 20x20x60 window of sipms that have detected
        photons in this event. the window is centered around the mean 
        position of E and then pushed into NEW (or desired) geometry.

        arguments: 
        E, all the electrons in one event

        returns:
        'window cut' if event does not fit in window, else

        returns:
        E[in_e], a subarray of E, containing all the electrons in or 
        close to the sliding window. 
        xpos, positions of sipms in x
        ypos, positions of sipms in y
        zpos, times of time slices
        """

        if len(E) == 0: raise ValueError('E is empty, 0 electrons')

        # Find boarders of window (inclusive) by calculating mean position 
        # of electrons and adding and subtracting ndim/2 (or maybe median?)
        xcenter = int(round(np.mean(E[:, 0]), - 1)) # value between 2 center sipms
        ycenter = int(round(np.mean(E[:, 1]), - 1)) # value between 2 center sipms
        zcenter = int(round((np.mean(E[:, 2]) - 1) / 2.0) * 2 + 1) 

        xb = np.array([xcenter - 95, xcenter + 95], dtype=np.int16) # mm
        yb = np.array([ycenter - 95, ycenter + 95], dtype=np.int16) # mm
        zb = np.array([zcenter - 59, zcenter + 59], dtype=np.int16) # us

        # Stuff inside 235mm x 235mm
        if   xb[0] < self.min_xp: xb = xb + (self.min_xp - xb[0])
        elif xb[1] > self.max_xp: xb = xb - (xb[1]  - self.max_xp)
        if   yb[0] < self.min_yp: yb = yb + (self.min_yp - yb[0])
        elif yb[1] > self.max_yp: yb = yb - (yb[1]  - self.max_yp)

        # Put inside NEW geometry
        if self.stuff_NEW: 

            # Eliminate the four 80x20 corners
            if xb[0] < -155:
                if y[0] < -195:
                    xb = xb + (-155 - xb[0])
                    yb = yb + (-195 - yb[0])
                elif y[1] > 195:
                    xb = xb + (-155 - xb[0])
                    yb = yb - (yb[1] - 195)
            elif xb[1] > 155:
                if yb[0] < -195:
                    xb = xb - (xb[1] - 155)
                    yb = yb + (-195 - yb[0])
                elif yb[1] > 195:
                    xb = xb - (xb[1] - 155)
                    yb = yb - (yb[1] - 195)

            # TODO: Finish stuffing inside NEW geometry
            # but it's a little more complicated

        # Correct time
        if zb[0] < self.min_zp:
            zb -= zb[0]
        elif zb[1] > self.max_zp / self.drift_speed:
            zb = zb - (zb[1] - np.ceil(self.max_zp / self.drift_speed)).astype(np.int32)

        # Get SiPM positions, adding pitch because boarders are inclusive
        # Notice maps will be organized as follows:
        # increasing x, increasing y, increasing z
        xpos = np.array(range(xb[0], xb[1] + self.xypitch, self.xypitch), dtype=np.int32 )
        ypos = np.array(range(yb[0], yb[1] + self.xypitch, self.xypitch), dtype=np.int32)
        zpos = np.array(range(zb[0], zb[1] + int(self.zpitch), int(self.zpitch)), dtype=np.int32)

        # Find indices of electrons not in/close to the window
        out_e = (E[:, 0] < xb[0] - self.d_cut) + (E[:, 0] > xb[1] + self.d_cut) \
              + (E[:, 1] < yb[0] - self.d_cut) + (E[:, 1] > yb[1] + self.d_cut) \
              + (E[:, 2] < zb[0] - self.el_traverse_time) \
              + (E[:, 2] > zb[1] + self.el_traverse_time)

        lect = float(len(E))

        # If more than 5% of energy outside window, 
        if (np.sum(out_e) / float(len(E))) > self.window_energy_threshold:

            # Discard evt
            return 'Window Cut'

        # Else return electrons in/near window
        else: return (E[np.logical_not(out_e)], xpos, ypos, zpos)
        

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
            nrow = 0 # need to track current row in pytable since events are not separated

            # Iterates over each event
            while not file_explored:

                # Call drift_electrons
                (electrons, nrow, file_explored) = drift_electrons(
                    ptab, nrow, self.max_energy, self.transverse_diffusion, 
                    self.longitudinal_diffusion, self.drift_speed, self.w_val, 
                    self.electrons_prod_F)

                # Find appropriate window
                try: (electrons, xpos, ypos, zpos) = self.sliding_window(electrons)   

                except ValueError:  

                    # Or discard
                    if self.sliding_window(electrons) == 'Window Cut':
                        discarded_events += 1
                        continue

                    else: raise

                # z in units of time bin
                TS  = (electrons[:, 2] - zpos[0]) / self.zpitch 
                if (TS < -1).any():
                    print(TS[np.where(TS < 0)[0]])
                    raise ValueError('Electrons are outside z window')    

                # Use EL_smear to get SiPM maps
                if self.zmear: 
                    ev_maps = EL_smear(TS, E, xpos, ypos, self.xydim, 
                                       self.el_width, self.el_sipm_d, 
                                       self.t_gain, self.gain_nf)

                # Call SiPM_response directly
                else:

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

                        if f_ts < 0: continue # Only relevant for zmear

                        ev_maps[:, :, f_ts] += SiPM_response(e, xpos, ypos, 
                            np.array([self.el_sipm_d + self.el_width, self.el_sipm_d], 
                            dtype=np.float32), g, self.xydim)

                # Add noise in detection probability
                if self.photon_detection_noise: ev_maps += np.random.poisson(ev_maps)

                # Save the event's maps
                self.tmaps.append([ev_maps])

                accepted_events += 1
                if accepted_events == self.NEVENTS: break

            f_mc.close()
            
        print(self.f_out)
        self.f_out.close()
      
    
    
    
conf = read_config_file('/home/abotas/IC-1/invisible_cities/config/anastasia.conf')


A = Anastasia(
    NEVENTS  = conf['NEVENTS'],
    files_in = glob(conf['FILE_IN']),
    file_out = conf['FILE_OUT'],
    )

A.set_geometry(conf['min_xp'], conf['max_xp']   , conf['min_yp'], 
               conf['max_yp'], conf['min_zp']   , conf['max_zp'],
               conf['xydim'] , conf['zdim']     , conf['xypitch'],
               conf['zpitch'], conf['el_sipm_d'], conf['el_width'],
               conf['el_traverse_time'])

A.set_drifting_params(conf['max_energy'],       
                      conf['electrons_prod_F'], 
                      conf['reduce_electrons'], conf['w_val'], 
                      conf['drift_speed'],      
                      conf['transverse_diffusion'],
                      conf['longitudinal_diffusion'])

A.set_sensor_response_params(conf['t_gain'],   conf['gain_nf'], 
                             conf['num_bins'], conf['zmear'], 
                             conf['photon_detection_noise'])

A.set_output_earray()

A.generate_s2()

    
      
        

