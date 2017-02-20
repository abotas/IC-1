
from __future__ import print_function

import sys
from glob import glob
from time import time

import numpy  as np
import tables as tb


import invisible_cities.core.tbl_functions as tbl
from   invisible_cities.core.configure     import configure, print_configuration
from   invisible_cities.cities.base_cities import DetectorResponseCity
from   invisible_cities.core               import drift_electrons
from   invisibile_cities.core.detector_response_functions import EL_smear, SiPM_response

class Anastasia(DetectorResponseCity):
    """
    The city of ANASTASIA
    """
    def __init__(self,
                 run_number = 0,
                 files_in   = None,
                 file_out   = None,
                 nprint     = 10000,
                 
                 # Detector Response City
                 ptab       = None,
                 wval       = 22.4,
                 transverse_diffusion   = 1.0,
                 longitudinal_diffusion = 0.3,
                 drift_speed            = 1.0,
                              
                 stuff_NEW = False,
                 d_cut = 15, # mm from window past which electron is ignored
                 window_energy_threshold = .05,  # max allowed fraction of
                                                 # electrons outside window
                 
                 # Sensor response params
                 min_xp = -235, # mm
                 max_xp =  235,
                 min_yp = -235,
                 max_yp =  235,
                 min_zp =  0,
                 max_zp =  530,
                 
                 photon_detection_noise = True,
                 zmear    = True, # photons sent to multiple time bins
                 num_bins = 2     # for the moment, this must be 2
    
                ):
                                   
        DetectorResponseCity.__init__(self,
                                      run_number = run_number,
                                      files_in   = files_in,
                                      file_out   = None,
                                      nprint     = nprint,
                                      ptab       = ptab,
                                      wval       = wval,
                                      transverse_diffusion   = transverse_diffusion,
                                      longitudinal_diffusion = longitudinal_diffusion,
                                      drift_speed            = drift_speed)
        
        
        self.stuff_NEW = stuff_NEW
        self.window_energy_threshold = window_energy_threshold
        self.d_cut  =  d_cut
        self.min_xp = min_xp 
        self.max_xp = max_xp
        self.min_yp = min_yp
        self.max_yp = max_yp
        self.min_zp = min_zp
        self.max_zp = max_zp
        self.photon_detection_noise = photon_detection_noise
        self.zmear    = zmear
        self.num_bins = num_bins

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
        if   xb[0] < min_xp: xb = xb + (self.min_xp - xb[0])
        elif xb[1] > max_xp: xb = xb - (xb[1]  - self.max_xp)
        if   yb[0] < min_yp: yb = yb + (self.min_yp - yb[0])
        elif yb[1] > max_yp: yb = yb - (yb[1]  - self.max_yp)

        # Put inside NEW geometry
        if stuff_NEW: 

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
            zb = zb - (zb[1] - np.ceil(max_zp / self.drift_speed)).astype(np.int32)

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
    
    
    
    # EArray parameters for signal and background
    filters = tb.Filters(complib='blosc', complevel=9, shuffle=False)
    atom    = tb.Atom.from_dtype(np.dtype('Float32'))
    f_out   = tb.open_file(self.output_file, 'w')
    maps    = f_out.create_earray(f_out.root, 'maps', atom, 
                                     (0, 20, 20, 60), filters=filters)
    
    # Iterate thru input files
    accepted_events  = 0
    discarded_events = 0
    for in_f in files_in:
        
        if accepted_events == NEVENTS: break
        
        print('\r', 'Processing file ' + str(fn), end="")
        
        #print('Processing file ' + str(fn))
        MC = tb.open_file(in_f, 'r')
        ptab = MC.root.MC.MCTracks

        file_explored = False
        nrow = 0 # need to track current row in pytable since events are not separated
        
        # Iterates over each event
        while not file_explored:
            
            # Call drift_electrons
            (electrons, nrow, file_explored) = drift_electrons(ptab, nrow)
            
            # Find appropriate window
            try: (electrons, xpos, ypos, zpos) = sliding_window(electrons)   
            
            except ValueError:  
                
                # Or discard
                if sliding_window(electrons) == 'Window Cut':
                    discarded_events += 1
                    continue
                    
                else: raise
             
            # z in units of time bin
            TS  = (electrons[:, 2] - zpos[0]) / zpitch 
            if (TS < -1).any(): raise ValueError('Electrons are outside z window')    

            # Use EL_smear to get SiPM maps
            if zmear: ev_maps = EL_smear(TS, electrons, xpos, ypos)
            
            # Call SiPM_response directly
            else:
                
                # SiPM maps for an event
                ev_maps = np.zeros((xydim, xydim, zdim), dtype=np.float32)
                
                if gain_nf != 0:
                    G = np.random.normal(t_gain, 
                        scale=np.sqrt(t_gain * gain_nf), 
                        size=(len(electrons),))
                else:
                    G = np.ones((len(electrons),), dtype=np.float32) * t_gain
                
                # Sum SiPM response for each electron
                for f_ts, e, g in zip(np.array(np.floor(TS), dtype=np.int8), electrons, G):
                    
                    if f_ts < 0: continue # Only relevant for zmear
                   
                    ev_maps[:, :, f_ts] += SiPM_response(
                        e, xpos, ypos, np.array([el_sipm_d + el_width, el_sipm_d], 
                        dtype=np.float32), g)

            # Add noise in detection probability
            if photon_detection_noise: ev_maps += np.random.poisson(ev_maps)
            
            # Save the event's maps
            maps.append([ev_maps])
                  
            accepted_events += 1
            if accepted_events == NEVENTS: break
                
        MC.close()         
        

