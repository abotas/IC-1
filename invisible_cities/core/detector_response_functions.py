
import numpy as np


##TODO: CONFIG

def SiPM_response(e, xpos, ypos, z_bound, gain, xydim=xydim):
    """
    # All photons are emitted from the point where electron
    # hit EL plane. The el plane is 5mm from
    # Vectorized SiPM response, but takes as input only 1 electron
    # at a time.
    """
    
    # Calculate the xy distances only once
    dx2 = (xpos - e[0])**2
    dy2 = (ypos - e[1])**2

    DX2 = np.array([dx2 for i in range(xydim)], dtype=np.float32).T
    DY2 = np.array([dy2 for i in range(xydim)], dtype=np.float32)
    
    return np.array(gain / (4.0 * (z_bound[0] - z_bound[1])) \
           * (1.0 / np.sqrt(DX2 + DY2 + z_bound[1]**2) \
           -  1.0 / np.sqrt(DX2 + DY2 + z_bound[0]**2)), dtype=np.float32)
    
def EL_smear(TS, E, xpos, ypos):
    """
    arguments:
    TS, time of electron arrival to start of EL plane in units
    of time bins
    E, ndarray of electrons for an event
    
    returns:
    ev_maps, a 20x20x60 np.array containing the SiPM maps for
    an event.
    
    -computes fraction of photons registered in each time bin
    -calls SiPM_response twice, once for each time bin
    """
    
    # SiPM maps for an event
    ev_maps = np.zeros((xydim, xydim, zdim), dtype=np.float32)
    
    # Fraction of photons detected in each time bin 
    FG   = np.empty((len(TS), 2), dtype=np.float32)
    f_TS = np.array(np.floor(TS), dtype=np.int8) # int16 if zdim > 127!
    FG[:, 0] = 1 - (TS - f_TS)
    FG[:, 1] = TS - f_TS
    
    # compute z integration boundaries for each time bin
    z_integ_bound = np.ones((len(TS), 3), dtype=np.float32)
    z_integ_bound[:, 0] = el_width + el_sipm_d
    z_integ_bound[:, 1] = FG[:, 0] * el_width + el_sipm_d
    z_integ_bound[:, 2] = el_sipm_d
    
    # compute gain for electron
    if gain_nf != 0:
        G = np.random.normal(t_gain, 
                      scale=np.sqrt(t_gain * gain_nf), 
                      size=(len(E),))
        
    else: G = np.ones((len(E),), dtype=np.float32) * t_gain

    for f_ts, fg, e, zbs, g in zip(f_TS, FG, E, z_integ_bound, G):
       
        if f_ts != -1: # if -1, photons observed in 1st 
                       # time bin are received by map 1 
                       # outside (before) z window
            # Map 1
            ev_maps[:, :, f_ts] += SiPM_response(
                e, xpos, ypos, zbs[:2], fg[0] * g)
        try:
            # Map 2
            ev_maps[:, :, f_ts + 1] += SiPM_response(
                e, xpos, ypos, zbs[1:], fg[1] * g)

        # Outside window in Z
        except IndexError:
            if f_ts != zdim - 1: raise

    return ev_maps

