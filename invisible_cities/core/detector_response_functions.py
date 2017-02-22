
## Most of the arguments for most of these functions are dependent only on
## the config file and do not change during a single run. Should I be passing
## them into these functions each time I call the funcitons? or should I somehow
## establish them as global vars

import numpy  as np
import tables as tb

def drift_electrons(ptab, nrow, max_energy, t_diff, l_diff, drift_speed, w_val, electrons_prod_F):
    """
    arguments: 
    pytable for 1 file of signal/background events,  
    current row, since there are many rows/event
    t_diff, num mm/sqrt(m) of transverse diffusion
    l_diff, num mm/sqrt(m) of longitudinal diffusion
    
    returns: 
    E, data for all electrons in one event in a numpy ndarray 
    of shape = (num electrons in event, 3). 
    Ex: E[n] = [xpos, ypos, time] 
    ** note: I think this can only be done one event at a time since
    ** there is a variable number of electrons in each event
    
    
    nrow: keep track fo current row
    """
    current_event = ptab[nrow]['event_indx']
    
    # Approximate number of electrons + some padding
    E = np.zeros((int(round(max_energy / w_val)), 3), 
                         dtype=np.float32)
    
    lect = len(E)
    
    # Track of position in electrons
    e_ind = 0  
   
    # Iterate over hits for an evt          
    for row in ptab.iterrows(start=nrow):
    
        # Note: event_indx not always consecutive
        if row['event_indx'] > current_event:
                
            # Delete excess space in electrons
            return (E[:e_ind], row.nrow, False)
        
        elif row['event_indx'] < current_event:
            raise ValueError('current_event skipped or poorly tracked')
                
        
        # e_indf - e_ind is num drifting electrons from hit
        e_indf = e_ind + int(round(row['hit_energy'] * 10**6 / w_val))
        
        # Low energy hit, reduce electrons is high
        if e_indf == e_ind: continue
        
        # add fluctuations in num drifting electrons produced
        if electrons_prod_F > 0:
            e_indf += int(np.round(np.random.normal(
                scale=np.sqrt((e_indf - e_ind) * electrons_prod_F))))
                
            
        # Throw error if e_indf greater than electrons len
        if e_indf >= lect: raise ValueError('electrons is not long enough')
        
        # Initial position of electrons 
        E[e_ind: e_indf] = row['hit_position'] 
        
        # Add diffusion
        E[e_ind: e_indf] = diffuse_electrons(E[e_ind: e_indf], t_diff, l_diff)

        # Time when electrons arrive at EL
        E[e_ind: e_indf, 2] /= drift_speed
                                             
        e_ind = e_indf
                
    # Return electrons and start row for next event 
    return (E, -1, True)


def diffuse_electrons(E_h, t_diff, l_diff):
    """
    Adds gausian noise to drifting electron position
    mu=0, sig=mm/sqrt(drifting distance in m) 
    Note, diffuse electrons is called for each hit, since
    each hit produces a variable number of electrons and has
    different coordinates. 
    
    E_h are elecrons for one hit
    """    
    
    # z distance in meters
    z_sqdist_from_el = np.sqrt(E_h[:, 2] / float(1000))
    num_E_h = len(E_h)
    if t_diff > 0:
        
        # mean=0, sigma=lat_diff * sqrt(m) 
        lateral_drift = np.random.normal(
            scale=t_diff * np.array([z_sqdist_from_el, z_sqdist_from_el], dtype=np.float32).T, 
            size=(num_E_h, 2))

        E_h[:, :2] += lateral_drift
    
    if l_diff > 0:
        longitudinal_drift = np.random.normal(
            scale=l_diff * z_sqdist_from_el, 
            size=(num_E_h,))

        E_h[:, 2] += longitudinal_drift
    
    # Note not entirely necessary since mod in place
    return E_h

def SiPM_response(e, xpos, ypos, xydim, z_bound, gain):
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
    
def bin_EL(TS, E, xpos, ypos, xydim, zdim, zpitch,
             el_traverse_time, el_width, el_sipm_d, t_gain, gain_nf):
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
    
    # num maps that will receive photons form each electron
    num_bins = max(2, int(el_traverse_time / zpitch) + 1)
    
    # Fraction of photons detected in each time bin 
    FG   = np.empty((len(TS), num_bins), dtype=np.float32)
    f_TS = np.array(np.floor(TS), dtype=np.int8) # int16 if zdim > 127!
    
    # Compute the fraction of gain in each relevant map
    FG[:,     0] = (1 + f_TS - TS) * zpitch / float(el_traverse_time)
    FG[:, 1: -1] = zpitch / float(el_traverse_time) # all the same
    FG[:,    -1] = 1 - FG[:, 0] \
                   - (num_bins - 2) * zpitch / float(el_traverse_time)
                    
    #** Note fraction of gain in each map is equivalent to fraction of 
    #   time photons received by each map which is equivalent to 
    #   fraction of EL traversed by each electron
    
    # compute z DISTANCE integration boundaries for each time bin
    z_integ_bound = np.ones((len(TS), num_bins + 1), dtype=np.float32)
    z_integ_bound[:, 0] = el_width + el_sipm_d
                    
    for i in range(num_bins):
        z_integ_bound[:, i + 1] = z_integ_bound[:, i] - FG[:, i] * el_width
    
    if not np.allclose(z_integ_bound[:, -1], el_sipm_d):
        raise ValueError('final z_integ_bound not el_sipm_d')
    
    # compute gain for electron
    if gain_nf != 0:
        G = np.random.normal(t_gain, 
                      scale=np.sqrt(t_gain * gain_nf), 
                      size=(len(E),))
        
    else: G = np.ones((len(E),), dtype=np.float32) * t_gain
    
    # for each electron
    for f_ts, fg_e, e, zbs, g in zip(f_TS, FG, E, z_integ_bound, G):
        
        # for each time bin
        for b, fg in enumerate(fg_e):
                        
            # 0th time bin when f_ts = -1 is outside z-window
            if not f_ts != -1 or b != 0:
                
                try:  
                    # get electron's contribution to this map
                    ev_maps[:, :, f_ts + b] = SiPM_response(e, xpos, ypos, 
                       xydim, zbs[[b, b + 1]], fg)
               
                # Outside z-window
                except IndexError:
                    
                    if f_ts >= zdim: 
                        raise

    return ev_maps

