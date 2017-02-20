
import numpy as np
import tables as tb

#TODO:CONFIGURE

def drift_electrons(ptab, nrow):
    """
    arguments: pytable for 1 file of signal/background events,  
    and current row. I think ptab is a pointer. 
    
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
    E = np.zeros((int(round(2.6 * 10**6 / w_val)), 3), 
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
        
        # Throw error if e_indf greater than electrons len
        if e_indf >= lect:
            raise ValueError('electrons is not long enough')
        
        E[e_ind: e_indf] = row['hit_position'] 
         
        if diffusion:
            
            # z distance in meters
            z_dist_from_el = np.sqrt(
                             E[e_ind: e_indf, 2] / float(1000))
            
            # Placate numpy
            Z_DIST_FROM_EL = np.array([z_dist_from_el, z_dist_from_el], dtype=np.float32).T
            
            
            # transverse
            if lat_diff > 0:
                
                # mean=0, sigma=lat_diff * sqrt(m) 
                lateral_drift = np.random.normal(
                                scale=lat_diff * Z_DIST_FROM_EL, 
                                size=(e_indf - e_ind, 2))
                
                E[e_ind: e_indf, :2] += lateral_drift
                
            # longitudinal 
            if long_diff > 0:
      
                longitudinal_drift = np.random.normal(
                                     scale=long_diff * Z_DIST_FROM_EL, 
                                     size=(e_indf - e_ind,))
                
                E[e_ind: e_indf, 2] += longitudinal_drift
        
        # electrons[:, 2] is (sort of) the distance travelled  
        if drift_speed != 1.0:
            
            # Time when electrons arrive at EL
            E[e_ind: e_indf, 2] /= drift_speed  

        e_ind = e_indf
                
    # Return electrons and start row for next event 
    return (E, -1, True)

