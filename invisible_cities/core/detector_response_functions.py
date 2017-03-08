
## Most of the arguments for most of these functions are dependent only on
## the config file and do not change during a single run. Should I be passing
## them into these functions each time I call the funcitons? or should I somehow
## establish them as global vars

import numpy  as np
import tables as tb
import pandas as pd

from   invisible_cities.core.system_of_units_c import units

class HPXeEL:
    """
    Defines a HPXe EL TPC
    EP   = E/P
    dV   = drift velocity
    P    = pressure
    d    = EL grid gap
    t    = dustance from the EL anode to the tracking plane
    L    = drift length
    Ws   = energy to produce a scintillation photon
    Wi   = energy to produce a ionization electron
    fano = sigma = sqrt(fano * hit energy / Wi)
    rf   = electron reduction factor. This is a number that multiplies
           the number of electrons and divides the number of photons, so
           that the product of both is the same (but the number of
           generated electrons is smaller)
    """
    def __init__(self,EP      =   3.5  * units.kilovolt/ (units.cm * units.bar),
                      dV      =   1.0  * units.mm/units.mus,
                      P       =  15    * units.bar,
                      d       =   5    * units.mm,
                      t       =   5    * units.mm,
                      L       = 530    * units.mm,
                      Ws      =  24    * units.eV,
                      Wi      =  16    * units.eV,
                      ie_fano =   0.15,
                      diff_xy =  10    * units.mm/np.sqrt(1 * units.m),
                      diff_z  =   3    * units.mm/np.sqrt(1 * units.m),
                      rf      =   1):

        self.EP      = EP
        u            = units.kilovolt/(units.cm*units.bar)
        ep           = EP/u
        self.dV      = dV
        self.P       = P
        self.d       = d
        self.L       = L
        self.YP      = 140 * ep - 116  # in photons per electron bar^-1 cm^-1
        self.Ng      = self.YP * self.d/units.cm * self.P/units.bar #photons/e
        self.Ws      = Ws
        self.Wi      = Wi
        self.rf      = rf
        self.ie_fano = ie_fano

    def scintillation_photons(self, E):
        return E / self.Ws

    def ionization_electrons(self, E):
        return self.rf * E / self.Wi

    def el_photons(self, E):
        return self.Ng * E / self.rf

def gather_montecarlo_hits(filepath):
    """
    gather_montecarlo_hits is a dictionary with where a key is an event number
    and a value is a pandas DataFrame with all of the hits (hit=[x,y,z,E]) for
    that event.
    """
    f      = tb.open_file(filepath, 'r')
    ptab   = f.root.MC.MCTracks
    Events = {}
    cols   = ['x', 'y', 'z', 'E']
    ev     = ptab[0]['event_indx']
    s_row  = 0

    # Iterate over the hits
    for row in ptab.iterrows():

        # Check for new events
        if ev != row['event_indx'] or row.nrow == ptab.nrows - 1:

            # Bad karma? A trick to record the last event in the file
            if row.nrow == ptab.nrows - 1: f_row = row.nrow + 1
            else:                          f_row = row.nrow

            ev_hits = np.empty((f_row - s_row, 4), dtype=np.float32)
            ev_hits[:, :3] = ptab[s_row : f_row]['hit_position'] * units.mm
            ev_hits[:,  3] = ptab[s_row : f_row]['hit_energy'  ] * units.MeV
            Events[ev] = pd.DataFrame(data=ev_hits, columns=cols)
            ev    = row['event_indx']
            s_row = row.nrow

    f.close()
    return Events

def generate_ionization_electrons(hits, Wi, ie_fano):
    """
    -Get the hits (a pandas DF contianing all the hits for an event)
    -generate ionization electrons for each hit
    -separate the electrons from each hit in a dictionary. keeping electrons
    from different hits separate because all the electrons from one hit will
    share the same TrackingPlaneResponseBox
    """
    H = {}
    cols = ['x', 'y', 'z']
    for i, h in enumerate(hits):
        n_ie  = h[3] / Wi
        n_ie += np.random.normal(scale=np.sqrt(n_ie * ie_fano))
        n_ie  = int(round(n_ie))

        H[i]  = pd.DataFrame(
                 data    = np.empty((n_ie, 3), dtype=np.float32),
                 columns = cols,
                 dtype   = np.float32)
        H[i] = h
    return H

def diffuse_electrons(E, dV, xy_diff, z_diff):
    """
    Adds gausian noise to drifting electron position
    mu=0, sig=mm/sqrt(drifting distance in m)
    """
    # Avoid modifying in place?
    E = np.copy(E)

    # z distance in meters
    z_sqdist_from_el = np.sqrt(E[:, 2] / float(units.m))
    n_ie = len(E)

    # mean=0, sigma=lat_diff * sqrt(m)
    lat_drift  = np.random.normal(scale = xy_diff * np.array(
                                       [z_sqdist_from_el, z_sqdist_from_el],
                                       dtype=np.float32).T,
                                  size  = (n_ie, 2))

    long_drift = np.random.normal(scale = z_diff * z_sqdist_from_el,
                                  size  = (n_ie,))

    E[:, :2] +=  lat_drift
    E[:,  2] += long_drift
    print('--')
    print(E[:, 2])
    E[:,  2] /= dV
    print(E[:, 2])
    print('--')

    # mod in place?
    return E

def sliding_window(E, xydim, zdim, xpitch, ypitch, zpitch, min_xp, max_xp,
                   min_yp, max_yp, min_zp, max_zp, d_cut, el_traverse_time,
                   dV, window_energy_threshold):
    """
    sliding window finds a xydim*xypitch by xydim*xypitch by zdim * zpitch
    window of sipms that have detected photons in this event.
    the window is centered around the mean position of E and then
    pushed into NEW (or desired) geometry.
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

    # Find event energy center between SiPMs in x,y,z
    xcenter = round(np.mean(E[:, 0]) / float(xypitch)) * xypitch
    ycenter = round(np.mean(E[:, 1]) / float(xypitch)) * xypitch
    zcenter = round(np.mean(E[:, 2]) / float(zpitch )) * zpitch

    # Find window boundaries
    xb = np.array(
        [xcenter - int(xydim * xypitch / 2.0 - xypitch / 2.0),
         xcenter + int(xydim * xypitch / 2.0 - xypitch / 2.0)],
                   dtype=np.int16) # mm
    yb = np.array(
        [ycenter - int(xydim * xypitch / 2.0 - xypitch / 2.0),
         ycenter + int(xydim * xypitch / 2.0 - xypitch / 2.0)],
                   dtype=np.int16) # mm
    zb = np.array(
        [zcenter - (zdim * zpitch / 2.0 - zpitch / 2.0),
         zcenter + (zdim * zpitch / 2.0 - zpitch / 2.0)],
                  dtype=np.int16) # us

    # Stuff inside 235mm x 235mm
    if   xb[0] < min_xp: xb = xb + (min_xp - xb[0])
    elif xb[1] > max_xp: xb = xb - (xb[1]  - max_xp)
    if   yb[0] < min_yp: yb = yb + (min_yp - yb[0])
    elif yb[1] > max_yp: yb = yb - (yb[1]  - max_yp)

    # Correct time
    if zb[0] < min_zp:
        zb -= zb[0]
    elif zb[1] > max_zp / dV:
        zb = zb - (zb[1] - np.ceil(max_zp / dV)).astype(np.int32)

    # Get SiPM positions
    xpos = np.array(range(xb[0], xb[1] + xypitch, xypitch), dtype=np.int32)
    ypos = np.array(range(yb[0], yb[1] + xypitch, xypitch), dtype=np.int32)
    zpos = np.arange(zb[0], zb[1] + zpitch, zpitch, dtype=np.float32)

    # Find indices of electrons not in/close to the window
    out_e = (E[:, 0] <= xb[0] - d_cut) + (E[:, 0] >= xb[1] + d_cut) \
          + (E[:, 1] <= yb[0] - d_cut) + (E[:, 1] >= yb[1] + d_cut) \
          + (E[:, 2] <= zb[0] - el_traverse_time) \
          + (E[:, 2] >= zb[1] + zpitch)

    lect = float(len(E))

    # If more than 5% of energy outside window, discard evt
    if (np.sum(out_e) / float(len(E))) >= window_energy_threshold:
        return 'Window Cut'

    # Else return electrons in/near window
    else: return (E[np.logical_not(out_e)], xpos, ypos, zpos)

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

    return np.array(
           gain / (4.0 * (z_bound[0] - z_bound[1])) \
           * (1.0 / np.sqrt(DX2 + DY2 + z_bound[1]**2) \
           -  1.0 / np.sqrt(DX2 + DY2 + z_bound[0]**2)),
           dtype=np.float32)

def bin_EL(E, xpos, ypos, zpos, xydim, zdim, zpitch,
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

    # z in units of time bin
    TS = (E[:, 2] - zpos[0]) / zpitch

    # Fraction of photons detected in each time bin
    FG   = np.empty((len(TS), num_bins), dtype=np.float32)
    f_TS = np.array(np.floor(TS), dtype=np.int8) # int16 if zdim > 127!

    # Don't divide by 0
    if el_traverse_time == 0:
        FG[:, 0] = 1 # All photons sent to first time bin
        FG[:, 1] = 0
    else:
        # Compute the fraction of gain in each relevant map
        FG[:,     0] = np.minimum(
                       (1 + f_TS - TS) * zpitch, el_traverse_time) \
                       / float(el_traverse_time)
        FG[:, 1: -1] = zpitch / float(el_traverse_time) # all the same
        FG[:,    -1] = 1 \
                       - FG[:, 0] \
                       - (num_bins - 2) * zpitch / float(el_traverse_time)

    #** Note fraction of gain in each map is equivalent to fraction of
    #   of EL traversed by electron during time bin associated with map
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

            # some electrons have bins before z window start
            if f_ts + b >= 0 and not np.isclose(zbs[b], zbs[b + 1]):
                try:
                    # get electron's contribution to this map
                    ev_maps[:, :, f_ts + b] = SiPM_response(e, xpos, ypos,
                       xydim, zbs[[b, b + 1]], fg * g)

                # Outside z-window
                except IndexError:
                    if f_ts > zdim: raise

    return ev_maps
