
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
    t    = distance from the EL anode to the tracking plane
    L    = drift length
    Ws   = energy to produce a scintillation photon
    Wi   = energy to produce a ionization electron
    ie_fano = sigma = sqrt(ie_fano * hit energy / Wi)
    g_fano = sigma = sqrt(g_fano * expected photons)
    rf   = electron reduction factor. This is a number that multiplies
           the number of electrons and divides the number of photons, so
           that the product of both is the same (but the number of
           generated electrons is smaller)
    """
    def __init__(self,EP      =   3.5  * units.kilovolt / (units.cm * units.bar),
                      dV      =   1.0  * units.mm/units.mus,
                      P       =  15    * units.bar,
                      d       =   5    * units.mm,
                      t       =   5    * units.mm,
                      t_el    =   2    * units.mus,
                      L       = 530    * units.mm,
                      Ws      =  24    * units.eV,
                      Wi      =  16    * units.eV,
                      ie_fano =   0.15           ,
                      g_fano  =   0.1            ,
                      diff_xy =  10    * units.mm/np.sqrt(1 * units.m),
                      diff_z  =   3    * units.mm/np.sqrt(1 * units.m),
                      rf      =   1):

        self.EP      = EP
        u            = units.kilovolt/(units.cm*units.bar)
        ep           = EP/u
        self.dV      = dV
        self.P       = P
        self.d       = d
        self.t       = t
        self.t_el    = t_el
        self.L       = L
        self.YP      = 140 * ep - 116  # in photons per electron bar^-1 cm^-1
        self.Ng      = self.YP * self.d/units.cm * self.P/units.bar #photons/e
        self.Ws      = Ws
        self.Wi      = Wi
        self.rf      = rf
        self.ie_fano = ie_fano
        self. g_fano =  g_fano

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
        n_ie += np.random.normal(scale=np.sqrt(h[3] / Wi * ie_fano))
        E     = np.empty((int(round(n_ie)), 3), dtype=np.float32)
        E[:]  = h[:3]
        H[i]  = E
        #H[i] = pd.DataFrame(data=E, columns=cols, dtype=np.float32)
    return H

def diffuse_electrons(E, dV, xy_diff, z_diff):
    """
    Adds gausian noise to drifting electron position
    mu=0, sig=mm/sqrt(drifting distance in m)
    """
    # Avoid modifying in place?
    E = np.copy(E)

    # sqrt dist from EL grid
    sd = np.sqrt(E[:, 2] / float(units.m))
    n_ie = len(E)

    # mu=0, sig=lat_diff * sqrt(m)
    lat_drift = np.random.normal(scale=xy_diff * np.array([sd, sd]).T, size=(n_ie,2))
    long_drift= np.random.normal(scale= z_diff * sd,                   size=(n_ie,))
    E[:, :2] +=  lat_drift
    E[:,  2] += long_drift
    E[:,  2] /= dV

    return E # mod in place?

def SiPM_response(rb, e, z_bound, gain):
    """
    # All photons are emitted from the point where electron
    # hit EL plane. The el plane is 5mm from
    # Vectorized SiPM response, but takes as input only 1 electron
    # at a time.
    """

    # Calculate the xy distances only once
    dx2 = (rb.x_pos - e[0])**2
    dy2 = (rb.y_pos - e[1])**2
    DX2 = np.array([dx2 for i in range(rb.x_dim)], dtype=np.float32).T
    DY2 = np.array([dy2 for i in range(rb.y_dim)], dtype=np.float32)

    return np.array(
           gain / (4.0 * (z_bound[0] -  z_bound[1])) \
           * (1.0 / np.sqrt(DX2 + DY2 + z_bound[1]**2) \
           -  1.0 / np.sqrt(DX2 + DY2 + z_bound[0]**2)),
           dtype=np.float32)

def bin_EL(E, hpxe, rb):
    """
    arguments:
    E   , a np.array of all the diffused electrons from a single hit
    rb, a TrackingPlaneResponseBox
    hpxe, a HPXeEL

    returns:
    the effect of the EL
    -G  the photons produced by each electron in each z-time bin
    -IB the integration boundaries (zdist from el from which photons are being
     cast for each electron for in each time bin)
    """
    # Fraction of Gain and Integration Boundaries for each e for each z in z_pos
    FG  = np.zeros((len(E), len(rb.z_pos)),     dtype=np.float32)
    IB  = np.zeros((len(E), len(rb.z_pos) , 2), dtype=np.float32) + hpxe.d + hpxe.t
    # z in units of time bin
    TS  = (E[:, 2] - rb.z_pos[0]) / rb.z_pitch
    fTS = np.array(np.floor(TS), dtype=np.int16) # int16 if zdim > 127!

    for i, (e, f, ts) in enumerate(zip(E, fTS, TS)):
        for j, z in enumerate(rb.z_pos):

            # electron enters EL between z and z + z_pitch
            if j == f:
                # Calculate gain in first responsive slice
                fg = min((j + 1 - ts) * rb.z_pitch / hpxe.t_el, 1)

                # Compute distance integration boundaries for time bin
                s_d = hpxe.d + hpxe.t
                ib  = [s_d, s_d - fg * hpxe.d]
                FG[i, j] = fg
                IB[i, j] = np.array(ib, dtype=np.float32)

            # electron is still crossing EL
            elif j > f and z < e[2] + hpxe.t_el and f > 0:
                # electron crossing EL over entire time bin
                if z + rb.z_pitch  <= e[2] + hpxe.t_el:
                    fg = rb.z_pitch / hpxe.t_el
                    s_d = hpxe.d + hpxe.t - (z - e[2]) / hpxe.t_el * hpxe.d
                    ib  = [s_d, s_d - fg * hpxe.d]
                    FG[i, j] = fg
                    IB[i, j] = np.array(ib, dtype=np.float32)

                # electron finishes crossing EL during time bin
                else:
                    fg  = (e[2] + hpxe.t_el - z) / hpxe.t_el
                    s_d = hpxe.d + hpxe.t - (z - e[2]) / hpxe.t_el * hpxe.d
                    ib  = [s_d, hpxe.t]
                    FG[i, j] = fg
                    IB[i, j] = np.array(ib, dtype=np.float32)
                    #if not np.allclose(hpxe.t, s_d - fg * hpxe.t_el):
                    #    raise ValueError('final integ boundary != hpxe.t')

    F  = FG * hpxe.Ng
    F += np.random.normal(scale=np.sqrt(F * hpxe.g_fano))
    return F, IB
