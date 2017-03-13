
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
                      diff_xy =  10    * units.mm/np.sqrt(units.m),
                      diff_z  =   3    * units.mm/np.sqrt(units.m),
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
        self.Ng      = self.YP * self.d/units.cm * self.P/units.bar #photons/e  TODO: divide by cm?
        self.Ws      = Ws
        self.Wi      = Wi
        self.rf      = rf
        self.ie_fano = ie_fano
        self. g_fano =  g_fano
        self.diff_xy = diff_xy
        self.diff_z  = diff_z

    def scintillation_photons(self, E):
        return E / self.Ws

    def ionization_electrons(self, E):
        return self.rf * E / self.Wi

    #Ng is photons / electron, so why is this dependent on E
    def el_photons(self):
        return self.Ng / self.rf

def gather_montecarlo_hits(filepath):
    """
    gather_montecarlo_hits gathers all the hits from a pytable in filepath and
    puts them in a dictionary. The dictionary maps event index to all of that
    event's hits.

    args
    filepath: the path to an input file

    returns
    file_hits: a dictionary mapping event_indx, named by nexus -->
               hits_ev, a np.array with all the hits in event event_indx
               hits_ev has shape (num hits, 4) and can by index as follows:
               hits_ev[hit, x coordinate, y coordinate, z coordinate, energy].
    """
    f      = tb.open_file(filepath, 'r')
    ptab   = f.root.MC.MCTracks
    hits_f = {}
    ev     = ptab[0]['event_indx']
    s_row  = 0

    # Iterate over the hits
    for row in ptab.iterrows():

        # Check for new events
        if ev != row['event_indx'] or row.nrow == ptab.nrows - 1:
            # Bad karma? A trick to record the last event in the file
            if row.nrow == ptab.nrows - 1: f_row = row.nrow + 1
            else:                          f_row = row.nrow
            hits_ev = np.empty((f_row - s_row, 4), dtype=np.float32)
            hits_ev[:, :3] = ptab[s_row : f_row]['hit_position'] * units.mm
            hits_ev[:,  3] = ptab[s_row : f_row]['hit_energy'  ] * units.MeV
            hits_f[ev]     = hits_ev
            ev    = row['event_indx']
            s_row = row.nrow

    f.close()
    return hits_f

def generate_ionization_electrons(hits_ev, hpxe):
    """
    generate_ionization_electrons generates all the ionization electrons for
    all the hits in one event.

    args
    hits_ev: a np array containing all the hits in an event. hits_ev[i] is the
             ith hit in hits_ev and hits_ev[i] = [xcoord, ycoord, zcoord, enrgy].
             Note, this is the same hits_ev as hits_ev in gather_montecarlo_hits
    hpxe   : an instance of HPXeEL
             .ionization_electrons is used to calculate the number of ionization
             electrons generated by each hit. .ie_fano is used to calculate the
             fluctuations in the number of ionization e- produced by each hit

    returns
    electrons_ev: a dictionary mapping hit index --> a np.array (electrons_h) of
                  (undiffused) ionization electrons produced by that hit.
                  electrons_h has shape (# electrons produced by hit, 3). Since
                  the ionization electrons in electrons_ev have not yet diffused
                  to the EL, each electron from the same hit shares the same
                  coordinates as the hit that produced it and the same coordinates
                  as every other ionization electron produced by the hit.

    """
    electrons_ev = {}
    for i, h in enumerate(hits_ev):
        n_ie  = hpxe.ionization_electrons(h[3])
        n_ie += np.random.normal(scale=np.sqrt(n_ie * hpxe.ie_fano))
        electrons_h     = np.empty((int(round(n_ie)), 3), dtype=np.float32)
        electrons_h[:]  = h[:3]
        electrons_ev[i]  = electrons_h
    return electrons_ev

def diffuse_electrons(electrons, hpxe):
    """
    diffuse_electrons drifts ionization electrons to the EL (converting their
    z coordinates to time of arrival) with diffusion in x, y, and z

    args
    electrons: electrons is a np array of undiffused ionization electrons. the
               numpy array has length equal to the number of ionization
               electrons and electrons[i] = [xcoord, ycoord, zcoord] where each
               of these coordinates has units of distance
    hpxe     : an instance of HPXeEL
               .xy_diff, .z_diff, diffusion constants are used to compute
               the sigmas of the gaussians used to simulate diffusion
               .dV, the drift velocity of ionization electrons in the active
               region is used to calculate the time of arrival of each
               ionization electron to the EL
    returns
    diffused_electrons: a numpy array, the same shape as electrons with the
               difference that gausian noise has been added to the electron
               positions and diffused_electrons[:, 2] is in units of time
    """
    # Avoid modifying in place?
    diffused_electrons = np.copy(electrons)

    # sqrt dist from EL grid
    sd = np.sqrt(electrons[:, 2] / units.mm * units.m)

    # mu=0, sig=lat_diff * sqrt(m)
    # TODO make sure sigma is correct
    latr_drift = np.random.normal(scale=hpxe.diff_xy * np.array([sd, sd]).T)
    long_drift = np.random.normal(scale=hpxe.diff_z  * sd, size=(len(electrons),))
    diffused_electrons[:, :2] += latr_drift
    diffused_electrons[:,  2] += long_drift
    diffused_electrons[:,  2] /= hpxe.dV
    return diffused_electrons

def SiPM_response(rb, e, z_bound, gain):
    """
    SiPM response computes the SiPM response of a plane of SiPMs in one time
    to one electron emmitting photons in the EL.

    args
    rb     : a MiniTrackingPlaneBox that should contain all the SiPMs that in
             close proximity to. .xpos and .ypos are used to calculate the
             distances of the SiPMs (in x and y) in this minibox to the electron
    e      : an electron, a numpy array [xcoord, ycoord, zcoord]
    z_bound: distance boundaries in z from the electron to the SiPMs between
             which the electron is producing photons cast toward the SiPMs in
             the current time bin
    gain   : the number of photons produced by e during this time bin

    returns
    a numpy array containing the SiPM responses to the photons produced by e
    during this time bin
    """

    # Calculate the xy distances only once
    dx2 = (rb.x_pos - e[0])**2
    dy2 = (rb.y_pos - e[1])**2
    DX2 = np.array([dx2 for i in range(rb.shape[0])], dtype=np.float32).T
    DY2 = np.array([dy2 for i in range(rb.shape[1])], dtype=np.float32)

    return np.array(
           gain / (4.0 * (z_bound[0] -  z_bound[1])) \
           * (1.0 / np.sqrt(DX2 + DY2 + z_bound[1]**2) \
           -  1.0 / np.sqrt(DX2 + DY2 + z_bound[0]**2)),
           dtype=np.float32)

def distribute_gain(electrons, hpxe, rb):
    """
    distribute_gain computes the fraction of the gain produced by each electron,
    that is received by each time bin in rb.
    """
    FG  = np.zeros((len(electrons), len(rb.z_pos)), dtype=np.float32)

    # electron z-coordinates in units of time bin
    TS  = (electrons[:, 2] - rb.z_pos[0]) / rb.z_pitch
    fTS = np.array(np.floor(TS), dtype=np.int16)

    for i, (e, f, ts) in enumerate(zip(electrons, fTS, TS)):
        for j, z in enumerate(rb.z_pos):
            # electron enters EL (between time=z and time=z + z_pitch)
            if j == f:
                # Calculate gain in first responsive slice
                FG[i, j] = min((j + 1 - ts) * rb.z_pitch / hpxe.t_el, 1)
            # electron is still crossing EL
            elif j > f and z < e[2] + hpxe.t_el and f > 0:
                # electron crossing EL over entire time bin
                if z + rb.z_pitch  <= e[2] + hpxe.t_el:
                    FG[i, j] = rb.z_pitch / hpxe.t_el
                # electron finishes crossing EL during time bin
                else: FG[i, j] = (e[2] + hpxe.t_el - z) / hpxe.t_el
    return FG

def compute_photon_emmission_boundaries(FG, hpxe, rb):
    """
    photon_emmission_boundaries calculates the
    """
    IB = np.zeros((len(FG), len(rb.z_pos), 2), dtype=np.float32) + (hpxe.d + hpxe.t)

    IB[:, 0, 1] -= FG[:, 0] * hpxe.d
    for i in range(1, len(rb.z_pos)):
        IB[:, i, 0] = IB[:, i - 1, 1]
        IB[:, i, 1] = IB[:, i, 0] - FG[:, i] * hpxe.d

    # Some electrons may have started traversing EL before the rb.zpos[0]
    # Then they've already traveled the distance below before rb.zpos[0]
    IB -= (IB[:, -1, -1] - hpxe.t)[:, None, None]
    return IB



def bin_EL(electrons, hpxe, rb): # TODO make this simpler
    """
    bin_EL allows Anastasia to simulate the effect of the EL

    The effect of the EL:
    (If it takes 2mus for an e- to cross the EL and time bins are 1mus thick,
     each time bin of the will only receive a fraction of the total gain
     produced by a single e-)

    bin_EL
    1) determines what number of photons produced by each ionization electron
    traveling through the EL is received by each time bin of SiPMs.

    2) computes the z-distance boundaries to the SiPMs from which the
    electron crossing the EL is producing photons.

    bin_EL does both of these things beacause the fraction of the gain
    produced by each time bin equals the fraction of the EL the electron has
    crossed during each time bin.

    I can probably split 1 and 2 into two functions by having 1 compute just the
    fraction of the gain, not the total number of photons, and using the
    fraction of the gain in each time bin to calculate the z-distance integration
    boundaries

    arguments:
    electrons, a np.array of all the diffused electrons from a single hit
    rb, a TrackingPlaneResponseBox
    hpxe, a HPXeEL

    returns:
    the effect of the EL
    -G  the photons produced by each electron in each z-time bin
    -IB the integration boundaries (zdist from el from which photons are being
     cast for each electron for in each time bin)
    """
    # Fraction of Gain and Integration Boundaries for each e for each z in z_pos
    n_e = len(electrons)
    FG  = np.zeros((n_e, len(rb.z_pos)),     dtype=np.float32)
    IB  = np.zeros((n_e, len(rb.z_pos) , 2), dtype=np.float32) + hpxe.d + hpxe.t
    # z in units of time bin
    TS  = (electrons[:, 2] - rb.z_pos[0]) / rb.z_pitch
    fTS = np.array(np.floor(TS), dtype=np.int16) # int16 if zdim > 127!

    for i, (e, f, ts) in enumerate(zip(electrons, fTS, TS)):
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

    F  = FG * hpxe.Ng / hpxe.rf
    F += np.random.normal(scale=np.sqrt(F * hpxe.g_fano))

    return F, IB
