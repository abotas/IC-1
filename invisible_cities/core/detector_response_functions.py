
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
    puts them in a dictionary. The dictionary maps event index to the hits in
    that event.

    args
    filepath: the path to an input file

    returns
    file_hits: a dictionary mapping event_indx, named by nexus -->
               hits_ev, a np.array with all the hits in event event_indx
               hits_ev has shape (num hits, 4) and can by index as follows:
               hits_ev[hit, x coordinate, y coordinate, z coordinate, energy]
               and file_hits[ev] = hits_ev
    """
    with tb.open_file(filepath, 'r') as f:
        ptab   = f.root.MC.MCTracks
        hits_f = {}
        ev     = ptab[0]['event_indx']
        s_row  = 0

        # Iterate over the rows in the pytable; each row contains a distinct hit
        for row in ptab.iterrows():

            # We want to gather the hits event by event so, here we check
            # for a change in event_indx to signal start of new event
            # * 'or row.nrow == ptab.nrows - 1' needed to gather hits from last
            # * event in the file
            if ev != row['event_indx'] or row.nrow == ptab.nrows - 1:

                # Find start and finish rows (hits) in previous event
                if row.nrow == ptab.nrows - 1: f_row = row.nrow + 1
                else: f_row = row.nrow
                hits_ev = np.empty((f_row - s_row, 4), dtype=np.float32)
                # Get extract hit position and energy from pytable
                hits_ev[:, :3] = ptab[s_row : f_row]['hit_position'] * units.mm
                hits_ev[:,  3] = ptab[s_row : f_row]['hit_energy'  ] * units.MeV
                hits_f[ev]     = hits_ev  # put hits for ev in the hits_f dict
                ev    = row['event_indx'] # update ev
                s_row = row.nrow          # update start row for ev

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

    # Create the ionization e- for each hit
    for i, h in enumerate(hits_ev):
        # determine number of ionization e- produced by h with energy h[3]
        n_ie  = hpxe.ionization_electrons(h[3])
        n_ie += np.random.normal(scale=np.sqrt(n_ie * hpxe.ie_fano))
        electrons_h     = np.empty((int(round(n_ie)), 3), dtype=np.float32)
        electrons_h[:]  = h[:3] # set ionization e- positions to hit position
        electrons_ev[i] = electrons_h # stick in dictionary
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
    diffused_electrons = np.copy(electrons) # TODO time profile
    sd = np.sqrt(electrons[:, 2] / units.mm * units.m)  # sqrt dist from EL grid

    # mu=0, sig=lat_diff * sqrt(m)
    # TODO make sure sigma is correct
    latr_drift = np.random.normal(scale=hpxe.diff_xy * np.array([sd, sd]).T)
    long_drift = np.random.normal(scale=hpxe.diff_z  * sd, size=(len(electrons),))
    diffused_electrons[:, :2] += latr_drift
    diffused_electrons[:,  2] += long_drift
    diffused_electrons[:,  2] /= hpxe.dV
    return diffused_electrons

def prob_SiPM_photon_detection(rb, e, z_bound):
    """
    prob_SiPM_photon_detection computes the SiPM response of a plane of SiPMs
    (one time bin) to one electron emmitting photons in the EL

    args
    rb     : an instance of MiniTrackingPlaneBox or TrackingPlaneBox from which
             .xpos and .ypos are used to calculate the distances of the SiPMs
             (in x and y) to the electron crossing the EL
    e      : an electron, a (3,) shaped numpy array [xcoord, ycoord, zcoord]
    z_bound: distance boundaries in z from the electron to the SiPMs between
             which the electron is producing photons cast toward the SiPMs in
             the current time bin. (e may have entered the EL or may exit the EL
             during another time bin. in this case photons are not being cast
             from across the entire width of the EL, just a subsection of it.)

    returns
    a numpy array of shape (len(rb.x_pos), len(rb.y_pos), len(rb.zpos)),
    containing the SiPM responses to the photons produced by e during this time
    bin.
    """

    # Calculate the xy distances only once
    dx2 = (rb.x_pos - e[0])**2
    dy2 = (rb.y_pos - e[1])**2
    DX2 = np.array([dx2 for i in range(rb.shape[0])], dtype=np.float32).T
    DY2 = np.array([dy2 for i in range(rb.shape[1])], dtype=np.float32)

    return np.array(
           1 / (4.0 * (z_bound[0] -  z_bound[1])) \
           * (1.0 / np.sqrt(DX2 + DY2 + z_bound[1]**2) \
           -  1.0 / np.sqrt(DX2 + DY2 + z_bound[0]**2)),
           dtype=np.float32)

def SiPM_response(electrons, photons, IB, rb):
    """
    SiPM_response gets the SiPM response of rb to electrons, calling
    prob_SiPM_photon_detection to get the SiPM response of each time bin of
    SiPMs to each electrons

    args
    electrons: a np array of electrons where
               electrons[e-] = [x, y, time reached EL]
    photons  : the np array containing the number of photons produced by each
               e- crossing the EL, during each time bin
               photons[e-] = [0, 9123, 1050] (if there were 3 time bins)

    IB       : contains the z_bound for each electron, for each time_bin
               (refer to prob_SiPM_photon_detection docstring for z_bound info)

    rb       : an instance of MiniTrackingPlaneBox. It us used to get the shape
               of the box of SiPMs that will respond to photons produced by
               electrons. It also passes parameters to prob_SiPM_photon_detection

    returns
    resp     : the cumulative response of rb to all the electrons. resp is a np
               array with the sape of rb, where each entry the response of a
               SiPM and resp is organized by resp[x-index, y-index, time-index]
               of SiPM within rb
    """
    resp = np.zeros_like(rb.resp_h)
    for e, f_e, ib_e   in zip(electrons, photons, IB):   # loop over electrons
        for i, (f, ib) in enumerate(zip(f_e, ib_e)):     # loop over time bins
            # Compute SiPM response if gain for this e- and time bin > 0
            if f > 0:
                # rb is the response box,
                # e  is an e- in electrons,
                # ib is an z-integration boundary for this e- for this time bin
                # f  is the number of photons produced by e- in this time bin
                resp[:,:, i] += prob_SiPM_photon_detection(rb, e, ib) * f
    return resp

def distribute_gain(electrons, hpxe, rb):
    """
    distribute_gain computes the fraction of the gain produced by each electron,
    that is received by each time bin of SiPMs in rb. (not all the photons
    produced by an ionization e- are received by one time bin necessarily)

    args
    electrons: electrons: a np array of electrons where
               electrons[e-] = [x, y, time reached EL]
    hpxe     : an instance of HPXeEL. used to get the time it takes to cross
               the EL (t_el), the width of the EL (d), the dist from the end
               of the EL to the SiPM plane (t)
    rb       : an instance of MiniTrackingPlaneBox or TrackingPlaneBox. It is
               used to access information about the z positions of SiPMs for
               which we plan to compute responses to electrons.
    returns
    FG       : a np array containing the fraction of the total gain produced
               during each time bin by each e- crossing the EL
               ex: FG[e-] = [0, .9, .1] (here rb has 3 time bins)
    """
    FG  = np.zeros((len(electrons), len(rb.z_pos)), dtype=np.float32)

    # TS is a np array containing the time that each e- reaches the EL in units
    # of time bin. ex: TS[e-] = 0.25 if e- reached the EL 1/4 of the way thru
    # the 0th time bin
    TS  = (electrons[:, 2] - rb.z_pos[0]) / rb.z_pitch
    fTS = np.array(np.floor(TS), dtype=np.int16)

    for i, (e, f, ts) in enumerate(zip(electrons, fTS, TS)):# loop over electrons
        for j, z in enumerate(rb.z_pos):                    # loop over time bins

            # There are 2 cases in which FG[i,j] != 0
            # Case 1: electron enters EL (between time=z and time=z + z_pitch)
            if j == f:
                # Calculate gain in first responsive slice
                FG[i, j] = min((j + 1 - ts) * rb.z_pitch / hpxe.t_el, 1)

            # Case 2: electron had already entered EL and is still crossing
            elif j > f and z < e[2] + hpxe.t_el and f > 0:

                # 2a: electron crossing EL over entire time bin
                if z + rb.z_pitch  <= e[2] + hpxe.t_el:

                    # FG == time bin size / t to cross EL
                    FG[i, j] = rb.z_pitch / hpxe.t_el

                # 2b: electron finishes crossing EL during time bin
                else:
                    # FG == (t e- crossed EL - t timebin start) / t to cross EL
                    FG[i, j] = (e[2] + hpxe.t_el - z) / hpxe.t_el
    return FG

def distribute_photons(FG, hpxe):
    """
    Uses FG from distribute_gain to distribute photons produced by e- across
    time bins.

    arguments
    FG       : a np array containing the fraction of the total gain produced
               during each time bin by each e- crossing the EL
               ex: FG[e-] = [0, .9, .1] (here rb has 3 time bins)
    hpxe     : an instance of HPXeEL. uses:
               .Ng, gain per ionization e-)
               .rf, reduction factor that reduce computational complexity
               by artificially reducing the number of ionization e- produced by
               each hit. gain produced by these e- is then correspondingly
               increased. This is float in (0, 1]
               g_fano, the fano factor that contributes to fluctuations in
               the number of photons produced by an ionization e-
    """
    # photons = fraction of gain from e- * (gain / e-) / reduction factor
    photons  = FG * hpxe.Ng / hpxe.rf
    photons += np.random.normal(scale=np.sqrt(photons * hpxe.g_fano))

    # TODO time profile
    return photons.round()

def compute_photon_emmission_boundaries(FG, hpxe):
    """
    compute_photon_emmission_boundaries uses FG to compute the z-distances
    between the e- crossing the EL and the SiPM plane during each time bin.
    More specifically, it computes the start distance and the end distance of
    each e- to the SiPM plane during each time bin. These z-distances are a
    necessary parameter to the current prob_SiPM_photon_detection function.
    * compute_photon_emmission_boundaries uses FG to compute these z-dists
    * because we (for now) assume that the e- travel at uniform speed across
    * the EL, and produce photons uniformly while crossing the EL. In this case
    * the fraction of the gain in produced by an e- in each time bin equals the
    * fraction of the EL width traversed during that time bin.

    args
    FG       : a np array containing the fraction of the total gain produced
               during each time bin by each e- crossing the EL
               ex: FG[e-] = [0, .9, .1] (here rb has 3 time bins)
    hpxe     : an instance of HPXeEL. this function uses,
               .d the width of the EL
               .t the distance from the end of the EL to the SiPM plane

    returns
    IB       : a np array containing the initial and final distances between
               each e- and the SiPM plane during each time bin (as the e- is
               crossing the EL)
               IB[e-, time bin] = [initial d to EL, final d to EL]
               These bounds are a necessary input to prob_SiPM_photon_detection
    """

    # Initializing I
    IB = np.zeros((*FG.shape, 2), dtype=np.float32) + (hpxe.d + hpxe.t)

    IB[:, 0, 1] -= FG[:, 0] * hpxe.d
    for i in range(1, FG.shape[1]):
        IB[:, i, 0] = IB[:, i - 1, 1]
        IB[:, i, 1] = IB[:, i,     0] - FG[:, i] * hpxe.d

    # Some electrons may have started traversing EL before the rb.zpos[0]
    # Then they've already traveled the distance below before rb.zpos[0]
    IB -= (IB[:, -1, -1] - hpxe.t)[:, None, None]
    return IB
