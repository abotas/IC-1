
import numpy as np
from   invisible_cities.core.system_of_units_c import units

class Box:

    def __init__(self,
                 x_min = -235 * units.mm,
                 x_max =  235 * units.mm,
                 y_min = -235 * units.mm,
                 y_max =  235 * units.mm,
                 z_min =    0 * units.mm,
                 z_max =  530 * units.mm):
        """
       Defines a Box
       """
        if x_min > x_max or y_min > y_max or z_min > z_max:
            raise ValueError('x,y,or z min > max!')

        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max

    def length_x(self):
        return self.x_max - self.x_min
    def length_y(self):
        return self.y_max - self.y_min
    def length_z(self):
        return self.z_max - self.z_min
    def length(self):
            return np.array([self.length_x(),
                             self.length_y(),
                             self.length_z()])
    def volume(self):
        return  self.length_x() * self.length_y() * self.length_z()

def sipm_pos(d_min, d_max, d_pitch):
    """returns a np array of the SiPM positions along one dimension"""
    return np.linspace(d_min, d_max, (d_max - d_min) / d_pitch + 1,
        dtype=np.float32)

class TrackingPlaneBox(Box):
    """
    Defines a Tracking Plane Box. Note in a TrackingPlaneBox
    the z dimension is in units of time.

    Additionally, a TrackingPlaneBox contains the positions of SiPMS
    """

    def __init__(self, x_min = -235 * units.mm,
                       x_max =  235 * units.mm,
                       y_min = -235 * units.mm,
                       y_max =  235 * units.mm,
                       z_min =    0 * units.mus,
                       z_max =  530 * units.mus,
                       x_pitch = 10 * units.mm,
                       y_pitch = 10 * units.mm,
                       z_pitch =  2 * units.mus):

        Box.__init__(self, x_min=x_min, x_max=x_max, y_min=y_min,
                           y_max=y_max, z_min=z_min, z_max=z_max)

        if z_min < 0:
            raise ValueError('time must be positive')
        if x_pitch <= 0 or y_pitch <= 0 or z_pitch <=0:
            raise ValueError('pitches must be positive')
        if ((x_max - x_min) % x_pitch != 0 or
            (y_max - y_min) % y_pitch != 0 or
            (z_max - z_min) % z_pitch != 0):
            raise ValueError('max - min not divisible by pitch')

        self.x_pitch = x_pitch
        self.y_pitch = y_pitch
        self.z_pitch = z_pitch
        self.x_pos = sipm_pos(x_min, x_max, x_pitch)
        self.y_pos = sipm_pos(y_min, y_max, y_pitch)
        self.z_pos = sipm_pos(z_min, z_max, z_pitch)
        self.P     =     (self.x_pos,      self.y_pos,      self.z_pos)
        self.shape = (len(self.x_pos), len(self.y_pos), len(self.z_pos))

def find_response_borders(center, dim, pitch, absmin, absmax):
    """
    find_response_borders helps initialize MiniTrackingPlaneBox, by
    finding the borders of MiniTrackingPlaneBox around a center, one dimension
    at a time.

    args
    center: an x,y, or z, coordinate around which to build the MiniTPB
    pitch : the pitch of the MiniTPB
    dim   : the number of SiPMs along this dimension of the MiniTPB
    absmin: the absolute minimum position of an SiPM along this dim in the tpbox
    absmax: the absolute maximum position of an SiPM along this dim in the tpbox

    returns
    r_center: the position of a/the SiPM closest to center in this dim
    d_min: the minimum border of the MiniTPB in this dim
    d_max: the maximum border of the MiniTPB in this dim
    """
    if dim <= 0: raise ValueError('dimension size must be greater than 0')

    # even, center box around position between two SiPMs
    elif dim % 2 == 0:
        r_center = absmin + np.floor((center - absmin) / pitch) * pitch + pitch / 2.0
        d_min    = r_center - (dim / 2.0 * pitch - pitch / 2.0)
        d_max    = r_center + (dim / 2.0 * pitch - pitch / 2.0)

    # odd, center box around SiPM
    elif dim % 2 == 1:
        r_center = absmin + round((center - absmin) / pitch) * pitch
        d_min    = r_center - ((dim - 1) / 2.0 * pitch)
        d_max    = r_center + ((dim - 1) / 2.0 * pitch)

    else: raise ValueError('x,y,z_dim must be whole number')

    # ensure within full sized TrackingPlaneBox
    if d_min < absmin:
        shift  = absmin - d_min
        d_min += shift; d_max += shift; r_center += shift
    elif d_max > absmax:
        shift  = d_max - absmax
        d_min -= shift; d_max -= shift; r_center -= shift

    return r_center, d_min, d_max

def determine_hrb_size(hit_zd, hpxe, tpbox, nsig=3):
    """
    determine_hrb_size determines the dimensions of the tracking plane box that
    are expected to respond to the photons produced by the ionization e-
    produced by hit.

    args
    hit_zd: the distance of the hit to the EL
    hpxe  : an instance of an HPXeEL, necessary to know the drift speed,
            and diffusion constants in the current configuration
    tpbox : an instance of TrackingPlaneBox, necessary to know the pitch
            between SiPMs in x,y,z.
    nsig  : number of sigma that we want

    returns
    x_dim, y_dim, z_dim: the dimensions of the tpbox expected to respond to hit
    """
    # 2x distance after diffusion to EL of ionization e- nsig sigma from mean
    xy_dist = 2 * nsig * hpxe.diff_xy * np.sqrt(hit_zd)
    z_time  = 2 * nsig * hpxe.diff_z  * np.sqrt(hit_zd) / hpxe.dV

    # Find distances in units of SiPMs or pitch x,y,z pitch
    # ** This can/should be refined...
    z_dim   = round((z_time  + hpxe.t_el ) / tpbox.z_pitch) # SiPMs continue to
                                                            # see light for t_el.
    x_dim   = round((xy_dist + 4*units.cm) / tpbox.x_pitch) # SiPMs respond from
    y_dim   = round((xy_dist + 4*units.cm) / tpbox.y_pitch) # to 2cm away on
    return int(max(1,x_dim)), int(max(1,y_dim)), int(max(1,z_dim)) # both sides.

class MiniTrackingPlaneBox:
    """
    A tracking plane box within a TrackingPlaneBox.

    Upon initialization, it takes a tracking plane box to know the absolute
    boundaries in x, y, and time, of the tracking plane responses for an event.

    It can then use a method, center, to self-center a mini tpbox around an
    x,y,z coordinate within the full-size tracking plane box it was initialized
    with.

    Finally, it can a
    """
    def __init__(self, tpbox):
        self.x_absmin = tpbox.x_min
        self.y_absmin = tpbox.y_min
        self.z_absmin = tpbox.z_min
        self.x_absmax = tpbox.x_max
        self.y_absmax = tpbox.y_max
        self.z_absmax = tpbox.z_max
        self.x_pitch  = tpbox.x_pitch
        self.y_pitch  = tpbox.y_pitch
        self.z_pitch  = tpbox.z_pitch

        # SiPM response to an event
        self.resp_ev  = np.zeros((tpbox.shape[0] ,
                                  tpbox.shape[1] ,
                                  tpbox.shape[2]), dtype=np.float32)

    def center(self, p_hit, shape):
        """
        center finds the position of the MiniTrackingPlaneBox within the tpbox
        the it was initialized with. It then sets attributes x,y,z_pos of the
        SiPMs in the mini box and initializes a zeros np array of for SiPM
        responses within the new positions of the mini box

        args
        self :
        p_hit: the position of a hit an with x,y, time positions of where the
               center of the hit is expected to reach the EL
        shape: the dimensions in SiPMs of the mini tracking box to create arond
               p_hit
        """

        if shape[0] * self.x_pitch + self.x_absmin > self.x_absmax:
            raise ValueError('xdim too large')
        if shape[1] * self.y_pitch + self.y_absmin > self.y_absmax:
            raise ValueError('ydim too large')
        if shape[2] * self.z_pitch + self.z_absmin > self.z_absmax:
            raise ValueError('zdim too large')

        self.rx_center, self.x_min, self.x_max = find_response_borders(
            p_hit[0], shape[0], self.x_pitch, self.x_absmin, self.x_absmax)
        self.ry_center, self.y_min, self.y_max = find_response_borders(
            p_hit[1], shape[1], self.y_pitch, self.y_absmin, self.y_absmax)
        self.rz_center, self.z_min, self.z_max = find_response_borders(
            p_hit[2], shape[2], self.z_pitch, self.z_absmin, self.z_absmax)

        self.x_pos  = sipm_pos(self.x_min, self.x_max, self.x_pitch)
        self.y_pos  = sipm_pos(self.y_min, self.y_max, self.y_pitch)
        self.z_pos  = sipm_pos(self.z_min, self.z_max, self.z_pitch)
        self.P      = (self.x_pos, self.y_pos, self.z_pos)
        self.shape  = shape
        self.resp_h = np.zeros(shape, dtype=np.float32)

    def add_hit_resp_to_event_resp(self):
        """
        add the SiPM response to the hit to the SiPM response of an event.
        This requires finding where the minitpbox is situated within the tpbox
        and then adding resp_h to resp_ev
        """

        # Compute min indices
        ix_s = (self.x_min - self.x_absmin) / self.x_pitch
        iy_s = (self.y_min - self.y_absmin) / self.y_pitch
        iz_s = (self.z_min - self.z_absmin) / self.z_pitch
        if not np.isclose(ix_s % 1, 0): raise ValueError('ix_s (indx) not an integer')
        if not np.isclose(iy_s % 1, 0): raise ValueError('iy_s (indx) not an integer')
        if not np.isclose(iz_s % 1, 0): raise ValueError('iz_s (indx) not an integer')

        # compute max indices --non-inclusive--
        ix_f = ix_s + self.shape[0]
        iy_f = iy_s + self.shape[1]
        iz_f = iz_s + self.shape[2]

        inds = np.array([ix_s, ix_f, iy_s, iy_f, iz_s, iz_f], dtype=np.float32)
        [xs, xf, ys, yf, zs, zf] = np.array(np.round(inds),   dtype=np.int32)

        self.resp_ev[xs: xf, ys: yf, zs: zf] += self.resp_h
        return -1

    def clear_event_response(self):
        self.resp_ev = np.zeros(self.resp_ev.shape, dtype=np.float32)
