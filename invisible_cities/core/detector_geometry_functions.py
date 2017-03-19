
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

class TrackingPlaneBox(Box):
    """
    Defines a Tracking Plane Box. Note in a TrackingPlaneBox
    the z dimension is in units of time.

    Additionally, a TrackingPlaneBox contains SiPMs at regular pitches
    in x, y,and z.
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
        self.x_pos = np.linspace(self.x_min,  self.x_max,
                                (self.x_max - self.x_min) / self.x_pitch + 1,
                                 dtype=np.float32)
        self.y_pos = np.linspace(self.y_min,  self.y_max,
                                (self.y_max - self.y_min) / self.y_pitch + 1,
                                 dtype=np.float32)
        self.z_pos = np.linspace(self.z_min,  self.z_max,
                                (self.z_max - self.z_min) / self.z_pitch + 1,
                                 dtype=np.float32)

        self.P     =     (self.x_pos,      self.y_pos,      self.z_pos)
        self.shape = (len(self.x_pos), len(self.y_pos), len(self.z_pos))
        self.resp  = np.zeros((self.shape[0], self.shape[1], self.shape[2]),
                              dtype=np.float32)

    def clear_response(self):
        self.resp = np.zeros((self.shape[0], self.shape[1], self.shape[2]),
                              dtype=np.float32)

    def in_sipm_plane(self, x, y):
       """Return True if xmin <= x <= xmax and ymin <= y <= ymax"""
       return ((self.x_min <= x <= self.x_max) and
               (self.y_min <= y <= self.y_max))

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
    if   dim <= 0: raise ValueError('dimension size must be greater than 0')

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

class MiniTrackingPlaneBox:
    """
    A box within a TrackingPlaneBox. It is a super class of TrackingPlaneBox.

    MiniTrackingPlaneBox is useful, more so than just a smaller instance of
    TrackingPlaneBox, because it is aware it is within a larger TrackingPlaneBox.

    Upon initialization, it self-centers around an x,y,z coordinate within a
    full-size tracking plane box and can describe its own position within the
    full-size tpbox with self.situate()
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

    def center(self, hit, shape):
        # I wonder if we want to allow shape to change from hit to hit, hits
        # that are farther away from EL will create ionization electrons that
        # will diffuse more.

        # TODO: Profile to see if this is slow
        if shape[0] * self.x_pitch + self.x_absmin > self.x_absmax:
            raise ValueError('xdim too large')
        if shape[1] * self.y_pitch + self.y_absmin > self.y_absmax:
            raise ValueError('ydim too large')
        if shape[2] * self.z_pitch + self.z_absmin > self.z_absmax:
            raise ValueError('zdim too large')

        self.shape = shape

        self.rx_center, self.x_min, self.x_max = find_response_borders(
            hit[0], shape[0], self.x_pitch, self.x_absmin, self.x_absmax)
        self.ry_center, self.y_min, self.y_max = find_response_borders(
            hit[1], shape[1], self.y_pitch, self.y_absmin, self.y_absmax)
        self.rz_center, self.z_min, self.z_max = find_response_borders(
            hit[2], shape[2], self.z_pitch, self.z_absmin, self.z_absmax)

        self.x_pos = np.linspace(self.x_min,  self.x_max,
                                (self.x_max - self.x_min) / self.x_pitch + 1,
                                 dtype=np.float32)
        self.y_pos = np.linspace(self.y_min,  self.y_max,
                                (self.y_max - self.y_min) / self.y_pitch + 1,
                                 dtype=np.float32)
        self.z_pos = np.linspace(self.z_min,  self.z_max,
                                (self.z_max - self.z_min) / self.z_pitch + 1,
                                 dtype=np.float32)

        self.P     = (self.x_pos, self.y_pos, self.z_pos)
        self.resp  = np.zeros((self.shape[0], self.shape[1], self.shape[2]),
                              dtype=np.float32)

    def situate(self, tpbox):
        """
        situate returns the indices indicating where in the larger
        TrackingPlaneBox (tpbox), TrackingPlaneResponseBox (self) is

        should situate not just 'situate' but integrate self.resp into
        tpb.resp?
        """
        if (self.x_pitch != tpbox.x_pitch or
            self.y_pitch != tpbox.y_pitch or
            self.z_pitch != tpbox.z_pitch):
            raise ValueError('self and tpb have incompatible pitch')

        # Compute min indices
        ix_s = (self.x_min - tpbox.x_min) / tpbox.x_pitch
        iy_s = (self.y_min - tpbox.y_min) / tpbox.y_pitch
        iz_s = (self.z_min - tpbox.z_min) / tpbox.z_pitch
        if not np.isclose(ix_s % 1, 0): raise ValueError('ix_s (indx) not an integer')
        if not np.isclose(iy_s % 1, 0): raise ValueError('iy_s (indx) not an integer')
        if not np.isclose(iz_s % 1, 0): raise ValueError('iz_s (indx) not an integer')

        # compute max indices --non inclusive--
        ix_f = ix_s + self.shape[0]
        iy_f = iy_s + self.shape[1]
        iz_f = iz_s + self.shape[2]

        inds = np.array([ix_s, ix_f, iy_s, iy_f, iz_s, iz_f], dtype=np.float32)
        inds = np.array(np.round(inds), dtype=np.int32)
        return inds
