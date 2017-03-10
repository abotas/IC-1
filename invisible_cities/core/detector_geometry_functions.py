
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
    Defines a Tracking Plane Box (z will actually be time).

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

        self.P = (self.x_pos, self.y_pos, self.z_pos)
        self.x_dim = len(self.x_pos)
        self.y_dim = len(self.y_pos)
        self.z_dim = len(self.z_pos)

    def in_sipm_plane(self, x, y):
       """Return True if xmin <= x <= xmax and ymin <= y <= ymax"""
       return ((self.x_min <= x <= self.x_max) and
               (self.y_min <= y <= self.y_max))

def find_response_borders(center, pitch, dim, absmin, absmax):
    """
    Helps initialize TrackingPlaneResponseBox, by finding the borders of the
    responsive SiPM box.

    May need to optimize because find_response_borders is run 3 times / hit

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

class TrackingPlaneResponseBox(TrackingPlaneBox):
    """
    A sub Box of TrackingPlaneBox. It is a super class of TrackingPlaneBox
    because it is a box, and it requires x_pitch, y_pitch, z_pitch params.

    (For now it seems useful to separate TrackingPlaneBox and
    TrackingPlaneRespBox because there are times when we want a box with pitch
    but it is fixed and there is no need for finding the center)

    x,y,z_dim define the dimensions of the responsive tracking box within
    the full size tracking box

    x,y,z_absmin, together with x,y,z pitch circumscribe the entire tracking
    box. they are necessary to determine the absolute positions of SiPMs in the
    TrackingPlaneResponseBox. Notice, with only x_pitch=10 and x_dim=3 I can
    not know if inside Tracking plane box there are SiPMs at x positions
    0cm, 1cm, 2cm or 0.5cm, 1.5cm, 2.5cm etc

    R is the response of the response box
    """
    def __init__(self, x_center, y_center, z_center,
                       x_pitch  =   10 * units.mm,
                       y_pitch  =   10 * units.mm,
                       z_pitch  =    2 * units.mus,

                       # Parameters added at this level
                       x_dim    =    8,
                       y_dim    =    8,
                       z_dim    =    2,
                       x_absmin = -235 * units.mm,
                       y_absmin = -235 * units.mm,
                       z_absmin =    0 * units.mus,
                       x_absmax =  235 * units.mm,
                       y_absmax =  235 * units.mm,
                       z_absmax =  530 * units.mus):

        if x_dim * x_pitch + x_absmin > x_absmax: raise ValueError('xdim too large')
        if y_dim * y_pitch + y_absmin > y_absmax: raise ValueError('ydim too large')
        if z_dim * z_pitch + z_absmin > z_absmax: raise ValueError('zdim too large')

        # Written in base class
        #self.x_dim    = x_dim
        #self.y_dim    = y_dim
        #self.z_dim    = z_dim   # initialize a response box
        self.R        = np.zeros((x_dim, y_dim, z_dim), dtype=np.float32)

        self.x_absmin = x_absmin
        self.y_absmin = y_absmin
        self.z_absmin = z_absmin
        self.x_absmax = x_absmax
        self.y_absmax = y_absmax
        self.z_absmax = z_absmax

        # Compute borders
        self.rx_center, self.x_min, self.x_max = find_response_borders(
            x_center, x_pitch, x_dim, x_absmin, x_absmax)
        self.ry_center, self.y_min, self.y_max = find_response_borders(
            y_center, y_pitch, y_dim, y_absmin, y_absmax)
        self.rz_center, self.z_min, self.z_max = find_response_borders(
            z_center, z_pitch, z_dim, z_absmin, z_absmax)

        TrackingPlaneBox.__init__(self,
                                  x_min   = self.x_min,
                                  x_max   = self.x_max,
                                  y_min   = self.y_min,
                                  y_max   = self.y_max,
                                  z_min   = self.z_min,
                                  z_max   = self.z_max,
                                  x_pitch = x_pitch,
                                  y_pitch = y_pitch,
                                  z_pitch = z_pitch)

    def situate(self, tpb):
        """
        situate returns the indices indicating where in the larger
        TrackingPlaneBox (tpb), TrackingPlaneResponseBox (self) is
        """
        if (self.x_pitch != tpb.x_pitch or
            self.y_pitch != tpb.y_pitch or
            self.z_pitch != tpb.z_pitch):
            print(self.x_pitch, tpb.x_pitch)
            print(self.y_pitch, tpb.y_pitch)
            print(self.z_pitch, tpb.z_pitch)
            raise ValueError('self and tpb have incompatible pitch')

        # Compute min indices
        ix_s = (self.x_min - tpb.x_min) / tpb.x_pitch
        iy_s = (self.y_min - tpb.y_min) / tpb.y_pitch
        iz_s = (self.z_min - tpb.z_min) / tpb.z_pitch
        if not np.isclose(ix_s % 1, 0): raise ValueError('ix_s (indx) is not an integer')
        if not np.isclose(iy_s % 1, 0): raise ValueError('iy_s (indx) is not an integer')
        if not np.isclose(iz_s % 1, 0): raise ValueError('iz_s (indx) is not an integer')

        # compute max indices --non inclusive--
        ix_f = ix_s + self.x_dim
        iy_f = iy_s + self.y_dim
        iz_f = iz_s + self.z_dim

        inds = np.array([ix_s, ix_f, iy_s, iy_f, iz_s, iz_f], dtype=np.float32)
        inds = np.array(np.round(inds), dtype=np.int32)
        return inds
