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
                       z_min =    0 * units.mm,
                       z_max =  530 * units.mm,
                       x_pitch = 10 * units.mm,
                       y_pitch = 10 * units.mm,
                       z_pitch = 2  * units.mus):

        Box.__init__(self, x_min=x_min, x_max=x_max, y_min=y_min,
                           y_max=y_max, z_min=z_min, z_max=z_max)

        self.x_pitch = x_pitch
        self.y_pitch = y_pitch
        self.z_pitch = z_pitch

    def in_sipm_plane(self, x, y):
       """Return True if xmin <= x <= xmax and ymin <= y <= ymax"""
       return ( (self.x_min <= x <= self.x_max) and
                (self.y_min <= y <= self.y_max))

def find_response_borders(center, pitch, dim):
    if pitch % 2 == 1: raise ValueError('Odd Pitch')
    if dim % 2 == 0:
        r_center = round(center / float(pitch)) * pitch
        d_min    = r_center - int(dim * pitch / 2.0 - pitch / 2.0)
        d_max    = r_center - int(dim * pitch / 2.0 - pitch / 2.0)
        return r_center, d_min, d_max

    elif dim % 2 == 1: raise ValueError('odd dim not yet supported')
    else:              raise ValueError('x,y,z_dim must be whole number')

class TrackingPlaneResponseBox(TrackingPlaneBox):
    """
    A sub Box of TrackingPlaneBox. It is a super class of TrackingPlaneBox
    because it is a box, and it requires x_pitch, y_pitch, z_pitch params.

    (For now it seems useful to separate TrackingPlaneBox and
    TrackingPlaneRespBox because there are times when we want a box with pitch
    but it is fixed and there is no need for finding the center)
    """
    def __init__(self, x_center = 0 * units.cm,
                       y_center = 0 * units.cm,
                       z_center = 0 * units.cm,
                       x_pitch  = 1 * units.cm,
                       y_pitch  = 1 * units.cm,
                       z_pitch  = 2 * units.mus,

                       # Parameters added at this level
                       x_dim    = 5,
                       y_dim    = 5,
                       z_dim    = 2):

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim

        # Compute borders
        self.rx_center, self.x_min, self.x_max = find_response_borders(
                                                     x_center, x_pitch, x_dim)
        self.ry_center, self.y_min, self.y_max = find_response_borders(
                                                     y_center, y_pitch, y_dim)
        self.rz_center, self.z_min, self.z_max = find_response_borders(
                                                     z_center, z_pitch, z_dim)

        TrackingPlaneBox.__init__(x_min   = self.x_min,
                                  x_max   = self.x_max,
                                  y_min   = self.y_min,
                                  y_max   = self.y_max,
                                  z_min   = self.z_min,
                                  z_max   = self.z_max,
                                  x_pitch = x_pitch,
                                  y_pitch = y_pitch,
                                  z_pitch = z_pitch)
