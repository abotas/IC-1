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

    def in_sipm_plane(self, x, y):
        """Return True if xmin <= x <= xmax and ymin <= y <= ymax"""
        return ( (self.x_min <= x <= self.x_max) and
                 (self.y_min <= y <= self.y_max))

class TrackingPlaneBox(Box):

    def __init__(self, x_min, x_max, y_min, y_max, z_min, z_max,
                       x_pitch = 1 * units.cm,
                       y_pitch = 1 * units.cm,
                       z_pitch = 2 * units.mus):
        """
       Defines a Tracking Plane Box (z will actually be time)
       """

       self.Box(x_min, x_max, y_min, y_max, z_min, z_max)
       self.x_pitch = x_pitch
       self.y_pitch = y_pitch
       self.z_pitch = z_pitch
