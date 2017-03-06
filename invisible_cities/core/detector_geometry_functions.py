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
    Defines a Tracking Plane Box (z will actually be time)
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

class TrackingPlaneResponseBox(TrackingPlaneBox):
    """
    A sub Box of TrackingPlaneBox
    xy_tol = distance from window in x OR y past which electrons are discarded
     z_tol = distance from window in  time  past which electrons are discarded
    ev_cut = energy fraction of evt outside window past which events are discarded

    x,y,z_dim are the dimensions of the SiPM window for which we will calculate
    responses. Here, x_min, x_max ... etc are functions of x_dim, y_dim etc not
    the other way around. This is because from -e to -e the borders change
    depending on the mean position of the drifting electrons,
    but x,y,z_dim stay the same
    """
    def __init__(self, x_center=0, y_center=0, z_center=0,
                       x_pitch = 1 * units.cm,
                       y_pitch = 1 * units.cm,
                       z_pitch = 2 * units.mus,

                       # Parameters addded at this level
                       x_dim   = 20,
                       y_dim   = 20,
                       z_dim   = 60):

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim

        # Compute borders



        TrackingPlaneBox.__init__(x_min   = self.x_min,
                                  x_max   = self.x_max,
                                  y_min   = self.y_min,
                                  y_max   = self.y_max,
                                  z_min   = self.z_min,
                                  z_max   = self.z_max,
                                  x_pitch = x_pitch,
                                  y_pitch = y_pitch,
                                  z_pitch = z_pitch)
