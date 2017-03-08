import numpy  as np
import tables as tb

from os.path import expandvars
from pytest  import fixture, mark


from invisible_cities.core.detector_response_functions import \
     generate_ionization_electrons, \
     diffuse_electrons, \
     sliding_window, \
     bin_EL, \
     SiPM_response, \
     HPXeEL, \
     gather_montecarlo_hits

from invisible_cities.core.detector_geometry_functions import Box, \
     TrackingPlaneBox, TrackingPlaneResponseBox, find_response_borders

from invisible_cities.core.configure         import configure
from invisible_cities.cities.anastasia       import Anastasia, ANASTASIA
from invisible_cities.core.system_of_units_c import units

config_file_format = """
# set_input_files
PATH_IN {PATH_IN}
FILE_IN {FILE_IN}

PATH_OUT {PATH_OUT}
FILE_OUT {FILE_OUT}
COMPRESSION {COMPRESSION}
RUN_NUMBER {RUN_NUMBER}
NPRINT {NPRINT}


NEVENTS {NEVENTS}
RUN_ALL {RUN_ALL}


electrons_prod_F {electrons_prod_F}
max_energy {max_energy}
reduce_electrons {reduce_electrons}
w_val {w_val}

longitudinal_diffusion {longitudinal_diffusion}
transverse_diffusion {transverse_diffusion}
drift_speed {drift_speed}
window_energy_threshold {window_energy_threshold}
d_cut {d_cut}
min_xp {min_xp}
max_xp {max_xp}
min_yp {min_yp}
max_yp {max_yp}
min_zp {min_zp}
max_zp {max_zp}
el_sipm_d {el_sipm_d}
el_width {el_width}
xydim {xydim}
zdim {zdim}
xypitch {xypitch}
zpitch {zpitch}

gain_nf {gain_nf}
t_gain {t_gain}
el_traverse_time {el_traverse_time}
photon_detection_noise {photon_detection_noise}
zmear {zmear}
"""

"""
def config_file_spec_with_tmpdir(tmpdir):
    return dict(PATH_IN  = '$ICDIR/database/test_data/',
                FILE_IN  = 'NEW_se_mc_1evt.h5',
                PATH_OUT = str(tmpdir),
                FILE_OUT = 'Anastasia_1evt.h',
                COMPRESSION = 'ZLIB4',
                RUN_NUMBER  =     0,
                NEVENTS     =     1,
                NPRINT      =     0,
                RUN_ALL     = False,

                electrons_prod_F = False,
                max_energy       = 2.6e6,
                reduce_electrons = 100  ,
                w_val            = 22.4 ,

                longitudinal_diffusion  = 3  ,
                transverse_diffusion    = 10 ,
                drift_speed             = 1.0,
                window_energy_threshold = 0.05,
                d_cut     =  15 ,
                min_xp    = -235,
                max_xp    =  235,
                min_yp    = -235,
                max_yp    =  235,
                min_zp    =  0  ,
                max_zp    =  530,
                el_sipm_d =  5  ,
                el_width  =  5  ,
                xydim     =  20 ,
                zdim      =  60 ,
                xypitch   =  10 ,
                zpitch    =  2.0,

                gain_nf   = 0.0,
                t_gain    = 1050.0,
                el_traverse_time = 2,
                photon_detection_noise = False,
                zmear = True)

def test_command_line_Anastasia(config_tmpdir):
    config_file_spec = config_file_spec_with_tmpdir(config_tmpdir)
    config_file_contents = config_file_format.format(**config_file_spec)
    conf_file_name = str(config_tmpdir.join('test-2-Anast.conf'))
    with open(conf_file_name, 'w') as conf_file:
        conf_file.write(config_file_contents)
    ANASTASIA(['ANASTASIA', '-c', conf_file_name])

def test_sliding_window(config_tmpdir):

    max_xp = 235; min_xp = -235
    max_yp = 235; min_yp = -235
    max_zp = 530; min_zp =  0
    d_cut  = 15
    drift_speed = 1.0
    el_traverse_time = 2
    xydim = 20; zdim = 60
    xypitch = 10; zpitch = 2.0
    window_energy_threshold = 0.05

    E_1     = np.empty((1000, 3), dtype=np.float32)
    E_1[:,] = np.array([max_xp + 1, 0, min_zp])

    # Exceed x boundaries
    E_1[0, 0] = max_xp + d_cut + 1
    E_1[2, 0] = min_xp - d_cut - 1

    # Exceed y boundaries
    E_1[5, 1] =  d_cut + max_yp + 1
    E_1[6, 1] = -d_cut + min_yp - 1

    # Check z boundaries
    E_1[8, 2]  =  el_traverse_time + max_zp / drift_speed + 1
    E_1[-1, 2] = -el_traverse_time - 1

    # Check redundancy
    E_1[2, 1] =  d_cut + max_yp + 1
    E_1[8, 0] = -d_cut + min_xp - 1

    # (4)
    EPOS = sliding_window(E_1, xydim, zdim, xypitch, zpitch, min_xp, max_xp,
                   min_yp, max_yp, min_zp, max_zp, d_cut, el_traverse_time,
                   drift_speed, window_energy_threshold)

    assert EPOS != 'Window Cut'

    try:    (E_1_out, xpos, ypos, zpos) = EPOS
    except: ValueError

    # (1, 2, 3)
    assert len(E_1_out) == 994
    assert xpos[ 0] == max_xp - (xydim - 1) * xypitch
    assert xpos[-1] == max_xp
    assert ypos[ 0] == -5 - (xydim / 2 - 1) * xypitch
    assert ypos[-1] == ypos[0] + xypitch * (xydim - 1)
    assert zpos[ 0] == min_zp
    assert zpos[-1] == min_zp + (zdim - 1) * zpitch

    # (4)
    assert sliding_window(E_1[:100],
                          xydim, zdim,
                          xypitch, zpitch,
                          min_xp, max_xp,
                          min_yp, max_yp,
                          min_zp, max_zp,
                          d_cut,
                          el_traverse_time,
                          drift_speed,
                          window_energy_threshold) == 'Window Cut'

"""

def test_SiPM_response():
    """
    Check that SiPM_response returns correct map for a couple
    electrons

    ** MUST UPDATE IF P(detection) CHANGED! **
    """

    # Sample electron/window
    E = np.array([[3, 41, 0]])
    xpos = np.array(range(5))*10 + 5
    ypos = np.array(range(5))*10 + 5
    gain = 1.0
    xydim = len(xpos)
    el_sipm_d = 5
    el_width  = 5
    m = np.zeros((len(xpos), len(ypos)), dtype=np.float32)

    # Get map
    for e in E:
        m += SiPM_response(e, xpos, ypos, xydim,
                           [el_width + el_sipm_d, el_sipm_d],
                           gain)

    # Check equality
    for row, x in zip(m, xpos):
        for resp, y in zip(row, ypos):

            check = gain * np.array([1.0 / np.sqrt(
                (E[0, 0] - x)**2 + (E[0, 1] - y)**2 + el_sipm_d**2) - \
                1  / np.sqrt((
                E[0, 0] - x)**2 + (E[0, 1] - y)**2 + (el_sipm_d + el_width)**2)],
                dtype=np.float32) / float(4) / float(el_width)

            # np.isclose because these are floats
            assert np.isclose(resp, check[0], rtol=1e-8, atol=1e-9)

def test_gather_event_hits():
    Events = gather_montecarlo_hits(
        expandvars('$ICDIR/database/test_data/NEW_se_mc_1evt.h5'))

    f = tb.open_file(
        expandvars('$ICDIR/database/test_data/NEW_se_mc_1evt.h5'), 'r')

    ptab = f.root.MC.MCTracks

    assert ptab.nrows == len(Events[ptab[0]['event_indx']])

    f.close()

def test_box_lengths(b=None):
    if b == None: b = Box(x_min = -335 * units.mm,
                          x_max =  235 * units.mm,
                          y_min =  -35 * units.mm,
                          y_max =    0 * units.mm,
                          z_min =    0 * units.mm,
                          z_max =   53 * units.mm)
    assert b.length_x() == b.x_max - b.x_min
    assert b.length_y() == b.y_max - b.y_min
    assert b.length_z() == b.z_max - b.z_min

def test_box_volume(b=None):
    if b == None: b = Box()
    assert b.volume() == b.length_x() * b.length_y() * b.length_z()

def test_tracking_plane_innerbox():
    b = TrackingPlaneBox()
    test_box_lengths(b)
    test_box_volume(b)

def test_tracking_plane_box_in_sipm_plane_method():
    b = TrackingPlaneBox()
    assert     b.in_sipm_plane(-235   * units.mm, -235 * units.mm)
    assert     b.in_sipm_plane( 235   * units.mm,  235 * units.mm)
    assert     b.in_sipm_plane( 235   * units.mm, -235 * units.mm)
    assert     b.in_sipm_plane(-235   * units.mm,  235 * units.mm)
    assert     b.in_sipm_plane(   1   * units.mm,    0 * units.mm)
    assert not b.in_sipm_plane(-235.1 * units.mm,    0 * units.mm)
    assert not b.in_sipm_plane( 236   * units.mm,    0 * units.mm)
    assert not b.in_sipm_plane(   0   * units.mm,  236 * units.mm)
    assert not b.in_sipm_plane(   0   * units.mm, -236 * units.mm)

def test_HPXeEL_attributes():
    D = HPXeEL()
    energy = np.array([2 * units.keV, 3 * units.keV], dtype=np.float32)
    assert np.isclose(D.YP, 140 * D.EP / units.kilovolt*units.cm*units.bar - 116)
    assert np.isclose(D.Ng, D.YP * D.d / units.cm * D.P / units.bar)

def test_HPXeEL_methods():
    D = HPXeEL()
    energy = np.array([2 * units.keV, 3 * units.keV], dtype=np.float32)
    assert (D.scintillation_photons(energy) == energy * D.rf / D.Ws).all()
    assert (D.ionization_electrons (energy) == energy * D.rf / D.Wi).all()
    assert (D.el_photons           (energy) == energy * D.Ng / D.rf).all()

def test_tracking_plane_response_box_helper_find_response_borders_even_dim():
    hits   =   5.2
    pitch  =  10
    dim    =   4
    absmin = -55
    absmax =  55
    rc, ma, mi = find_response_borders(hits, pitch, dim, absmin, absmax)
    assert rc == 10
    assert ma == rc + (dim / 2.0 * pitch - pitch / 2.0)
    assert mi == rc - (dim / 2.0 * pitch - pitch / 2.0)
    assert len(list(range(int(mi), int(ma + pitch), int(pitch)))) == dim

def test_tracking_plane_response_box_helper_find_response_borders_oddd_dim():
    hits   =   5.2
    pitch  =  10
    dim    =   3
    absmin = -55
    absmax =  55
    rc, ma, mi = find_response_borders(hits, pitch, dim, absmin, absmax)
    assert rc == 5
    assert ma == rc + (dim - 1) / 2.0 * pitch
    assert mi == rc - (dim - 1) / 2.0 * pitch
    assert len(list(range(int(mi), int(ma + pitch), int(pitch)))) == dim
