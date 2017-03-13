import numpy  as np
import tables as tb
import pandas as pd

from os.path import expandvars
from pytest  import fixture, mark

from invisible_cities.core.detector_response_functions import \
     gather_montecarlo_hits, \
     generate_ionization_electrons, \
     diffuse_electrons, \
     distribute_gain, \
     compute_photon_emmission_boundaries, \
     SiPM_response, \
     HPXeEL

from invisible_cities.core.detector_geometry_functions import Box, \
     TrackingPlaneBox, MiniTrackingPlaneBox, find_response_borders

from invisible_cities.core.configure         import configure
from invisible_cities.cities.anastasia       import Anastasia, ANASTASIA
from invisible_cities.core.system_of_units_c import units

config_file_format = """
PATH_IN {PATH_IN}
FILE_IN {FILE_IN}
PATH_OUT {PATH_OUT}
FILE_OUT {FILE_OUT}
COMPRESSION {COMPRESSION}
RUN_NUMBER {RUN_NUMBER}
NPRINT {NPRINT}
NEVENTS {NEVENTS}
ie_fano {ie_fano}
g_fano {g_fano}
rf {rf}
Wi {Wi}
diff_z {diff_z}
diff_xy {diff_xy}
dV {dV}
x_min {x_min}
x_max {x_max}
y_min {y_min}
y_max {y_max}
z_min {z_min}
z_max {z_max}
x_pitch {x_pitch}
y_pitch {y_pitch}
z_pitch {z_pitch}
wx_dim {wx_dim}
wy_dim {wy_dim}
wz_dim {wz_dim}
t_el {t_el}
t {t}
d {d}
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
                ie_fano     =  0.1 ,
                g_fano      =  0.15,
                rf          =  1.0 ,
                Wi          = 22.4 ,
                diff_z      =  3   ,
                diff_xy     = 10   ,
                dV          =  1.0 ,
                x_min = -235,
                x_max =  235,
                y_min = -235,
                y_max =  235,
                z_min =  0  ,
                z_max =  530,
                wx_dim = 8,
                wy_dim = 8,
                wz_dim = 2,
                x_pitch = 10,
                y_pitch = 10,
                z_pitch =  2,
                t_el = 2,
                t    = 5,
                d    = 5)

@mark.slow
def test_command_line_anastasia(config_tmpdir):
    config_file_spec = config_file_spec_with_tmpdir(config_tmpdir)
    config_file_contents = config_file_format.format(**config_file_spec)
    conf_file_name = str(config_tmpdir.join('test-2-Anast.conf'))
    with open(conf_file_name, 'w') as conf_file:
        conf_file.write(config_file_contents)
    ANASTASIA(['ANASTASIA', '-c', conf_file_name])

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
    t = 5; d = 5
    m = np.zeros((len(xpos), len(ypos)), dtype=np.float32)
    tpb = TrackingPlaneBox(x_min=xpos[0], x_max=xpos[-1],
                           y_min=ypos[0], y_max=ypos[-1])
    tpb.shape = (xydim, xydim, 4)

    # Get map
    for e in E: m += SiPM_response(tpb, e, [t + d, d], gain)

    # Check equality
    for row, x in zip(m, xpos):
        for resp, y in zip(row, ypos):

            check = gain * np.array([1.0 / np.sqrt(
                (E[0, 0] - x)**2 + (E[0, 1] - y)**2 + t**2) - \
                1  / np.sqrt((
                E[0, 0] - x)**2 + (E[0, 1] - y)**2 + (t + d)**2)],
                dtype=np.float32) / float(4) / float(d)

            # np.isclose because these are floats
            assert np.isclose(resp, check[0], rtol=1e-8, atol=1e-9)

def test_gather_correct_number_of_hits():
    fp     = expandvars('$ICDIR/database/test_data/NEW_se_mc_1evt.h5')
    Events = gather_montecarlo_hits(fp)
    f      = tb.open_file(fp, 'r')
    ptab   = f.root.MC.MCTracks
    assert ptab.nrows == len(Events[ptab[0]['event_indx']])
    f.close()

def test_correct_number_of_ionization_electrons_generated():
    hE1 = 100 * units.eV
    hE2 =  31 * units.eV
    hits = pd.DataFrame(
        data = np.array([[1, 2, 3, hE1],
                         [4, 5, 6, hE2]], dtype=np.float32))

    Wi = 10 * units.eV
    hpxe = HPXeEL(Wi=10*units.eV, ie_fano=0)
    H = generate_ionization_electrons(hits.values, hpxe)
    assert len(H)    ==  len(hits.values)
    assert len(H[0]) ==  round(hE1 / Wi)
    assert len(H[1]) ==  round(hE2 / Wi)

def test_correct_diffuse_electrons_time_coordinate():
    dV      = 1.11 * units.mm / units.mus
    E       = np.zeros((10, 3)  , dtype=np.float32) * units.mm
    E[:, 2] = np.array(range(10), dtype=np.float32) * units.mm
    d_E     = diffuse_electrons(np.copy(E), HPXeEL(dV=dV, diff_xy=0, diff_z=0))
    assert (E[:, 2] / dV == d_E[:, 2]).all()

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

def test_tracking_plane_box_positons():
    x_min   = -20 * units.mm ; x_max = 20 * units.mm
    y_min   = -20 * units.mm ; y_max = 20 * units.mm
    z_min   =   0 * units.mus; z_max =  6 * units.mus
    x_pitch =   10 * units.mm ;y_pitch = 4 * units.mm; z_pitch = 2 * units.mus
    b = TrackingPlaneBox(x_min = x_min, x_max = x_max,
                         y_min = y_min, y_max = y_max,
                         z_min = z_min, z_max = z_max,
                         x_pitch = x_pitch, y_pitch = y_pitch, z_pitch = z_pitch)
    P = b.P
    print(P[0])
    print('')
    print(np.linspace(x_min, x_max,  9))
    #print(P[1])
    #print(P[2])
    assert(P[0] == np.linspace(x_min, x_max, b.length_x() / b.x_pitch + 1)).all()
    assert(P[1] == np.linspace(y_min, y_max, b.length_y() / b.y_pitch + 1)).all()
    assert(P[2] == np.linspace(z_min, z_max, b.length_z() / b.z_pitch + 1)).all()

def test_HPXeEL_attributes():
    D = HPXeEL()
    energy = np.array([2 * units.keV, 3 * units.keV], dtype=np.float32)
    assert np.isclose(D.YP, 140 * D.EP / units.kilovolt*units.cm*units.bar - 116)
    assert np.isclose(D.Ng, D.YP * D.d / units.cm * D.P / units.bar)

def test_HPXeEL_methods():
    D = HPXeEL(rf=.1)
    energy = np.array([2 * units.keV, 3 * units.keV], dtype=np.float32)
    assert (D.scintillation_photons(energy) == energy        / D.Ws).all()
    assert (D.ionization_electrons (energy) == energy * D.rf / D.Wi).all()
    assert (D.el_photons()                  == D.Ng / D.rf)

def test_mini_tracking_plane_box_helper_find_response_borders_even_dim():
    hits   =   5.2
    pitch  =  10 * units.mm
    dim    =   4
    absmin = -55 * units.mm
    absmax =  55 * units.mm
    rc, mi, ma = find_response_borders(hits, pitch, dim, absmin, absmax)
    assert rc == 10
    assert mi == rc - (dim / 2.0 * pitch - pitch / 2.0)
    assert ma == rc + (dim / 2.0 * pitch - pitch / 2.0)
    assert len(list(range(int(mi), int(ma + pitch), int(pitch)))) == dim

def test_mini_tracking_plane_box_helper_find_response_borders_oddd_dim():
    hits   =   5.2
    pitch  =  10 * units.mm
    dim    =   3
    absmin = -55 * units.mm
    absmax =  55 * units.mm
    rc, mi, ma = find_response_borders(hits, pitch, dim, absmin, absmax)
    assert rc == 5
    assert mi == rc - (dim - 1) / 2.0 * pitch
    assert ma == rc + (dim - 1) / 2.0 * pitch
    assert len(list(range(int(mi), int(ma + pitch), int(pitch)))) == dim

def test_mini_tracking_plane_box_dim():
    shape = (6,7,3)
    b  = TrackingPlaneBox()
    rb = MiniTrackingPlaneBox([0,0,0], b, shape=shape)
    assert rb.shape[0]==shape[0]
    assert rb.shape[1]==shape[1]
    assert rb.shape[2]==shape[2]

def test_mini_tracking_plane_box_situate():
    for x,y,z in [[-235*units.mm, -235*units.mm,   0*units.mus],
                  [   0*units.mm,    0*units.mm,  50*units.mus],
                  [ 235*units.mm,  235*units.mm, 530*units.mus]]:
        tp   = 10 * units.mm
        zp   =  2 * units.mus
        tpb  = TrackingPlaneBox(x_pitch=tp, y_pitch=tp, z_pitch=zp)
        rb   = MiniTrackingPlaneBox([x,y,z],tpb)
        inds = rb.situate(tpb)
        assert (rb.x_pos == tpb.x_pos[inds[0]: inds[1]]).all()
        assert (rb.y_pos == tpb.y_pos[inds[2]: inds[3]]).all()
        assert (rb.z_pos == tpb.z_pos[inds[4]: inds[5]]).all()

def test_distribute_gain_3mus():
    z   = 5.3 * units.mus
    E   = np.array([[0, 0, z]], dtype=np.float32)
    EL  = HPXeEL(ie_fano=0, g_fano=0, t_el=3*units.mus)
    tpbox = TrackingPlaneBox(z_pitch=2*units.mus)
    b0  = MiniTrackingPlaneBox([0,0,5.5*units.mus], tpbox, shape=(5,5,5))
    FG  = distribute_gain(E, EL, b0)
    gf1 = (b0.z_pos[1] + b0.z_pitch - z) / EL.t_el
    gf2 =  b0.z_pitch                    / EL.t_el
    gf3 = 1 - gf1 - gf2
    assert np.allclose(FG[0, 0], 0)
    assert np.allclose(FG[0, 1], gf1)
    assert np.allclose(FG[0, 2], gf2)
    assert np.allclose(FG[0, 3], gf3)
    assert np.allclose(FG[0, 4], 0)

def test_distribute_gain_2mus():
    z   = 5.3 * units.mus
    E   = np.array([[0, 0, z]], dtype=np.float32)
    EL  = HPXeEL(ie_fano=0, g_fano=0, t_el=2*units.mus)
    tpbox = TrackingPlaneBox(z_pitch=2*units.mus)
    b0  = MiniTrackingPlaneBox([0, 0, 5.5*units.mus], tpbox, shape=(5,5,5))
    FG  = distribute_gain(E, EL, b0)
    gf1 = min((b0.z_pos[1] + b0.z_pitch - z) / EL.t_el, 1)
    gf2 = 1-gf1
    assert np.allclose(FG[0, 0], 0)
    assert np.allclose(FG[0, 1], gf1)
    assert np.allclose(FG[0, 2], gf2)
    assert np.allclose(FG[0, 3], 0)
    assert np.allclose(FG[0, 4], 0)

def test_distribute_gain_noELt():
    z   = 5.3 * units.mus
    E   = np.array([[0, 0, z]], dtype=np.float32)
    EL  = HPXeEL(ie_fano=0, g_fano=0, t_el=.01*units.mus)
    tpbox = TrackingPlaneBox(z_pitch=2*units.mus)
    b0  = MiniTrackingPlaneBox([0, 0, 5.5*units.mus], tpbox, shape=(5,5,5))
    FG  = distribute_gain(E, EL, b0)
    assert np.allclose(FG[0, 0], 0)
    assert np.allclose(FG[0, 1], 1)
    assert np.allclose(FG[0, 2], 0)
    assert np.allclose(FG[0, 3], 0)
    assert np.allclose(FG[0, 4], 0)

def test_bin_EL_integration_boundaries():
    z   = 5.3 * units.mus
    E   = np.array([[0, 0, z]], dtype=np.float32)
    EL  = HPXeEL(ie_fano=0, g_fano=0, t_el=3*units.mus)
    tpbox = TrackingPlaneBox()
    b0  = MiniTrackingPlaneBox([0, 0,  5.5 * units.mus], tpbox, shape=(5,5,5))
    gf1 = (b0.z_pos[1] + b0.z_pitch - z) / EL.t_el
    gf2 =  b0.z_pitch                    / EL.t_el
    gf3 = 1 - gf1 - gf2
    FG  = np.array([[0, gf1, gf2, gf3, 0]])
    IB  = compute_photon_emmission_boundaries(FG, EL, b0)
    ib0 = np.array([EL.d + EL.t, EL.d + EL.t],    dtype=np.float32)
    ib1 = np.array([ib0[0], ib0[1] - gf1 * EL.d], dtype=np.float32)
    ib2 = np.array([ib1[1], ib1[1] - gf2 * EL.d], dtype=np.float32)
    ib3 = np.array([ib2[1], ib2[1] - gf3 * EL.d], dtype=np.float32)
    assert np.allclose(IB[0, 0], ib0)
    assert np.allclose(IB[0, 1], ib1)
    assert np.allclose(IB[0, 2], ib2)
    assert np.allclose(IB[0, 3], ib3)
    assert IB[0, 4, 0] == IB[0, 4, 1]
