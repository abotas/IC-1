import numpy  as np

from invisible_cities.core.detector_response_functions import generate_ionization_electrons, \
     diffuse_electrons, sliding_window, bin_EL, SiPM_response
    
from invisible_cities.core.configure import configure
from   invisible_cities.cities.anastasia import Anastasia, ANASTASIA

    
    
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
    """
    
    1) check that sliding window is inside bounds perscribed by
    min_xp, min_yp etc
    
    2) check that electrons of distance greater than d_cut
    from the window in x or y are discarded
    
    3) check that electrons that occur more than EL_traverse_time
    from window in z are discarded
    
    4) check to see that sliding window cuts events with
    more than x% of energy outside the window
    
    5) check to see that window creates correct window
    
    kwargs:
    T, is the outside-window-energy-fraction, above which
    events are cut. 
    """
    
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
    EPOS = sliding_window(E_1, 20, 60, 
                          10, 2.0, 
                          -235, 235,
                          -235, 235, 
                          0, 530, 
                          15, 
                          2.0,
                          1.0, 
                          0.05)
    
    assert EPOS != 'Window Cut'
    
    try:    (E_1_out, xpos, ypos, zpos) = EPOS
    except: ValueError
    
    # (1, 2, 3)
    assert len(E_1_out) == 994                      
    assert xpos[0]  == max_xp - (xydim - 1) * xypitch
    assert xpos[-1] == max_xp
    assert ypos[0]  == -5 - (xydim / 2 - 1) * xypitch
    assert ypos[-1] == -ypos[0]
    assert zpos[0]  == min_zp
    assert zpos[-1] == min_zp + (zdim - 1) * int(zpitch)
    
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
    

    
    
