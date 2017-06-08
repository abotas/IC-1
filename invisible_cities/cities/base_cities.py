"""
This module defines base classes for the IC cities. The classes are:
City: Handles input and output files, compression, and access to data base
DeconvolutionCity: A City that performs deconvolution of the PMT RWFs
CalibratedCity: A DeconvolutionCity that perform the calibrated sum of the
                PMTs and computes the calibrated signals in the SiPM plane.
PmapCity: A CalibratedCity that computes S1, S2 and S2Si that togehter
          constitute a PMAP.
SensorResponseCity: A city that describes sensor response

Authors: J.J. Gomez-Cadenas and J. Generowicz.
Feburary, 2017.
"""

import sys

import numpy  as np
import tables as tb

from .. core.configure         import print_configuration
from .. core.exceptions        import NoInputFiles
from .. core.exceptions        import NoOutputFile
from .. core.system_of_units_c import units
from .. core                   import fit_functions        as fitf

from .. database import load_db

from ..io                 import pmap_io          as pio
from ..io.dst_io          import KrEvent

from ..reco               import peak_functions_c as cpf
from ..reco               import peak_functions   as pf
from ..reco               import pmaps_functions  as pmp
from ..reco               import tbl_functions    as tbf
from ..reco               import wfm_functions    as wfm
from ..reco.params        import SensorParams
from ..reco.nh5           import DECONV_PARAM
from ..reco.corrections   import Correction
from ..reco.corrections   import Fcorrection
from ..reco.corrections   import LifetimeCorrection
from ..reco.xy_algorithms import barycenter

from ..sierpe             import blr
from ..sierpe             import fee as FE

if sys.version_info >= (3,5):
    # Exec to avoid syntax errors in older Pythons
    exec("""def merge_two_dicts(a,b):
               return {**a, **b}""")
else:
    def merge_two_dicts(a,b):
        c = a.copy()
        c.update(b)
        return c


class City:
    """Base class for all cities.
       An IC city consumes data stored in the input_files and produce new data
       which is stored in the output_file. In addition to setting input and
       output files, the base class sets the print frequency and accesses
       the data base, storing as attributed several calibration coefficients

     """

    def __init__(self,
                 run_number  = 0,
                 files_in    = None,
                 file_out    = None,
                 compression = 'ZLIB4',
                 nprint      = 10000):

        self.run_number     = run_number
        self.nprint         = nprint  # default print frequency
        self.input_files    = files_in
        self.output_file    = file_out
        self.compression    = compression
        # access data base
        DataPMT             = load_db.DataPMT (run_number)
        DataSiPM            = load_db.DataSiPM(run_number)
        self.det_geo        = load_db.DetectorGeo()

        # This is JCK-1: text reveals symmetry!
        self.xs              = DataSiPM.X.values
        self.ys              = DataSiPM.Y.values
        self.pmt_active      = np.nonzero(DataPMT.Active.values)[0].tolist()
        self.adc_to_pes      = abs(DataPMT.adc_to_pes.values).astype(np.double)
        self.sipm_adc_to_pes = DataSiPM.adc_to_pes.values    .astype(np.double)
        self.coeff_c         = DataPMT.coeff_c.values        .astype(np.double)
        self.coeff_blr       = DataPMT.coeff_blr.values      .astype(np.double)
        self.noise_rms       = DataPMT.noise_rms.values      .astype(np.double)

        self.DataPMT  = DataPMT
        self.DataSiPM = DataSiPM

    @property
    def monte_carlo(self):
        return self.run_number <= 0

    def check_files(self):
        if not self.input_files:
            raise NoInputFiles('input file list is empty, must set before running')
        if not self.output_file:
            raise NoOutputFile('must set output file before running')

    def conditional_print(self, evt, n_events_tot):
        if n_events_tot % self.nprint == 0:
            print('event in file = {}, total = {}'
                  .format(evt, n_events_tot))

    def max_events_reached(self, nmax, n_events_in):
        if nmax < 0:
            return False
        if n_events_in == nmax:
            print('reached max nof of events (= {})'
                  .format(nmax))
            return True
        return False


    def display_IO_info(self, nmax):
        print("""
                 {} will run a max of {} events
                 Input Files = {}
                 Output File = {}
                          """.format(self.__class__.__name__,
                                     nmax, self.input_files, self.output_file))

    def print_configuration(self, sp):
        print_configuration({"# PMT"        : sp.NPMT,
                             "PMT WL"       : sp.PMTWL,
                             "# SiPM"       : sp.NSIPM,
                             "SIPM WL"      : sp.SIPMWL})

    def get_rwf_vectors(self, h5in):
        "Return RWF vectors and sensor data."
        pmtrwf, sipmrwf, pmtblr  = self._get_rwf(h5in)
        NEVT_pmt , NPMT,   PMTWL = pmtrwf .shape
        NEVT_simp, NSIPM, SIPMWL = sipmrwf.shape
        assert NEVT_simp == NEVT_pmt
        NEVT = NEVT_pmt

        return NEVT, pmtrwf, sipmrwf, pmtblr

    def get_rd_vectors(self, h5in):
        """Return MC RD vectors and sensor data.

        PMTWL is the length of the RWF computed by divinging the
        length of the MCRD for wavelength by sampling time. """

        pmtrd, sipmrd = self._get_rd(h5in)

        NEVT_pmt , NPMT,   PMTWL = pmtrd .shape
        NEVT_simp, NSIPM, SIPMWL = sipmrd.shape
        assert NEVT_simp == NEVT_pmt
        NEVT = NEVT_pmt
        return NEVT, pmtrd, sipmrd

    def get_sensor_rd_params(self, filename):
        with tb.open_file(filename, "r") as h5in:
            pmtrd, sipmrd = self._get_rd(h5in)
            _, NPMT,   PMTWL = pmtrd .shape
            _, NSIPM, SIPMWL = sipmrd.shape
            PMTWL_FEE = int(PMTWL // self.FE_t_sample)
        return SensorParams(NPMT=NPMT, PMTWL=PMTWL_FEE, NSIPM=NSIPM, SIPMWL=SIPMWL)

    def get_sensor_params(self, filename):
        with tb.open_file(filename, "r") as h5in:
            pmtrwf, sipmrwf, _ = self._get_rwf(h5in)
            _, NPMT,   PMTWL   = pmtrwf .shape
            _, NSIPM, SIPMWL   = sipmrwf.shape
        return SensorParams(NPMT=NPMT, PMTWL=PMTWL, NSIPM=NSIPM, SIPMWL=SIPMWL)

    @staticmethod
    def get_run_and_event_info(h5in):
        return h5in.root.Run.events

    # TODO Replace this with
    # tbl_functions.get_event_numbers_and_timestamps_from_file_name,
    # maybe
    @staticmethod
    def event_and_timestamp(evt, events_info):
        return events_info[evt]

    @staticmethod
    def event_number_from_input_file_name(filename):
        file_base_name = filename.split('/')[-1]
        base_hash = hash(file_base_name)
        # Something, somewhere, is preventing us from using the full
        # 64 bit space, and limiting us to 32 bits. TODO find and
        # eliminate it.
        limited_hash = base_hash % int(1e9)
        return limited_hash

    def _get_rwf(self, h5in):
        "Return raw waveforms for SIPM and PMT data"
        return (h5in.root.RD.pmtrwf,
                h5in.root.RD.sipmrwf,
                h5in.root.RD.pmtblr)

    def _get_rd(self, h5in):
        "Return (MC) raw data waveforms for SIPM and PMT data"
        return (h5in.root.pmtrd,
                h5in.root.sipmrd)


class SensorResponseCity(City):
    """A SensorResponseCity city extends the City base class adding the
       response (Monte Carlo simulation) of the energy plane and
       tracking plane sensors (PMTs and SiPMs).
    """

    def __init__(self,
                 run_number  = 0,
                 files_in    = None,
                 file_out    = None,
                 compression = 'ZLIB4',
                 nprint      = 10000,
                 # Parameters added at this level
                 sipm_noise_cut = 3 * units.pes,
                 first_evt = 0):

        City.__init__(self,
                      run_number  = run_number,
                      files_in    = files_in,
                      file_out    = file_out,
                      compression = compression,
                      nprint      = nprint)

        self.sipm_noise_cut = sipm_noise_cut
        self.first_evt      = first_evt

    def simulate_sipm_response(self, event, sipmrd,
                               sipms_noise_sampler):
        """Add noise with the NoiseSampler class and return
        the noisy waveform (in pes)."""
        # add noise (in PES) to true waveform
        dataSiPM = sipmrd[event] + sipms_noise_sampler.Sample()
        # return total signal in adc counts
        return wfm.to_adc(dataSiPM, self.sipm_adc_to_pes)

    def simulate_pmt_response(self, event, pmtrd):
        """ Full simulation of the energy plane response
        Input:
         1) extensible array pmtrd
         2) event_number

        returns:
        array of raw waveforms (RWF) obtained by convoluting pmtrd with the PMT
        front end electronics (LPF, HPF filters)
        array of BLR waveforms (only decimation)
        """
        # Single Photoelectron class
        spe = FE.SPE()
        # FEE, with noise PMT
        fee  = FE.FEE(noise_FEEPMB_rms=FE.NOISE_I, noise_DAQ_rms=FE.NOISE_DAQ)
        NPMT = pmtrd.shape[1]
        RWF  = []
        BLRX = []

        for pmt in range(NPMT):
            # normalize calibration constants from DB to MC value
            cc = self.adc_to_pes[pmt] / FE.ADC_TO_PES
            # signal_i in current units
            signal_i = FE.spe_pulse_from_vector(spe, pmtrd[event, pmt])
            # Decimate (DAQ decimation)
            signal_d = FE.daq_decimator(FE.f_mc, FE.f_sample, signal_i)
            # Effect of FEE and transform to adc counts
            signal_fee = FE.signal_v_fee(fee, signal_d, pmt) * FE.v_to_adc()
            # add noise daq
            signal_daq = cc * FE.noise_adc(fee, signal_fee)
            # signal blr is just pure MC decimated by adc in adc counts
            signal_blr = cc * FE.signal_v_lpf(fee, signal_d) * FE.v_to_adc()
            # raw waveform stored with negative sign and offset
            RWF.append(FE.OFFSET - signal_daq)
            # blr waveform stored with positive sign and no offset
            BLRX.append(signal_blr)
        return np.array(RWF), np.array(BLRX)

    @property
    def FE_t_sample(self):
        return FE.t_sample


class DeconvolutionCity(City):
    """A Deconvolution city extends the City base class adding the
       deconvolution step, which transforms RWF into CWF.
       The parameters of the deconvolution are the number of samples
       used to compute the baseline (n_baseline) and the threshold to
       thr_trigger in the rising signal (thr_trigger)
    """

    def __init__(self,
                 run_number            = 0,
                 files_in              = None,
                 file_out              = None,
                 compression           = 'ZLIB4',
                 nprint                = 10000,
                 # Parameters added at this level
                 n_baseline            = 28000,
                 thr_trigger           = 5 * units.adc,
                 acum_discharge_length = 5000):

        City.__init__(self,
                      run_number  = run_number,
                      files_in    = files_in,
                      file_out    = file_out,
                      compression = compression,
                      nprint      = nprint)

        # BLR parameters
        self.n_baseline            = n_baseline
        self.thr_trigger           = thr_trigger
        self.acum_discharge_length = acum_discharge_length

    def write_deconv_params(self, ofile):
        group = ofile.create_group(ofile.root, "DeconvParams")

        table = ofile.create_table(group,
                                   "DeconvParams",
                                   DECONV_PARAM,
                                   "deconvolution parameters",
                                   tbf.filters(self.compression))

        row = table.row
        row["N_BASELINE"]            = self.n_baseline
        row["THR_TRIGGER"]           = self.thr_trigger
        row["ACUM_DISCHARGE_LENGTH"] = self.acum_discharge_length
        table.flush()

    def deconv_pmt(self, RWF):
        """Deconvolve the RWF of the PMTs"""
        return blr.deconv_pmt(RWF,
                              self.coeff_c,
                              self.coeff_blr,
                              pmt_active            = self.pmt_active,
                              n_baseline            = self.n_baseline,
                              thr_trigger           = self.thr_trigger,
                              acum_discharge_length = self.acum_discharge_length)


class CalibratedCity(DeconvolutionCity):
    """A calibrated city extends a DeconvCity, performing two actions.
       1. Compute the calibrated sum of PMTs, in two flavours:
          a) csum: PMTs waveforms are equalized to photoelectrons (pes) and
             added
          b) csum_mau: waveforms are equalized to photoelectrons;
             compute a MAU that follows baseline and add PMT samples above
             MAU + threshold
       2. Compute the calibrated signal in the SiPMs:
          a) equalize to pes;
          b) compute a MAU that follows baseline and keep samples above
             MAU + threshold.
       """

    def __init__(self,
                 run_number            = 0,
                 files_in              = None,
                 file_out              = None,
                 compression           = 'ZLIB4',
                 nprint                = 10000,
                 n_baseline            = 28000,
                 thr_trigger           = 5 * units.adc,
                 acum_discharge_length = 5000,
                 # Parameters added at this level
                 n_MAU                 = 100,
                 thr_MAU               = 3.0 * units.adc,
                 thr_csum_s1           = 0.2 * units.pes,
                 thr_csum_s2           = 1.0 * units.pes,
                 n_MAU_sipm            = 100,
                   thr_sipm            = 5.0 * units.pes):

        DeconvolutionCity.__init__(self,
                                   run_number            = run_number,
                                   files_in              = files_in,
                                   file_out              = file_out,
                                   compression           = compression,
                                   nprint                = nprint,
                                   n_baseline            = n_baseline,
                                   thr_trigger           = thr_trigger,
                                   acum_discharge_length = acum_discharge_length)

        # Parameters of the PMT csum.
        self.n_MAU       = n_MAU
        self.thr_MAU     = thr_MAU
        self.thr_csum_s1 = thr_csum_s1
        self.thr_csum_s2 = thr_csum_s2

        # Parameters of the SiPM signal
        self.n_MAU_sipm = n_MAU_sipm
        self.  thr_sipm =   thr_sipm

    def calibrated_pmt_sum(self, CWF):
        """Return the csum and csum_mau calibrated sums."""
        return cpf.calibrated_pmt_sum(CWF,
                                      self.adc_to_pes,
                                      pmt_active = self.pmt_active,
                                           n_MAU = self.  n_MAU   ,
                                         thr_MAU = self.thr_MAU   )

    def csum_zs(self, csum, threshold):
        """Zero Suppression over csum"""
        return cpf.wfzs(csum, threshold=threshold)

    def calibrated_signal_sipm(self, SiRWF):
        """Return the calibrated signal in the SiPMs."""
        return cpf.signal_sipm(SiRWF,
                               self.sipm_adc_to_pes,
                               thr   = self.  thr_sipm,
                               n_MAU = self.n_MAU_sipm)


class PmapCity(CalibratedCity):
    """A PMAP city extends a CalibratedCity, computing the S1, S2 and S2Si
       objects that togehter constitute a PMAP.

    """

    def __init__(self,
                 run_number            = 0,
                 files_in              = None,
                 file_out              = None,
                 compression           = 'ZLIB4',
                 nprint                = 10000,
                 n_baseline            = 28000,
                 thr_trigger           = 5 * units.adc,
                 acum_discharge_length = 5000,
                 n_MAU                 = 100,
                 thr_MAU               = 3.0 * units.adc,
                 thr_csum_s1           = 0.2 * units.adc,
                 thr_csum_s2           = 1.0 * units.adc,
                 n_MAU_sipm            = 100,
                 thr_sipm              = 5.0 * units.pes,
                 # Parameters added at this level
                 s1_params             = None,
                 s2_params             = None,
                 thr_sipm_s2           = 30 * units.pes):

        CalibratedCity.__init__(self,
                                run_number            = run_number,
                                files_in              = files_in,
                                file_out              = file_out,
                                compression           = compression,
                                nprint                = nprint,
                                n_baseline            = n_baseline,
                                thr_trigger           = thr_trigger,
                                acum_discharge_length = acum_discharge_length,
                                n_MAU                 = n_MAU,
                                thr_MAU               = thr_MAU,
                                thr_csum_s1           = thr_csum_s1,
                                thr_csum_s2           = thr_csum_s2,
                                n_MAU_sipm            = n_MAU_sipm,
                                  thr_sipm            =   thr_sipm)

        self.s1_params   = s1_params
        self.s2_params   = s2_params
        self.thr_sipm_s2 = thr_sipm_s2

    def find_S12(self, s1_ene, s1_indx, s2_ene, s2_indx):
        """Return S1 and S2."""
        S1 = cpf.find_S12(s1_ene,
                          s1_indx,
                          **self.s1_params._asdict())

        S2 = cpf.find_S12(s2_ene,
                          s2_indx,
                          **self.s2_params._asdict())
        return S1, S2

    def correct_S1_ene(self, S1, csum):
        return cpf.correct_S1_ene(S1, csum)

    def find_S2Si(self, S2, sipmzs):
        """Return S2Si."""
        SIPM = cpf.select_sipm(sipmzs)
        S2Si = pf.sipm_s2_dict(SIPM, S2, thr = self.thr_sipm_s2)
        return pio.S2Si(S2Si)

    def check_s1s2_params(self):
        if (not self.s1_params) or (not self.s2_params):
            raise IOError('must set S1/S2 parameters before running')


    def select_event(self, evt_number, evt_time, S1, S2, Si):
        evt       = KrEvent()
        evt.event = evt_number
        evt.time  = evt_time * 1e-3 # s

        from .. filters.s1s2_filter import s1s2_filter
        if not s1s2_filter(S1, S2, Si):
            return

        evt.nS1 = len(S1)
        for peak_no, (t, e) in sorted(S1.items()):
            evt.S1w.append(pmp.width(t))
            evt.S1h.append(np.max(e))
            evt.S1e.append(np.sum(e))
            evt.S1t.append(t[np.argmax(e)])

        evt.nS2 = len(S2)
        for peak_no, (t, e) in sorted(S2.items()):
            s2time  = t[np.argmax(e)]

            evt.S2w.append(pmp.width(t, to_mus=True))
            evt.S2h.append(np.max(e))
            evt.S2e.append(np.sum(e))
            evt.S2t.append(s2time)

            IDs, Qs = pmp.integrate_charge(Si[peak_no])
            xsipms  = self.xs[IDs]
            ysipms  = self.ys[IDs]
            x       = np.average(xsipms, weights=Qs)
            y       = np.average(ysipms, weights=Qs)
            q       = np.sum    (Qs)

            evt.Nsipm.append(len(IDs))
            evt.S2q  .append(q)

            evt.X    .append(x)
            evt.Y    .append(y)

            evt.Xrms .append((np.sum(Qs * (xsipms-x)**2) / (q - 1))**0.5)
            evt.Yrms .append((np.sum(Qs * (ysipms-y)**2) / (q - 1))**0.5)

            evt.R    .append((x**2 + y**2)**0.5)
            evt.Phi  .append(np.arctan2(y, x))

            dt  = s2time - evt.S1t[0] if len(evt.S1t) > 0 else -1e3
            dt *= units.ns / units.mus
            evt.DT   .append(dt)
            evt.Z    .append(dt * units.mus * self.drift_v)

        return evt


class MapCity(City):
    def __init__(self,
                 lifetime           ,
                 u_lifetime   =    1,

                 xbins        =  100,
                 xmin         = None,
                 xmax         = None,

                 ybins        =  100,
                 ymin         = None,
                 ymax         = None):

        self.  _lifetimes = [lifetime]   if not np.shape(  lifetime) else   lifetime
        self._u_lifetimes = [u_lifetime] if not np.shape(u_lifetime) else u_lifetime
        self._lifetime_corrections = tuple(map(LifetimeCorrection, self._lifetimes, self._u_lifetimes))

        xmin = self.det_geo.XMIN[0] if xmin is None else xmin
        xmax = self.det_geo.XMAX[0] if xmax is None else xmax
        ymin = self.det_geo.YMIN[0] if ymin is None else ymin
        ymax = self.det_geo.YMAX[0] if ymax is None else ymax

        self._xbins  = xbins
        self._ybins  = ybins
        self._xrange = xmin, xmax
        self._yrange = ymin, ymax

    def xy_correction(self, X, Y, E):
        xs, ys, es, us = \
        fitf.profileXY(X, Y, E, self._xbins, self._ybins, self._xrange, self._yrange)

        norm_index = xs.size//2, ys.size//2
        return Correction((xs, ys), es, us, norm_strategy="index", index=norm_index)

    def xy_statistics(self, X, Y):
        return np.histogram2d(X, Y, (self._xbins, self._ybins), (self._xrange, self._yrange))


class HitCollectionCity(City):
    def __init__(self,
                 rebin            = 1,
                  z_corr_filename = None,
                 xy_corr_filename = None,
                 lifetime         = None,
                 reco_algorithm   = barycenter):

        self.rebin          = rebin
        self. z_corr        = LifetimeCorrection(lifetime)\
                              if lifetime else\
                              dstf.load_z_corrections(z_corr_filename)

        self.xy_corr        = dstf.load_xy_corrections(xy_corr_filename)
        self.reco_algorithm = reco_algorithm

    # TODO: remove from here
    def rebin_s2(self, S2, Si):
        if self.rebin <= 1:
            return S2, Si

        S2_rebin = {}
        Si_rebin = {}
        for peak in S2:
            t, e, sipms = cpf.rebin_S2(S2[peak][0], S2[peak][1], Si[peak], self.rebin)
            S2_rebin[peak] = Peak(t, e)
            Si_rebin[peak] = sipms
        return S2_rebin, Si_rebin

    def split_energy(self, e, clusters):
        if len(clusters) == 1:
            return [e]
        qs = np.array([c.Q for c in clusters])
        return e * qs / np.sum(qs)

    def correct_energy(self, e, x, y, z):
        ecorr = e * self.z_corr([z])[0][0]
        if not np.isnan([x, y]).any():
            ecorr *= self.xy_corr([x], [y])[0][0]
        return ecorr

    def compute_xy_position(self, si, slice_no):
        si      = pmp.select_si_slice(si, slice_no)
        IDs, Qs = map(list, zip(*si.items()))
        xs, ys  = self.xs[IDs], self.ys[IDs]
        return self.reco_algorithm(np.stack((xs, ys)), Qs)

    def select_event(self, evt_number, evt_time, S1, S2, Si):
        hitc = HitCollection()

        S1     = self.select_S1(S1)
        S2, Si = self.select_S2(S2, Si)

        if len(S1) != 1 or not self.S2_Nmin <= len(S2) <= self.S2_Nmax:
            return None

        hitc.evt   = evt_number
        hitc.time  = evt_time * 1e-3 # s

        t, e = next(iter(S1.values()))
        S1t  = t[np.argmax(e)]

        S2, Si = self.rebin_s2(S2, Si)

        npeak = 0
        for peak_no, (t_peak, e_peak) in sorted(S2.items()):
            si = Si[peak_no]
            for slice_no, (t_slice, e_slice) in enumerate(zip(t_peak, e_peak)):
                clusters = self.compute_xy_position(si, slice_no)
                es       = self.split_energy(e_slice, clusters)
                z        = (t_slice - S1t) * units.ns * self.drift_v
                for c, e in zip(clusters, es):
                    hit       = Hit()
                    hit.Npeak = npeak
                    hit.X     = c.X
                    hit.Y     = c.Y
                    hit.R     = (c.X**2 + c.Y**2)**0.5
                    hit.Phi   = np.arctan2(c.Y, c.X)
                    hit.Z     = z
                    hit.Q     = c.Q
                    hit.E     = e
                    hit.Ecorr = self.correct_energy(e, c.X, c.Y, z)
                    hit.Nsipm = c.Nsipm
                    hitc.append(hit)
            npeak += 1

        return hitc
