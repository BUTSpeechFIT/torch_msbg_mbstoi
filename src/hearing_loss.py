# This module provides a differentiable implementation of simulation of hearing
# loss. It follows MSBG hearing loss model provided with the Clarity challenge
# at https://github.com/claritychallenge/clarity_CEC1/tree/master/projects/MSBG

import math
import json
import os
import numpy as np
import scipy
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import MSBG
from clarity_core.config import CONFIG
from scipy.signal import firwin

FS = 44100
SRC_POS = 'ff'
SRC_CORR = MSBG.Ear.get_src_correction(SRC_POS)
UPPER_CUTOFF_HZ = 18000

winlen = 2 * math.floor(0.0015 * FS) + 1
LPF44D1 = firwin(winlen, UPPER_CUTOFF_HZ / int(FS / 2), window=("kaiser", 8))

HL_PARAMS = {                                                                   
    "SEVERE": {                                                                 
        "gtfbank_file": "GT4FBank_Brd3.0E_Spaced2.3E_44100Fs",                  
        "smear_params": (4, 2),  # asymmetric severe smearing                   
    },                                                                          
    "MODERATE": {                                                               
        "gtfbank_file": "GT4FBank_Brd2.0E_Spaced1.5E_44100Fs",                  
        "smear_params": (2.4, 1.6),  # asymmetric moderate smearing             
    },                                                                          
    "MILD": {                                                                   
        "gtfbank_file": "GT4FBank_Brd1.5E_Spaced1.1E_44100Fs",                  
        "smear_params": (1.6, 1.1),  # asymmetric mild smearing                 
    },                                                                          
    "NOTHING": {                                                                
        "gtfbank_file": "GT4FBank_Brd1.5E_Spaced1.1E_44100Fs",                  
        "smear_params": (1.0, 1.0),  # No smearing                              
    },                                                                          
}

smear_mats = {severity: MSBG.smearing.make_smear_mat3(
                        *HL_PARAMS[severity]['smear_params'], FS
                        )
              for severity in HL_PARAMS}

MSBG_DATADIR = Path(os.getenv('CLARITY_ROOT')) / 'projects/MSBG/data'

def read_gtf_file(gtf_file):
    """Read a gammatone filterbank file.

    List data is converted into numpy arrays.

    """
    with open(gtf_file, "r") as fp:
        data = json.load(fp)
    for key in data:
        if type(data[key]) == list:
            data[key] = np.array(data[key])
    return data

gtfbank_files = {severity: HL_PARAMS[severity]["gtfbank_file"]
                 for severity in HL_PARAMS}
gtfbank_params_all = {severity: read_gtf_file(
                        str(MSBG_DATADIR / f'{gtfbank_files[severity]}.json'))
                      for severity in HL_PARAMS}

def precompute_envelope_filters(gtfbank_params, imp_len = 2000):
    n_chans = gtfbank_params['GTn_denoms'].shape[0]

    chan_lpf_imp = np.zeros((n_chans, imp_len))
    for ixch in range(n_chans):
        fc_envlp = (30 / 40) * min(100, gtfbank_params['ERBn_CentFrq'][ixch])
        chan_lpfB, chan_lpfA = scipy.signal.ellip(2, 0.25, 35, 
                                                    fc_envlp / (FS / 2))
        chan_lpf_imp[ixch,:] = scipy.signal.lfilter(chan_lpfB, 
                                                    chan_lpfA, 
                                                    [1] + [0]*(imp_len-1))
    return torch.tensor(chan_lpf_imp)

envelope_filters = {severity: precompute_envelope_filters(gtfbank_params_all[severity])
                    for severity in HL_PARAMS}

class HearingLoss(nn.Module):
    '''Differentiable simulation of hearing loss.

    Implements the same functionality as class Ear in MSBG hearing loss model.
    '''
    def __init__(self, audiogram):
        super().__init__()
        self.audiogram = audiogram
        self.cochlea = Cochlea(audiogram)
        self.lpf44d1 = nn.Parameter(torch.tensor(LPF44D1))
        self.src_to_cochlea_filters_fwd = self.compute_src_to_cochlea_filters(False)
        self.src_to_cochlea_filters_bwd = self.compute_src_to_cochlea_filters(True)

    def forward(self, x):
        levelreFS = 10 * torch.log10((x**2).mean())
        equiv_0dB_SPL = CONFIG.equiv0dBSPL + CONFIG.ahr
        leveldBSPL = equiv_0dB_SPL + levelreFS
        CALIB_DB_SPL = leveldBSPL
        TARGET_SPL = leveldBSPL
        REF_RMS_DB = CALIB_DB_SPL - equiv_0dB_SPL

        calculated_rms = self.measure_rms(x[0], -12)

        change_dB = TARGET_SPL - (equiv_0dB_SPL + 20 * torch.log10(calculated_rms))
        x = x * torch.pow(10, 0.05 * change_dB)

        x_cf = [self.src_to_cochlea_filt(x1) for x1 in x]
        x_c = [self.cochlea(x1, equiv_0dB_SPL) for x1 in x_cf]
        x_cfb = [self.src_to_cochlea_filt(x1, backward = True) for x1 in x_c]

        x_o = [F.conv1d(x1[None,None], 
                      self.lpf44d1[None,None].flip(dims = (2,)), 
                      padding = self.lpf44d1.shape[0] - 1
                     )[0,0,:-self.lpf44d1.shape[0]+1]
             for x1 in x_cfb
            ]

        return x_o

    def measure_rms(self, x, dB_rel_rms):
        '''Measure RMS of the inputs signal.

        Args:
            x: signal of which to measure rms
            dB_rel_rms (float): threshold for frames to track

        Returns:
            rms (float): overall calculated RMS
        '''
        WIN_SECS = 0.01

        siglen = x.shape[0]
        winlen = round(WIN_SECS * FS)
        totframes = math.floor(siglen / winlen)
        
        first_stage_rms = torch.sqrt(torch.sum(
                            torch.clamp(torch.square(x),min=1e-6) / x.shape[0]))
        key_thr_dB = torch.clamp(20 * torch.log10(first_stage_rms) + dB_rel_rms, 
                                 min = -80)
        non_zero = torch.pow(10, (key_thr_dB - 30) / 10)
        every_dB = 10 * torch.log10(non_zero + 
                        torch.sum(torch.square(
                            torch.reshape(x[:winlen*totframes], 
                                          (totframes, winlen))
                        ), dim = 1) / winlen
                    )
        frame_index = torch.where(every_dB >= key_thr_dB)[0]
        key = ((frame_index * winlen)[:,None] + torch.arange(winlen)[None,:].to(x.device)
                ).reshape(-1)
        rms = torch.sqrt(torch.sum(
            torch.clamp(torch.square(x[key]), min=1e-6)) / key.shape[0])
        return rms

    def compute_src_to_cochlea_filters(self, backward):
        '''Precomputes filters simulating middle and outer tranfer ear functions.

        Args:
            backward (bool): if true then cochlea to src
        '''
        data = MSBG.data.src_to_cochlea_filters

        # make sure that response goes only up to fs/2
        nyquist = int(FS / 2)
        ixf_useful = np.nonzero(data.HZ < nyquist)
        hz_used = data.HZ[ixf_useful]
        hz_used = np.append(hz_used, nyquist)

        # sig from free field to cochlea
        correction = SRC_CORR - data.MIDEAR
        f = scipy.interpolate.interp1d(data.HZ, correction, kind='linear')
        last_correction = f(nyquist)
        correction_used = np.append(correction[ixf_useful], last_correction)
        if backward:
            correction_used = -correction_used
        correction_used = np.power(10, (0.05 * correction_used))
        correction_used = correction_used.flatten()

        # create filter
        n_wdw = 2 * math.floor((FS / 16e3) * 368 / 2)
        hz_used = hz_used / nyquist
        b = MSBG.firwin2(n_wdw + 1, hz_used.flatten(), 
                         correction_used, window=("kaiser", 4))
        b = torch.tensor(b)
        return b

    def src_to_cochlea_filt(self, x, backward = False):
        '''Applies filter simulating middle and outer ear transfer functions.

        Args:
            x: signal to process
            backward (bool): if true then cochlea to src
        '''
        b = self.src_to_cochlea_filters_fwd if not backward \
               else self.src_to_cochlea_filters_bwd
        b = b.to(x.device)
        y = torch.nn.functional.conv1d(x[None,None,:],
                                       b[None,None,:],
                                       padding=b.shape[0]-1)[0,0,:-b.shape[0]+1]
        return y


class Cochlea(nn.Module):
    '''Simulates the cochlea based on listener's audiogram'''
    def __init__(self, audiogram):
        super().__init__()
        self.audiogram = audiogram
        self.gtfbank_params = gtfbank_params_all[audiogram.severity]

        self.recr_params = MSBG.cochlea.compute_recruitment_parameters(
                                self.gtfbank_params['GTn_CentFrq'], 
                                audiogram, 105.0)
        self.recr_params = nn.ParameterList([nn.Parameter(torch.tensor(x)) 
                                             for x in self.recr_params])

        gtn_imp, hp_imp = self.precompute_gammatone_filters()
        self.gtn_imp = nn.Parameter(gtn_imp)
        self.hp_imp = nn.Parameter(hp_imp)
        self.chan_lpf_imp = nn.Parameter(envelope_filters[audiogram.severity])

    def forward(self, x, equiv_0dB_file_SPL):
        '''Pass signal through the cochlea.

        Args:
            x: signal to process
            equiv_0dB_file_SPL (float): equivalent level in dB SPL of 0 dB Full Scale

        Returns:
            torch.tensor: cochlea output signal
        '''
        if self.audiogram.severity != 'NOTHING':
            x_sm = self.smear(x)
        else:
            x_sm = x

        x_g = self.gammatone_filterbank(x_sm) 
        envelope = self.compute_envelope(x_g)
        x_r = self.recruitment(x_g, envelope, equiv_0dB_file_SPL, 
                             *self.recr_params)
        power = torch.pow(10, 
                -0.05 * torch.tensor(self.gtfbank_params["Recombination_dB"]))
        x_r2 = x_r.sum(dim = 0) * power

        return x_r2

    def precompute_gammatone_filters(self, imp_len = 1000):
        '''Precomputes FIR filters approximating gammatone filterbank.
        
        Args:
            imp_len (int): Length of the FIR filters.
        '''
        gtn_denoms = self.gtfbank_params["GTn_denoms"]
        gtn_nums = self.gtfbank_params["GTn_nums"]
        hp_denoms = self.gtfbank_params["HP_denoms"]
        hp_nums = self.gtfbank_params["HP_nums"]

        n_chans = gtn_denoms.shape[0]
        gtn_imp = torch.zeros((n_chans, imp_len), dtype=torch.float64)
        for ixch in range(n_chans):
            gtn_imp[ixch] = torch.tensor(
                    scipy.signal.lfilter(gtn_nums[ixch,:], 
                                         gtn_denoms[ixch,:], 
                                         [1] + [0]*(imp_len-1))
                    )
        hp_imp = torch.zeros((len(hp_denoms), imp_len), dtype=torch.float64)
        for ixch in range(len(hp_denoms)):
            hp_imp[ixch] = torch.tensor(
                    scipy.signal.lfilter(hp_nums[ixch,:], 
                                         hp_denoms[ixch,:], 
                                         [1] + [0]*(imp_len-1))
                    )
        return gtn_imp, hp_imp


    def gammatone_filterbank(self, x):
        '''Passes signal through gammatone filterbank.

        We use FIR signals approximating the gammatone filterbank as
        the convolution operation can be computed efficiently on GPU.
        '''
        n_samples = len(x)
        n_chans, imp_len = self.gtn_imp.shape

        pass_n = x[None].repeat(n_chans, 1)
        for ig in range(self.gtfbank_params['NGAMMA']):
            pass_n = F.conv1d(pass_n[None], 
                              self.gtn_imp[:,None].flip(dims = (2,)), 
                              padding = imp_len - 1,
                              groups = n_chans)[0,:,:-imp_len+1]

        pass_n_del = torch.zeros_like(pass_n)
        for ixch in range(n_chans):
            dly = self.gtfbank_params['GTnDelays'][ixch]
            pass_n_del[ixch,:n_samples-dly] = pass_n[ixch,dly:n_samples]
            pass_n_del[ixch, n_samples-dly:n_samples] = 0

        start2poleHP = self.gtfbank_params['Start2PoleHP']
        pass_n_del[start2poleHP-1:] = F.conv1d(
                       pass_n_del[start2poleHP-1:][None].clone(),
                       self.hp_imp[:,None].flip(dims = (2,)),
                       padding = imp_len - 1,
                       groups = self.hp_imp.shape[0]
                       )[0,:,:-imp_len+1]
        return pass_n_del

    def compute_envelope(self, x):
        '''Computes signal envelope.
        
        See `precompute_envelope_filters` for approximation of the envelope 
        filters using FIR filter.
        '''
        n_chans, imp_len = self.chan_lpf_imp.shape
        out_fwd = F.conv1d(x[None].abs(), 
                           self.chan_lpf_imp[:,None].flip(dims = (2,)), 
                           padding = imp_len - 1,
                           groups = n_chans)[0,:,:-imp_len+1]
        out_bwd = F.conv1d(out_fwd.flip(-1)[None].abs(), 
                           self.chan_lpf_imp[:,None].flip(dims = (2,)), 
                           padding = imp_len - 1,
                           groups = n_chans)[0,:,:-imp_len+1].flip(-1)
        return out_bwd

    def recruitment(self, x, envelope, SPL_equiv_0dB, expansion_ratios, eq_loud_db):
        '''Computes loudness recruitment.
        
        Args:
            x (torch.tensor): input signal
            envelope (torch.tensor): signal envelope
            SPL_equiv_0dB: equivalnt level in dB SPL of 0 dB Full scale
            expansion_ratios: expansion ratios for expanding channel signals
            eq_loud_db: loudness catch-up level in dB

        Returns:
            torch.tensor: cochlear output signal
        '''
        envlp_max = torch.pow(10, 0.05 * (eq_loud_db - SPL_equiv_0dB))
        envlp_max = envlp_max[:,None].repeat(1,envelope.shape[1])
        idx_max = envelope > envlp_max
        envelope_cl = torch.clamp(envelope, min = 1e-9)
        envelope_cl = torch.minimum(envelope_cl, envlp_max)
        envelope_norm = torch.clamp(envelope_cl / envlp_max, min = 1e-6)
        gain = envelope_norm ** (torch.clamp(expansion_ratios[:,None],-5) - 1)
        gain_cl = torch.clamp(gain, max=100)
        coch_sig_out = x * gain_cl
        return coch_sig_out

    def smear(self, x, fft_size = 512, frame_size = 256, shift = 64):
        '''Simualtes smearing of frequencies.'''
        f_smear = smear_mats[self.audiogram.severity]
        nyquist = int(fft_size / 2)                                                 
        window = 0.5 - 0.5 * np.cos(                                                
            2 * np.pi * (np.arange(1, frame_size + 1) - 0.5) / frame_size
        )                                                                           
        window = window / math.sqrt(1.5)  
        window = torch.tensor(window).to(x.device)
        f_smear = torch.tensor(f_smear).to(x.device)

        orig_length = x.shape[0]
        x = F.pad(x, (0, (x.shape[0] // shift + 1) * shift - x.shape[0]))
        x_ad = torch.cat((x[:shift],)*3 + (x,) + (x[-shift:],)*3)
        x_resh = torch.as_strided(x_ad, 
                                  size = (
                                    (x_ad.shape[0] - frame_size) // shift + 1,
                                    frame_size),
                                  stride = (x_ad.stride()[0] * shift, x_ad.stride()[0])
        )
        wx_resh = x_resh * window[None,:]
        spectrum = torch.fft.fft(wx_resh, fft_size, dim = 1)
        power = torch.real(spectrum[:,:nyquist] * torch.conj(spectrum[:,:nyquist]))
        power = torch.clamp(power, min=1e-6)
        mag = torch.sqrt(power)
        phasor = torch.where(mag!=0, spectrum[:,:nyquist] / mag, spectrum[:,:nyquist])
        smeared = torch.matmul(f_smear, power.T).T
        smeared = torch.clamp(smeared, min=1e-6)
        spectrum_new = spectrum.clone()
        spectrum_new[:,:nyquist] = torch.sqrt(smeared) * phasor
        spectrum_new[:,nyquist] = 0
        spectrum_new[:,(nyquist + 1) : fft_size] = torch.conj(
                                torch.flip(spectrum_new[:,1:nyquist], dims = (1,)))
        wxwave = torch.real(
                    torch.fft.ifft(spectrum_new, fft_size, axis = 1)
                 )[:,:frame_size] * window[None]

        rows, cols = np.ogrid[:wxwave.shape[0],:wxwave.shape[1]]
        r = np.array([-3]*shift + [-2]*shift + [-1]*shift + [0]*shift)
        r = r + wxwave.shape[0]
        rows = rows - r[None]
        out = wxwave[rows,cols].reshape(-1,4,shift)[:-3].sum(axis=1).reshape(-1)

        return out
