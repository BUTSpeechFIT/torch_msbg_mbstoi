# This module provides a differentiable implementation of MBSTOI intelligibility
# metric. 
# It is a Pytorch approximation of an implementation 
# of the Modified Binaural Short-Time Objective Intelligibility (MBSTOI) 
# measure as described in: 
# A. H. Andersen, J. M. de Haan, Z.-H. Tan, and J. Jensen, “Refinement and validation of the binaural short time objective intelligibility measure for spatially diverse conditions,” Speech Communication, vol. 102, pp. 1-13, Sep. 2018. - Asger Heidemann Andersen, 10/12-2018. 
# All title, copyrights and pending patents in and to the original MATLAB Software are owned by Oticon A/S and/or Aalborg University. Please see details at http://ah-andersen.net/code/"

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from clarity_core.config import CONFIG
import math
import MBSTOI as baselineMBSTOI
from functools import partial
import mbstoi_constants

gridcoarseness = 1

class MBSTOI(nn.Module):
    '''Differentiable implementation of MBSTOI intelligbility metric.

    Based on:
    https://github.com/claritychallenge/clarity_CEC1/tree/master/projects/MBSTOI

    The MBSTOI metrix is described in 
    A. H. Andersen, J. M. de Haan, Z.-H. Tan, and J. Jensen, “Refinement        
    and validation of the binaural short time objective intelligibility         
    measure for spatially diverse conditions,” Speech Communication,            
    vol. 102, pp. 1-13, Sep. 2018. A. H. Andersen, 10/12-2018                   
                                                                                
    All title, copyrights and pending patents in and to the original MATLAB     
    Software are owned by Oticon A/S and/or Aalborg University. Please see      
    details at http://ah-andersen.net/code/
    '''
    def __init__(self, add_internal_noise = False):
        super().__init__()
        self.len2int_noise = {}
        self.add_internal_noise = add_internal_noise
        pass

    def _signal_to_frames(self, x, framelen, hop):
        w = torch.tensor(np.hanning(framelen +2))[1:-1].to(x.device)
        return F.unfold(x[:, None, None, :], 
                        kernel_size=(1, framelen),
                        stride=(1, hop))[..., :-1] * w[None,:,None]

    def _detect_silent_frames(self, x, dyn_range, framelen, hop):
        """ Detects silent frames on input tensor.
        Implementation taken from https://github.com/mpariente/pytorch_stoi with
        only slight modification to include hanning window and factoring out
        _signal_to_frames method.
        
        A frame is excluded if its energy is lower than max(energy) - dyn_range
        Args:
            x (torch.Tensor): batch of original speech wav file  (batch, time)
            dyn_range : Energy range to determine which frame is silent
            framelen : Window size for energy evaluation
            hop : Hop size for energy evaluation
        Returns:
            torch.BoolTensor, framewise mask.
        """
        EPS = 1e-8
        x_frames = self._signal_to_frames(x, framelen, hop)
        # Compute energies in dB
        x_energies = 20 * torch.log10(torch.norm(x_frames, dim=1,
                                                keepdim=True) + EPS)
        # Find boolean mask of energies lower than dynamic_range dB
        # with respect to maximum clean speech energy frame
        mask = (torch.max(x_energies, dim=2, keepdim=True)[0] - dyn_range -
                x_energies) < 0
        return mask, x_frames

    def _remove_silent_frames(self, xl, xr, yl, yr, 
                                    fs, dyn_range, 
                                    N_frame, hop):
        '''Remove silent frames of x and y based on x.

        Args:
            xl (torch.tensor): clean input signal left channel
            xr (torch.tensor): clean input signal right channel
            yl (torch.tensor): processed signal left channel
            yr (torch.tensor): processed signal right channel
            dyn_range: energy range to determine which frame is silent
            framelen: window size for energy evaluation
            hop: hop size for energy evaluation
        '''
        maskxl, xl_frames = self._detect_silent_frames(xl[None], 
                                    dyn_range, 
                                    N_frame, hop)
        maskxr, xr_frames = self._detect_silent_frames(xr[None], 
                                    dyn_range, 
                                    N_frame, hop)
        yl_frames = self._signal_to_frames(yl[None], N_frame, hop)
        yr_frames = self._signal_to_frames(yr[None], N_frame, hop)
        mask = torch.logical_or(maskxl, maskxr)

        xl_frames = xl_frames[:,:,mask[0][0]]
        xr_frames = xr_frames[:,:,mask[0][0]]
        yl_frames = yl_frames[:,:,mask[0][0]]
        yr_frames = yr_frames[:,:,mask[0][0]]

        n_sil = (xl_frames.shape[-1] - 1) * hop + N_frame
        xl_sil = torch.zeros(n_sil).to(xl.device)
        xr_sil = torch.zeros(n_sil).to(xl.device)
        yl_sil = torch.zeros(n_sil).to(xl.device)
        yr_sil = torch.zeros(n_sil).to(xl.device)

        for i in range(xl_frames.shape[-1]):
            xl_sil[i*hop : i*hop+N_frame] += xl_frames[0,:,i]
            xr_sil[i*hop : i*hop+N_frame] += xr_frames[0,:,i]
            yl_sil[i*hop : i*hop+N_frame] += yl_frames[0,:,i]
            yr_sil[i*hop : i*hop+N_frame] += yr_frames[0,:,i]
        return xl_sil, xr_sil, yl_sil, yr_sil

    def _normalise(self, seg_x):
        Lx = torch.sum(torch.conj(seg_x) * seg_x, dim = 1)
        Lx = torch.real(Lx)
        return Lx - Lx.mean(dim = 1, keepdim = True)

    def _get_rho(self, seg_xr, seg_xl):
        rho = torch.sum(torch.conj(seg_xr) * seg_xl, dim = 1)
        return rho - rho.sum(dim=1, keepdim=True) / rho.shape[1]

    def _firstpartfunc(self, L1, L2, R1, R2, ntaus, gammas, epsexp):
        result = (
            torch.ones((1, ntaus, 1), dtype=torch.float64).to(L1.device) * ((
                10 ** (2 * gammas) * torch.sum(L1 * L2, dim=1)[:,None] 
                + 10 ** (-2 * gammas) * torch.sum(R1 * R2, dim=1)[:,None]
            )*epsexp)[:,None,:] + 
            torch.sum(L1*R2, dim=1)[:,None,None] + torch.sum(R1*L2, dim=1)[:,None,None]
        )
        return result

    def _secondpartfunc(self, L1, L2, rho1, rho2, tauexp, epsdelexp, gammas):
        result = (
                2* (((L1.type(torch.complex128)[:,:,None] *
                    (torch.real(rho1[:,:,None] * tauexp[None])).type(torch.complex128)
                     ).sum(dim=1)[:,None]
                    +(L2.type(torch.complex128)[:,:,None] *
                    (torch.real(rho2[:,:,None] * tauexp[None])).type(torch.complex128)
                    ).sum(dim=1)[:,None]).permute(0,2,1)
                * 10 ** gammas[None])
            * epsdelexp[None]
        )
        return result

    def _thirdpartfunc(self, R1, R2, rho1, rho2, tauexp, epsdelexp, gammas):
        result = (
            2 * ((
                    R1.type(torch.complex128)[:,:,None] *
                    torch.real(
                        rho1.type(torch.complex128)[:,:,None] * tauexp[None]
                        ).type(torch.complex128)
                 ).sum(dim=1)
                + (
                    R2.type(torch.complex128)[:,:,None] *
                    torch.real(
                        rho2.type(torch.complex128)[:,:,None] * tauexp[None]
                        ).type(torch.complex128)
                 ).sum(dim=1)
                )[:,:,None]
            * 10 ** -gammas[None]
            * epsdelexp[None]
        )
        return result

    def _fourthpartfunc(self, rho1, rho2, tauexp2, ngammas, deltexp):
        result = (
            2 * (
                torch.real(
                    (
                        rho1.type(torch.complex128) *
                        torch.conj(rho2).type(torch.complex128)
                    ).sum(dim=1)
                ).type(torch.complex128)[:,None]
                + deltexp.type(torch.complex128) * 
                torch.real(
                    (
                        rho1[:,:,None].type(torch.complex128) * 
                        rho2[:,:,None] * tauexp2[None]
                    ).sum(dim=1)).type(torch.complex128)
            )[:,:,None]
            * torch.ones((1, 1, ngammas)).to(rho1.device)
        )
        return result

    def _ec(self, xl_hat, xr_hat, yl_hat, yr_hat,
            J, N, fids, cf, taus, ntaus, gammas, ngammas,
            d, p_ec_max, sigma_epsilon, sigma_delta):
        '''Run the equalisation-cancellation stage of the MBSTOI metric.
        
        Args:
            xl_hat: clean L short-time DFT coefficients (single-sided) per frequency bin and frame
            xr_hat: clean R short-time DFT coefficients (single-sided) per frequency bin and frame
            yl_hat: proc. L short-time DFT coefficients (single-sided) per frequency bin and frame
            yr_hat: proc. R short-time DFT coefficients (single-sided) per frequency bin and frame
            J: number of one-third octave bands
            N: number of frames for intermediate intelligibility measure
            fids: indices of frequency band edges
            cf: centre frequencies
            taus: interaural delay (tau) values
            ntaus: number tau values
            gammas: interaural level difference (gamma) values
            ngammas: number gamma values
            d: grid for intermediate intelligibility measure
            p_ec_max: empty grid for maximum values
            sigma_epsilon: jitter for gammas
            sigma_delta: jitter for taus
        Returns:
            d: updated grid for intermediate intelligibility measure
            p_ec_max: grid containing maximum values
        '''
        taus = taus[None]
        sigma_delta = sigma_delta[None]
        sigma_epsilon = sigma_epsilon[None]
        gammas = gammas[None]
        epsexp = torch.exp(2 * np.log(10) ** 2 * sigma_epsilon ** 2)

        for i in range(J):
            tauexp = torch.exp(-1j * cf[i] * taus)
            tauexp2 = torch.exp(-1j * 2 * cf[i] * taus)
            deltexp = torch.exp(-1 * cf[i] ** 2 * sigma_delta **2)
            epsdelexp = torch.exp(
                0.5 * torch.ones(ntaus,1).to(xl_hat.device) * (
                    np.log(10) ** 2 * sigma_epsilon ** 2
                    - cf[i] ** 2 * sigma_delta.T ** 2
                ) * torch.ones(1, ngammas).to(xl_hat.device))

            str1, str2 = xl_hat[int(fids[i,0] - 1) : int(fids[i,1])].stride()
            seg_xl = torch.as_strided(
                            xl_hat[int(fids[i,0] - 1) : int(fids[i,1])],
                            (d.shape[1], int(fids[i,1]) - int(fids[i,0]-1), N),
                            (str2,str1,str2)
                            )
            seg_xr = torch.as_strided(
                            xr_hat[int(fids[i,0] - 1) : int(fids[i,1])],
                            (d.shape[1], int(fids[i,1]) - int(fids[i,0]-1), N),
                            (str2,str1,str2)
                            )
            seg_yl = torch.as_strided(
                            yl_hat[int(fids[i,0] - 1) : int(fids[i,1])],
                            (d.shape[1], int(fids[i,1]) - int(fids[i,0]-1), N),
                            (str2,str1,str2)
                            )
            seg_yr = torch.as_strided(
                            yr_hat[int(fids[i,0] - 1) : int(fids[i,1])],
                            (d.shape[1], int(fids[i,1]) - int(fids[i,0]-1), N),
                            (str2,str1,str2)
                            )
            # Normalization by substracting mean
            Lx, Rx = self._normalise(seg_xl), self._normalise(seg_xr)
            Ly, Ry = self._normalise(seg_yl), self._normalise(seg_yr)
            rhox, rhoy = self._get_rho(seg_xr, seg_xl), self._get_rho(seg_yr, seg_yl)
            
            # These correspong to equations 7 and 8 in Andersen et al. 2018
            # Calculate Exy
            firstpart = self._firstpartfunc(Lx, Ly, Rx, Ry, ntaus, gammas, epsexp)    
            secondpart = self._secondpartfunc(Lx, Ly, rhoy, rhox, tauexp, epsdelexp, gammas)
            thirdpart = self._thirdpartfunc(Rx, Ry, rhoy, rhox, tauexp, epsdelexp, gammas)
            fourthpart = self._fourthpartfunc(rhox, rhoy, tauexp2, ngammas, deltexp)  
            exy = torch.real(firstpart - secondpart - thirdpart + fourthpart)

            # Calculate Exx
            firstpart = self._firstpartfunc(Lx, Lx, Rx, Rx, ntaus, gammas, epsexp)    
            secondpart = self._secondpartfunc(Lx, Lx, rhox, rhox, tauexp, epsdelexp, gammas)
            thirdpart = self._thirdpartfunc(Rx, Rx, rhox, rhox, tauexp, epsdelexp, gammas)
            fourthpart = self._fourthpartfunc(rhox, rhox, tauexp2, ngammas, deltexp)  
            exx = torch.real(firstpart - secondpart - thirdpart + fourthpart)

            # Calcualte Eyy
            firstpart = self._firstpartfunc(Ly, Ly, Ry, Ry, ntaus, gammas, epsexp)    
            secondpart = self._secondpartfunc(Ly, Ly, rhoy, rhoy, tauexp, epsdelexp, gammas)
            thirdpart = self._thirdpartfunc(Ry, Ry, rhoy, rhoy, tauexp, epsdelexp, gammas)
            fourthpart = self._fourthpartfunc(rhoy, rhoy, tauexp2, ngammas, deltexp)  
            eyy = torch.real(firstpart - secondpart - thirdpart + fourthpart)

            # Ensure that intermediate correlation will be sensible and compute it
            idx_small = (torch.min(torch.abs(exx*eyy), dim=1).values < 1e-40).all(dim=1)
            d[i,idx_small] = -1

            if not idx_small.all():
                p = torch.divide(exx[~idx_small], eyy[~idx_small])
                tmp = p.max(dim=1).values
                idx1 = p.argmax(dim=1)
                p_ec_max[i,~idx_small] = tmp.max(dim=1).values.type(torch.float32)
                idx2 = tmp.argmax(dim=1)
                idx3 = idx1[np.arange(idx1.shape[0]),idx2]
                d[i,~idx_small] = torch.divide(
                    exy[~idx_small,idx3,idx2],
                    torch.sqrt(exx[~idx_small,idx3,idx2] * eyy[~idx_small,idx3,idx2]
                        )
                    ).type(torch.float32)
        return d, p_ec_max

    def resample_approx(self, x, fso, fst):
        k = fso / fst
        length = x.shape[0]
        length_t = math.floor(length / k)
        idx = [round(i*k) for i in range(length_t)]
        return x[idx]

    def create_internal_noise(self, yl_len, yr_len):
        """This is a procedure from Andersen et al. 2017 as described in paper cited
        below. This was developed to represent internal noise for an unimpaired listener
        in Non-intrusive STOI and is provided as an experimental option here.
        Use with caution.

        A. H. Andersen, J. M. de Haan, Z.-H. Tan, and J. Jensen, A non-
        intrusive Short-Time Objective Intelligibility measure, IEEE
        International Conference on Acoustics, Speech and Signal Processing
        (ICASSP), March 2017.

        Args:
        yl (ndarray): noisy/processed speech signal from left ear.
        yr (ndarray): noisy/processed speech signal from right ear.

        Returns
            ndarray: nl, noise signal, left ear
            ndarray: nr, noise signal, right ear
        """

        import numpy as np
        from scipy.signal import lfilter

        # Set constants for internal noise: pure tone threshold filter coefficients w. thirdoct weighting
        b = mbstoi_constants.b

        if yl_len in self.len2int_noise:
            nl = self.len2int_noise[yl_len]
        else:
            nl = lfilter(b, 1, np.random.randn(yl_len))
        if yr_len in self.len2int_noise:
            nr = self.len2int_noise[yr_len]
        else:
            nr = lfilter(b, 1, np.random.randn(yr_len))
        
        self.len2int_noise.update({yl_len: nl, yr_len: nr})

        return nl, nr
        

    def forward(self, xl, xr, yl, yr):
        fs_signal = CONFIG.fs                                                       
        fs = 10000  # Sample rate of proposed intelligibility measure in Hz         
        N_frame = 256  # Window support in samples  
        hop = N_frame // 2
        K = 512  # FFT size in samples                                              
        J = 15  # Number of one-third octave bands                                  
        mn = 150  # Centre frequency of first 1/3 octave band in Hz                 
        N = 30  # Number of frames for intermediate intelligibility measure (length analysis window)
        dyn_range = 40  # Speech dynamic range in dB                                

        # Values to define EC grid                                                  
        tau_min = -0.001  # Minumum interaural delay compensation in seconds. B: -0.01.
        tau_max = 0.001  # Maximum interaural delay compensation in seconds. B: 0.01.
        ntaus = math.ceil(100 / gridcoarseness)  # Number of tau values to try out  
        gamma_min = -20  # Minumum interaural level compensation in dB              
        gamma_max = 20  # Maximum interaural level compensation in dB               
        ngammas = math.ceil(40 / gridcoarseness)  # Number of gamma values to try out
        
        # Constants for jitter                                                      
        # ITD compensation standard deviation in seconds. Equation 6 Andersen et al. 2018 Refinement
        sigma_delta_0 = 65e-6                                                       
        # ILD compensation standard deviation.  Equation 5 Andersen et al. 2018     
        sigma_epsilon_0 = 1.5                                                       
        # Constant for level shift deviation in dB. Equation 5 Andersen et al. 2018 
        alpha_0_db = 13                                                             
        # Constant for time shift deviation in seconds. Equation 6 Andersen et al. 2018
        tau_0 = 1.6e-3                                                              
        # Constant for level shift deviation. Power for calculation of sigma delta gamma in equation 5 Andersen et al. 2018.
        p = 1.6
        
        # Resample signals to 10 kHz
        res = partial(self.resample_approx, fso=CONFIG.fs, fst=fs)
        xl, xr, yl, yr = res(xl), res(xr), res(yl), res(yr)

        # Remove silent frames
        xl, xr, yl, yr = self._remove_silent_frames(xl, xr, yl, yr, fs, dyn_range,
                                            N_frame, hop)

        if self.add_internal_noise:
            nl, nr = sfloorcreate_internal_noise(yl.shape[0], yr.shape[0])
            nl = torch.tensor(nl).to(yl.device)
            nr = torch.tensor(nr).to(yl.device)
            yl = yl + 10 ** (4 / 20) * nl
            yr = yr + 10 ** (4 / 20) * nr
        
        # Get 1/3 octave band matrix
        H, cf, fids, freq_low, freq_high = baselineMBSTOI.thirdoct(fs, K, J, mn)
        cf = torch.tensor(cf).to(xl.device)
        cf = 2 * math.pi * cf
        H = torch.tensor(H).to(xl.device)
        
        # Apply short time DFT
        win = torch.tensor(np.hanning(N_frame + 2)[1:-1]).to(xl.device)
        xl_hat = torch.stft(xl, K, win_length = N_frame, hop_length = hop,
                        return_complex = True, center = False,
                        window = win)
        xr_hat = torch.stft(xr, K, win_length = N_frame, hop_length = hop,
                        return_complex = True, center = False,
                        window = win)
        yl_hat = torch.stft(yl, K, win_length = N_frame, hop_length = hop,
                        return_complex = True, center = False,
                        window = win)
        yr_hat = torch.stft(yr, K, win_length = N_frame, hop_length = hop,
                        return_complex = True, center = False,
                        window = win)
        
        # Compute intermediate correlation via EC search
        taus = torch.linspace(tau_min, tau_max, ntaus).to(xl.device)
        gammas = torch.linspace(gamma_min, gamma_max, ngammas).to(xl.device)
        sigma_epsilon = (
                np.sqrt(2) * sigma_epsilon_0 * (
                    1 + (torch.abs(gammas) / alpha_0_db) ** p) / 20
            )
        gammas = gammas / 20
        sigma_delta = np.sqrt(2) * sigma_delta_0 * (
                        1 + (torch.abs(taus) / tau_0))

        d = torch.zeros(J, xl_hat.shape[1] - N + 1).to(xl.device)
        p_ec_max = torch.zeros(J, xl_hat.shape[1] - N + 1).to(xl.device)
        d, p_ec_max = self._ec(
            xl_hat, xr_hat, yl_hat, yr_hat,
            J, N, fids, cf.flatten(), taus, ntaus,
            gammas, ngammas, d, p_ec_max,
            sigma_epsilon, sigma_delta
        )
        
        # Compute the better ear STOI
        # Apply 1/3 octave bands as described in Eq.(1) of the STOI article
        Xl = torch.matmul(H.type(torch.complex128), (torch.abs(xl_hat)**2).type(torch.complex128))
        Xr = torch.matmul(H.type(torch.complex128), (torch.abs(xr_hat)**2).type(torch.complex128))
        Yl = torch.matmul(H.type(torch.complex128), (torch.abs(yl_hat)**2).type(torch.complex128))
        Yr = torch.matmul(H.type(torch.complex128), (torch.abs(yr_hat)**2).type(torch.complex128))
        
        # Compute temporary better-ear correlations
        Xl_seg = torch.as_strided(Xl, (J,xl_hat.shape[1] - N, N), (xl_hat.shape[1],1,1))
        Xr_seg = torch.as_strided(Xr, (J,xl_hat.shape[1] - N, N), (xl_hat.shape[1],1,1))
        Yl_seg = torch.as_strided(Yl, (J,xl_hat.shape[1] - N, N), (xl_hat.shape[1],1,1))
        Yr_seg = torch.as_strided(Yr, (J,xl_hat.shape[1] - N, N), (xl_hat.shape[1],1,1))

        xln = Xl_seg - torch.sum(Xl_seg, dim = 2, keepdim = True) / N
        xrn = Xr_seg - torch.sum(Xr_seg, dim = 2, keepdim = True) / N
        yln = Yl_seg - torch.sum(Yl_seg, dim = 2, keepdim = True) / N
        yrn = Yr_seg - torch.sum(Yr_seg, dim = 2, keepdim = True) / N

        pl = (torch.sum(xln*xln, dim = 2) / (torch.sum(yln*yln, dim = 2) + 1e-10)).real
        pr = (torch.sum(xrn*xrn, dim = 2) / (torch.sum(yrn*yrn, dim = 2) + 1e-10)).real

        nom = torch.sum(xln*yln, dim=2)
        denom = torch.linalg.norm(xln.abs(), dim=2) * torch.linalg.norm(yln.abs(), dim=2) + 1e-10
        dl_interm = (nom / denom).real

        nom2 = torch.sum(xrn*yrn, dim=2)
        denom2 = torch.linalg.norm(xrn.abs(), dim=2) * torch.linalg.norm(yrn.abs(), dim=2) + 1e-10
        dr_interm = (nom2 / denom2).real

        # Get the better ear intermediate coefficients
        dl_interm[denom < 1e-6] = 0
        dr_interm[denom2 < 1e-6] = 0

        pl = torch.cat((pl, torch.zeros(J,1).to(pl.device)), dim = 1)
        pr = torch.cat((pr, torch.zeros(J,1).to(pl.device)), dim = 1)
        dl_interm = torch.cat((dl_interm, torch.zeros(J,1).to(dl_interm.device)), dim = 1)
        dr_interm = torch.cat((dr_interm, torch.zeros(J,1).to(dr_interm.device)), dim = 1)
        
        p_be_max = torch.maximum(pl,pr)
        dbe_interm = torch.zeros_like(dl_interm)
        
        idx = pl > pr
        dbe_interm[idx] = dl_interm[idx]
        dbe_interm[~idx] = dr_interm[~idx]

        # Compute STOI measure        
        idx = p_be_max > p_ec_max
        d[idx] = dbe_interm[idx].type(torch.float32)
        sii = torch.mean(d)
        return sii
