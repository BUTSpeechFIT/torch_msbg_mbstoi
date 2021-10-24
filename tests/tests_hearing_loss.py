import json
from pathlib import Path
import torch
import torch.nn.functional as F
import soundfile as sf
import numpy as np
import MSBG
import clarity_core
from hearing_loss import HearingLoss, Cochlea, gtfbank_params_all

clarity_metadata_dir = Path('/clarity/metadata')
clarity_data_dir = Path('/clarity_CEC1_data/clarity_data')

# example signal
with open(clarity_metadata_dir / 'scenes.dev.json') as f:
    scenes = json.load(f)
scene = scenes[0]
s, fs = sf.read(clarity_data_dir / f'dev/scenes/{scene["scene"]}_mixed_CH0.wav')
s_torch = torch.tensor(s)

# example audiograms
std_audiograms = MSBG.audiogram.standard_audiograms()

def test_measure_rms():
    calculated_rms_orig,_,_,_ = MSBG.measure_rms(s[:,0], fs, -12)
    hl = HearingLoss(std_audiograms[0])
    calculated_rms = hl.measure_rms(s_torch[:,0], -12)
    diff = np.abs(calculated_rms.detach().numpy() - calculated_rms_orig).mean()
    print(f'Test measure_rms, mean difference {diff}')
    assert diff < 1e-6

def test_src_to_cochlea_filt():
    ear = MSBG.Ear('ff', std_audiograms[1])
    src_corr = ear.get_src_correction('ff')
    out_orig = ear.src_to_cochlea_filt(s[:,0], src_corr, fs, backward=False)
    out_bwd_orig = ear.src_to_cochlea_filt(s[:,0], src_corr, fs, backward=True)
    
    hl = HearingLoss(std_audiograms[1])
    out = hl.src_to_cochlea_filt(s_torch[:,0], backward=False)
    out_bwd = hl.src_to_cochlea_filt(s_torch[:,0], backward=True)

    diff = np.abs(out.detach().numpy() - out_orig).mean()
    diff_bwd = np.abs(out_bwd.detach().numpy() - out_bwd_orig).mean()
    print(f'Test src_to_cochlea_filt forward, mean difference {diff}')
    print(f'Test src_to_cochlea_filt backward, mean difference {diff_bwd}')
    assert diff < 1e-6
    assert diff_bwd < 1e-6

def test_smear():
    audiogram = std_audiograms[2]
    smear_params = MSBG.cochlea.HL_PARAMS[audiogram.severity]['smear_params']
    out_orig = MSBG.Smearer(*smear_params, fs).smear(s[:,0])

    out = Cochlea(audiogram).smear(s_torch[:,0])

    # there is an intentional inconsistency between MSBG smearing and our smearing
    # we correct for the inconsistency here
    # (this inconsistency does not really affect output in later stages)
    shift = 64
    out = torch.cat((torch.zeros(shift*3).to(out.device), out))
    tgt_length = int(np.ceil(s_torch.shape[0] / shift) + 3) * shift
    out = F.pad(out, (0, out.shape[0] - tgt_length))
    out = out[:tgt_length]

    diff = np.abs(out.detach().numpy() - out_orig).mean()
    print(f'Test smear, mean difference {diff}')
    assert diff < 1e-3

def test_gammatone_filterbank():
    audiogram = std_audiograms[2]
    gtfbank_params = gtfbank_params_all[audiogram.severity]
    args = [
        gtfbank_params["NGAMMA"],                                      
        gtfbank_params["GTn_denoms"],                                  
        gtfbank_params["GTn_nums"],                                    
        gtfbank_params["GTnDelays"],                                   
        gtfbank_params["Start2PoleHP"],                                
        gtfbank_params["HP_denoms"],                                   
        gtfbank_params["HP_nums"],
    ]
    out_orig = MSBG.cochlea.gammatone_filterbank(s[:,0], *args)
    dev = torch.device('cuda')
    out = Cochlea(audiogram).to(dev).gammatone_filterbank(s_torch[:,0].to(dev))
    diff = np.abs(out.detach().cpu().numpy() - out_orig).mean()
    print(f'Test gammatone_filterbank, mean difference {diff}')
    assert diff < 1e-6

def test_compute_envelope():
    audiogram = std_audiograms[2]
    gtfbank_params = gtfbank_params_all[audiogram.severity]
    args = [
        gtfbank_params["NGAMMA"],                                      
        gtfbank_params["GTn_denoms"],                                  
        gtfbank_params["GTn_nums"],                                    
        gtfbank_params["GTnDelays"],                                   
        gtfbank_params["Start2PoleHP"],                                
        gtfbank_params["HP_denoms"],                                   
        gtfbank_params["HP_nums"],
    ]
    out_gm = MSBG.cochlea.gammatone_filterbank(s[:,0], *args)
    out_gm_torch = torch.tensor(out_gm)

    out_orig = MSBG.cochlea.compute_envelope(out_gm, gtfbank_params['ERBn_CentFrq'], fs)
    dev = torch.device('cuda')
    out = Cochlea(audiogram).to(dev).compute_envelope(out_gm_torch.to(dev))

    diff = np.abs(out.detach().cpu().numpy() - out_orig).mean()
    print(f'Test compute_envelope, mean difference {diff}')
    assert diff < 1e-4

def test_recruitment():
    audiogram = std_audiograms[2]
    gtfbank_params = gtfbank_params_all[audiogram.severity]
    args = [
        gtfbank_params["NGAMMA"],                                      
        gtfbank_params["GTn_denoms"],                                  
        gtfbank_params["GTn_nums"],                                    
        gtfbank_params["GTnDelays"],                                   
        gtfbank_params["Start2PoleHP"],                                
        gtfbank_params["HP_denoms"],                                   
        gtfbank_params["HP_nums"],
    ]
    out_gm = MSBG.cochlea.gammatone_filterbank(s[:,0], *args)
    out_gm_torch = torch.tensor(out_gm)
    env = MSBG.cochlea.compute_envelope(out_gm, gtfbank_params['ERBn_CentFrq'], fs)
    env_torch = torch.tensor(env)

    recr_params = MSBG.cochlea.compute_recruitment_parameters(
                        gtfbank_params['GTn_CentFrq'], audiogram, 105.0)
    
    config = clarity_core.config.CONFIG
    out_orig = MSBG.cochlea.recruitment(out_gm, env, 
                                        config.equiv0dBSPL + config.ahr,
                                        *recr_params)
    out = Cochlea(audiogram).recruitment(out_gm_torch, env_torch,
                                         config.equiv0dBSPL + config.ahr,
                                         torch.tensor(recr_params[0]),
                                         torch.tensor(recr_params[1]))
    diff = np.abs(out.detach().numpy() - out_orig).mean()
    print(f'Test recruitment, mean difference {diff}')
    assert diff < 1e-6
                                         
def test_cochlea():
    audiogram = std_audiograms[2]
    config = clarity_core.config.CONFIG
    out_orig = MSBG.cochlea.Cochlea(audiogram).simulate(s[:,0], config.equiv0dBSPL + config.ahr)
    dev = torch.device('cuda')
    out = Cochlea(audiogram).to(dev)(s_torch[:,0].to(dev), config.equiv0dBSPL + config.ahr)
    diff = np.abs(out.detach().cpu().numpy() - out_orig[-out.shape[0]:]).mean()
    print(f'Test cochlea, mean difference {diff}')
    assert diff < 1e-5

def test_hearing_loss():
    audiogram = std_audiograms[2]
    ear = MSBG.Ear('ff', std_audiograms[2])
    out_orig = ear.process(s[:,0])
    dev = torch.device('cuda')
    out = HearingLoss(audiogram).to(dev)(s_torch[:,0][None].to(dev))
    diff = np.abs(out[0].detach().cpu().numpy() - out_orig[0][-out[0].shape[0]:]).mean()
    print(f'Test hearing_loss, mean difference {diff}')
    assert diff < 1e-5

if __name__ == '__main__':
    test_measure_rms()
    test_src_to_cochlea_filt()
    test_smear()
    test_gammatone_filterbank()
    test_compute_envelope()
    test_recruitment()
    test_cochlea()
    test_hearing_loss()
