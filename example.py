# This script shows an exampleof how to use the HearingLoss and MBSTOI module
# as a loss function for training a neural network. For simplicity, we use
# a small three layer neural network and overfit on one example. 
#
# The example uses an audio file from Clarity challenge dataset. If you have
# the data, replace `clarity_metadata_dir` and `clarity_data_dir` with paths
# to the data. If not, you can replace the loading of the example signal with 
# loading of any binaural signal.

from pathlib import Path
import json
import soundfile as sf
import sys

import torch
import torch.nn as nn
from mbstoi import MBSTOI
from hearing_loss import HearingLoss
from MSBG.audiogram import standard_audiograms

from tqdm import tqdm

clarity_metadata_dir = Path('/mnt/matylda6/izmolikova/Corpora/Clarity/clarity/metadata')
clarity_data_dir = Path('/mnt/matylda6/izmolikova/Corpora/Clarity/clarity_full_new/clarity_CEC1_data/clarity_data')

# example signal
with open(clarity_metadata_dir / 'scenes.dev.json') as f:
    scenes = json.load(f)
scene = scenes[0]
s, fs = sf.read(clarity_data_dir / f'dev/scenes/{scene["scene"]}_mixed_CH1.wav')
tgt, fs = sf.read(clarity_data_dir / f'dev/scenes/{scene["scene"]}_target_anechoic.wav')
s_torch = torch.tensor(s) # (n_samples, 2)
tgt_torch = torch.tensor(tgt) # (n_samples, 2)

# example audiograms
audiograms_all = standard_audiograms()
audiogram = {'left': audiograms_all[1], 'right': audiograms_all[2]}

# loss function
class HearingIntelligibilityLoss(nn.Module):
    def __init__(self, add_internal_noise = False):
        super().__init__()
        self.MBSTOI = MBSTOI(add_internal_noise)
        
    def forward(self, xs, targets, audiograms):
        siis = []

        for x, target, audiogram in zip(xs, targets, audiograms):
            # 1) simulation of hearing loss
            hl_left = HearingLoss(audiogram['left']).to(xs.device)
            hl_right = HearingLoss(audiogram['right']).to(xs.device)
            out_left = hl_left(x[0][None].type(torch.float64))
            out_right = hl_right(x[1][None].type(torch.float64))
            out_left_shift = torch.zeros_like(out_left[0])
            out_right_shift = torch.zeros_like(out_right[0])
            # approximate compensation for delay introduced by hearing loss model
            out_left_shift[:-1080] = out_left[0][1080:]
            out_right_shift[:-1080] = out_right[0][1080:]
            # 2) intelligibility model
            sii = self.MBSTOI(target[0], target[1], 
                            out_left_shift, out_right_shift)
            siis.append(sii)
        return -torch.stack(siis)

# simple model
class SmallNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Conv1d(2, 50, 20, 10)
        self.dec = nn.ConvTranspose1d(50, 2, 20, 10)
        self.mid = nn.Sequential(
                nn.Linear(50,100),
                nn.ReLU(),
                nn.Linear(100,50)
                )

    def forward(self, x):
        h = self.enc(x)
        h = self.mid(h.permute(0,2,1)).permute(0,2,1)
        out = self.dec(h)
        return nn.functional.pad(out, [0, x.shape[2] - out.shape[2]])

# training
model = SmallNN()
lossf = HearingIntelligibilityLoss()
optim = torch.optim.SGD(model.parameters(), lr=0.01)

device = torch.device('cuda')
s_torch = s_torch.to(device)
tgt_torch = tgt_torch.to(device)
model = model.to(device)

for i in tqdm(range(3000)):
    out = model(s_torch.T[None].float())
    loss = lossf(out, tgt_torch.T[None].float(), [audiogram])
    loss.backward()
    print(loss.item())
    optim.step()
