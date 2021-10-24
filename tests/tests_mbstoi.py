from pathlib import Path
import json
import soundfile as sf
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from clarity_core.signal import find_delay_impulse
from clarity_core.config import CONFIG
from mbstoi import MBSTOI
import MBSTOI as baselineMBSTOI

clarity_metadata_dir = Path('/clarity/metadata')
clarity_data_dir = Path('/clarity_CEC1_data/clarity_data')

with open(clarity_metadata_dir / 'scenes.dev.json') as f:
    scenes = json.load(f)
with open(clarity_metadata_dir / 'scenes_listeners.dev.json') as f:
    scenes_listeners = json.load(f)
with open(clarity_metadata_dir / 'listeners.json') as f:
    listeners = json.load(f)

def test_mbstoi():
    sii_origs = []
    siis = []
    for scene in tqdm(scenes):
        for listener_id in scenes_listeners[scene['scene']]:
            listener = listeners[listener_id]
            clean,_ = sf.read(str(clarity_data_dir / 'dev' / 'scenes' /
                                  f'{scene["scene"]}_target_anechoic.wav'))
            proc,_ = sf.read(str(clarity_data_dir / 'dev' / 'scenes' /
                                  f'{scene["scene"]}_{listener["name"]}_HL-output.wav'))
            ddf,_ = sf.read(str(clarity_data_dir / 'dev' / 'scenes' /
                            f'{scene["scene"]}_{listener["name"]}_HLddf-output.wav'))

            delay = find_delay_impulse(ddf, initial_value = int(CONFIG.fs / 2))
            maxdelay = int(np.max(delay))
            cleanpad = np.zeros((len(clean) + maxdelay, 2))
            procpad = np.zeros((len(clean) + maxdelay, 2))
            cleanpad[int(delay[0]) : int(len(clean) + int(delay[0])), 0] = clean[:, 0]  
            cleanpad[int(delay[1]) : int(len(clean) + int(delay[1])), 1] = clean[:, 1]  
            procpad[:len(proc)] = proc

            sii_orig = baselineMBSTOI.mbstoi(cleanpad[:,0], cleanpad[:,1],
                                             procpad[:,0], procpad[:,1],
                                             gridcoarseness=1)
            sii = MBSTOI()(torch.tensor(cleanpad[:,0]),
                         torch.tensor(cleanpad[:,1]),
                         torch.tensor(procpad[:,0]),
                         torch.tensor(procpad[:,1]))
            sii_origs.append(sii_orig)
            siis.append(sii.item())
            print(f'Difference {sii_orig - sii.item()} {sii_orig} {sii.item()}')

    sii_origs = np.array(sii_origs)
    siis = np.array(siis)

    plt.figure()
    plt.scatter(sii_origs, siis, marker = '+')
    line = np.linspace(0,1)
    plt.plot(line, line, color='red', linewidth=1)
    plt.gca().set_xlabel('Baseline MBSTOI')
    plt.gca().set_ylabel('torch MBSTOI')
    plt.savefig('tests/mbstoi_correspondance.png')

    diff = np.abs(sii_origs - siis).mean()
    print(f'Mean difference between MBSTOIs: {diff}')

if __name__ == '__main__':
    test_mbstoi()
