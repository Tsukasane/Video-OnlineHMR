import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio
import torch.nn.functional as F


def plot_spectrogram(keypoints_3d, sr=30, save_name="mspec.png", align_interpolate=True):
    '''
    Args:
        - keypoints_3d: seqlen, 24, 3
        - sr: sample rate
        - save_name: figure name
        - align_interpolate: whether align time to original sequence (frame)
    '''
    device = keypoints_3d.device
    y = keypoints_3d.reshape(-1,)
    
    seqlen = keypoints_3d.shape[0]

    n_fft = int(sr * 1)   # 1 second window size
    hop_length = n_fft // 4  # 75% overlap

    if y.ndim == 1:
        y = y.unsqueeze(0)  # [1, samples]

    window = torch.hann_window(n_fft).to(device) # cpu-->gpu
    transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        power=None,
        window_fn=lambda n_fft: window
    )

    D = transform(y)  # [channel, freq, time]
    amplitude_raw = torch.abs(D)

    if align_interpolate:
        amplitude = F.interpolate(
            amplitude_raw,
            size=seqlen,
            mode="linear",
            align_corners=True)[0]
    else:
        amplitude = amplitude_raw[0]

    plot_amplitude(amplitude, save_name)

    return amplitude


def plot_amplitude(amplitude, save_name):
    plt.figure(figsize=(10, 4))
    plt.imshow(amplitude.cpu(), origin='lower', aspect='auto', cmap='inferno')
    plt.colorbar(format='%+2.0f')
    plt.title('Motion Spectrogram')
    plt.tight_layout()
    plt.savefig(save_name)
    plt.close()
