import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio
import torch.nn.functional as F


def plot_spectrogram(motion_ts, sr=30, save_name="mspec.png", align_interpolate=True):
    '''
    Args:
        - motion_ts: keypoints_3d (seqlen, 24, 3) or vertices or sparse vertices
        - sr: sample rate
        - save_name: figure name
        - align_interpolate: whether align time to original sequence (frame)
    '''
    device = motion_ts.device
    y = motion_ts.reshape(-1,)
    # y = motion_ts.norm(dim=2).mean(dim=1).unsqueeze(0)
    print(f'debug -- freq_motion {y.shape}')
    
    seqlen = motion_ts.shape[0]

    n_fft = 128 #int(sr * 1)   # 1 second window size
    hop_length =32 #n_fft // 4  # 75% overlap

    if y.ndim == 1:
        y = y.unsqueeze(0)  # [1, samples]

    # NOTE has complex number, more stable on CPU
    window = torch.hann_window(n_fft)
    transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        power=None,
        window_fn=lambda n_fft: window
    )
    D = transform(y.cpu())
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
    plt.imshow(amplitude.cpu(), origin='lower', aspect='auto', cmap='plasma') # cmap='nipy_spectral'
    plt.colorbar(format='%.2f')
    plt.xlabel("Frame")
    plt.ylabel("Frequency Bin")
    plt.title('Motion Spectrogram')
    plt.tight_layout()
    plt.savefig(save_name)
    plt.close()
