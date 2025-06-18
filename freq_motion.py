import pickle
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
import torchaudio


filename = "/Users/yiwenzhao/Desktop/research_general/STFT/_P-JWcq1ewI_05_0_1380_slice28.pkl" #"/Users/yiwenzhao/Desktop/research_general/STFT/uitvYa5NsJI_07_0_1200_slice31.pkl"
# with open (filename, 'rb') as f:
#     inputs = pickle.load(f)
#     smpl_poses = inputs["smpl_poses"]
#     smpl_trans = inputs["smpl_trans"]
#     full_pose = inputs["full_pose"]
    
#     y = smpl_poses[2, :, ].reshape(-1,) # T first_person
    # D = librosa.stft(y)  
    # S_db = np.abs(D)
    # S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)  
    # librosa.display.specshow(magnitude[0].cpu(), sr=sr, x_axis='time', y_axis='linear') # linear scale, instead of log scale in audio
    # plt.colorbar(format='%+2.0f')
    # plt.title('Motion Spectrogram')
    # plt.tight_layout()
    # plt.savefig(save_name)
    

def plot_spectrogram(keypoints_3d, sr=30, save_name="mspec.png"):
    '''
    Args:
        - keypoints_3d: seqlen, 24, 3
        - sr: sample rate
        - save_name: figure name
    '''
    # sr = 30 * 72 # 30FPS * 72Dpose

    device = keypoints_3d.device
    y = keypoints_3d.reshape(-1,)
    

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
    amplitude = torch.abs(D)

    plt.figure(figsize=(10, 4))
    plt.imshow(amplitude[0].cpu(), origin='lower', aspect='auto', cmap='inferno')
    plt.colorbar(format='%+2.0f')
    plt.title('Motion Spectrogram')
    plt.tight_layout()
    plt.savefig(save_name)

    return amplitude[0]

    # import pdb
    # pdb.set_trace()