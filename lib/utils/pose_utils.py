"""
Code adapted from: https://github.com/akanazawa/hmr/blob/master/src/benchmark/eval_util.py
"""

import torch
import numpy as np
from typing import Optional, Dict, List, Tuple

from lib.core import constants


from freq_motion import plot_spectrogram, plot_amplitude

def cal_spectrogram_similarity(gt_amp, pred_amp):
    # 1) MSE
    mse = torch.mean((gt_amp - pred_amp) ** 2)

    # 2) LSD
    lsd = torch.sqrt(torch.mean((20 * torch.log10(gt_amp + 1e-6) - 20 * torch.log10(pred_amp + 1e-6)) ** 2))

    # 3) Corr
    gt_mean = torch.mean(gt_amp)
    pred_mean = torch.mean(pred_amp)
    numerator = torch.sum((gt_amp - gt_mean) * (pred_amp - pred_mean))
    denominator = torch.sqrt(torch.sum((gt_amp - gt_mean) ** 2) * torch.sum((pred_amp - pred_mean) ** 2))
    corr = numerator / (denominator + 1e-8)

    print(f'spectrogram similarity -- MSE:{mse}, LSD:{lsd}, CORR:{corr}')

def select_valid(batch_tensor, valid_range):
    batch_tensor = batch_tensor.reshape(-1, 3, *batch_tensor.shape[1:])[:,valid_range[0]:valid_range[1]+1]
    batch_tensor = batch_tensor.reshape(-1, *batch_tensor.shape[2:])

    return batch_tensor

def compute_error_accel(joints_gt, joints_pred, vis=None):
    """
    Computes acceleration error:
        1/(n-2) sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
    Note that for each frame that is not visible, three entries in the
    acceleration error should be zero'd out.
    Args:
        joints_gt (Nx14x3).
        joints_pred (Nx14x3).
        vis (N).
    Returns:
        error_accel (N-2).
    """
    # (N-2)x14x3
    accel_gt = joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
    accel_pred = joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]

    normed = np.linalg.norm(accel_pred - accel_gt, axis=2)

    if vis is None:
        new_vis = np.ones(len(normed), dtype=bool)
    else:
        invis = np.logical_not(vis)
        invis1 = np.roll(invis, -1)
        invis2 = np.roll(invis, -2)
        new_invis = np.logical_or(invis, np.logical_or(invis1, invis2))[:-2]
        new_vis = np.logical_not(new_invis)

    return np.mean(normed[new_vis], axis=1)

def compute_similarity_transform(S1: torch.Tensor, S2: torch.Tensor) -> torch.Tensor:
    """
    Computes a similarity transform (sR, t) in a batched way that takes
    a set of 3D points S1 (B, N, 3) closest to a set of 3D points S2 (B, N, 3),
    where R is a 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    Args:
        S1 (torch.Tensor): First set of points of shape (B, N, 3).
        S2 (torch.Tensor): Second set of points of shape (B, N, 3).
    Returns:
        (torch.Tensor): The first set of points after applying the similarity transformation.
    """

    batch_size = S1.shape[0]
    S1 = S1.permute(0, 2, 1)
    S2 = S2.permute(0, 2, 1)
    # 1. Remove mean.
    mu1 = S1.mean(dim=2, keepdim=True)
    mu2 = S2.mean(dim=2, keepdim=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = (X1**2).sum(dim=(1,2))

    # 3. The outer product of X1 and X2.
    K = torch.matmul(X1, X2.permute(0, 2, 1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are singular vectors of K.
    U, s, V = torch.svd(K)
    Vh = V.permute(0, 2, 1)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1]).unsqueeze(0).repeat(batch_size, 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.linalg.det(torch.matmul(U, Vh)))

    # Construct R.
    R = torch.matmul(torch.matmul(V, Z), U.permute(0, 2, 1))

    # 5. Recover scale.
    trace = torch.matmul(R, K).diagonal(offset=0, dim1=-1, dim2=-2).sum(dim=-1)
    scale = (trace / var1).unsqueeze(dim=-1).unsqueeze(dim=-1)

    # 6. Recover translation.
    t = mu2 - scale*torch.matmul(R, mu1)

    # 7. Error:
    S1_hat = scale*torch.matmul(R, S1) + t

    return S1_hat.permute(0, 2, 1)


def add_noise_to_seq(motion_seq, num_frames=2, random=True):
    # predefined Gaussian noise hyperparameters
    mean = 0.0
    std = 0.1
    noised_motion = motion_seq.clone()
    T, J, D = motion_seq.shape  # motion_seq 是 torch.Tensor，形状 (T, J, D)

    if random:
        torch.manual_seed(42)
        frames_to_noise = torch.randperm(T)[:num_frames]
        print("noise", frames_to_noise)

        for frame in frames_to_noise:
            noise = torch.randn(J, D, device=motion_seq.device) * std + mean
            noised_motion[frame] += noise

    return noised_motion



def reconstruction_error(S1, S2) -> np.array:
    """
    Computes the mean Euclidean distance of 2 set of points S1, S2 after performing Procrustes alignment.
    Args:
        S1 (torch.Tensor): First set of points of shape (B, N, 3).
        S2 (torch.Tensor): Second set of points of shape (B, N, 3).
    Returns:
        (np.array): Reconstruction error.
    """
    S1_hat = compute_similarity_transform(S1, S2)
    re = torch.sqrt( ((S1_hat - S2)** 2).sum(dim=-1)).mean(dim=-1)
    return re.cpu().numpy()

def eval_pose(pred_joints, gt_joints) -> Tuple[np.array, np.array]:
    """
    Compute joint errors in mm before and after Procrustes alignment.
    Args:
        pred_joints (torch.Tensor): Predicted 3D joints of shape (B, N, 3).
        gt_joints (torch.Tensor): Ground truth 3D joints of shape (B, N, 3).
    Returns:
        Tuple[np.array, np.array]: Joint errors in mm before and after alignment.
    """
    # Absolute error (MPJPE)
    mpjpe = torch.sqrt(((pred_joints - gt_joints) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()

    # Reconstuction_error
    r_error = reconstruction_error(pred_joints.cpu(), gt_joints.cpu())
    return 1000 * mpjpe, 1000 * r_error

class Evaluator:

    def __init__(self, dataset_length=None, seq_len=None):
        """
        Class used for evaluating trained models on different 3D pose datasets.
        Args:
            dataset_length (int): Total dataset length.
            keypoint_list [List]: List of keypoints used for evaluation.
            pelvis_ind (int): Index of pelvis keypoint; used for aligning the predictions and ground truth.
            metrics [List]: List of evaluation metrics to record.
        """
        self.dataset_length = dataset_length
        self.seq_len = seq_len
        
        self.mpjpe = np.zeros((dataset_length,))
        self.re = np.zeros((dataset_length,))
        self.pve = np.zeros((dataset_length,))
        self.acc = np.zeros((dataset_length,))
        self.counter = 0

        self.J24_TO_J17 = constants.J24_TO_J17
        self.J24_TO_J14 = constants.J24_TO_J14
        self.H36M_TO_J17 = constants.H36M_TO_J17
        self.H36M_TO_J14 = constants.H36M_TO_J14
        self.all_acc = []

        self.valid_range = (1,1)
        self.chunk_size = 16

        self.visualize_spec = True


    def __call__(self, gt_keypoints_3d, pred_keypoints_3d, dataset='3dpw', 
                gt_verts=None, pred_verts=None):
        
        '''
        Args:
            - gt_keypoints_3d(tensor): bs * 3, 24, 4
            - pred_keypoints_3d(tensor): bs * 3, 24, 3
        '''
        # batch_size = gt_keypoints_3d.shape[0] # 128, 24, 4

        gt_keypoints_3d = gt_keypoints_3d[:, :, :3].detach()
        pred_keypoints_3d = pred_keypoints_3d[:, :, :3].detach()
        num_j = gt_keypoints_3d.shape[1]
        
        gt_valid, pred_valid = self.get_valid_joints(gt_keypoints_3d, 
                                                     pred_keypoints_3d, 
                                                     dataset)
        # 48, 24, 3 --> 16, 24, 3 (T, J, 3)
        gt_valid = select_valid(gt_valid, self.valid_range)
        pred_valid = select_valid(pred_valid, self.valid_range)

        if self.visualize_spec: # one time for each validation pass
            gtnoise_amplitude = plot_spectrogram(add_noise_to_seq(gt_valid), sr=30*24, save_name="vis_GTnoised.png")

            gt_amplitude = plot_spectrogram(gt_valid, sr=30*24, save_name="vis_GT.png")
            pred_amplitude = plot_spectrogram(pred_valid, sr=30*24, save_name="vis_Pred.png")

            plot_amplitude(gt_amplitude-pred_amplitude, save_name="gt-pred.png")
            plot_amplitude(gt_amplitude-gtnoise_amplitude, save_name="gt-noise.png")

            cal_spectrogram_similarity(gt_amplitude, pred_amplitude)
            cal_spectrogram_similarity(gtnoise_amplitude, pred_amplitude)
            self.visualize_spec = False
        
        batch_size = self.chunk_size # NOTE(yiwen) only count the current frame (stacked 16)
        # Compute joint errors
        mpjpe, re = eval_pose(pred_valid, gt_valid) # only pass current frame to eval pose

        self.mpjpe[self.counter:self.counter+batch_size] = mpjpe
        self.re[self.counter:self.counter+batch_size] = re
        
        if gt_verts is not None and pred_verts is not None:
            pve = (pred_verts - gt_verts).norm(dim=-1).mean(dim=-1).cpu().numpy()
            self.pve[self.counter:self.counter+batch_size] = pve * 1000

        if self.seq_len is not None:
            gt = gt_keypoints_3d.reshape(self.chunk_size, -1, num_j, 3).cpu()[:,1:2].reshape(-1, num_j, 3) # 2, 16, 24, 3
            pred = pred_keypoints_3d.reshape(self.chunk_size, -1, num_j, 3).cpu()[:,1:2].reshape(-1, num_j, 3)
            acc = 0 # NOTE(yiwen) originally calculate the acc error in each window

            acc += compute_error_accel(gt, pred).mean() / 1.0 # len 16 chunk accer calculation
            self.acc[self.counter:self.counter+batch_size] = acc * 1000 #(30**2)
            
        self.counter += batch_size


    def get_valid_joints(self, gt_keypoints_3d, pred_keypoints_3d, dataset):
        if 'emdb' in dataset:
            gt_valid = gt_keypoints_3d
            pred_valid = pred_keypoints_3d

        else:
            j_mapper = self.get_gt_mapper(dataset)
            gt_valid = gt_keypoints_3d[:, j_mapper]

            j_mapper = self.get_pred_mapper(dataset)
            pred_valid = pred_keypoints_3d[:, j_mapper]

        return gt_valid, pred_valid


    def get_gt_mapper(self, dataset):

        if dataset == 'mpi-inf-3dhp':
            j_mapper = self.J24_TO_J17

        elif dataset == 'h36m':
            j_mapper = self.J24_TO_J14

        elif dataset == '3dpw':
            j_mapper = self.H36M_TO_J14

        return j_mapper


    def get_pred_mapper(self, dataset):
        
        if dataset == 'mpi-inf-3dhp':
            j_mapper = self.H36M_TO_J17

        elif dataset == 'h36m':
            j_mapper = self.H36M_TO_J14

        elif dataset == '3dpw':
            j_mapper = self.H36M_TO_J14

        return j_mapper


    def log(self):
        """
        Print current evaluation metrics
        """
        if self.counter == 0:
            print('Evaluation has not started')
            return

        print(f'{self.counter} / {self.dataset_length} samples')
        print(f're: {self.re[:self.counter].mean()} mm')
        print(f'mpjpe: {self.mpjpe[:self.counter].mean()} mm')
        print(f'pve: {self.pve[:self.counter].mean()} mm')
        print(f'accel: {self.acc[:self.counter].mean()} mm')
        print('***')


