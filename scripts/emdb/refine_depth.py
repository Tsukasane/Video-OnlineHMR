import numpy as np
import torch
import cv2
from torchmin import minimize

import torch.nn.functional as F
import time


def gmof(x, sigma=100):
    """
    Geman-McClure error function
    """
    x_squared =  x ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)

def est_scale_hybrid(slam_depth_raw: np.ndarray, 
                     pred_depth: np.ndarray, 
                     sigma=0.5, 
                     msk=None, 
                     far_thresh=10):
    """ Depth-align by iterative + robust least-square """
    start_time = time.time()
    if msk is None:
        msk = np.zeros_like(pred_depth)
    else:
        msk = cv2.resize(msk, (pred_depth.shape[1], pred_depth.shape[0]))

    H2, W2 = pred_depth.shape
    # slam_depth: (H1, W1)
    # pred_depth: (H2, W2)
    slam_depth = cv2.resize(
        slam_depth_raw,
        (W2, H2),
        interpolation=cv2.INTER_LINEAR
    )

    # Stage 1: Iterative steps
    # Avoid division by zero: replace zeros in slam_depth with a small epsilon
    eps = np.finfo(slam_depth.dtype).eps
    slam_depth_safe = np.where(slam_depth > 0, slam_depth, eps)
    s = pred_depth / slam_depth_safe

    robust = (msk<0.5) * (eps<pred_depth) * (pred_depth<10) # if there is mask (for human) ==1, ignore those regions
    s_est = s[robust]
    scale_median = np.median(s_est) # use the median value as initial scale
    
    for _ in range(10): # seems no big difference after several iterations ~0.001
        slam_depth_0 = slam_depth * scale_median
        # Filter out invalid depths: check both original and scaled depths > eps to be consistent with epsilon replacement
        robust = (msk<0.5) * (eps<slam_depth) * (eps<slam_depth_0) * (slam_depth_0<far_thresh) * (eps<pred_depth) * (pred_depth<far_thresh)
        s_est = s[robust]
        scale_median = np.median(s_est)

        # print(f"Depth scale estimation: median {scale_median:.4f}")
    scale = scale_median

    # Stage 2: Robust optimization
    # Filter out invalid depths: check both original and scaled depths > eps to be consistent with epsilon replacement
    robust = (msk<0.5) * (eps<slam_depth) * (eps<slam_depth_0) * (slam_depth_0<far_thresh) * (eps<pred_depth) * (pred_depth<far_thresh)
    pm = torch.from_numpy(pred_depth[robust])
    sm = torch.from_numpy(slam_depth[robust])

    def f(x):
        loss = sm * x - pm
        loss = gmof(loss, sigma=sigma).mean()
        return loss

    x0 = torch.tensor([scale])
    result = minimize(f, x0,  method='bfgs')
    scale = result.x.detach().cpu().item()
    # print(f"Depth scale estimation: robust opt {scale:.4f}")
    # print(f"Depth scale estimation time: {end_time - start_time:.2f} s")
    
    return scale