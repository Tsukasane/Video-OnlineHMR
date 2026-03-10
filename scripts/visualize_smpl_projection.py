"""
Visualize SMPL mesh projected onto images using camera extrinsics.
"""

import argparse
import numpy as np
import torch
import cv2
from glob import glob
from tqdm import tqdm
import os
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.models.smpl import SMPL
from lib.utils.geometry import perspective_projection
from pytorch3d.transforms import quaternion_to_matrix


def quaternion_to_rotation_matrix(q):
    """Convert quaternion [qx, qy, qz, qw] to rotation matrix."""
    qx, qy, qz, qw = q
    # Normalize quaternion
    norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
    
    # Convert to rotation matrix
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])
    return R


def load_camera_extrinsics(txt_path):
    """
    Load camera extrinsics from text file.
    Format: scale x y z qx qy qz qw (per line)
    Returns: list of dicts with 'scale', 'translation', 'rotation' (as quaternion)
    """
    extrinsics = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8:
                scale = float(parts[0])
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                extrinsics.append({
                    'scale': scale,
                    'translation': np.array([x, y, z]),
                    'quaternion': np.array([qx, qy, qz, qw]),
                })
    return extrinsics


def load_smpl_data(npz_path):
    """Load SMPL parameters from npz file."""
    data = np.load(npz_path)
    return {
        'pred_rotmat': torch.from_numpy(data['pred_rotmat']),  # T, 24, 3, 3
        'pred_shape': torch.from_numpy(data['pred_shape']),     # T, 10
        'pred_trans': torch.from_numpy(data['pred_trans']),     # T, 1, 3 or T, 3
    }


def world_to_camera_coords(vertices_world, R_cw, t_cw):
    """
    Transform vertices from world coordinates to camera coordinates.
    
    Args:
        vertices_world: (N, 3) vertices in world coordinates
        R_cw: (3, 3) rotation matrix from world to camera
        t_cw: (3,) translation vector from world to camera (camera position in world)
    
    Returns:
        vertices_cam: (N, 3) vertices in camera coordinates
    """
    # R_cw is rotation from world to camera
    # t_cw is camera position in world coordinates
    vertices_cam = (R_cw @ vertices_world.T).T + t_cw.reshape(1, 3)
    return vertices_cam


def draw_smpl_mesh(img, vertices_2d, faces, color=(0, 255, 0), line_thickness=1):
    """
    Draw SMPL mesh on image.
    
    Args:
        img: (H, W, 3) image
        vertices_2d: (N, 2) 2D projected vertices
        faces: (F, 3) face indices
        color: (B, G, R) color for edges
        line_thickness: thickness of edges
    """
    # Filter faces that are visible (all vertices in front of camera)
    # For simplicity, draw all edges
    for face in faces:
        v0, v1, v2 = vertices_2d[face[0]], vertices_2d[face[1]], vertices_2d[face[2]]
        
        # Check if vertices are within image bounds
        h, w = img.shape[:2]
        if (0 <= v0[0] < w and 0 <= v0[1] < h and
            0 <= v1[0] < w and 0 <= v1[1] < h and
            0 <= v2[0] < w and 0 <= v2[1] < h):
            # Draw triangle edges
            cv2.line(img, (int(v0[0]), int(v0[1])), (int(v1[0]), int(v1[1])), color, line_thickness)
            cv2.line(img, (int(v1[0]), int(v1[1])), (int(v2[0]), int(v2[1])), color, line_thickness)
            cv2.line(img, (int(v2[0]), int(v2[1])), (int(v0[0]), int(v0[1])), color, line_thickness)
    
    return img


def draw_smpl_joints(img, joints_2d, color=(255, 0, 0), radius=3):
    """
    Draw SMPL joints on image.
    
    Args:
        img: (H, W, 3) image
        joints_2d: (24, 2) 2D projected joints
        color: (B, G, R) color for joints
        radius: radius of joint circles
    """
    h, w = img.shape[:2]
    for joint in joints_2d:
        x, y = int(joint[0]), int(joint[1])
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(img, (x, y), radius, color, -1)
    return img


def main():
    parser = argparse.ArgumentParser(description='Visualize SMPL projection on images')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing input images')
    parser.add_argument('--camera_txt', type=str, required=True,
                        help='Text file with camera extrinsics (scale x y z qx qy qz qw per line)')
    parser.add_argument('--smpl_npz', type=str, required=True,
                        help='NPZ file with SMPL parameters')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for visualization')
    parser.add_argument('--focal_length', type=float, default=None,
                        help='Camera focal length (optional)')
    parser.add_argument('--camera_center', type=float, nargs=2, default=None,
                        help='Camera center (cx, cy) (optional, will use image center if not provided)')
    parser.add_argument('--draw_mesh', action='store_true',
                        help='Draw mesh edges (slower but more detailed)')
    parser.add_argument('--draw_joints', action='store_true', default=True,
                        help='Draw joints (default: True)')
    parser.add_argument('--start_frame', type=int, default=0,
                        help='Start frame index (default: 0)')
    parser.add_argument('--end_frame', type=int, default=None,
                        help='End frame index (default: all frames)')
    parser.add_argument('--smpl_in_world_coords', action='store_true',
                        help='If set, assumes SMPL pred_trans is in world coordinates and needs transformation')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    img_files = sorted(glob(f"{args.image_dir}/*.jpg"))
    if len(img_files) == 0:
        img_files = sorted(glob(f"{args.image_dir}/*.png"))
    
    camera_extrinsics = load_camera_extrinsics(args.camera_txt)
    smpl_data = load_smpl_data(args.smpl_npz)
    
    print(f"Found {len(img_files)} images")
    print(f"Found {len(camera_extrinsics)} camera poses")
    print(f"SMPL data shape: rotmat={smpl_data['pred_rotmat'].shape}, shape={smpl_data['pred_shape'].shape}, trans={smpl_data['pred_trans'].shape}")
    
    # Initialize SMPL model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    smpl = SMPL().to(device)
    
    # Get image dimensions from first image
    first_img = cv2.imread(img_files[0])
    img_h, img_w = first_img.shape[:2]
    print(f"Image size: {img_w}x{img_h}")
    
    # Calculate camera intrinsics from image dimensions
    if args.focal_length is None:
        img_focal = max(img_w, img_h) * 0.8  # Rough estimate: 80% of max dimension
        print(f"Estimated focal length: {img_focal}")
    else:
        img_focal = args.focal_length
        print(f"Using provided focal length: {img_focal}")
    
    if args.camera_center is None:
        img_center = np.array([img_w / 2., img_h / 2.])  # Image center
        print(f"Using image center as camera center: {img_center}")
    else:
        img_center = np.array(args.camera_center)
        print(f"Using provided camera center: {img_center}")
    
    # Process frames
    end_frame = args.end_frame if args.end_frame is not None else min(len(img_files), len(camera_extrinsics), len(smpl_data['pred_rotmat']))
    
    print(f"Processing frames {args.start_frame} to {end_frame-1}...")
    
    for frame_idx in tqdm(range(args.start_frame, end_frame)):
        # Load image
        img = cv2.imread(img_files[frame_idx])
        if img is None:
            print(f"Warning: Could not load image {img_files[frame_idx]}")
            continue
        
        # Get camera extrinsics
        if frame_idx >= len(camera_extrinsics):
            print(f"Warning: No camera pose for frame {frame_idx}")
            continue
        
        cam_ext = camera_extrinsics[frame_idx]

        t_wc = cam_ext['translation']  # Camera position in world coordinates (T_WC translation)
        q_wc = cam_ext['quaternion']   # Camera orientation quaternion (T_WC rotation)
        
        # Convert quaternion to rotation matrix
        # q_wc represents camera orientation in world frame (R_WC)
        # Format: [qx, qy, qz, qw] -> rotation matrix R_WC
        R_wc = quaternion_to_rotation_matrix(q_wc)  # R_WC: rotation from world to camera frame
        
        # Get SMPL parameters for this frame
        if frame_idx >= len(smpl_data['pred_rotmat']):
            print(f"Warning: No SMPL data for frame {frame_idx}")
            continue
        
        pred_rotmat = smpl_data['pred_rotmat'][frame_idx:frame_idx+1].to(device)  # (1, 24, 3, 3)
        pred_shape = smpl_data['pred_shape'][frame_idx:frame_idx+1].to(device)   # (1, 10)
        
        # Handle pred_trans shape
        pred_trans = smpl_data['pred_trans'][frame_idx]
        if pred_trans.ndim == 2 and pred_trans.shape[0] == 1:
            pred_trans = pred_trans.squeeze(0)  # (3,)
        pred_trans = pred_trans.unsqueeze(0).to(device)  # (1, 3)
        
        # Generate SMPL mesh
        with torch.no_grad():
            smpl_output = smpl(
                body_pose=pred_rotmat[:, 1:],
                global_orient=pred_rotmat[:, [0]],
                betas=pred_shape,
                transl=pred_trans,
                pose2rot=False,
                default_smpl=True
            )
        
        vertices = smpl_output.vertices[0].cpu().numpy()  # (6890, 3)
        joints = smpl_output.joints[0, :24].cpu().numpy()  # (24, 3)
        
        # Handle coordinate system
        if args.smpl_in_world_coords:
            # SMPL mesh is in world coordinates, transform to camera coordinates
            R_cw = R_wc.T  # R_CW = R_WC^T: rotation from world to camera
            vertices_cam = (R_cw @ (vertices - t_wc.reshape(1, 3)).T).T
            joints_cam = (R_cw @ (joints - t_wc.reshape(1, 3)).T).T
        else:
            # SMPL mesh is already in camera coordinates (default, as from HMR)
            vertices_cam = vertices
            joints_cam = joints
        
        # Project to image plane
        vertices_cam_torch = torch.from_numpy(vertices_cam).float().unsqueeze(0).to(device)  # (1, 6890, 3)
        joints_cam_torch = torch.from_numpy(joints_cam).float().unsqueeze(0).to(device)  # (1, 24, 3)
        
        focal_length = torch.tensor([img_focal], device=device)
        camera_center = torch.tensor([img_center], device=device)  # (1, 2)
        
        # Project vertices
        vertices_2d = perspective_projection(
            vertices_cam_torch,
            rotation=None,  # Already in camera coordinates
            translation=None,
            focal_length=focal_length,
            camera_center=camera_center
        )[0].cpu().numpy()  # (6890, 2)
        
        # Project joints
        joints_2d = perspective_projection(
            joints_cam_torch,
            rotation=None,
            translation=None,
            focal_length=focal_length,
            camera_center=camera_center
        )[0].cpu().numpy()  # (24, 2)
        
        # Draw on image
        if args.draw_mesh:
            faces = smpl.faces
            img = draw_smpl_mesh(img, vertices_2d, faces, color=(0, 255, 0), line_thickness=1)
        
        if args.draw_joints:
            img = draw_smpl_joints(img, joints_2d, color=(255, 0, 0), radius=3)
        
        # Save visualization
        output_path = os.path.join(args.output_dir, f"frame_{frame_idx:04d}.jpg")
        cv2.imwrite(output_path, img)
    
    print(f"Visualization saved to {args.output_dir}")


if __name__ == '__main__':
    main()

