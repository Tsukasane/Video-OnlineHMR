import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/..')

import open3d as o3d
import numpy as np
import imageio
import torch
import trimesh
from pytorch3d.transforms import quaternion_to_matrix

import argparse

rainbow_colors = [
    [1.0, 0.0, 0.0],     # red
    [1.0, 0.2, 0.0],
    [1.0, 0.4, 0.0],
    [1.0, 0.6, 0.0],
    [1.0, 0.8, 0.0],
    [1.0, 1.0, 0.0],     # yellow
    [0.8, 1.0, 0.0],
    [0.6, 1.0, 0.0],
    [0.4, 1.0, 0.2],
    [0.2, 1.0, 0.4],
    [0.0, 1.0, 0.6],
    [0.0, 1.0, 0.8],
    [0.0, 1.0, 1.0],     # cyan
    [0.0, 0.8, 1.0],
    [0.0, 0.6, 1.0],
    [0.0, 0.4, 1.0],
    [0.2, 0.2, 1.0],
    [0.4, 0.0, 1.0],
    [0.6, 0.0, 1.0],
    [0.8, 0.0, 1.0],     # purple
]

def load_camera_poses(txt_file, scale=0.2):
    """
    Load camera poses from text file.
    Each line: tx ty tz qx qy qz qw
    Returns a list of LineSet pyramids and camera poses (t, q) for human mesh transformation.
    """
    poses = []
    cam_poses = []  # (t, q) for each frame
    
    with open(txt_file, "r") as f:
        lines = f.readlines()

    depth_scaler = 1.0
    for l_id, line in enumerate(lines):
        # print(f"debug -- l_id {l_id}")
        vals = list(map(float, line.strip().split()))
        if l_id == 0:
            depth_scaler = vals[0]
        tx, ty, tz = vals[1:4]
        qx, qy, qz, qw = vals[4:]
        
        # Store camera pose
        cam_poses.append((depth_scaler * np.array([tx, ty, tz]), np.array([qw, qx, qy, qz])))

        # quaternion → rotation matrix
        R = o3d.geometry.get_rotation_matrix_from_quaternion([qw, qx, qy, qz])
        t = depth_scaler * np.array([tx, ty, tz])

        # pyramid points (camera frustum)
        cam_points = np.array([
            [0, 0, 0],
            [scale, scale, scale * 1.5],
            [scale, -scale, scale * 1.5],
            [-scale, -scale, scale * 1.5],
            [-scale, scale, scale * 1.5],
        ])
        cam_lines = [
            [0, 1], [0, 2], [0, 3], [0, 4],
            [1, 2], [2, 3], [3, 4], [4, 1]
        ]

        cam = o3d.geometry.LineSet()
        cam.points = o3d.utility.Vector3dVector(cam_points @ R.T + t)
        cam.lines = o3d.utility.Vector2iVector(cam_lines)
        cam.colors = o3d.utility.Vector3dVector([rainbow_colors[l_id%20]] * len(cam_lines))  # red lines
        poses.append(cam)

    return poses, cam_poses, depth_scaler


def render_scene_with_cameras(ply_file, pose_file, out_gif="scene.gif", angle=180, 
                              human_npz_path=None, render_interval=2):
    """
    Args:
        angle: 0-360
        human_npz_path: Path to npz file containing human mesh predictions (pred_cam, pred_pose, pred_shape, pred_rotmat, pred_trans)
        render_interval: Interval for rendering human meshes (skip frames to reduce memory)
    """
    # Load cameras and camera poses (get scaler from trajectory file)
    cams, cam_poses, depth_scaler = load_camera_poses(pose_file)
    print(f"Loaded camera trajectory with scaler: {depth_scaler:.6f}")
    
    # Load point cloud
    pcd = o3d.io.read_point_cloud(ply_file)
    if not pcd.has_colors():
        pcd.paint_uniform_color([0.8, 0.8, 0.8])  # fallback if no RGB in ply
    
    # Apply scaler to point cloud to match world scale (camera poses already have scaler applied)
    # SLAM point cloud is in original SLAM scale, need to scale to world/metric scale
    pcd_points = np.asarray(pcd.points)
    pcd_points_scaled = pcd_points * depth_scaler
    pcd.points = o3d.utility.Vector3dVector(pcd_points_scaled)
    print(f"Applied scaler {depth_scaler:.6f} to point cloud (scene)")
    
    # Load human mesh data if provided
    human_data = None
    smpl = None
    if human_npz_path is not None and os.path.exists(human_npz_path):
        print(f"Loading human mesh data from {human_npz_path}")
        human_data = np.load(human_npz_path)
        from lib.models.smpl import SMPL
        smpl = SMPL()
        print(f"Loaded {len(human_data['pred_rotmat'])} frames of human mesh data")

    # Offscreen renderer
    w, h = 800, 600
    render = o3d.visualization.rendering.OffscreenRenderer(w, h)

    # Background black
    render.scene.set_background([0, 0, 0, 1])

    # Point cloud material
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.point_size = 3.0
    mat.base_color = [0.8, 0.8, 0.8, 1.0]

    # Camera line material
    cam_mat = o3d.visualization.rendering.MaterialRecord()
    cam_mat.shader = "unlitLine"
    
    # Human mesh material
    human_mat = o3d.visualization.rendering.MaterialRecord()
    human_mat.shader = "defaultLit"

    # Add point cloud
    render.scene.add_geometry("pcd", pcd, mat)

    # Camera view setup - increased distance for better view
    center = pcd.get_center()
    up = [0, 1, 0]
    view_radius = 5  # Increased from 5 to 10 for wider view
    eye0 = center + np.array([view_radius, view_radius, view_radius])

    imgs = []
    radius = np.linalg.norm(eye0 - center)

    # Yaw around the vertical (Y) axis, keeping camera above the object
    theta = np.deg2rad(angle)
    eye = center + radius * np.array([np.sin(theta), -0.2, np.cos(theta)])  # 0.2: small height above ground
    up = np.array([0, 1, 0])  # keep world Y up
    render.setup_camera(60, center, eye, up)
    
    set_render_camera = False

    for i, cam in enumerate(cams):
        # Remove previous human mesh if exists (keep only current and previous frame)
        # Similar to run_custom.py, remove i-2 geometry
        if human_data is not None:
            num_keep_frames = 1
            prev_frame_idx = i - render_interval * num_keep_frames
            if prev_frame_idx >= 0:
                try:
                    render.scene.remove_geometry(f"human{prev_frame_idx}")
                    render.scene.remove_geometry(f"cam{prev_frame_idx}")
                except:
                    pass
        
        # Add camera
        render.scene.add_geometry(f"cam{i}", cam, cam_mat)
        
        # Add human mesh if available and at render_interval
        if human_data is not None and smpl is not None and i < len(human_data['pred_rotmat']) and i % render_interval == 0:
            
            # Get camera pose (already scaled by depth_scaler in load_camera_poses)
            # cam_t is in world coordinates with correct scale
            cam_t, cam_q = cam_poses[i]
            current_camt = torch.tensor(cam_t).unsqueeze(0).float()
            current_camq = torch.tensor(cam_q).unsqueeze(0).float()
            
            # Convert quaternion to rotation matrix
            current_camr = quaternion_to_matrix(current_camq)
            
            # Get human mesh parameters for this frame (in camera coordinates)
            pred_rotmat = torch.tensor(human_data['pred_rotmat'][i:i+1])  # [1, 24, 3, 3]
            pred_shape = torch.tensor(human_data['pred_shape'][i:i+1])  # [1, 10]
            pred_trans = torch.tensor(human_data['pred_trans'][i:i+1])  # [1, 1, 3]
            
            # Generate SMPL mesh (in camera coordinates)
            pred = smpl(body_pose=pred_rotmat[:, 1:], 
                       global_orient=pred_rotmat[:, [0]], 
                       betas=pred_shape, 
                       transl=pred_trans.squeeze(1),
                       pose2rot=False, 
                       default_smpl=True)
            pred_vert = pred.vertices  # [1, 6890, 3] in camera coordinates
              
            # Transform to world coordinates using scaled camera pose
            # This ensures human mesh is in the same scale as scene and camera trajectory
            pred_vert_w = torch.einsum('bij,bnj->bni', current_camr, pred_vert) + current_camt[:, None]  # [1, 6890, 3] in world coordinates
            
            # Convert to Open3D mesh
            mesh = trimesh.Trimesh(vertices=pred_vert_w[0].cpu().numpy(), faces=smpl.faces)
            human_mesh = o3d.geometry.TriangleMesh()
            human_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
            human_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
            human_mesh.compute_vertex_normals()
            human_mesh.paint_uniform_color([0.8, 0.6, 0.6])
            
            # Add human mesh to scene
            render.scene.add_geometry(f"human{i}", human_mesh, human_mat)
            
            # Update camera view center based on human mesh if not set yet
            if not set_render_camera:
                human_center = human_mesh.get_center()
                # Use average of point cloud center and human center
                center = (pcd.get_center() + human_center) / 2
                eye0 = center + np.array([view_radius, view_radius, view_radius])
                radius = np.linalg.norm(eye0 - center)
                eye = center + radius * np.array([np.sin(theta), -0.2, np.cos(theta)])
                render.setup_camera(60, center, eye, up)
                set_render_camera = True
        
        img_o3d = render.render_to_image()

        # flip vertically for correct image orientation
        img_np = np.asarray(img_o3d)
        img_np = np.flipud(img_np)
        img_np = np.fliplr(img_np)
        imgs.append(img_np)

    # Save gif
    imageio.mimsave(out_gif, imgs, fps=10)
    print(f"✅ Saved gif to {out_gif}")


if __name__ == "__main__":

    """
    python vis_tools/render_movingcamhuman.py --cam_traj_path ./logs/tdance1_images_incremental_all.txt \
        --scene_ply_path ./res_human_camera/global_results/tdance1_images.ply \
        --save_prefix tdance1 \
        --human_npz_path ./res_human_camera/tdance1.npz \
        --render_interval 1
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--cam_traj_path', type=str, default="./logs/annab2_images_incremental_all.txt")
    parser.add_argument("--scene_ply_path", type=str, default="./res_human_camera/global_results/annab2_images.ply")
    parser.add_argument("--save_prefix", type=str, default="annab2")
    parser.add_argument("--human_npz_path", type=str, default="./res_human_camera/annab2.npz", help="Path to npz file with human mesh predictions")
    parser.add_argument("--render_interval", type=int, default=1, help="Interval for rendering human meshes")

    args = parser.parse_args()
    
    cam_traj_path = args.cam_traj_path
    scene_ply_path = args.scene_ply_path
    render_scene_with_cameras(scene_ply_path, cam_traj_path, 
                              f"output_{args.save_prefix}_camera_scene.gif",
                              angle=180,
                              human_npz_path=args.human_npz_path,
                              render_interval=args.render_interval)



