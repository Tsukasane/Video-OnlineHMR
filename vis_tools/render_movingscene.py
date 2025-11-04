import open3d as o3d
import numpy as np
import imageio

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
    Returns a list of LineSet pyramids.
    """
    poses = []
    with open(txt_file, "r") as f:
        lines = f.readlines()

    for l_id, line in enumerate(lines):
        print(f"debug -- l_id {l_id}")
        vals = list(map(float, line.strip().split()))
        timestep = vals[0]
        tx, ty, tz = vals[1:4]
        qx, qy, qz, qw = vals[4:]

        # quaternion → rotation matrix
        R = o3d.geometry.get_rotation_matrix_from_quaternion([qw, qx, qy, qz])
        t = np.array([tx, ty, tz])

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

    return poses


def render_scene_with_cameras(ply_file, pose_file, out_gif="scene.gif"):
    # Load point cloud
    pcd = o3d.io.read_point_cloud(ply_file)
    if not pcd.has_colors():
        pcd.paint_uniform_color([0.8, 0.8, 0.8])  # fallback if no RGB in ply

    # Load cameras
    cams = load_camera_poses(pose_file)

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

    # Add point cloud + cameras
    render.scene.add_geometry("pcd", pcd, mat)
    for i, cam in enumerate(cams):
        render.scene.add_geometry(f"cam{i}", cam, cam_mat)

    # Camera view setup
    center = pcd.get_center()
    up = [0, 1, 0]
    eye0 = center + np.array([3, 3, 3])

    imgs = []
    radius = np.linalg.norm(eye0 - center)

    for angle in range(0, 360, 10):
        # Yaw around the vertical (Y) axis, keeping camera above the object
        theta = np.deg2rad(angle)
        eye = center + radius * np.array([np.sin(theta), -0.2, np.cos(theta)])  # 0.2: small height above ground

        up = np.array([0, 1, 0])  # keep world Y up

        render.setup_camera(60, center, eye, up)
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
    cam_traj_path = "/ocean/projects/cis240055p/yzhao16/MASt3R-SLAM/logs/P0_09_outdoor_walk_video_incremental_kf.txt"
    scene_ply_path = "/ocean/projects/cis240055p/yzhao16/MASt3R-SLAM/logs/P0_09_outdoor_walk_video.ply"
    render_scene_with_cameras(scene_ply_path, cam_traj_path, "output_P0_09_incremental.gif")



