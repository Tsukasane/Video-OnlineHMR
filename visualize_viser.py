import argparse
import os
import time
from typing import Any, Optional

import numpy as np
import open3d as o3d
import torch

from lib.models.smpl import SMPL


SMPL_MODEL: SMPL = SMPL()
SMPL_MODEL.eval()
SMPL_FACES: Optional[np.ndarray] = None
faces_attr = getattr(SMPL_MODEL, "faces", None)
if faces_attr is not None:
    SMPL_FACES = np.asarray(faces_attr, dtype=np.int32)


def read_camera_poses(txt_file: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read camera poses from text file.
    Format per line: <timestep> tx ty tz qx qy qz qw
    Returns translations, rotation matrices, quaternions.
    """
    cam_r, cam_t, cam_q = [], [], []
    with open(txt_file, "r") as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) < 8:
                continue
            _, tx, ty, tz, qx, qy, qz, qw = vals[:8]

            R = o3d.geometry.get_rotation_matrix_from_quaternion([qw, qx, qy, qz])
            t = np.array([tx, ty, tz], dtype=float)
            q = np.array([qx, qy, qz, qw], dtype=float)

            cam_r.append(R)
            cam_t.append(t)
            cam_q.append(q)

    return np.asarray(cam_t, dtype=float), np.asarray(cam_r, dtype=float), np.asarray(cam_q, dtype=float)


def _load_camera_transforms(camera_txt_path: str) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Load camera translations and rotations from a simple pose text file."""
    try:
        cam_t_np, cam_R_np, _ = read_camera_poses(camera_txt_path)
    except Exception as exc:
        print(f"Failed to read camera poses from {camera_txt_path}: {exc}")
        return None, None
    if cam_t_np.size == 0 or cam_R_np.size == 0:
        print(f"Camera trajectory is empty: {camera_txt_path}")
        return None, None
    return torch.from_numpy(cam_t_np).float(), torch.from_numpy(cam_R_np).float()


def load_human_world_vertices(
    human_npz_path: str,
    camera_txt_path: str,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load predicted humans in world coordinates using associated camera poses."""
    if not human_npz_path or not os.path.exists(human_npz_path):
        print(f"Human prediction file not found or invalid: {human_npz_path}")
        return None, None

    cam_t, cam_R = _load_camera_transforms(camera_txt_path)
    if cam_t is None or cam_R is None:
        return None, None

    faces = SMPL_FACES
    smpl_model = SMPL_MODEL

    try:
        with np.load(human_npz_path) as data:
            pred_rotmat = torch.from_numpy(data["pred_rotmat"]).float()
            pred_shape = torch.from_numpy(data["pred_shape"]).float()
            pred_trans = torch.from_numpy(data["pred_trans"]).float()
    except Exception as exc:
        print(f"Failed to load human predictions from {human_npz_path}: {exc}")
        return None, None

    if pred_trans.ndim == 3:
        pred_trans = pred_trans.squeeze(1)
    if pred_trans.ndim == 1:
        pred_trans = pred_trans.unsqueeze(0)

    mean_shape = pred_shape.mean(dim=0, keepdim=True)
    pred_shape = mean_shape.expand(pred_rotmat.shape[0], -1).contiguous()

    num_frames = pred_rotmat.shape[0]

    with torch.no_grad():
        target_joints = 24
        joint_count = pred_rotmat.shape[1]
        if joint_count > target_joints:
            pred_rotmat = pred_rotmat[:, :target_joints]
        elif joint_count < target_joints:
            pad = pred_rotmat[:, -1:].repeat(1, target_joints - joint_count, 1, 1)
            pred_rotmat = torch.cat([pred_rotmat, pad], dim=1)
        smpl_out = smpl_model(
            body_pose=pred_rotmat[:, 1:],
            global_orient=pred_rotmat[:, [0]],
            betas=pred_shape,
            transl=pred_trans,
            pose2rot=False,
            default_smpl=True,
        )
        vertices_local = smpl_out.vertices

    vertices_world = torch.einsum("bij,bnj->bni", cam_R, vertices_local) + cam_t.unsqueeze(1)
    vertices_world_np = vertices_world.cpu().numpy().astype(np.float32, copy=False)
    return vertices_world_np, faces


def _load_camera_for_vis(camera_path: str) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load camera poses for visualization from a generic pose text file."""
    try:
        cam_t_np, cam_R_np, _ = read_camera_poses(camera_path)
    except Exception as exc:
        print(f"Failed to load camera trajectory from {camera_path}: {exc}")
        return None, None
    if cam_t_np.size == 0 or cam_R_np.size == 0:
        print(f"Camera trajectory is empty: {camera_path}")
        return None, None
    return cam_t_np, cam_R_np


def render_viser_scene(
    *,
    human_npz_path: Optional[str],
    camera_path: str,
    stride: int = 5,
    human_stride: int = 20,
    static: bool = False,
) -> None:
    import viser

    cam_t, cam_R = _load_camera_for_vis(camera_path)
    if cam_t is None or cam_R is None:
        print(f"Could not load camera trajectory from {camera_path}")
        return

    stride = max(int(stride), 1)
    positions = cam_t[::stride]
    rotations = cam_R[::stride]
    if positions.size == 0 or rotations.size == 0:
        print("Camera trajectory is empty after applying stride.")
        return

    human_vertices_seq: Optional[np.ndarray] = None
    human_faces: Optional[np.ndarray] = None
    if human_npz_path:
        verts_world, faces = load_human_world_vertices(human_npz_path, camera_path)
        if verts_world is not None and faces is not None:
            human_stride = max(int(human_stride), 1)
            human_vertices_seq = verts_world[::human_stride]
            human_faces = faces.astype(np.int32, copy=False)

    cam_steps = positions.shape[0]
    human_steps = human_vertices_seq.shape[0] if human_vertices_seq is not None else 0
    timeline_steps = max(
        (cam_steps - 1) * stride + 1 if cam_steps else 0,
        (human_steps - 1) * human_stride + 1 if human_steps else 0,
        cam_steps,
        human_steps,
    )
    world_origin = positions[0]

    def _make_line_segments(points_xyz: np.ndarray) -> np.ndarray:
        if points_xyz.shape[0] < 2:
            return np.empty((0, 2, 3), dtype=np.float32)
        return np.stack([points_xyz[:-1], points_xyz[1:]], axis=1).astype(np.float32)

    def _make_frustum_segments(cam_r: np.ndarray, cam_t_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cam_points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.2, 0.2, 0.3],
                [0.2, -0.2, 0.3],
                [-0.2, -0.2, 0.3],
                [-0.2, 0.2, 0.3],
            ],
            dtype=np.float32,
        )
        cam_lines = np.array(
            [
                [0, 1],
                [0, 2],
                [0, 3],
                [0, 4],
                [1, 2],
                [2, 3],
                [3, 4],
                [4, 1],
            ],
            dtype=np.int32,
        )
        segments = []
        time_ids: list[np.ndarray] = []
        for idx, (R, t) in enumerate(zip(cam_r, cam_t_xyz)):
            frustum_points = cam_points @ R.T + t
            segments.append(frustum_points[cam_lines])
            time_ids.append(np.full(cam_lines.shape[0], idx, dtype=np.int32))
        if not segments:
            return np.empty((0, 2, 3), dtype=np.float32), np.empty((0,), dtype=np.int32)
        return np.concatenate(segments, axis=0), np.concatenate(time_ids, axis=0)

    def _viridis_colors(count: int) -> np.ndarray:
        stops = np.array([[52, 25, 121], [29, 144, 168], [213, 236, 104]], dtype=np.float32) / 255.0
        if count <= 0:
            return np.empty((0, 3), dtype=np.float32)
        if count == 1:
            return stops[[0]]
        stop_pos = np.linspace(0.0, 1.0, stops.shape[0])
        positions_lin = np.linspace(0.0, 1.0, count)
        channels = [np.interp(positions_lin, stop_pos, stops[:, ch]) for ch in range(3)]
        colors = np.stack(channels, axis=1).astype(np.float32)
        return np.clip(colors, 0.0, 1.0)

    def _frustum_segment_colors(time_ids: np.ndarray, total_steps: int) -> np.ndarray:
        if time_ids.size == 0 or total_steps <= 0:
            return np.empty((0, 2, 3), dtype=np.float32)
        colors_lut = _viridis_colors(total_steps)
        capped_ids = np.clip(time_ids, 0, max(colors_lut.shape[0] - 1, 0))
        frustum_colors = colors_lut[capped_ids]
        return np.repeat(frustum_colors[:, None, :], 2, axis=1)

    server = viser.ViserServer()
    server.scene.set_up_direction("-y")

    color_cache: dict[int, np.ndarray] = {}
    def _color_array(n: int) -> np.ndarray:
        if n in color_cache:
            return color_cache[n]
        base = np.asarray((249, 199, 155), dtype=np.float32) / 255.0
        arr = np.broadcast_to(base, (n, 2, 3)).copy()
        color_cache[n] = arr
        return arr

    with server.gui.add_folder("Playback"):
        gui_show_traj = server.gui.add_checkbox("Show Trajectory", True)
        gui_show_frustum = server.gui.add_checkbox("Show Frustum", True)
        gui_show_humans = server.gui.add_checkbox(
            "Show Humans",
            bool(human_vertices_seq is not None and human_faces is not None),
            disabled=human_vertices_seq is None or human_faces is None,
        )
        gui_timestep = server.gui.add_slider(
            "Timestep",
            min=0,
            max=max(0, timeline_steps - 1),
            step=1,
            initial_value=0,
            disabled=timeline_steps == 0,
        )
        gui_next_frame = server.gui.add_button("Next Frame", disabled=timeline_steps == 0 or static)
        gui_prev_frame = server.gui.add_button("Prev Frame", disabled=timeline_steps == 0 or static)
        gui_playing = server.gui.add_checkbox("Playing", not static, disabled=static or timeline_steps == 0)
        gui_fps = server.gui.add_slider("FPS", min=1, max=60, step=0.5, initial_value=30.0)

    traj_handle = server.scene.add_line_segments(
        name="/trajectory/segments",
        points=np.empty((0, 2, 3), dtype=np.float32),
        colors=np.empty((0, 2, 3), dtype=np.float32),
        line_width=2.0,
    )
    frustum_handle = server.scene.add_line_segments(
        name="/trajectory/frustum",
        points=np.empty((0, 2, 3), dtype=np.float32),
        colors=np.empty((0, 2, 3), dtype=np.float32),
        line_width=1.2,
    )

    human_mesh_handle: Optional[Any] = None
    static_human_mesh_handles: list[Any] = []
    if human_vertices_seq is not None and human_faces is not None:
        if static:
            for idx, verts in enumerate(human_vertices_seq):
                handle = server.scene.add_mesh_simple(
                    name=f"/humans/{idx}",
                    vertices=verts,
                    faces=human_faces,
                    flat_shading=False,
                    wireframe=False,
                    opacity=None,
                    color=(249.0 / 255.0, 199.0 / 255.0, 155.0 / 255.0),
                    side="double",
                )
                handle.visible = gui_show_humans.value
                static_human_mesh_handles.append(handle)
        else:
            human_mesh_handle = server.scene.add_mesh_simple(
                name="/humans",
                vertices=human_vertices_seq[0],
                faces=human_faces,
                flat_shading=False,
                wireframe=False,
                opacity=None,
                color=(249.0 / 255.0, 199.0 / 255.0, 155.0 / 255.0),
                side="double",
            )
            human_mesh_handle.visible = gui_show_humans.value

    frustum_segments_full, frustum_time_ids_full = _make_frustum_segments(rotations, positions)
    trajectory_segments_full = _make_line_segments(positions)

    def _update_step(base_step_idx: int) -> None:
        if static:
            traj_handle.visible = gui_show_traj.value and trajectory_segments_full.size > 0
            frustum_handle.visible = gui_show_frustum.value and frustum_segments_full.size > 0
            return

        cam_idx = min(base_step_idx // stride, max(cam_steps - 1, 0))
        traj_segments = _make_line_segments(positions[: cam_idx + 1])
        if gui_show_traj.value and traj_segments.size:
            traj_handle.visible = True
            traj_handle.points = traj_segments
            traj_handle.colors = _color_array(traj_segments.shape[0])
        else:
            traj_handle.visible = gui_show_traj.value and traj_segments.size > 0

        frustum_segments, frustum_time_ids = _make_frustum_segments(
            rotations[: cam_idx + 1],
            positions[: cam_idx + 1],
        )
        if gui_show_frustum.value and frustum_segments.size:
            frustum_handle.visible = True
            frustum_handle.points = frustum_segments
            frustum_colors = _frustum_segment_colors(frustum_time_ids, rotations.shape[0])
            if frustum_colors.size:
                frustum_handle.colors = frustum_colors
        else:
            frustum_handle.visible = gui_show_frustum.value and frustum_segments.size > 0

        if human_mesh_handle is not None and human_vertices_seq is not None and human_vertices_seq.size:
            human_idx = min(base_step_idx // human_stride, human_vertices_seq.shape[0] - 1)
            human_mesh_handle.vertices = human_vertices_seq[human_idx]
            human_mesh_handle.visible = gui_show_humans.value

    if static:
        traj_handle.points = trajectory_segments_full
        if trajectory_segments_full.size:
            traj_handle.colors = _color_array(trajectory_segments_full.shape[0])
        traj_handle.visible = gui_show_traj.value and trajectory_segments_full.size > 0

        frustum_handle.points = frustum_segments_full
        if frustum_segments_full.size:
            frustum_handle.colors = _frustum_segment_colors(frustum_time_ids_full, rotations.shape[0])
        frustum_handle.visible = gui_show_frustum.value and frustum_segments_full.size > 0

        if static_human_mesh_handles:
            for handle in static_human_mesh_handles:
                handle.visible = gui_show_humans.value

    _update_step(0)

    @gui_show_traj.on_update
    def _(_) -> None:
        _update_step(gui_timestep.value)
        server.flush()

    @gui_show_frustum.on_update
    def _(_) -> None:
        _update_step(gui_timestep.value)
        server.flush()

    if static_human_mesh_handles:
        @gui_show_humans.on_update
        def _(_) -> None:
            for handle in static_human_mesh_handles:
                handle.visible = gui_show_humans.value
            server.flush()
    elif human_mesh_handle is not None:
        @gui_show_humans.on_update
        def _(_) -> None:
            human_mesh_handle.visible = gui_show_humans.value
            server.flush()

    @gui_timestep.on_update
    def _(_) -> None:
        _update_step(gui_timestep.value)
        server.flush()

    @gui_next_frame.on_click
    def _(_) -> None:
        if timeline_steps == 0:
            return
        gui_timestep.value = (gui_timestep.value + 1) % timeline_steps

    @gui_prev_frame.on_click
    def _(_) -> None:
        if timeline_steps == 0:
            return
        gui_timestep.value = (gui_timestep.value - 1) % timeline_steps

    @gui_playing.on_update
    def _(_) -> None:
        gui_timestep.disabled = gui_playing.value or timeline_steps == 0
        gui_next_frame.disabled = gui_playing.value or timeline_steps == 0
        gui_prev_frame.disabled = gui_playing.value or timeline_steps == 0

    base_step = 0.0
    last_time = time.time()
    while True:
        now = time.time()
        dt = now - last_time
        last_time = now
        if not static and gui_playing.value and timeline_steps > 0:
            frame_advance = dt * gui_fps.value
            base_step = (base_step + frame_advance) % timeline_steps
            gui_timestep.value = int(base_step)
        else:
            base_step = float(gui_timestep.value)
        time.sleep(1.0 / max(gui_fps.value, 1e-3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal Viser demo for human trajectories.")
    parser.add_argument("--human_npz_path", type=str, required=False, help="Path to human prediction npz.")
    parser.add_argument("--camera_path", type=str, required=True, help="Camera trajectory txt for the human prediction.")
    parser.add_argument("--stride", type=int, default=5, help="Stride for sampling camera poses.")
    parser.add_argument("--human_stride", type=int, default=1, help="Stride for subsampling human meshes.")
    parser.add_argument("--static", action="store_true", help="Use a static viser view (no autoplay).")
    args = parser.parse_args()

    render_viser_scene(
        human_npz_path=args.human_npz_path,
        camera_path=args.camera_path,
        stride=args.stride,
        human_stride=args.human_stride,
        static=args.static,
    )
