"""
Quick test script to check data format and coordinate transformation.
"""

import numpy as np
import torch

# Test loading camera extrinsics
def test_load_camera():
    txt_path = "logs/annab2n2_demo_images_incremental_all.txt"
    with open(txt_path, 'r') as f:
        first_line = f.readline().strip()
        parts = first_line.split()
        print(f"First camera line: {first_line}")
        print(f"Parts: {len(parts)}")
        if len(parts) >= 8:
            scale = float(parts[0])
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            print(f"Scale: {scale}")
            print(f"Translation: [{x}, {y}, {z}]")
            print(f"Quaternion: [{qx}, {qy}, {qz}, {qw}]")

# Test loading SMPL data
def test_load_smpl():
    npz_path = "res_human_camera/5802630_annab2n2_demo.npz"
    data = np.load(npz_path)
    print(f"\nSMPL data keys: {list(data.keys())}")
    for key in data.keys():
        print(f"  {key}: shape={data[key].shape}, dtype={data[key].dtype}")
        if key == 'pred_trans':
            print(f"    First frame pred_trans: {data[key][0]}")
        if key == 'pred_rotmat':
            print(f"    First frame pred_rotmat shape: {data[key][0].shape}")

if __name__ == '__main__':
    print("Testing camera extrinsics format...")
    test_load_camera()
    print("\nTesting SMPL data format...")
    test_load_smpl()

