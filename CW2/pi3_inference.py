import sys
sys.path.append('external/Pi3')
import torch
import argparse
import numpy as np
import os
import json
from pi3.utils.basic import load_multimodal_data, write_ply
from pi3.utils.geometry import depth_edge
from pi3.models.pi3x import Pi3X
from pi3.pipe.pi3x_vo import Pi3XVO
from utils import *


def write_transforms_json(transforms_file, camera_poses, image_paths, intrinsics, width, height, ply_file_path, depth_paths=None, aabb_scale=16):
    """Write nerfstudio transforms.json file.
    
    Args:
        transforms_file: Path to output transforms.json file
        camera_poses: numpy array of shape (N, 4, 4) - camera-to-world poses (OpenCV convention)
        image_paths: List of image filenames
        intrinsics: numpy array of shape (N, 3, 3) or (3, 3) - camera intrinsics, or None
        width: Image width
        height: Image height
        ply_file_path: Relative path to PLY file
        depth_paths: List of depth image filenames (optional)
        aabb_scale: AABB scale (default: 16)
    """
    os.makedirs(os.path.dirname(transforms_file), exist_ok=True)
    
    # Extract intrinsics (use first frame's intrinsics if available)
    if intrinsics is not None:
        if len(intrinsics.shape) == 3:
            K = intrinsics[0]  # Use first frame's intrinsics
        else:
            K = intrinsics


        ## YOUR CODE HERE: Extract focal lengths and principal point from K
        fl_x = float(K[0, 0])
        fl_y = float(K[1, 1])
        cx   = float(K[0, 2])
        cy   = float(K[1, 2])


    else:
        # Default intrinsics (assume centered, square pixels)
        fl_x = fl_y = float(width)
        cx = float(width) / 2.0
        cy = float(height) / 2.0
    
    # Create frames array
    frames = []
    for i, (pose, img_path) in enumerate(zip(camera_poses, image_paths)):


        ## YOUR CODE HERE: Convert pose from Pi3's coordinate convention (OpenCV) to nerfstudio's convention to get `transform_matrix`
        # OpenCV convention: X-Right, Y-Down, Z-Forward
        # nerfstudio/OpenGL convention: X-Right, Y-Up, Z-Backward
        # Flip Y and Z axes (columns 1 and 2) of the camera-to-world matrix
        transform_matrix = pose.copy()
        transform_matrix[:3, 1] *= -1  # flip Y axis
        transform_matrix[:3, 2] *= -1  # flip Z axis
        transform_matrix = transform_matrix.tolist()


        frame = {
            "file_path": f"./images/{img_path}",
            "transform_matrix": transform_matrix
        }
        
        # Add depth file path if available
        if depth_paths is not None and i < len(depth_paths):
            frame["depth_file_path"] = f"./depths/{depth_paths[i]}"
        
        frames.append(frame)
    
    # Setting applied transform to the same as `ns-process-data` outputs. This does not affect NeRF/gsplat training.
    applied_transform = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [-0.0, -1.0, -0.0, -0.0]
    ]
    
    # Create transforms dictionary
    transforms = {
        "fl_x": fl_x,
        "fl_y": fl_y,
        "cx": cx,
        "cy": cy,
        "w": int(width),
        "h": int(height),
        "aabb_scale": aabb_scale,
        "frames": frames,
        "applied_transform": applied_transform,
        "ply_file_path": ply_file_path
    }
    
    # Write JSON file
    with open(transforms_file, 'w') as f:
        json.dump(transforms, f, indent=4)


if __name__ == '__main__':
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run inference with the Pi3 model.")
    
    parser.add_argument("--data_path", type=str, default='data/static_scene.mp4',
                        help="Path to the input image directory or a video file.")
    parser.add_argument("--conditions_path", type=str, default=None,
                        help="Optional path to a .npz file containing 'poses', 'depths', 'intrinsics'.")
    parser.add_argument("--save_path", type=str, default='results/pi3',
                        help="Path to save the output .ply file.")
    parser.add_argument("--interval", type=int, default=-1,
                        help="Interval to sample image. Default: 1 for images dir, 10 for video")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to the model checkpoint file. Default: None")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Device to run inference on ('cuda' or 'cpu'). Default: 'cuda'")
    parser.add_argument("--downsample_pc", type=str, default='voxel:0.1',
                        help="Downsample point cloud: 'voxel:SIZE' (e.g., 'voxel:0.01'), 'random:NUM' (e.g., 'random:100000'), or 'uniform:STEP' (e.g., 'uniform:2'). Default: None (no downsampling)")
    
    args = parser.parse_args()
    if args.interval < 0:
        args.interval = 10 if args.data_path.endswith('.mp4') else 1
    print(f'Sampling interval: {args.interval}')
    
    # Determine output file name from input video/image directory name
    data_basename = os.path.splitext(os.path.basename(args.data_path))[0]
    
    # If save_path is a directory or doesn't end with .ply, use video name
    if os.path.isdir(args.save_path) or not args.save_path.endswith('.ply'):
        save_dir = args.save_path
        args.save_path = os.path.join(save_dir, f'{data_basename}')
    else:
        save_dir = os.path.dirname(args.save_path) or '.'
    
    # Create COLMAP directory structure
    colmap_dir = os.path.join(save_dir, data_basename, 'colmap', 'sparse', '0')
    images_dir = os.path.join(save_dir, data_basename, 'images')
    images_4_dir = os.path.join(save_dir, data_basename, 'images_4')
    depths_dir = os.path.join(save_dir, data_basename, 'depths')

    # 1. Prepare model
    print(f"Loading model...")
    device = torch.device(args.device)
    if args.ckpt is not None:
        model = Pi3X().to(device).eval()
        if args.ckpt.endswith('.safetensors'):
            from safetensors.torch import load_file
            weight = load_file(args.ckpt)
        else:
            weight = torch.load(args.ckpt, map_location=device, weights_only=False)
        
        model.load_state_dict(weight, strict=False)
    else:
        model = Pi3X.from_pretrained("yyfz233/Pi3X").to(device).eval()
        # or download checkpoints from `https://huggingface.co/yyfz233/Pi3X/resolve/main/model.safetensors`, and `--ckpt ckpts/model.safetensors`

    pipe = Pi3XVO(model)

    # 2. Prepare input data
    # Load optional conditions from .npz
    poses = None
    depths = None
    intrinsics = None

    if args.conditions_path is not None and os.path.exists(args.conditions_path):
        print(f"Loading conditions from {args.conditions_path}...")
        data_npz = np.load(args.conditions_path, allow_pickle=True)

        poses = data_npz['poses']             # Expected (N, 4, 4) OpenCV camera-to-world
        depths = data_npz['depths']           # Expected (N, H, W)
        intrinsics = data_npz['intrinsics']   # Expected (N, 3, 3)

    conditions = dict(
        intrinsics=intrinsics,
        poses=poses,
        depths=depths
    )

    # Load images (Required)
    imgs, conditions = load_multimodal_data(args.data_path, conditions, interval=args.interval, device=device) 

    """
    Args:
        imgs (torch.Tensor): Input RGB images valued in [0, 1].
            Shape: (B, N, 3, H, W).
        intrinsics (torch.Tensor, optional): Camera intrinsic matrices.
            Shape: (B, N, 3, 3).
            Values are in pixel coordinates (not normalized).
        rays (torch.Tensor, optional): Pre-computed ray directions (unit vectors).
            Shape: (B, N, H, W, 3).
            Can replace `intrinsics` as a geometric condition.
        poses (torch.Tensor, optional): Camera-to-World matrices.
            Shape: (B, N, 4, 4).
            Coordinate system: OpenCV convention (Right-Down-Forward).
        depths (torch.Tensor, optional): Ground truth or prior depth maps.
            Shape: (B, N, H, W).
            Invalid values (e.g., sky or missing data) should be set to 0.
        mask_add_depth (torch.Tensor, optional): Mask for depth condition.
            Shape: (B, N, N).
        mask_add_ray (torch.Tensor, optional): Mask for ray/intrinsic condition.
            Shape: (B, N, N).
        mask_add_pose (torch.Tensor, optional): Mask for pose condition.
            Shape: (B, N, N).
            Note: Requires at least two frames to be True to establish a meaningful
            coordinate system (absolute pose for a single frame provides no relative constraint).
    """

    # 3. Infer
    print("Running model inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    with torch.no_grad():
        res = pipe(
            imgs=imgs, 
            dtype=dtype,
        )

    pose = res['camera_poses'] # camera-to-world poses in OpenCV convention

    # 4. process mask
    masks = res['conf'][0] > 0.05

    # 5. Save points
    print(f"Saving point cloud to: {args.save_path}/sparse_points.ply")
    if os.path.dirname(args.save_path):
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    points_masked = res['points'][0][masks].cpu().numpy()
    colors_masked = (imgs[0].permute(0, 2, 3, 1)[masks].cpu().numpy() * 255).astype(np.uint8)
    
    # Apply downsampling if requested
    if args.downsample_pc is not None:
        original_count = len(points_masked)
        print(f"Original point cloud size: {original_count} points")
        
        if args.downsample_pc.startswith('voxel:'):
            voxel_size = float(args.downsample_pc.split(':')[1])
            print(f"Downsampling point cloud using voxel grid (voxel_size={voxel_size})...")
            points_masked, colors_masked = downsample_pointcloud_voxel(points_masked, colors_masked, voxel_size)
        elif args.downsample_pc.startswith('random:'):
            num_points = int(args.downsample_pc.split(':')[1])
            print(f"Downsampling point cloud using random sampling (target={num_points} points)...")
            points_masked, colors_masked = downsample_pointcloud_random(points_masked, colors_masked, num_points)
        elif args.downsample_pc.startswith('uniform:'):
            step = int(args.downsample_pc.split(':')[1])
            print(f"Downsampling point cloud using uniform sampling (step={step})...")
            points_masked, colors_masked = downsample_pointcloud_uniform(points_masked, colors_masked, step)
        else:
            print(f"Warning: Unknown downsampling method '{args.downsample_pc}'. Supported: 'voxel:SIZE', 'random:NUM', 'uniform:STEP'")
        
        downsampled_count = len(points_masked)
        print(f"Downsampled point cloud size: {downsampled_count} points (reduction: {100 * (1 - downsampled_count / original_count):.1f}%)")
    
    # Keep OpenCV copy for COLMAP points3D (before OpenGL conversion)
    points_masked_opencv = points_masked.copy()
    
    # 6. Save images (original and downsampled)
    print(f"Saving images to: {images_dir}")
    image_paths = save_images(imgs, images_dir, downsample_factor=1)
    
    print(f"Saving downsampled images (4x) to: {images_4_dir}")
    image_paths_4 = save_images(imgs, images_4_dir, downsample_factor=4)
    
    # 6.5. Compute and save depth maps
    print("Computing depth maps from points and poses...")
    depth_maps = compute_depth_maps(res['points'], pose)  # (N, H, W)
    
    print(f"Saving depth maps to: {depths_dir}")
    depth_paths = save_depth_maps(depth_maps, depths_dir)
    
    # 7. Save COLMAP format files
    print(f"Saving COLMAP format to: {colmap_dir}")
    camera_poses_np = pose[0].cpu().numpy()  # (N, 4, 4)
    
    # Get image dimensions from first image
    # imgs shape: (B, N, 3, H, W)
    H, W = imgs.shape[3], imgs.shape[4]
    
    # Get intrinsics: from conditions or estimate from point map (MoGE-style)
    intrinsics_np = None
    if conditions.get('intrinsics') is not None:
        intrinsics_tensor = conditions['intrinsics']
        if intrinsics_tensor is not None:
            intrinsics_np = intrinsics_tensor[0].cpu().numpy()  # (N, 3, 3) -> use first batch
    if intrinsics_np is None:
        mask_intrinsics = res['conf'] > 0.05
        intrinsics_np = estimate_intrinsics_from_points_moge(
            res['points'], pose, mask_intrinsics.float(), 0.1, device
        )
        if intrinsics_np is not None:
            print(f"Estimated intrinsics: fx={intrinsics_np[0,0]:.2f}, fy={intrinsics_np[1,1]:.2f}, cx={intrinsics_np[0,2]:.2f}, cy={intrinsics_np[1,2]:.2f}")
        else:
            print("Intrinsics estimation skipped (MoGE not available or failed). Using default.")
    
    os.makedirs(colmap_dir, exist_ok=True)
    cameras_file = os.path.join(colmap_dir, 'cameras.bin')
    write_colmap_cameras(cameras_file, intrinsics_np, W, H)
    images_file = os.path.join(colmap_dir, 'images.bin')
    write_colmap_images(images_file, camera_poses_np, image_paths)
    points3d_file = os.path.join(colmap_dir, 'points3D.bin')
    write_colmap_points3d(points3d_file, points_masked_opencv, colors_masked)
    write_ply(points_masked, colors_masked, args.save_path + '/sparse_pc.ply')
    
    # 8. Save transforms.json (nerfstudio format)
    transforms_json_path = os.path.join(save_dir, data_basename, 'transforms.json')
    ply_relative_path = os.path.relpath(args.save_path + '/sparse_pc.ply', os.path.dirname(transforms_json_path))
    print(f"Saving transforms.json to: {transforms_json_path}")
    write_transforms_json(
        transforms_json_path,
        camera_poses_np,
        image_paths,
        intrinsics_np,
        W,
        H,
        ply_relative_path,
        depth_paths=depth_paths
    )

    print("Done.")
