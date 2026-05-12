import os
import sys
import numpy as np
from typing import Tuple
import struct
from PIL import Image
import torch
import torch.nn.functional as F
sys.path.append('external/MoGe')


def save_images(imgs, output_dir, downsample_factor=1):
    """Save images from tensor to disk.
    
    Args:
        imgs: torch.Tensor of shape (B, N, 3, H, W) with values in [0, 1]
        output_dir: Directory to save images
        downsample_factor: Factor to downsample images (default: 1, no downsampling)
    """
    os.makedirs(output_dir, exist_ok=True)
    imgs_np = imgs[0].permute(0, 2, 3, 1).cpu().numpy()  # (N, H, W, 3)
    imgs_np = (imgs_np * 255).astype(np.uint8)
    
    image_paths = []
    for i in range(imgs_np.shape[0]):
        filename = f'frame_{i+1:05d}.png'
        filepath = os.path.join(output_dir, filename)
        
        img = Image.fromarray(imgs_np[i])
        
        # Downsample if needed
        if downsample_factor > 1:
            new_width = img.width // downsample_factor
            new_height = img.height // downsample_factor
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        img.save(filepath)
        image_paths.append(filename)
    
    return image_paths


def save_depth_maps(depth_maps, output_dir):
    """Save depth maps as 16-bit PNG files.
    
    Args:
        depth_maps: numpy array of shape (N, H, W) - depth values
        output_dir: Directory to save depth maps
        
    Returns:
        depth_paths: List of depth image filenames
    """
    os.makedirs(output_dir, exist_ok=True)
    depth_paths = []
    
    for i in range(len(depth_maps)):
        depth = depth_maps[i]
        
        # Divide by 1000 and convert to 16-bit
        depth_scaled = (depth * 1000.0).astype(np.float32)
        # Clip to valid 16-bit range (0-65535)
        depth_uint16 = np.clip(depth_scaled, 0, 65535).astype(np.uint16)
        
        filename = f'depth_{i+1:05d}.png'
        filepath = os.path.join(output_dir, filename)
        
        # Save as 16-bit PNG
        depth_img = Image.fromarray(depth_uint16, mode='I;16')
        depth_img.save(filepath)
        depth_paths.append(filename)
    
    return depth_paths


def downsample_pointcloud_voxel(points, colors, voxel_size):
    """Downsample point cloud using voxel grid method.
    
    Args:
        points: numpy array of shape (N, 3) - 3D points
        colors: numpy array of shape (N, 3) - RGB colors (0-255)
        voxel_size: float - size of each voxel
        
    Returns:
        points_downsampled: numpy array of shape (M, 3) - downsampled points
        colors_downsampled: numpy array of shape (M, 3) - downsampled colors
    """
    if len(points) == 0:
        return points, colors
    
    # Compute voxel indices for each point
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)
    
    # Use dictionary to store unique voxels (using tuple as key)
    voxel_dict = {}
    
    for i in range(len(points)):
        voxel_key = tuple(voxel_indices[i])
        
        if voxel_key not in voxel_dict:
            # Store first point in this voxel
            voxel_dict[voxel_key] = {
                'point': points[i],
                'color': colors[i],
                'count': 1
            }
        else:
            # Average with existing point in voxel
            existing = voxel_dict[voxel_key]
            count = existing['count'] + 1
            # Weighted average
            existing['point'] = (existing['point'] * existing['count'] + points[i]) / count
            existing['color'] = (existing['color'] * existing['count'] + colors[i]) / count
            existing['count'] = count
    
    # Extract downsampled points and colors
    points_downsampled = np.array([v['point'] for v in voxel_dict.values()])
    colors_downsampled = np.array([v['color'] for v in voxel_dict.values()]).astype(np.uint8)
    
    return points_downsampled, colors_downsampled


def downsample_pointcloud_random(points, colors, num_points):
    """Downsample point cloud using random sampling.
    
    Args:
        points: numpy array of shape (N, 3) - 3D points
        colors: numpy array of shape (N, 3) - RGB colors (0-255)
        num_points: int - target number of points (if > N, returns all points)
        
    Returns:
        points_downsampled: numpy array of shape (M, 3) - downsampled points
        colors_downsampled: numpy array of shape (M, 3) - downsampled colors
    """
    if len(points) == 0:
        return points, colors
    
    num_points = min(num_points, len(points))
    
    # Randomly sample indices
    indices = np.random.choice(len(points), size=num_points, replace=False)
    indices = np.sort(indices)  # Sort for consistency
    
    return points[indices], colors[indices]


def downsample_pointcloud_uniform(points, colors, step):
    """Downsample point cloud using uniform sampling (every Nth point).
    
    Args:
        points: numpy array of shape (N, 3) - 3D points
        colors: numpy array of shape (N, 3) - RGB colors (0-255)
        step: int - take every Nth point
        
    Returns:
        points_downsampled: numpy array of shape (M, 3) - downsampled points
        colors_downsampled: numpy array of shape (M, 3) - downsampled colors
    """
    if len(points) == 0:
        return points, colors
    
    step = max(1, int(step))
    indices = np.arange(0, len(points), step)
    
    return points[indices], colors[indices]


def write_colmap_cameras(camera_file, intrinsics, width, height):
    """Write COLMAP cameras.bin file.
    
    Args:
        camera_file: Path to output cameras.bin file
        intrinsics: numpy array of shape (N, 3, 3) - camera intrinsic matrices
        width: Image width
        height: Image height
    """
    os.makedirs(os.path.dirname(camera_file), exist_ok=True)
    
    with open(camera_file, 'wb') as f:
        # Write number of cameras (assuming all cameras have same intrinsics)
        num_cameras = 1
        f.write(struct.pack('Q', num_cameras))
        
        # Camera model: PINHOLE = 1 (fx, fy, cx, cy)
        camera_id = 1
        model_id = 1  # PINHOLE
        width_i = int(width)
        height_i = int(height)
        
        # Use first camera's intrinsics (or average if multiple)
        if intrinsics is not None:
            if len(intrinsics.shape) == 3:
                K = intrinsics[0]  # Use first frame's intrinsics
            else:
                K = intrinsics
            fx = float(K[0, 0])
            fy = float(K[1, 1])
            cx = float(K[0, 2])
            cy = float(K[1, 2])
        else:
            # Default intrinsics (assume centered, square pixels)
            fx = fy = float(width)
            cx = float(width) / 2.0
            cy = float(height) / 2.0
        
        # Write camera: id (i), model (i), width (Q), height (Q) = 24 bytes, then params (COLMAP/viser format)
        f.write(struct.pack('i', camera_id))
        f.write(struct.pack('i', model_id))
        f.write(struct.pack('QQ', width_i, height_i))
        f.write(struct.pack('dddd', fx, fy, cx, cy))


def write_colmap_images(images_file, camera_poses, image_paths, camera_id=1):
    """Write COLMAP images.bin file.
    
    Args:
        images_file: Path to output images.bin file
        camera_poses: numpy array of shape (N, 4, 4) - camera-to-world poses (OpenCV convention)
        image_paths: List of image filenames
        camera_id: Camera ID (default: 1)
    """
    os.makedirs(os.path.dirname(images_file), exist_ok=True)
    
    with open(images_file, 'wb') as f:
        num_images = len(image_paths)
        f.write(struct.pack('Q', num_images))
        
        for i, (pose, img_path) in enumerate(zip(camera_poses, image_paths)):
            image_id = i + 1
            
            # Convert OpenCV camera-to-world to COLMAP world-to-camera
            # COLMAP uses world-to-camera, so we need to invert
            c2w = pose
            w2c = np.linalg.inv(c2w)
            
            # Extract rotation (3x3) and translation (3,)
            R = w2c[:3, :3]
            t = w2c[:3, 3]
            
            # Convert rotation matrix to quaternion (w, x, y, z)
            # COLMAP uses (w, x, y, z) format
            # Using standard matrix-to-quaternion conversion
            trace = np.trace(R)
            if trace > 0:
                s = np.sqrt(trace + 1.0) * 2
                qw = 0.25 * s
                qx = (R[2, 1] - R[1, 2]) / s
                qy = (R[0, 2] - R[2, 0]) / s
                qz = (R[1, 0] - R[0, 1]) / s
            elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
                s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                qw = (R[2, 1] - R[1, 2]) / s
                qx = 0.25 * s
                qy = (R[0, 1] + R[1, 0]) / s
                qz = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                qw = (R[0, 2] - R[2, 0]) / s
                qx = (R[0, 1] + R[1, 0]) / s
                qy = 0.25 * s
                qz = (R[1, 2] + R[2, 1]) / s
            else:
                s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                qw = (R[1, 0] - R[0, 1]) / s
                qx = (R[0, 2] + R[2, 0]) / s
                qy = (R[1, 2] + R[2, 1]) / s
                qz = 0.25 * s
            
            # Normalize quaternion
            q_norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
            if q_norm > 0:
                qw /= q_norm
                qx /= q_norm
                qy /= q_norm
                qz /= q_norm
            
            # Write image: id (i), qw,qx,qy,qz (4d), tx,ty,tz (3d), camera_id (i) = 64 bytes (COLMAP/viser format)
            f.write(struct.pack('i', image_id))
            f.write(struct.pack('dddd', qw, qx, qy, qz))
            f.write(struct.pack('ddd', t[0], t[1], t[2]))
            f.write(struct.pack('i', camera_id))
            
            # Write image name (null-terminated string)
            img_name_bytes = img_path.encode('utf-8')
            f.write(img_name_bytes)
            f.write(b'\x00')
            
            # Write num_points2D (0 for now, as we don't have 2D feature points)
            f.write(struct.pack('Q', 0))  # No 2D feature points


def write_colmap_points3d(points3d_file, points, colors):
    """Write COLMAP points3D.bin file.
    
    Args:
        points3d_file: Path to output points3D.bin file
        points: numpy array of shape (M, 3) - 3D points in world coordinates
        colors: numpy array of shape (M, 3) - RGB colors (0-255)
    """
    os.makedirs(os.path.dirname(points3d_file), exist_ok=True)
    
    with open(points3d_file, 'wb') as f:
        num_points = len(points)
        f.write(struct.pack('Q', num_points))
        
        for i, (point, color) in enumerate(zip(points, colors)):
            point_id = i + 1
            
            # Write point: id (Q), x, y, z (3d), r, g, b (3B), error (d), track_length (Q)
            f.write(struct.pack('Q', point_id))
            f.write(struct.pack('ddd', point[0], point[1], point[2]))
            f.write(struct.pack('BBB', int(color[0]), int(color[1]), int(color[2])))
            f.write(struct.pack('d', 0.0))  # error
            f.write(struct.pack('Q', 0))  # track_length (no 2D correspondences)


def compute_depth_maps(points_world, camera_poses):
    """Compute depth maps in camera coordinates from world points and camera poses.
    
    Args:
        points_world: torch.Tensor of shape (B, N, H, W, 3) - 3D points in world coordinates
        camera_poses: torch.Tensor of shape (B, N, 4, 4) - camera-to-world poses (OpenCV convention)
        
    Returns:
        depth_maps: numpy array of shape (N, H, W) - depth values in camera coordinates
    """
    B, N, H, W, _ = points_world.shape
    depth_maps = []
    
    # Convert to numpy for easier manipulation
    points_world_np = points_world[0].cpu().numpy()  # (N, H, W, 3)
    camera_poses_np = camera_poses[0].cpu().numpy()  # (N, 4, 4)
    
    for i in range(N):
        # Get points in world coordinates for this frame
        points_w = points_world_np[i]  # (H, W, 3)
        
        # Get camera pose (camera-to-world)
        c2w = camera_poses_np[i]  # (4, 4)
        
        # Convert to world-to-camera
        w2c = np.linalg.inv(c2w)
        
        # Convert points to homogeneous coordinates
        points_w_hom = np.concatenate([
            points_w,
            np.ones((H, W, 1))
        ], axis=-1)  # (H, W, 4)
        
        # Transform to camera coordinates
        points_w_hom_flat = points_w_hom.reshape(-1, 4).T  # (4, H*W)
        points_cam_hom = w2c @ points_w_hom_flat  # (4, H*W)
        points_cam = points_cam_hom[:3].T.reshape(H, W, 3)  # (H, W, 3)
        
        # Depth is the Z coordinate in camera space (positive Z is forward in OpenCV)
        depth = points_cam[:, :, 2]  # (H, W)
        
        # Set invalid points (zero or negative depth) to 0
        depth[depth <= 0] = 0
        
        depth_maps.append(depth)
    
    return np.array(depth_maps)


## from https://github.com/microsoft/MoGe/blob/07444410f1e33f402353b99d6ccd26bd31e469e8/moge/model/v1.py:353-369
def estimate_intrinsics_from_points_moge(points_world, camera_poses, mask, conf_thre, device):
    """Estimate fx, fy (and build intrinsics) from point map using MoGE-style focal recovery.

    Follows https://github.com/microsoft/MoGe/blob/07444410f1e33f402353b99d6ccd26bd31e469e8/moge/model/v1.py:353-369: recover focal from camera-space points and mask, then compute fx, fy from aspect ratio.

    Args:
        points_world: torch.Tensor (B, N, H, W, 3) - points in world coordinates
        camera_poses: torch.Tensor (B, N, 4, 4) - camera-to-world (OpenCV)
        mask: torch.Tensor (B, N, H, W) - confidence mask (True = valid)
        conf_thre: float - already applied to get mask
        device: torch device

    Returns:
        intrinsics: numpy (3, 3) or None if estimation fails
    """
    from moge.utils.geometry_torch import recover_focal_shift

    B, N, H, W, _ = points_world.shape
    pts_w = points_world[0, 0].to(device)
    c2w = camera_poses[0, 0].to(device)
    w2c = torch.linalg.inv(c2w)
    pts_w_flat = pts_w.reshape(-1, 3).T
    pts_w_hom = torch.cat([pts_w_flat, torch.ones(1, pts_w_flat.shape[1], device=device, dtype=pts_w.dtype)], dim=0)
    pts_cam = (w2c @ pts_w_hom)[:3].T.reshape(H, W, 3)
    mask_binary = mask[0, 0].to(device).float()
    pts_cam_batch = pts_cam.unsqueeze(0).float()
    mask_batch = mask_binary.unsqueeze(0)
    focal, shift = recover_focal_shift(pts_cam_batch, mask_batch)
    focal_val = focal.item() if focal.numel() == 1 else focal[0].item()
    aspect_ratio = W / H
    fx_norm = (focal_val / 2.0) * ((1 + aspect_ratio ** 2) ** 0.5) / aspect_ratio
    fy_norm = (focal_val / 2.0) * ((1 + aspect_ratio ** 2) ** 0.5)
    fx_pix = float(fx_norm * W)
    fy_pix = float(fy_norm * H)
    cx_pix = 0.5 * W
    cy_pix = 0.5 * H
    K = np.array([
        [fx_pix, 0, cx_pix],
        [0, fy_pix, cy_pix],
        [0, 0, 1]
    ], dtype=np.float64)
    return K
