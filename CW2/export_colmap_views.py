import json
import os
import numpy as np
import trimesh
import matplotlib.pyplot as plt

SCENE_DIR = "results/colmap/juice"
OUT_DIR = os.path.join(SCENE_DIR, "exported_views")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- Load sparse point cloud ----------
ply_path = os.path.join(SCENE_DIR, "sparse_pc.ply")
pc = trimesh.load(ply_path)

if hasattr(pc, "vertices"):
    pts = np.asarray(pc.vertices)
else:
    raise ValueError("Could not read vertices from sparse_pc.ply")

# Optional colors
colors = None
if hasattr(pc, "colors") and pc.colors is not None and len(pc.colors) == len(pts):
    colors = np.asarray(pc.colors)[:, :3] / 255.0

# ---------- Load camera poses ----------
json_path = os.path.join(SCENE_DIR, "transforms.json")
with open(json_path, "r") as f:
    meta = json.load(f)

cam_centers = []
for fr in meta["frames"]:
    T = np.array(fr["transform_matrix"], dtype=float)  # camera-to-world
    cam_centers.append(T[:3, 3])
cam_centers = np.array(cam_centers)

# ---------- Downsample points for faster plotting ----------
max_points = 30000
if len(pts) > max_points:
    idx = np.random.choice(len(pts), max_points, replace=False)
    pts_plot = pts[idx]
    colors_plot = colors[idx] if colors is not None else None
else:
    pts_plot = pts
    colors_plot = colors

# ---------- Helper to draw ----------
def save_view(elev, azim, name):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    if colors_plot is not None:
        ax.scatter(
            pts_plot[:, 0], pts_plot[:, 1], pts_plot[:, 2],
            c=colors_plot, s=1, alpha=0.7
        )
    else:
        ax.scatter(
            pts_plot[:, 0], pts_plot[:, 1], pts_plot[:, 2],
            s=1, alpha=0.5
        )

    ax.scatter(
        cam_centers[:, 0], cam_centers[:, 1], cam_centers[:, 2],
        c="red", s=18, label="camera centers"
    )

    ax.set_title(f"COLMAP sparse reconstruction ({name})")
    ax.legend(loc="upper right")
    ax.view_init(elev=elev, azim=azim)

    # Make axes roughly equal
    all_pts = np.vstack([pts_plot, cam_centers])
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    centers = (mins + maxs) / 2
    radius = (maxs - mins).max() / 2

    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)

    out_path = os.path.join(OUT_DIR, f"{name}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")

# ---------- Export several views ----------
save_view(elev=20, azim=30, name="view1")
save_view(elev=20, azim=120, name="view2")
save_view(elev=70, azim=45, name="top_view")