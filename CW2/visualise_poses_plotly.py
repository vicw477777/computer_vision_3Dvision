import os
import json
import argparse
import numpy as np
import trimesh
import plotly.graph_objects as go


def load_scene(scene_path: str):
    tf_path = os.path.join(scene_path, "transforms.json")
    ply_path = os.path.join(scene_path, "sparse_pc.ply")

    if not os.path.exists(tf_path):
        raise FileNotFoundError(f"Missing transforms.json: {tf_path}")
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"Missing sparse_pc.ply: {ply_path}")

    with open(tf_path, "r") as f:
        meta = json.load(f)

    pc = trimesh.load(ply_path)
    if not hasattr(pc, "vertices"):
        raise ValueError(f"Could not read vertices from {ply_path}")

    pts = np.asarray(pc.vertices)
    colors = None
    if hasattr(pc, "colors") and pc.colors is not None and len(pc.colors) == len(pts):
        colors = np.asarray(pc.colors)[:, :3]

    cam_centers = []
    cam_dirs = []

    for fr in meta["frames"]:
        T = np.array(fr["transform_matrix"], dtype=float)  # camera-to-world
        center = T[:3, 3]
        # camera forward direction in world coords: local -Z axis mapped by rotation
        forward = -T[:3, 2]
        cam_centers.append(center)
        cam_dirs.append(forward)

    cam_centers = np.array(cam_centers)
    cam_dirs = np.array(cam_dirs)

    return pts, colors, cam_centers, cam_dirs


def make_equal_axis_ranges(all_pts):
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    centers = (mins + maxs) / 2
    radius = (maxs - mins).max() / 2
    return [
        [centers[0] - radius, centers[0] + radius],
        [centers[1] - radius, centers[1] + radius],
        [centers[2] - radius, centers[2] + radius],
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-path", required=True, help="Path to scene folder containing transforms.json and sparse_pc.ply")
    parser.add_argument("--output-html", default=None, help="Output html path")
    parser.add_argument("--max-points", type=int, default=30000, help="Max number of sparse points to plot")
    parser.add_argument("--camera-scale", type=float, default=0.05, help="Length of camera direction arrows relative to scene size")
    args = parser.parse_args()

    scene_path = args.scene_path
    pts, colors, cam_centers, cam_dirs = load_scene(scene_path)

    if len(pts) > args.max_points:
        idx = np.random.choice(len(pts), args.max_points, replace=False)
        pts_plot = pts[idx]
        colors_plot = colors[idx] if colors is not None else None
    else:
        pts_plot = pts
        colors_plot = colors

    all_pts = np.vstack([pts_plot, cam_centers])
    xr, yr, zr = make_equal_axis_ranges(all_pts)
    scene_radius = max(xr[1] - xr[0], yr[1] - yr[0], zr[1] - zr[0]) / 2
    arrow_len = args.camera_scale * scene_radius

    if colors_plot is not None:
        point_trace = go.Scatter3d(
            x=pts_plot[:, 0],
            y=pts_plot[:, 1],
            z=pts_plot[:, 2],
            mode="markers",
            marker=dict(
                size=1.5,
                color=[f"rgb({r},{g},{b})" for r, g, b in colors_plot],
                opacity=0.7,
            ),
            name="Sparse point cloud",
        )
    else:
        point_trace = go.Scatter3d(
            x=pts_plot[:, 0],
            y=pts_plot[:, 1],
            z=pts_plot[:, 2],
            mode="markers",
            marker=dict(size=1.5, color="gray", opacity=0.7),
            name="Sparse point cloud",
        )

    cam_trace = go.Scatter3d(
        x=cam_centers[:, 0],
        y=cam_centers[:, 1],
        z=cam_centers[:, 2],
        mode="markers",
        marker=dict(size=4, color="red"),
        name="Camera centers",
    )

    arrow_x = []
    arrow_y = []
    arrow_z = []
    for c, d in zip(cam_centers, cam_dirs):
        end = c + arrow_len * d
        arrow_x += [c[0], end[0], None]
        arrow_y += [c[1], end[1], None]
        arrow_z += [c[2], end[2], None]

    dir_trace = go.Scatter3d(
        x=arrow_x,
        y=arrow_y,
        z=arrow_z,
        mode="lines",
        line=dict(color="orange", width=3),
        name="Camera directions",
    )

    fig = go.Figure(data=[point_trace, cam_trace, dir_trace])
    fig.update_layout(
        title=os.path.basename(os.path.abspath(scene_path)),
        scene=dict(
            xaxis=dict(title="X", range=xr),
            yaxis=dict(title="Y", range=yr),
            zaxis=dict(title="Z", range=zr),
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(x=0.01, y=0.99),
    )

    output_html = args.output_html
    if output_html is None:
        output_html = os.path.join(scene_path, "pose_viewer.html")

    fig.write_html(output_html, include_plotlyjs="cdn")
    print(f"Saved: {output_html}")


if __name__ == "__main__":
    main()