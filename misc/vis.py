import matplotlib.pyplot as plt
import numpy as np
from misc.skeleton import JOINT_COLOR_MAP

def get_bounds(points):
    all_points = points[..., :3].reshape(-1, 3)
    min_x, max_x = np.min(all_points[:, 0]), np.max(all_points[:, 0])
    min_y, max_y = np.min(all_points[:, 1]), np.max(all_points[:, 1])
    min_z, max_z = np.min(all_points[:, 2]), np.max(all_points[:, 2])
    return min_x, max_x, min_y, max_y, min_z, max_z

def set_3d_ax_limits(ax, bounds, padding=0.1):
    min_x, max_x, min_y, max_y, min_z, max_z = bounds
    range_x = max_x - min_x
    range_y = max_y - min_y
    range_z = max_z - min_z

    ax.set_box_aspect([range_x, range_y, range_z])
    ax.set_xlim(min_x - padding * range_x, max_x + padding * range_x)
    ax.set_ylim(min_y - padding * range_y, max_y + padding * range_y)
    ax.set_zlim(min_z - padding * range_z, max_z + padding * range_z)

def set_2d_ax_limits(ax, bounds, dims=[0, 1], padding=0.1):
    min_x, max_x, min_y, max_y, min_z, max_z = bounds
    ax.set_aspect('equal')
    if dims == [0, 1]:
        range_x = max_x - min_x
        range_y = max_y - min_y
        ax.set_xlim(min_x - padding * range_x, max_x + padding * range_x)
        ax.set_ylim(min_y - padding * range_y, max_y + padding * range_y)
    elif dims == [0, 2]:
        range_x = max_x - min_x
        range_z = max_z - min_z
        ax.set_xlim(min_x - padding * range_x, max_x + padding * range_x)
        ax.set_ylim(min_z - padding * range_z, max_z + padding * range_z)
    elif dims == [1, 2]:
        range_y = max_y - min_y
        range_z = max_z - min_z
        ax.set_xlim(min_y - padding * range_y, max_y + padding * range_y)
        ax.set_ylim(min_z - padding * range_z, max_z + padding * range_z)

def plot_3d_skeleton(ax, keypoints, edges=None, color_map=JOINT_COLOR_MAP, linewidth=2, s=20):
    if edges is not None:
        for i, j in edges:
            ax.plot([keypoints[i, 0], keypoints[j, 0]],
                    [keypoints[i, 1], keypoints[j, 1]],
                    [keypoints[i, 2], keypoints[j, 2]], color='0.5', linewidth=linewidth)
    
    for i, (x, y, z) in enumerate(keypoints):
        ax.scatter(x, y, z, color=color_map[i], marker='o', s=s)

def plot_2d_skeleton(ax, keypoints, dims=[0, 1], edges=None, color_map=JOINT_COLOR_MAP, linewidth=2, s=20):
    if edges is not None:
        for i, j in edges:
            ax.plot([keypoints[i, dims[0]], keypoints[j, dims[0]]],
                    [keypoints[i, dims[1]], keypoints[j, dims[1]]], color='0.5', linewidth=linewidth)
    
    for i, (x, y) in enumerate(keypoints[:, dims]):
        ax.scatter(x, y, color=color_map[i], marker='o', s=s)

def plot_3d_point_cloud(ax, points, color='k', s=1):
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, s=s)

def plot_2d_point_cloud(ax, points, dims=[0, 1], color='k', s=1):
    ax.scatter(points[:, dims[0]], points[:, dims[1]], c=color, s=s)

def visualize_sample(sample, edges=None, point_size=1, joint_size=20, linewidth=2, padding=0.1):
    x, y, y_hat = sample
    # x: NxD point cloud
    # y: Jx3 ground truth keypoints
    # y_hat: Jx3 predicted keypoints

    fig = plt.figure(figsize=(16, 8)) # 4 x 2 
    ax_3d_gt = fig.add_subplot(2, 4, 1, projection='3d')
    ax_2d_gt_xy = fig.add_subplot(2, 4, 2)
    ax_2d_gt_xz = fig.add_subplot(2, 4, 3)
    ax_2d_gt_yz = fig.add_subplot(2, 4, 4)
    ax_3d_pred = fig.add_subplot(2, 4, 5, projection='3d')
    ax_2d_pred_xy = fig.add_subplot(2, 4, 6)
    ax_2d_pred_xz = fig.add_subplot(2, 4, 7)
    ax_2d_pred_yz = fig.add_subplot(2, 4, 8)

    all_ps = np.concatenate([x[..., :3].reshape(-1, 3), y[..., :3].reshape(-1, 3), y_hat[..., :3].reshape(-1, 3)], axis=0)
    bounds = get_bounds(all_ps)

    set_3d_ax_limits(ax_3d_gt, bounds, padding)
    set_2d_ax_limits(ax_2d_gt_xy, bounds, dims=[0, 1], padding=padding)
    set_2d_ax_limits(ax_2d_gt_xz, bounds, dims=[0, 2], padding=padding)
    set_2d_ax_limits(ax_2d_gt_yz, bounds, dims=[1, 2], padding=padding)
    set_3d_ax_limits(ax_3d_pred, bounds, padding)
    set_2d_ax_limits(ax_2d_pred_xy, bounds, dims=[0, 1], padding=padding)
    set_2d_ax_limits(ax_2d_pred_xz, bounds, dims=[0, 2], padding=padding)
    set_2d_ax_limits(ax_2d_pred_yz, bounds, dims=[1, 2], padding=padding)

    plot_3d_point_cloud(ax_3d_gt, x[..., :3].reshape(-1, 3), color='k', s=point_size)
    plot_2d_point_cloud(ax_2d_gt_xy, x[..., :3].reshape(-1, 3), dims=[0, 1], color='k', s=point_size)
    plot_2d_point_cloud(ax_2d_gt_xz, x[..., :3].reshape(-1, 3), dims=[0, 2], color='k', s=point_size)
    plot_2d_point_cloud(ax_2d_gt_yz, x[..., :3].reshape(-1, 3), dims=[1, 2], color='k', s=point_size)
    
    plot_3d_skeleton(ax_3d_gt, y[..., :3], edges=edges, linewidth=linewidth, s=joint_size)
    plot_2d_skeleton(ax_2d_gt_xy, y[..., :3], dims=[0, 1], edges=edges, linewidth=linewidth, s=joint_size)
    plot_2d_skeleton(ax_2d_gt_xz, y[..., :3], dims=[0, 2], edges=edges, linewidth=linewidth, s=joint_size)
    plot_2d_skeleton(ax_2d_gt_yz, y[..., :3], dims=[1, 2], edges=edges, linewidth=linewidth, s=joint_size)

    plot_3d_point_cloud(ax_3d_pred, x[..., :3].reshape(-1, 3), color='k', s=point_size)
    plot_2d_point_cloud(ax_2d_pred_xy, x[..., :3].reshape(-1, 3), dims=[0, 1], color='k', s=point_size)
    plot_2d_point_cloud(ax_2d_pred_xz, x[..., :3].reshape(-1, 3), dims=[0, 2], color='k', s=point_size)
    plot_2d_point_cloud(ax_2d_pred_yz, x[..., :3].reshape(-1, 3), dims=[1, 2], color='k', s=point_size)

    plot_3d_skeleton(ax_3d_pred, y_hat[..., :3], edges=edges, linewidth=linewidth, s=joint_size)
    plot_2d_skeleton(ax_2d_pred_xy, y_hat[..., :3], dims=[0, 1], edges=edges, linewidth=linewidth, s=joint_size)
    plot_2d_skeleton(ax_2d_pred_xz, y_hat[..., :3], dims=[0, 2], edges=edges, linewidth=linewidth, s=joint_size)
    plot_2d_skeleton(ax_2d_pred_yz, y_hat[..., :3], dims=[1, 2], edges=edges, linewidth=linewidth, s=joint_size)

    ax_3d_gt.set_title('GT 3D')
    ax_2d_gt_xy.set_title('GT XY')
    ax_2d_gt_xz.set_title('GT XZ')
    ax_2d_gt_yz.set_title('GT YZ')
    ax_3d_pred.set_title('Pred 3D')
    ax_2d_pred_xy.set_title('Pred XY')
    ax_2d_pred_xz.set_title('Pred XZ')
    ax_2d_pred_yz.set_title('Pred YZ')

    return fig