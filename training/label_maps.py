import math
import numpy as np

from training.dataflow import JointsLoader


def create_heatmap(num_maps, height, width, all_joints, sigma, stride):
    """
    Creates stacked heatmaps for all joints + background. For 18 joints
    we would get an array height x width x 19.
    Size width and height should be the same as output from the network
    so this heatmap can be used to evaluate a loss function.

    :param num_maps: number of maps. for coco dataset we have 18 joints + 1 background
    :param height: height dimension of the network output
    :param width: width dimension of the network output
    :param all_joints: list of all joints (for coco: 18 items)
    :param sigma: parameter used to calculate a gaussian
    :param stride: parameter used to scale down the coordinates of joints. Those coords
            relate to the original image size
    :return: heat maps (height x width x num_maps)
    """
    heatmap = np.zeros((height, width, num_maps), dtype=np.float64)

    for joints in all_joints:
        for plane_idx, joint in enumerate(joints):
            if joint:
                _put_heatmap_on_plane(heatmap, plane_idx, joint, sigma, height, width, stride)

    # background
    heatmap[:, :, -1] = np.clip(1.0 - np.amax(heatmap, axis=2), 0.0, 1.0)

    return heatmap


def create_paf(num_maps, height, width, all_joints, threshold, stride):
    """
    Creates stacked paf maps for all connections. One connection requires
    2 maps because of paf vectors along dx and dy axis. For coco we have
    19 connections -> x2 it gives 38 maps

    :param num_maps: number of maps. for coco dataset we have 19 connections
    :param height: height dimension of the network output
    :param width: width dimension of the network output
    :param all_joints: list of all joints (for coco: 18 items)
    :param threshold: parameter determines the "thickness" of a limb within a paf
    :param stride: parameter used to scale down the coordinates of joints. Those coords
            relate to the original image size
    :return: paf maps (height x width x 2*num_maps)
    """
    vectormap = np.zeros((height, width, num_maps*2), dtype=np.float64)
    countmap = np.zeros((height, width, num_maps), dtype=np.uint8)
    for joints in all_joints:
        for plane_idx, (j_idx1, j_idx2) in enumerate(JointsLoader.joint_pairs):
            center_from = joints[j_idx1]
            center_to = joints[j_idx2]

            # skip if no valid pair of keypoints
            if center_from is None or center_to is None:
                continue

            x1, y1 = (center_from[0] / stride, center_from[1] / stride)
            x2, y2 = (center_to[0] / stride, center_to[1] / stride)

            _put_paf_on_plane(vectormap, countmap, plane_idx, x1, y1, x2, y2,
                              threshold, height, width)

    return vectormap


def _put_heatmap_on_plane(heatmap, plane_idx, joint, sigma, height, width, stride):
    start = stride / 2.0 - 0.5

    center_x, center_y = joint

    for g_y in range(height):
        for g_x in range(width):
            x = start + g_x * stride
            y = start + g_y * stride
            d2 = (x-center_x) * (x-center_x) + (y-center_y) * (y-center_y)
            exponent = d2 / 2.0 / sigma / sigma
            if exponent > 4.6052:
                continue

            heatmap[g_y, g_x, plane_idx] += math.exp(-exponent)
            if heatmap[g_y, g_x, plane_idx] > 1.0:
                heatmap[g_y, g_x, plane_idx] = 1.0


def _put_paf_on_plane(vectormap, countmap, plane_idx, x1, y1, x2, y2,
                     threshold, height, width):

    min_x = max(0, int(round(min(x1, x2) - threshold)))
    max_x = min(width, int(round(max(x1, x2) + threshold)))

    min_y = max(0, int(round(min(y1, y2) - threshold)))
    max_y = min(height, int(round(max(y1, y2) + threshold)))

    vec_x = x2 - x1
    vec_y = y2 - y1

    norm = math.sqrt(vec_x ** 2 + vec_y ** 2)
    if norm < 1e-8:
        return

    vec_x /= norm
    vec_y /= norm

    for y in range(min_y, max_y):
        for x in range(min_x, max_x):
            bec_x = x - x1
            bec_y = y - y1
            dist = abs(bec_x * vec_y - bec_y * vec_x)

            if dist > threshold:
                continue

            cnt = countmap[y][x][plane_idx]

            if cnt == 0:
                vectormap[y][x][plane_idx * 2 + 0] = vec_x
                vectormap[y][x][plane_idx * 2 + 1] = vec_y
            else:
                vectormap[y][x][plane_idx*2+0] = (vectormap[y][x][plane_idx*2+0] * cnt + vec_x) / (cnt + 1)
                vectormap[y][x][plane_idx*2+1] = (vectormap[y][x][plane_idx*2+1] * cnt + vec_y) / (cnt + 1)

            countmap[y][x][plane_idx] += 1
