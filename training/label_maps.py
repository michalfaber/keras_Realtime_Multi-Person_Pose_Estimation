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
    heatmap = np.zeros((height, width, num_maps), dtype=np.float32)

    for joints in all_joints:
        for plane_idx, joint in enumerate(joints):
            if joint:
                _put_heatmap_on_plane(heatmap, plane_idx, joint, sigma, height, width, stride)

    # background
    heatmap[:, :, -1] = np.clip(1.0 - np.amax(heatmap, axis=2), 0.0, 1.0)

    return heatmap.astype(np.float16)


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
    vectormap = np.zeros((height, width, num_maps*2), dtype=np.float32)
    countmap = np.zeros((height, width, num_maps), dtype=np.int16)
    for joints in all_joints:
        for plane_idx, (j_idx1, j_idx2) in enumerate(JointsLoader.joint_pairs):
            center_from = joints[j_idx1]
            center_to = joints[j_idx2]

            if not center_from or not center_to:
                continue

            _put_paf_on_plane(vectormap, countmap, plane_idx, center_from,
                              center_to, threshold, height, width, stride)

    nonzeros = np.nonzero(countmap)
    for y, x, p in zip(nonzeros[0], nonzeros[1], nonzeros[2]):
        if countmap[y][x][p] <= 0:
            continue
        vectormap[y][x][p*2+0] /= countmap[y][x][p]
        vectormap[y][x][p*2+1] /= countmap[y][x][p]

    return vectormap.astype(np.float16)


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


def _put_paf_on_plane(vectormap, countmap, plane_idx, center_from, center_to,
                     threshold, height, width, stride):
    center_from = (center_from[0] // stride, center_from[1] // stride)
    center_to = (center_to[0] // stride, center_to[1] // stride)

    vec_x = center_to[0] - center_from[0]
    vec_y = center_to[1] - center_from[1]

    min_x = max(0, int(min(center_from[0], center_to[0]) - threshold))
    min_y = max(0, int(min(center_from[1], center_to[1]) - threshold))

    max_x = min(width, int(max(center_from[0], center_to[0]) + threshold))
    max_y = min(height, int(max(center_from[1], center_to[1]) + threshold))

    norm = math.sqrt(vec_x ** 2 + vec_y ** 2)
    if norm < 1e-8:
        return

    vec_x /= norm
    vec_y /= norm

    for y in range(min_y, max_y):
        for x in range(min_x, max_x):
            bec_x = x - center_from[0]
            bec_y = y - center_from[1]
            dist = abs(bec_x * vec_y - bec_y * vec_x)

            if dist > threshold:
                continue

            countmap[y][x][plane_idx] += 1

            vectormap[y][x][plane_idx*2+0] = vec_x
            vectormap[y][x][plane_idx*2+1] = vec_y
