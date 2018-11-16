import os
import cv2
import numpy as np

from pycocotools.coco import maskUtils

from tensorpack.dataflow.common import BatchData, MapData
from tensorpack.dataflow.common import TestDataSpeed
from tensorpack.dataflow.parallel import PrefetchDataZMQ

from training.augmentors import ScaleAug, RotateAug, CropAug, FlipAug, \
    joints_to_point8, point8_to_joints, AugImgMetadata
from training.dataflow import CocoDataFlow, JointsLoader, COCODataPaths
from training.label_maps import create_heatmap, create_paf


ALL_PAF_MASK = np.repeat(
    np.ones((46, 46, 1), dtype=np.uint8), 38, axis=2)

ALL_HEATMAP_MASK = np.repeat(
    np.ones((46, 46, 1), dtype=np.uint8), 19, axis=2)

AUGMENTORS_LIST = [
        ScaleAug(scale_min=0.5,
                 scale_max=1.1,
                 target_dist=0.6,
                 interp=cv2.INTER_CUBIC),

        RotateAug(rotate_max_deg=40,
                  interp=cv2.INTER_CUBIC,
                  border=cv2.BORDER_CONSTANT,
                  border_value=(128, 128, 128), mask_border_val=1),

        CropAug(368, 368, center_perterb_max=40, border_value=(128, 128, 128),
                 mask_border_val=1),

        FlipAug(num_parts=18, prob=0.5),
    ]


def read_img(components):
    """
    Loads image from meta.img_path. Assigns the image to
    the field img of the same meta instance.

    :param components: components
    :return: updated components
    """
    meta = components[0]
    img_buf = open(meta.img_path, 'rb').read()

    if not img_buf:
        raise Exception('image not read, path=%s' % meta.img_path)

    arr = np.fromstring(img_buf, np.uint8)
    meta.img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    meta.height, meta.width = meta.img.shape[:2]

    return components


def gen_mask(components):
    """
    Generate masks based on the coco mask polygons.

    :param components: components
    :return: updated components
    """
    meta = components[0]
    if meta.masks_segments:
        mask_miss = np.ones((meta.height, meta.width), dtype=np.uint8)
        for seg in meta.masks_segments:
            bin_mask = maskUtils.decode(seg)
            bin_mask = np.logical_not(bin_mask)
            mask_miss = np.bitwise_and(mask_miss, bin_mask)

        meta.mask = mask_miss

    return components


def augment(components):
    """
    Augmenting of images.

    :param components: components
    :return: updated components.
    """
    meta = components[0]

    aug_center = meta.center.copy()
    aug_joints = joints_to_point8(meta.all_joints)

    for aug in AUGMENTORS_LIST:
        (im, mask), params = aug.augment_return_params(
            AugImgMetadata(img=meta.img,
                            mask=meta.mask,
                            center=aug_center,
                            scale=meta.scale))

        # augment joints
        aug_joints = aug.augment_coords(aug_joints, params)

        # after flipping horizontaly the left side joints and right side joints are also
        # flipped so we need to recover their orginal orientation.
        if isinstance(aug, FlipAug):
            aug_joints = aug.recover_left_right(aug_joints, params)

        # augment center position
        aug_center = aug.augment_coords(aug_center, params)

        meta.img = im
        meta.mask = mask

    meta.aug_joints = point8_to_joints(aug_joints)
    meta.aug_center = aug_center

    return components


def apply_mask(components):
    """
    Applies the mask (if exists) to the image.

    :param components: components
    :return: updated components
    """
    meta = components[0]
    if meta.mask is not None:
        meta.img[:, :, 0] = meta.img[:, :, 0] * meta.mask
        meta.img[:, :, 1] = meta.img[:, :, 1] * meta.mask
        meta.img[:, :, 2] = meta.img[:, :, 2] * meta.mask
    return components


def create_all_mask(mask, num, stride):
    """
    Helper function to create a stack of scaled down mask.

    :param mask: mask image
    :param num: number of layers
    :param stride: parameter used to scale down the mask image because it has
    the same size as orginal image. We need the size of network output.
    :return:
    """
    scale_factor = 1.0 / stride
    small_mask = cv2.resize(mask, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    small_mask = small_mask[:, :, np.newaxis]
    return np.repeat(small_mask, num, axis=2)


def build_sample(components):
    """
    Builds a sample for a model.

    :param components: components
    :return: list of final components of a sample.
    """
    meta = components[0]
    image = meta.img

    if meta.mask is None:
        mask_paf = ALL_PAF_MASK
        mask_heatmap = ALL_HEATMAP_MASK
    else:
        mask_paf = create_all_mask(meta.mask, 38, stride=8)
        mask_heatmap = create_all_mask(meta.mask, 19, stride=8)

    heatmap = create_heatmap(JointsLoader.num_joints_and_bkg, 46, 46,
                             meta.aug_joints, 7.0, stride=8)

    pafmap = create_paf(JointsLoader.num_connections, 46, 46,
                        meta.aug_joints, 1, stride=8)

    # release reference to the image/mask/augmented data. Otherwise it would easily consume all memory at some point
    meta.mask = None
    meta.img = None
    meta.aug_joints = None
    meta.aug_center = None
    return [image.astype(np.uint8), mask_paf, mask_heatmap, pafmap, heatmap]


def get_dataflow(coco_data_paths):
    """
    This function initializes the tensorpack dataflow and serves generator
    for training operation.

    :param coco_data_paths: paths to the coco files: annotation file and folder with images
    :return: dataflow object
    """
    df = CocoDataFlow((368, 368), coco_data_paths)
    df.prepare()
    df = MapData(df, read_img)
    df = MapData(df, gen_mask)
    df = MapData(df, augment)
    df = MapData(df, apply_mask)
    df = MapData(df, build_sample)
    df = PrefetchDataZMQ(df, nr_proc=4) #df = PrefetchData(df, 2, 1)

    return df


def batch_dataflow(df, batch_size):
    """
    The function builds batch dataflow from the input dataflow of samples

    :param df: dataflow of samples
    :param batch_size: batch size
    :return: dataflow of batches
    """
    df = BatchData(df, batch_size, use_list=False)
    df = MapData(df, lambda x: (
        [x[0], x[1], x[2]],
        [x[3], x[4], x[3], x[4], x[3], x[4], x[3], x[4], x[3], x[4], x[3], x[4]])
                 )
    df.reset_state()
    return df


if __name__ == '__main__':
    """
    Run this script to check speed of generating samples. Tweak the nr_proc
    parameter of PrefetchDataZMQ. Ideally it should reflect the number of cores 
    in your hardware
    """
    batch_size = 10
    curr_dir = os.path.dirname(__file__)
    annot_path = os.path.join(curr_dir, '../dataset/annotations/person_keypoints_val2017.json')
    img_dir = os.path.abspath(os.path.join(curr_dir, '../dataset/val2017/'))
    df = CocoDataFlow((368, 368), COCODataPaths(annot_path, img_dir))#, select_ids=[1000])
    df.prepare()
    df = MapData(df, read_img)
    df = MapData(df, gen_mask)
    df = MapData(df, augment)
    df = MapData(df, apply_mask)
    df = MapData(df, build_sample)
    df = PrefetchDataZMQ(df, nr_proc=4)
    df = BatchData(df, batch_size, use_list=False)
    df = MapData(df, lambda x: (
        [x[0], x[1], x[2]],
        [x[3], x[4], x[3], x[4], x[3], x[4], x[3], x[4], x[3], x[4], x[3], x[4]])
    )

    TestDataSpeed(df, size=100).start()
