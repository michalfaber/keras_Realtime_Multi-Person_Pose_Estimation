import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorpack.dataflow.common import MapData
from tensorpack.dataflow.parallel import PrefetchData

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from training.dataflow import CocoDataFlow, JointsLoader, COCODataPaths
from training.dataset import read_img, gen_mask, augment, apply_mask, \
    create_all_mask, ALL_HEATMAP_MASK, ALL_PAF_MASK
from training.label_maps import create_heatmap, create_paf


def _get_bgimg(inp, target_size=None):
    """
    Get a RGB image from cv2 BGR

    :param inp:
    :param target_size:
    :return: RGB image
    """
    inp = cv2.cvtColor(inp.astype(np.uint8), cv2.COLOR_BGR2RGB)
    if target_size:
        inp = cv2.resize(inp, target_size, interpolation=cv2.INTER_AREA)
    return inp


def display_image(img, heatmap, vectmap):
    """
    Displays an image and associated heatmaps and pafs (all)

    :param img:
    :param heatmap:
    :param vectmap:
    :return:
    """
    fig = plt.figure()
    a = fig.add_subplot(2, 2, 1)
    a.set_title('Image')
    plt.imshow(_get_bgimg(img))

    a = fig.add_subplot(2, 2, 2)
    a.set_title('Heatmap')
    plt.imshow(_get_bgimg(img, target_size=(heatmap.shape[1], heatmap.shape[0])), alpha=0.5)
    tmp = np.amax(heatmap, axis=2)
    plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
    plt.colorbar()

    tmp2 = vectmap.transpose((2, 0, 1))
    tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
    tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)

    a = fig.add_subplot(2, 2, 3)
    a.set_title('paf-x')
    plt.imshow(_get_bgimg(img, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
    plt.imshow(tmp2_odd, cmap=plt.cm.gray, alpha=0.5)
    plt.colorbar()

    a = fig.add_subplot(2, 2, 4)
    a.set_title('paf-y')
    plt.imshow(_get_bgimg(img, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
    plt.imshow(tmp2_even, cmap=plt.cm.gray, alpha=0.5)
    plt.colorbar()

    plt.show()


def display_masks(center, img, mask):
    """
    Displays mask for a given image and marks the center of a main person.

    :param center:
    :param img:
    :param mask:
    :return:
    """
    fig = plt.figure()

    a = fig.add_subplot(2, 2, 1)
    a.set_title('Image')
    i = _get_bgimg(img)
    cv2.circle(i, (int(center[0, 0]), int(center[0, 1])), 9, (0, 255, 0), -1)
    plt.imshow(i)

    if mask is not None:
        a = fig.add_subplot(2, 2, 2)
        a.set_title('Mask')
        plt.imshow(mask * 255, cmap=plt.cm.gray)

        a = fig.add_subplot(2, 2, 3)
        a.set_title('Image + Mask')
        plt.imshow(_get_bgimg(img), alpha=0.5)
        plt.imshow(mask * 255, cmap=plt.cm.gray, alpha=0.5)

    plt.show()


def show_image_mask_center_of_main_person(g):
    """
    Displays window with image,mask and center point of main person

    :param g: generated sample
    """
    img = g[0].img
    mask = g[0].mask
    center = g[0].aug_center
    display_masks(center, img, mask)


def show_image_heatmap_paf(g):
    """
    Displays window with image, heatmap and paf

    :param g: generated sample
    """
    img = g[0].img
    paf = g[3].astype(np.float32)
    heatmap = g[4].astype(np.float32)

    display_image(img, heatmap, paf)


def build_debug_sample(components):
    """
    Builds a sample for a model ONLY FOR DEBUGING. It returns the full meta
    instance instead of an image

    :param components: components
    :return: list of final components of a sample.
    """
    meta = components[0]

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

    return [meta, mask_paf, mask_heatmap, pafmap, heatmap]


if __name__ == '__main__':
    batch_size = 10
    curr_dir = os.path.dirname(__file__)
    annot_path = os.path.join(curr_dir, '../dataset/annotations/person_keypoints_val2017.json')
    img_dir = os.path.abspath(os.path.join(curr_dir, '../dataset/val2017/'))
    df = CocoDataFlow((368, 368),
                      COCODataPaths(annot_path, img_dir))#, select_ids=[1000])
    df.prepare()
    df = MapData(df, read_img)
    df = MapData(df, gen_mask)
    df = MapData(df, augment)
    df = MapData(df, apply_mask)
    df = MapData(df, build_debug_sample)
    df = PrefetchData(df, nr_prefetch=2, nr_proc=1)

    df.reset_state()
    gen = df.get_data()

    for g in gen:
        show_image_mask_center_of_main_person(g)
        #show_image_heatmap_paf(g)
