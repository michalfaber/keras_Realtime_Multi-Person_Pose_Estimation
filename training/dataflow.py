import os
import numpy as np

from scipy.spatial.distance import cdist
from pycocotools.coco import COCO
from tensorpack.dataflow.base import RNGDataFlow


class JointsLoader:
    """
    Loader for joints from coco keypoints
    """
    @staticmethod
    def _get_neck(coco_parts, idx1, idx2):

        p1 = coco_parts[idx1]
        p2 = coco_parts[idx2]
        if p1 and p2:
            return (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        else:
            return None

    num_joints = 18

    num_joints_and_bkg = num_joints + 1

    num_connections = 19

    idx_in_coco = [0, lambda x: JointsLoader._get_neck(x, 5, 6), 6, 8,
                   10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]

    idx_in_coco_str = [
        'Nose','Neck','RShoulder','RElbow','RWrist','LShoulder','LElbow','LWrist',
        'RHip','RKnee','RAnkle','LHip','LKnee','LAnkle','REye','LEye','REar','LEar']

    joint_pairs = list(zip(
        [1, 8, 9, 1, 11, 12, 1, 2, 3, 2, 1, 5, 6, 5, 1, 0, 0, 14, 15],
        [8, 9, 10, 11, 12, 13, 2, 3, 4, 16, 5, 6, 7, 17, 0, 14, 15, 16, 17]))

    @staticmethod
    def from_coco_keypoints(all_keypoints, w ,h):
        """
        Creates list of joints based on the list of coco keypoints vectors.

        :param all_keypoints: list of coco keypoints vector [[x1,y1,v1,x2,y2,v2,....], []]
        :param w: image width
        :param h: image height
        :return: list of joints [[(x1,y1), (x1,y1), ...], [], []]
        """
        all_joints = []
        for keypoints in all_keypoints:
            kp = np.array(keypoints)
            xs = kp[0::3]
            ys = kp[1::3]
            vs = kp[2::3]

            # filter and loads keypoints to the list

            keypoints_list = []
            for idx, (x, y, v) in enumerate(zip(xs, ys, vs)):
                # only visible and occluded keypoints are used
                if v >= 1 and x >=0 and y >= 0 and x < w and y < h:
                    keypoints_list.append((x, y))
                else:
                    keypoints_list.append(None)

            # build the list of joints. It contains the same coordinates
            # of body parts like in the orginal coco keypoints plus
            # additional body parts interpolated from coco
            # keypoints (ex. a neck)

            joints = []
            for part_idx in range(len(JointsLoader.idx_in_coco)):
                coco_kp_idx = JointsLoader.idx_in_coco[part_idx]

                if callable(coco_kp_idx):
                    p = coco_kp_idx(keypoints_list)
                else:
                    p = keypoints_list[coco_kp_idx]

                joints.append(p)
            all_joints.append(joints)

        return all_joints


class Meta(object):
    """
    Metadata representing a single data point for training.
    """
    __slots__ = (
        'img_path',
        'height',
        'width',
        'center',
        'bbox',
        'area',
        'num_keypoints',
        'masks_segments',
        'scale',
        'all_joints',
        'img',
        'mask',
        'aug_center',
        'aug_joints')

    def __init__(self, img_path, height, width, center, bbox,
                 area, scale, num_keypoints):

        self.img_path = img_path
        self.height = height
        self.width = width
        self.center = center
        self.bbox = bbox
        self.area = area
        self.scale = scale
        self.num_keypoints = num_keypoints

        # updated after iterating over all persons
        self.masks_segments = None
        self.all_joints = None

        # updated during augmentation
        self.img = None
        self.mask = None
        self.aug_center = None
        self.aug_joints = None


class COCODataPaths:
    """
    Holder for coco dataset paths
    """
    def __init__(self, annot_path, img_dir):
        self.annot = COCO(annot_path)
        self.img_dir = img_dir


class CocoDataFlow(RNGDataFlow):
    """
    Tensorpack dataflow serving coco data points.
    """
    def __init__(self, target_size, coco_data, select_ids=None):
        """
        Initializes dataflow.

        :param target_size:
        :param coco_data: paths to the coco files: annotation file and folder with images
        :param select_ids: (optional) identifiers of images to serve (for debugging)
        """
        self.coco_data = coco_data if isinstance(coco_data, list) else [coco_data]
        self.all_meta = []
        self.select_ids = select_ids
        self.target_size = target_size

    def prepare(self):
        """
        Loads coco metadata. Partially populates meta objects (image path,
        scale of main person, bounding box, area, joints) Remaining fields
        are populated in next steps - MapData tensorpack tranformer.
        """
        for coco in self.coco_data:

            print("Loading dataset {} ...".format(coco.img_dir))

            if self.select_ids:
                ids = self.select_ids
            else:
                ids = list(coco.annot.imgs.keys())

            for i, img_id in enumerate(ids):
                img_meta = coco.annot.imgs[img_id]

                # load annotations

                img_id = img_meta['id']
                img_file = img_meta['file_name']
                h, w = img_meta['height'], img_meta['width']
                img_path = os.path.join(coco.img_dir, img_file)
                ann_ids = coco.annot.getAnnIds(imgIds=img_id)
                anns = coco.annot.loadAnns(ann_ids)

                total_keypoints = sum([ann.get('num_keypoints', 0) for ann in anns])
                if total_keypoints == 0:
                    continue

                persons = []
                prev_center = []
                masks = []
                keypoints = []

                # sort from the biggest person to the smallest one

                persons_ids = np.argsort([-a['area'] for a in anns], kind='mergesort')

                for id in list(persons_ids):
                    person_meta = anns[id]

                    if person_meta["iscrowd"]:
                        masks.append(coco.annot.annToRLE(person_meta))
                        continue

                    # skip this person if parts number is too low or if
                    # segmentation area is too small

                    if person_meta["num_keypoints"] < 5 or person_meta["area"] < 32 * 32:
                        masks.append(coco.annot.annToRLE(person_meta))
                        continue

                    person_center = [person_meta["bbox"][0] + person_meta["bbox"][2] / 2,
                                     person_meta["bbox"][1] + person_meta["bbox"][3] / 2]

                    # skip this person if the distance to existing person is too small

                    too_close = False
                    for pc in prev_center:
                        a = np.expand_dims(pc[:2], axis=0)
                        b = np.expand_dims(person_center, axis=0)
                        dist = cdist(a, b)[0]
                        if dist < pc[2]*0.3:
                            too_close = True
                            break

                    if too_close:
                        # add mask of this person. we don't want to show the network
                        # unlabeled people
                        masks.append(coco.annot.annToRLE(person_meta))
                        continue

                    pers = Meta(
                        img_path=img_path,
                        height=h,
                        width=w,
                        center=np.expand_dims(person_center, axis=0),
                        bbox=person_meta["bbox"],
                        area=person_meta["area"],
                        scale=person_meta["bbox"][3] / self.target_size[0],
                        num_keypoints=person_meta["num_keypoints"])

                    keypoints.append(person_meta["keypoints"])
                    persons.append(pers)
                    prev_center.append(np.append(person_center, max(person_meta["bbox"][2],
                                                                    person_meta["bbox"][3])))

                if len(persons) > 0:
                    main_person = persons[0]
                    main_person.masks_segments = masks
                    main_person.all_joints = JointsLoader.from_coco_keypoints(keypoints, w, h)
                    self.all_meta.append(main_person)

                if i % 1000 == 0:
                    print("Loading image annot {}/{}".format(i, len(ids)))

    def save(self, path):
        raise NotImplemented

    def load(self, path):
        raise NotImplemented

    def size(self):
        """
        :return: number of items
        """
        return len(self.all_meta)

    def get_data(self):
        """
        Generator of data points

        :return: instance of Meta
        """
        idxs = np.arange(self.size())
        self.rng.shuffle(idxs)
        for idx in idxs:
            yield [self.all_meta[idx]]
