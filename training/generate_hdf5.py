from pycocotools.coco import COCO
from scipy.spatial.distance import cdist
import numpy as np
import cv2
import os
import os.path
import struct
import h5py

dataset_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'dataset'))

tr_anno_path = os.path.join(dataset_dir, "annotations/person_keypoints_train2017.json")
tr_img_dir = os.path.join(dataset_dir, "train2017")
tr_mask_dir = os.path.join(dataset_dir, "trainmask2017")

val_anno_path = os.path.join(dataset_dir, "annotations/person_keypoints_val2017.json")
val_img_dir = os.path.join(dataset_dir, "val2017")
val_mask_dir = os.path.join(dataset_dir, "valmask2017")

datasets = [
    (tr_anno_path, tr_img_dir, tr_mask_dir, "COCO"),
    (val_anno_path, val_img_dir, val_mask_dir, "COCO")
]

# datasets = [
#     (val_anno_path, val_img_dir, val_mask_dir, "COCO")
# ]

joint_all = []
tr_hdf5_path = os.path.join(dataset_dir, "train_dataset.h5")
val_hdf5_path = os.path.join(dataset_dir, "val_dataset.h5")

val_size = 2645 # size of validation set
#val_size = 300

def process():
    count = 0
    for _, ds in enumerate(datasets):

        anno_path = ds[0]
        images_dir = ds[1]
        masks_dir = ds[2]
        dataset_type = ds[3]

        coco = COCO(anno_path)
        ids = list(coco.imgs.keys())
        max_images = len(ids)
        #for i, img_id in enumerate(ids[744:745]):
        for i, img_id in enumerate(ids):
            ann_ids = coco.getAnnIds(imgIds=img_id)
            img_anns = coco.loadAnns(ann_ids)

            numPeople = len(img_anns)
            image = coco.imgs[img_id]
            h, w = image['height'], image['width']

            print("Image ID ", img_id)

            persons = []
            prev_center = []

            for p in range(numPeople):

                # skip this person if parts number is too low or if
                # segmentation area is too small
                if img_anns[p]["num_keypoints"] < 5 or img_anns[p]["area"] < 32 * 32:
                    continue

                anno = img_anns[p]["keypoints"]

                pers = dict()

                person_center = [img_anns[p]["bbox"][0] + img_anns[p]["bbox"][2] / 2,
                                 img_anns[p]["bbox"][1] + img_anns[p]["bbox"][3] / 2]

                # skip this person if the distance to exiting person is too small
                flag = 0
                for pc in prev_center:
                    a = np.expand_dims(pc[:2], axis=0)
                    b = np.expand_dims(person_center, axis=0)
                    dist = cdist(a, b)[0]
                    if dist < pc[2]*0.3:
                        flag = 1
                        continue

                if flag == 1:
                    continue

                pers["objpos"] = person_center
                pers["bbox"] = img_anns[p]["bbox"]
                pers["segment_area"] = img_anns[p]["area"]
                pers["num_keypoints"] = img_anns[p]["num_keypoints"]

                pers["joint"] = np.zeros((17, 3))
                for part in range(17):
                    pers["joint"][part, 0] = anno[part * 3]
                    pers["joint"][part, 1] = anno[part * 3 + 1]

                    if anno[part * 3 + 2] == 2:
                        pers["joint"][part, 2] = 1
                    elif anno[part * 3 + 2] == 1:
                        pers["joint"][part, 2] = 0
                    else:
                        pers["joint"][part, 2] = 2

                pers["scale_provided"] = img_anns[p]["bbox"][3] / 368

                persons.append(pers)
                prev_center.append(np.append(person_center, max(img_anns[p]["bbox"][2], img_anns[p]["bbox"][3])))


            if len(persons) > 0:

                joint_all.append(dict())

                joint_all[count]["dataset"] = dataset_type

                if count < val_size:
                    isValidation = 1
                else:
                    isValidation = 0

                joint_all[count]["isValidation"] = isValidation

                joint_all[count]["img_width"] = w
                joint_all[count]["img_height"] = h
                joint_all[count]["image_id"] = img_id
                joint_all[count]["annolist_index"] = i

                # set image path
                joint_all[count]["img_paths"] = os.path.join(images_dir, '%012d.jpg' % img_id)
                joint_all[count]["mask_miss_paths"] = os.path.join(masks_dir,
                                                                   'mask_miss_%012d.png' % img_id)
                joint_all[count]["mask_all_paths"] = os.path.join(masks_dir,
                                                                  'mask_all_%012d.png' % img_id)

                # set the main person
                joint_all[count]["objpos"] = persons[0]["objpos"]
                joint_all[count]["bbox"] = persons[0]["bbox"]
                joint_all[count]["segment_area"] = persons[0]["segment_area"]
                joint_all[count]["num_keypoints"] = persons[0]["num_keypoints"]
                joint_all[count]["joint_self"] = persons[0]["joint"]
                joint_all[count]["scale_provided"] = persons[0]["scale_provided"]

                # set other persons
                joint_all[count]["joint_others"] = []
                joint_all[count]["scale_provided_other"] = []
                joint_all[count]["objpos_other"] = []
                joint_all[count]["bbox_other"] = []
                joint_all[count]["segment_area_other"] = []
                joint_all[count]["num_keypoints_other"] = []

                for ot in range(1, len(persons)):
                    joint_all[count]["joint_others"].append(persons[ot]["joint"])
                    joint_all[count]["scale_provided_other"].append(persons[ot]["scale_provided"])
                    joint_all[count]["objpos_other"].append(persons[ot]["objpos"])
                    joint_all[count]["bbox_other"].append(persons[ot]["bbox"])
                    joint_all[count]["segment_area_other"].append(persons[ot]["segment_area"])
                    joint_all[count]["num_keypoints_other"].append(persons[ot]["num_keypoints"])

                joint_all[count]["people_index"] = 0
                lenOthers = len(persons) - 1

                joint_all[count]["numOtherPeople"] = lenOthers

                count += 1


def writeHDF5():

    tr_h5 = h5py.File(tr_hdf5_path, 'w')
    tr_grp = tr_h5.create_group("datum")
    tr_write_count = 0

    val_h5 = h5py.File(val_hdf5_path, 'w')
    val_grp = val_h5.create_group("datum")
    val_write_count = 0

    data = joint_all
    numSample = len(data)

    isValidationArray = [data[i]['isValidation'] for i in range(numSample)]
    val_total_write_count = isValidationArray.count(0.0)
    tr_total_write_count = len(data) - val_total_write_count

    print("Num samples " , numSample)

    random_order = [ i for i,el in enumerate(range(len(data)))] #np.random.permutation(numSample).tolist()

    for count in range(numSample):
        idx = random_order[count]

        img = cv2.imread(data[idx]['img_paths'])
        mask_all = cv2.imread(data[idx]['mask_all_paths'], 0)
        mask_miss = cv2.imread(data[idx]['mask_miss_paths'], 0)

        isValidation = data[idx]['isValidation']

        height = img.shape[0]
        width = img.shape[1]
        if (width < 64):
            img = cv2.copyMakeBorder(img, 0, 0, 0, 64 - width, cv2.BORDER_CONSTANT,
                                     value=(128, 128, 128))
            print('saving padded image!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            cv2.imwrite('padded_img.jpg', img)
            width = 64
        # no modify on width, because we want to keep information
        meta_data = np.zeros(shape=(height, width, 1), dtype=np.uint8)
        # print type(img), img.shape
        # print type(meta_data), meta_data.shape
        clidx = 0  # current line index
        # dataset name (string)
        for i in range(len(data[idx]['dataset'])):
            meta_data[clidx][i] = ord(data[idx]['dataset'][i])
        clidx = clidx + 1
        # image height, image width
        height_binary = float2bytes(data[idx]['img_height'])
        for i in range(len(height_binary)):
            meta_data[clidx][i] = height_binary[i]
        width_binary = float2bytes(data[idx]['img_width'])
        for i in range(len(width_binary)):
            meta_data[clidx][4 + i] = width_binary[i]
        clidx = clidx + 1
        # (a) isValidation(uint8), numOtherPeople (uint8), people_index (uint8), annolist_index (float), writeCount(float), totalWriteCount(float)
        meta_data[clidx][0] = data[idx]['isValidation']
        meta_data[clidx][1] = data[idx]['numOtherPeople']
        meta_data[clidx][2] = data[idx]['people_index']
        annolist_index_binary = float2bytes(data[idx]['annolist_index'])
        for i in range(len(annolist_index_binary)):  # 3,4,5,6
            meta_data[clidx][3 + i] = annolist_index_binary[i]
        if isValidation:
            count_binary = float2bytes(float(val_write_count))
        else:
            count_binary = float2bytes(float(tr_write_count))
        for i in range(len(count_binary)):
            meta_data[clidx][7 + i] = count_binary[i]
        if isValidation:
            totalWriteCount_binary = float2bytes(float(val_total_write_count))
        else:
            totalWriteCount_binary = float2bytes(float(tr_total_write_count))
        for i in range(len(totalWriteCount_binary)):
            meta_data[clidx][11 + i] = totalWriteCount_binary[i]
        nop = int(data[idx]['numOtherPeople'])
        clidx = clidx + 1
        # (b) objpos_x (float), objpos_y (float)
        objpos_binary = float2bytes(data[idx]['objpos'])
        for i in range(len(objpos_binary)):
            meta_data[clidx][i] = objpos_binary[i]
        clidx = clidx + 1
        # (c) scale_provided (float)
        scale_provided_binary = float2bytes(data[idx]['scale_provided'])
        for i in range(len(scale_provided_binary)):
            meta_data[clidx][i] = scale_provided_binary[i]
        clidx = clidx + 1
        # (d) joint_self (3*16) (float) (3 line)
        joints = np.asarray(data[idx]['joint_self']).T.tolist()  # transpose to 3*16
        for i in range(len(joints)):
            row_binary = float2bytes(joints[i])
            for j in range(len(row_binary)):
                meta_data[clidx][j] = row_binary[j]
            clidx = clidx + 1
        # (e) check nop, prepare arrays
        if (nop != 0):
            joint_other = data[idx]['joint_others']
            objpos_other = data[idx]['objpos_other']
            scale_provided_other = data[idx]['scale_provided_other']
            # (f) objpos_other_x (float), objpos_other_y (float) (nop lines)
            for i in range(nop):
                objpos_binary = float2bytes(objpos_other[i])
                for j in range(len(objpos_binary)):
                    meta_data[clidx][j] = objpos_binary[j]
                clidx = clidx + 1
            # (g) scale_provided_other (nop floats in 1 line)
            scale_provided_other_binary = float2bytes(scale_provided_other)
            for j in range(len(scale_provided_other_binary)):
                meta_data[clidx][j] = scale_provided_other_binary[j]
            clidx = clidx + 1
            # (h) joint_others (3*16) (float) (nop*3 lines)
            for n in range(nop):
                joints = np.asarray(joint_other[n]).T.tolist()  # transpose to 3*16
                for i in range(len(joints)):
                    row_binary = float2bytes(joints[i])
                    for j in range(len(row_binary)):
                        meta_data[clidx][j] = row_binary[j]
                    clidx = clidx + 1

        # print meta_data[0:12,0:48]
        # total 7+4*nop lines
        if "COCO" in data[idx]['dataset']:
            img4ch = np.concatenate((img, meta_data, mask_miss[..., None], mask_all[..., None]),
                                    axis=2)
        elif "MPI" in data[idx]['dataset']:
            img4ch = np.concatenate((img, meta_data, mask_miss[..., None]), axis=2)

        img4ch = np.transpose(img4ch, (2, 0, 1))

        if isValidation:
            key = '%07d' % val_write_count
            val_grp.create_dataset(key, data=img4ch, chunks=None)
            val_write_count += 1
        else:
            key = '%07d' % tr_write_count
            tr_grp.create_dataset(key, data=img4ch, chunks=None)
            tr_write_count += 1

        print('Writing sample %d/%d' % (count, numSample))

def float2bytes(floats):
    if type(floats) is float:
        floats = [floats]
    if type(floats) is int:
        floats = [float(floats)]

    if type(floats) is list and len(floats) > 0 and type(floats[0]) is list:
        floats = floats[0]

    return struct.pack('%sf' % len(floats), *floats)


if __name__ == '__main__':
    process()
    writeHDF5()
