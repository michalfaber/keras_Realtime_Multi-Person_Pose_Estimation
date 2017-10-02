import threading
import h5py
import numpy as np

class DataIterator(object):

    def __init__(self, h5file, batch_size, data_shape, mask_shape, label_shape, vec_num, heat_num, shuffle=False, seed=None):
        h5 = h5py.File(h5file, 'r')
        self.data_group = h5["data"]
        self.label_group = h5["label"]
        self.mask_group = h5["mask"]
        self.keys = np.array([key for key in self.label_group.keys()])
        self.N = len(self.keys)

        self.data_shape = data_shape
        self.mask_shape = mask_shape
        self.label_shape = label_shape
        self.vec_num = vec_num
        self.heat_num = heat_num
        self.split_point = vec_num

        self.batch_size = batch_size
        self.batch_index = 0
        self.total_batches = self.N // batch_size
        self.lock = threading.Lock()
        self.shuffle = shuffle
        self.seed = seed

    def reset(self): self.batch_index = 0

    def next(self):
        with self.lock:
            if self.batch_index == 0:
                if self.seed is not None:
                    np.random.seed(self.seed + self.total_batches)
                self.index_array = (np.random.permutation(self.N) if self.shuffle
                    else np.arange(self.N))

            batches_x, batches_x1, batches_x2, batches_y1, batches_y2 = \
                [], [], [],[],[]

            start = self.batch_index * self.batch_size
            end = (self.batch_index + 1) * self.batch_size

            idx = self.index_array[start:end]
            batch_keys = self.keys[idx]

            # add to batch all samples from batch_keys

            for k in batch_keys:
                # image
                dta = self.data_group[k]
                dta_img = np.reshape(dta, self.data_shape)
                dta_img = np.transpose(dta_img, (1, 2, 0))

                batches_x.append(dta_img[np.newaxis, ...])

                # masks
                mask = self.mask_group[k]
                mask_img = np.reshape(mask, self.mask_shape)
                mask_img = np.transpose(mask_img, (1, 2, 0))
                mask1 = np.repeat(mask_img, self.vec_num, axis=2)
                mask2 = np.repeat(mask_img, self.heat_num, axis=2)

                batches_x1.append(mask1[np.newaxis, ...])
                batches_x2.append(mask2[np.newaxis, ...])

                # labels
                lbl = self.label_group[k]
                lbl_all = np.reshape(lbl, self.label_shape)
                lbl1 = lbl_all[:self.split_point, :, :]
                lbl1 = np.transpose(lbl1, (1, 2, 0))
                lbl2 = lbl_all[self.split_point:, :, :]
                lbl2 = np.transpose(lbl2, (1, 2, 0))

                batches_y1.append(lbl1[np.newaxis, ...])
                batches_y2.append(lbl2[np.newaxis, ...])

            self.batch_index += 1

            if self.batch_index == self.total_batches:
                self.batch_index = 0

            batch_x = np.concatenate(batches_x)
            batch_x1 = np.concatenate(batches_x1)
            batch_x2 = np.concatenate(batches_x2)
            batch_y1 = np.concatenate(batches_y1)
            batch_y2 = np.concatenate(batches_y2)

            return [batch_x, batch_x1,  batch_x2], \
                   [batch_y1, batch_y2,
                    batch_y1, batch_y2,
                    batch_y1, batch_y2,
                    batch_y1, batch_y2,
                    batch_y1, batch_y2,
                    batch_y1, batch_y2]

    def __iter__(self): return self

    def __next__(self, *args, **kwargs): return self.next(*args, **kwargs)