import numpy as np
import zmq
from ast import literal_eval as make_tuple

import six
if six.PY3:
  buffer_ = memoryview
else:
  buffer_ = buffer  # noqa


class DataGeneratorClient(object):

    def __init__(self, host, port, hwm=20, batch_size=10):
        """
        :param host:
        :param port:
        :param hwm:, optional
          The `ZeroMQ high-water mark (HWM)
          <http://zguide.zeromq.org/page:all#High-Water-Marks>`_ on the
          sending socket. Increasing this increases the buffer, which can be
          useful if your data preprocessing times are very random.  However,
          it will increase memory usage. There is no easy way to tell how
          many batches will actually be queued with a particular HWM.
          Defaults to 10. Be sure to set the corresponding HWM on the
          receiving end as well.
        :param batch_size:
        :param shuffle:
        :param seed:
        """
        self.host = host
        self.port = port
        self.hwm = hwm
        self.socket = None

        self.split_point = 38
        self.vec_num = 38
        self.heat_num = 19

        self.batch_size = batch_size

    def _recv_arrays(self):
        """Receive a list of NumPy arrays.
        Parameters
        ----------
        socket : :class:`zmq.Socket`
        The socket to receive the arrays on.
        Returns
        -------
        list
        A list of :class:`numpy.ndarray` objects.
        Raises
        ------
        StopIteration
        If the first JSON object received contains the key `stop`,
        signifying that the server has finished a single epoch.
        """
        headers = self.socket.recv_json()
        if 'stop' in headers:
            raise StopIteration
        arrays = []

        for header in headers:
            data = self.socket.recv()
            buf = buffer_(data)
            array = np.frombuffer(buf, dtype=np.dtype(header['descr']))
            array.shape = make_tuple(header['shape'])

            if header['fortran_order']:
                array.shape = header['shape'][::-1]
                array = array.transpose()
            arrays.append(array)

        return arrays

    def gen(self):
        batches_x, batches_x1, batches_x2, batches_y1, batches_y2 = \
            [None]*self.batch_size, [None]*self.batch_size, [None]*self.batch_size, \
            [None]*self.batch_size, [None]*self.batch_size

        sample_idx = 0

        while True:
            data_img, mask_img, label = tuple(self._recv_arrays())

            # image
            dta_img = np.transpose(data_img, (1, 2, 0))
            batches_x[sample_idx]=dta_img[np.newaxis, ...]

            # mask - the same for vec_weights, heat_weights
            vec_weights = np.repeat(mask_img[:,:,np.newaxis], self.vec_num, axis=2)
            heat_weights = np.repeat(mask_img[:,:,np.newaxis], self.heat_num, axis=2)

            batches_x1[sample_idx]=vec_weights[np.newaxis, ...]
            batches_x2[sample_idx]=heat_weights[np.newaxis, ...]

            # label
            vec_label = label[:self.split_point, :, :]
            vec_label = np.transpose(vec_label, (1, 2, 0))
            heat_label = label[self.split_point:, :, :]
            heat_label = np.transpose(heat_label, (1, 2, 0))

            batches_y1[sample_idx]=vec_label[np.newaxis, ...]
            batches_y2[sample_idx]=heat_label[np.newaxis, ...]

            sample_idx += 1

            if sample_idx == self.batch_size:
                sample_idx = 0

                batch_x = np.concatenate(batches_x)
                batch_x1 = np.concatenate(batches_x1)
                batch_x2 = np.concatenate(batches_x2)
                batch_y1 = np.concatenate(batches_y1)
                batch_y2 = np.concatenate(batches_y2)

                yield [batch_x, batch_x1,  batch_x2], \
                       [batch_y1, batch_y2,
                        batch_y1, batch_y2,
                        batch_y1, batch_y2,
                        batch_y1, batch_y2,
                        batch_y1, batch_y2,
                        batch_y1, batch_y2]

    def start(self):
        context = zmq.Context()
        self.socket = context.socket(zmq.PULL)
        self.socket.set_hwm(self.hwm)
        self.socket.connect("tcp://{}:{}".format(self.host, self.port))

    def stop(self):
        if self.socket:
            self.socket.__del__()

    def restart(self):
        self.stop()
        self.start()