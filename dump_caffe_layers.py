#
# Run this file from docker:
#
# docker run -v [absolute path to your keras_Realtime_Multi-Person_Pose_Estimation folder]:/workspace -it bvlc/caffe:cpu python dump_caffe_layers.py
#

from __future__ import division, print_function
import caffe
import numpy as np
import os

layers_output = 'model/caffe/layers'
caffe_model = 'model/caffe/_trained_COCO/pose_iter_440000.caffemodel'
caffe_proto = 'model/caffe/_trained_COCO/pose_deploy.prototxt'

caffe.set_mode_cpu()
net = caffe.Net(caffe_proto, caffe_model, caffe.TEST)

# layer names and output shapes
for layer_name, blob in net.blobs.iteritems():
    print(layer_name, blob.data.shape)

# write out weight matrices and bias vectors
for k, v in net.params.items():
    print(k, v[0].data.shape, v[1].data.shape)
    np.save(os.path.join(layers_output, "W_{:s}.npy".format(k)), v[0].data)
    np.save(os.path.join(layers_output, "b_{:s}.npy".format(k)), v[1].data)

print("Done !")
