import os
import numpy as np

from model.cmu_model import get_testing_model

CAFFE_LAYERS_DIR = "model/caffe/layers"
KERAS_MODEL_FILE = "model/keras/model.h5"

m = get_testing_model()

for layer in m.layers:
    layer_name = layer.name
    if (os.path.exists(os.path.join(CAFFE_LAYERS_DIR, "W_%s.npy" % layer_name))):
        w = np.load(os.path.join(CAFFE_LAYERS_DIR, "W_%s.npy" % layer_name))
        b = np.load(os.path.join(CAFFE_LAYERS_DIR, "b_%s.npy" % layer_name))

        w = np.transpose(w, (2, 3, 1, 0))

        layer_weights = [w, b]
        layer.set_weights(layer_weights)

m.save_weights(KERAS_MODEL_FILE)

print("Done !")