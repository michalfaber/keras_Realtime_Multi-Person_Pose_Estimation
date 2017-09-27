#
#  TODO: work in progress
#
import sys
sys.path.append("..")
from model import get_model
from ds_iterator import DataIterator
from keras import optimizers
from optimizers import MultiSGD

model = get_model()


batch_size = 10

base_lr = 4e-5 # 2e-5
momentum = 0.9
weight_decay = 5e-4
lr_policy =  "step"
gamma = 0.333
stepsize = 136106 #68053   // after each stepsize iterations update learning rate: lr=lr*gamma
max_iter = 600000

lr_mult_distro = [1.0, 1.0, 4.0, 1]


di = DataIterator("../dataset/val_dataset.h5", data_shape=(3, 368, 368), label_shape=(57, 46, 46),
                  split_point=38, batch_size=batch_size, shuffle=True)

x, y1, y2 = di.next()

print("x  : ", x.shape)
print("y1 : ", y1.shape)
print("y2 : ", y2.shape)

# as suggested in: https://github.com/fchollet/keras/issues/5920

last_layer_variables = list()
for layer in model.layers:
    #print(layer.weights)
    if layer.name in ['prediction']:
        last_layer_variables.extend(layer.weights)


multisgd = MultiSGD(lr=base_lr, momentum=momentum, decay=weight_decay, nesterov=False,
    exception_vars=last_layer_variables, multiplier=0.1)

model.compile(loss='mean_squared_error',
                  optimizer=multisgd,
                  metrics=['accuracy'])
