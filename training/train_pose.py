#
#  TODO: work in progress
#
import sys
sys.path.append("..")
from model import get_model
from ds_iterator import DataIterator
from keras import optimizers
from optimizers import MultiSGD
from keras.callbacks import LearningRateScheduler
from keras.layers.convolutional import Conv2D
import re
import math

batch_size = 10
base_lr = 4e-5 # 2e-5
momentum = 0.9
weight_decay = 5e-4
lr_policy =  "step"
gamma = 0.333
stepsize = 136106 #68053   // after each stepsize iterations update learning rate: lr=lr*gamma
max_iter = 600000


model = get_model(training=True, weight_decay=weight_decay)

di = DataIterator("../dataset/val_dataset.h5", data_shape=(3, 368, 368),
                  mask_shape=(1, 46, 46),
                  label_shape=(57, 46, 46),
                  vec_num=38, heat_num=19, batch_size=batch_size, shuffle=True)

# x, y1, y2 = di.next()
#
# print("x  : ", x.shape)
# print("y1 : ", y1.shape)
# print("y2 : ", y2.shape)

# setup lr multipliers for conv layers

lr_mult=dict()
for layer in model.layers:

    if isinstance(layer, Conv2D):

        # stage = 1
        if re.match("Mconv\d_stage1.*", layer.name):
            kernel_name = layer.weights[0].name
            bias_name = layer.weights[1].name
            lr_mult[kernel_name] = 1
            lr_mult[bias_name] = 2

        # stage > 1
        elif re.match("Mconv\d_stage.*", layer.name):
            kernel_name = layer.weights[0].name
            bias_name = layer.weights[1].name
            lr_mult[kernel_name] = 4
            lr_mult[bias_name] = 8

        # vgg
        else:
           kernel_name = layer.weights[0].name
           bias_name = layer.weights[1].name
           lr_mult[kernel_name] = 1
           lr_mult[bias_name] = 2

losses = {}
losses["weight_stage1_L1"] = "mean_squared_error"
losses["weight_stage1_L2"] = "mean_squared_error"
losses["weight_stage2_L1"] = "mean_squared_error"
losses["weight_stage2_L2"] = "mean_squared_error"
losses["weight_stage3_L1"] = "mean_squared_error"
losses["weight_stage3_L2"] = "mean_squared_error"
losses["weight_stage4_L1"] = "mean_squared_error"
losses["weight_stage4_L2"] = "mean_squared_error"
losses["weight_stage5_L1"] = "mean_squared_error"
losses["weight_stage5_L2"] = "mean_squared_error"
losses["weight_stage6_L1"] = "mean_squared_error"
losses["weight_stage6_L2"] = "mean_squared_error"

loss_weights = {}
loss_weights["weight_stage1_L1"] = 1
loss_weights["weight_stage1_L2"] = 1
loss_weights["weight_stage2_L1"] = 1
loss_weights["weight_stage2_L2"] = 1
loss_weights["weight_stage3_L1"] = 1
loss_weights["weight_stage3_L2"] = 1
loss_weights["weight_stage4_L1"] = 1
loss_weights["weight_stage4_L2"] = 1
loss_weights["weight_stage5_L1"] = 1
loss_weights["weight_stage5_L2"] = 1
loss_weights["weight_stage6_L1"] = 1
loss_weights["weight_stage6_L2"] = 1

# learning rate schedule - equivalent of caffe lr_policy =  "step"
def step_decay(epoch):
    initial_lrate = base_lr
    drop = gamma
    epochs_drop = stepsize
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))

    return lrate

lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]

# sgd optimizer with lr multipliers
multisgd = MultiSGD(lr=base_lr, momentum=momentum, decay=0.0, nesterov=False, lr_mult=lr_mult)

model.compile(loss=losses, loss_weights=loss_weights, optimizer=multisgd, metrics=['accuracy'])

model.fit_generator(di, steps_per_epoch=di.N // batch_size, epochs=max_iter,
                    callbacks=callbacks_list)