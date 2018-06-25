# Realtime Multi-Person Pose Estimation
This is a keras version of [Realtime Multi-Person Pose Estimation](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation) project  

## Introduction
Code repo for reproducing [2017 CVPR](https://arxiv.org/abs/1611.08050) paper using keras.  

This is a new improved version. The main objective was to remove
dependency on separate c++ server which besides the complexity
of compiling also contained some bugs... and was very slow.
The old version utilizing [rmpe_dataset_server](https://github.com/michalfaber/rmpe_dataset_server) is
still available under the tag [v0.1](https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation/releases/tag/v0.1) if you really would like to take a look.

## Results

<p align="center">
<img src="https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation/blob/master/readme/dance.gif", width="720">
</p>

<div align="center">
<img src="https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation/blob/master/sample_images/ski.jpg", width="300", height="300">
&nbsp;
<img src="https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation/blob/master/readme/result.png", width="300", height="300">
</div>


## Contents
1. [Converting caffe model](#converting-caffe-model-to-keras-model)
2. [Testing](#testing-steps)
3. [Training](#training-steps)
3. [Changes](#changes)

## Require
1. [Keras](https://keras.io/)
2. [Caffe - docker](https://hub.docker.com/r/bvlc/caffe/) required if you would like to convert caffe model to keras model. You 
 don't have to compile/install caffe on your local machine.

## Converting Caffe model to Keras model
Authors of [original implementation](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation) released already trained caffe model 
which you can use to extract weights data.   

- Download caffe model `cd model; sh get_caffe_model.sh`
- Dump caffe layers to numpy data `cd ..; docker run -v [absolute path to your keras_Realtime_Multi-Person_Pose_Estimation folder]:/workspace -it bvlc/caffe:cpu python dump_caffe_layers.py`
  Note that docker accepts only absolute paths so you have to set the full path to the folder containing this project.
- Convert caffe model (from numpy data) to keras model `python caffe_to_keras.py`  

## Testing steps
- Convert caffe model to keras model or download already converted keras model https://www.dropbox.com/s/llpxd14is7gyj0z/model.h5
- Run the notebook `demo.ipynb`.
- `python demo_image.py --image sample_images/ski.jpg` to run the picture demo. Result will be stored in the file result.png. You can use
any image file as an input.

## Training steps


- Install gsutil `curl https://sdk.cloud.google.com | bash`. This is a really helpful tool for downloading large datasets. 
- Download the data set (~25 GB) `cd dataset; sh get_dataset.sh`,
- Download [COCO official toolbox](https://github.com/pdollar/coco) in `dataset/coco/` . 
- `cd coco/PythonAPI; sudo python setup.py install` to install pycocotools.
- Go to the "training" folder `cd ../../../training`.
- Optionally, you can set the number of processes used to generate samples in parallel
  `dataset.py` -> find the line `df = PrefetchDataZMQ(df, nr_proc=4)`
- Run the command in terminal `python train_pose.py`

## Changes
**25/06/2018**

- Performance improvement thanks to replacing c++ server rmpe_dataset_server
with [tensorpack dataflow](http://tensorpack.readthedocs.io/tutorial/dataflow.html).
Tensorpack is a very efficient library for preprocessing and data loading for tensorflow models.
Dataflow object behaves like a normal Python iterator but it can generate samples using many processes.
This significantly reduces latency when GPU waits for
the next sample to be processed.

- Masks generated on the fly - no need to run separate scripts to generate masks.
In fact most of the mask were only positive (nothing to mask out)

- Masking out the discarded persons who are too close to the main person in the
picture, so that the network never sees unlabelled people. Previously we filtered out
keypoints of such smaller persons but they were still visible in the picture.

- Incorrect handling of masks has been fixed. The rmpe_dataset_server
sometimes assigned a wrong mask to the image, misleading the network.


**26/10/2017**

Fixed problem with the training procedure.
 Here are my results after training for 5 epochs = 25000 iterations (1 epoch is ~5000 batches)
 The loss values are quite similar as in the original training - [output.txt](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/blob/master/training/example_loss/output.txt)

<p align="center">
<img src="https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation/blob/master/readme/losses.png", width="700">
</p>

Results of running `demo_image --image sample_images/ski.jpg --model training/weights.best.h5` with model trained only 25000 iterations. Not too bad !!! Training on my single 1070 GPU took around 10 hours.

<p align="center">
<img src="https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation/blob/master/readme/5ep_result.png", width="300">
</p>

**22/10/2017**

Augmented samples are fetched from the [server](https://github.com/michalfaber/rmpe_dataset_server). The network never sees the same image twice
  which was a problem in previous approach (tool rmpe_dataset_transformer)
  This allows you to run augmentation locally or on separate node.
  You can start 2 instances, one serving training set and a second one serving validation set (on different port if locally)

## Related repository
- CVPR'16, [Convolutional Pose Machines](https://github.com/shihenw/convolutional-pose-machines-release).
- CVPR'17, [Realtime Multi-Person Pose Estimation](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation).

## Citation
Please cite the paper in your publications if it helps your research:    

    @InProceedings{cao2017realtime,
      title = {Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields},
      author = {Zhe Cao and Tomas Simon and Shih-En Wei and Yaser Sheikh},
      booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year = {2017}
      }
	  
