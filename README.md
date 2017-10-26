# Realtime Multi-Person Pose Estimation
This is a keras version of [Realtime Multi-Person Pose Estimation](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation) project  

## Introduction
Code repo for reproducing [2017 CVPR](https://arxiv.org/abs/1611.08050) paper using keras.  

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
- `python demo_camera.py` to run the web demo.

## Training steps

**UPDATE 26/10/2017**

**Fixed problem with the training procedure. 
 Here are my results after training for 5 epochs = 25000 iterations (1 epoch is ~5000 batches)
 The loss values are quite similar as in the original training - [output.txt](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/blob/master/training/example_loss/output.txt)**
   
<p align="center">
<img src="https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation/blob/master/readme/losses.png", width="700">
</p>

**Results of running `demo_image --image sample_images/ski.jpg --model training/weights.best.h5` with model trained only 25000 iterations. Not too bad !!! Training on my single 1070 GPU took around 10 hours.**

<p align="center">
<img src="https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation/blob/master/readme/5ep_result.png", width="300">
</p>

**UPDATE 22/10/2017:**

**Augmented samples are fetched from the [server](https://github.com/michalfaber/rmpe_dataset_server). The network never sees the same image twice
  which was a problem in previous approach (tool rmpe_dataset_transformer)
  This allows you to run augmentation locally or on separate node. 
  You can start 2 instances, one serving training set and a second one serving validation set (on different port if locally)** 
  
- Install gsutil `curl https://sdk.cloud.google.com | bash`. This is a really helpful tool for downloading large datasets. 
- Download the data set (~25 GB) `cd dataset; sh get_dataset.sh`,
- Download [COCO official toolbox](https://github.com/pdollar/coco) in `dataset/coco/` . 
- `cd coco/PythonAPI; sudo python setup.py install` to install pycocotools.
- Go to the "training" folder `cd ../../../training`.
- Generate masks `python generate_masks.py`. Note: set the parameter "mode" in generate_masks.py (validation or training) 
- Create intermediate dataset `python generate_hdf5.py`. This tool creates a dataset in hdf5 format. The structure of this dataset is very similar to the 
    original lmdb dataset where a sample is represented as an array: 5 x width x height (3 channels for image, 1 channel for metedata, 1 channel for miss masks)
    For MPI dataset there are 6 channels with additional all masks.
    Note: set the parameters `datasets` and `val_size` in `generate_hdf5.py`
- Download and compile the dataset server [rmpe_dataset_server](https://github.com/michalfaber/rmpe_dataset_server).
  This server generates augmented samples on the fly. Source samples are retrieved from previously generated hdf5 dataset file.                           
- Start training data server in the first terminal session. 
    `./rmpe_dataset_server ../../keras_Realtime_Multi-Person_Pose_Estimation/dataset/train_dataset.h5 5555`
- Start validation data server in a second terminal session.
    `./rmpe_dataset_server ../../keras_Realtime_Multi-Person_Pose_Estimation/dataset/val_dataset.h5 5556`
- Optionally you can verify the datasets `inspect_dataset.ipynb`
- Set the correct number of samples within `python train_pose.py` - variables "train_samples = ???" and "val_samples = ???".  
 This number is used by keras to determine how many samples are in 1 epoch.
- Train the model in a third terminal `python train_pose.py`
    
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
	  
