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
- `python demo_image.py --image sample_images\ski.jpg` to run the picture demo. Result will be stored in the file result.png. You can use
any image file as an input.
- `python demo_camera.py` to run the web demo.

## Training steps
- Install gsutil `curl https://sdk.cloud.google.com | bash`. This is a really helpful tool for downloading large datasets. 
- Download the data set (~25 GB) `cd dataset; sh get_dataset.sh`,
- Download [COCO official toolbox](https://github.com/pdollar/coco) in `dataset/coco/` . 
- `cd coco/PythonAPI; sudo python setup.py install` to install pycocotools.
- Run `cd ../..; python train_pose.py` to start training. 

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
	  
