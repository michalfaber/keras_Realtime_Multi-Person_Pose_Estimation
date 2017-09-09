#!/usr/bin/env bash

wget -nc --directory-prefix=./caffe/_trained_COCO/ 		http://posefs1.perception.cs.cmu.edu/Users/ZheCao/pose_iter_440000.caffemodel
wget -nc --directory-prefix=./caffe/_trained_MPI/ 		http://posefs1.perception.cs.cmu.edu/Users/ZheCao/pose_iter_146000.caffemodel
