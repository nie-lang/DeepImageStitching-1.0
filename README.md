# DeepImageStitching v1 [paper](https://www.sciencedirect.com/science/article/pii/S1047320320301784)
The official implementation of "A view-free image stitching network based on global homography" in Tensorflow.

## Abstract
Image stitching is a traditional but challenging computer vision task, aiming to obtain a seamless panoramic image. Recently, researchers begin to study the image stitching task using deep learning. However, the existing learning methods assume a relatively fixed view during the image capturing, thus show a poor generalization ability to flexible view cases. To address the above problem, we present a cascaded view-free image stitching network based on a global homography. This novel image stitching network does not have any restriction on the view of images and it can be implemented in three stages. In particular, we first estimate a global homography between two input images from different views. And then we propose a structure stitching layer to obtain the coarse stitching result using the global homography. In the last stage, we design a content revision network to eliminate ghosting effects and refine the content of the stitching result. To enable efficient learning on various views, we also present a method to generate synthetic datasets for network training. Experimental results demonstrate that our method can achieve almost 100% elimination of artifacts in overlapping areas at the cost of acceptable slight distortions in non-overlapping areas, compared with traditional methods. In addition, the proposed method is view-free and more robust especially in a scene where feature points are difficult to detect.

## Requirements
* python 3.6
* numpy 1.18.1
* tensorflow 1.13.1
* tensorlayer 1.8.0

## Dataset
