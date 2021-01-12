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


## Training

## Testing

## Limitation

## Meta
NIE Lang - nielang@bjtu.edu.cn

All code is provided for research purposes only and without any warranty. Any commercial use requires our consent. If you use this code or ideas from the paper for your research, please cite our paper:

```
@article{nie2020view,
  title={A view-free image stitching network based on global homography},
  author={Nie, Lang and Lin, Chunyu and Liao, Kang and Liu, Meiqin and Zhao, Yao},
  journal={Journal of Visual Communication and Image Representation},
  volume={73},
  pages={102950},
  year={2020},
  publisher={Elsevier}
}
```

## Reference
[1] D. DeTone, T. Malisiewicz, and A. Rabinovich. Deep image homography estimation. arXiv preprint arXiv:1606.03798, 2016.
[2] T. Nguyen, S. W. Chen, S. S. Shivakumar, C. J. Taylor, and V. Kumar. Unsupervised deep homography: A fast and robust homography estimation model. IEEE Robotics and Automation Letters, 3(3):2346–2353, 2018.  
