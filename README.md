# A view-free image stitching network based on global homography [paper](https://www.sciencedirect.com/science/article/pii/S1047320320301784)
<p align="center">Lang Nie*, Chunyu Lin*, Kang Liao*, Meiqin Liu*, Yao Zhao*</p>
<p align="center">* Institute of Information Science, Beijing Jiaotong University</p>

The official implementation of "A view-free image stitching network based on global homography" **(VFISNet)** in Tensorflow.

## Abstract
Image stitching is a traditional but challenging computer vision task, aiming to obtain a seamless panoramic image. Recently, researchers begin to study the image stitching task using deep learning. However, the existing learning methods assume a relatively fixed view during the image capturing, thus show a poor generalization ability to flexible view cases. To address the above problem, we present a cascaded view-free image stitching network based on a global homography. This novel image stitching network does not have any restriction on the view of images and it can be implemented in three stages. In particular, we first estimate a global homography between two input images from different views. And then we propose a structure stitching layer to obtain the coarse stitching result using the global homography. In the last stage, we design a content revision network to eliminate ghosting effects and refine the content of the stitching result. To enable efficient learning on various views, we also present a method to generate synthetic datasets for network training. Experimental results demonstrate that our method can achieve almost 100% elimination of artifacts in overlapping areas at the cost of acceptable slight distortions in non-overlapping areas, compared with traditional methods. In addition, the proposed method is view-free and more robust especially in a scene where feature points are difficult to detect.

![image](https://github.com/nie-lang/DeepImageStitching-1.0/blob/main/pipeline.png)

## Requirement
* python 3.6
* numpy 1.18.1
* tensorflow 1.13.1
* tensorlayer 1.8.0

## Dataset
1. Create folders as follows:
* dataset
  * training  
  * testing  

2. Set the path for row images, training samples, and testing samples in ./Dataset_gen/dataset.py. Then run this script：
```
cd Dataset_gen/
python dataset.py
```
It may take several days to generate the dataset since our code is not optimized for speed. You can change the number of samples according to your needs.

## For windows system
For windows OS users, you have to change '/' to '\\\\' in 'line 61 and line 120 of Codes/H_Net/utils.py' and 'line 60 of Codes/Stitch_Net/utils.py'.

## Training
Step 1. Train the deep homography network
```
cd Codes/H_Net/

python train_H.py  --train_folder  ../../dataset/training
                   --test_folder  ../../dataset/testing
                   --summary_dir  ../summary/homography
                   --snapshot_dir  ../checkpoints/homography
                   --gpu  0
                   --batch 4
                   --iters    600000
```

Step 2. And download the pretrained model of vgg19 from:
```
https://github.com/machrisaa/tensorflow-vgg
```
Then move the vgg model to ./Codes/checkpoints/vgg19/ .

Step 3. Train the content revision network
```
cd Codes/Stitch_Net/

python train.py  --train_folder  ../../dataset/training
                 --test_folder  ../../dataset/testing
                 --summary_dir  ../summary/stitch
                 --snapshot_dir  ../checkpoints/stitch
                 --gpu  0
                 --batch 4
                 --iters    600000
```

## Testing
Test with your retrained model. Or you can get our pretrained model for image stitching in [Google Drive](https://drive.google.com/drive/folders/1JD89Nu4DbEqiBdIkYQNs5KOqhpne5-IQ?usp=sharing) or [Baidu Cloud](https://pan.baidu.com/s/18aW6LTVg4-qQtOQF_GKfHg)(Extraction code: 1234). Move the model to ./Codes/checkpoints/stitch/, and run:
```
cd Codes/Stitch_Net/

python inference.py   --test_folder  ../../dataset/testing
                      --gpu  0    
```
## Visualization
Visualization on TensorBoard for training and validation is supported.
```
tensorboard --logdir= path_of_summary --port port_number
```
 
## Limitation
To the best of our knowledge, this is the first work that can stitch images from arbitrary views in a complete deep learning framework. However, it also has the following two limitations：

1. It cannot handle input of arbitrary resolution.

2. There is no parallax in the synthetic dataset. Therefore, the generalization ability of the proposed network in real scenes is not ideal.

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
