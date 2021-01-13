import tensorflow as tf

import unet
import H_model
import SSL
import vgg19
from tensorDLT import solve_DLT


slim = tf.contrib.slim

def H_estimator(train_inputs, is_training):
    return H_model.H_model(train_inputs, is_training)


def generator(inputs, is_training, layers=4, features_root=64, filter_size=3, pool_size=2, output_channel=3):
    #step 1. predict shift
    shift = H_estimator(inputs, is_training)
    
    #step 2. solve homography
    H = solve_DLT(shift, 128.)  
    
    #step 3. Structure Stitching Layer
    CoarseStitching = SSL.StructureStitchingLayer(inputs, H)
    
    #step 4. Content Revision Network
    output = unet.unet(CoarseStitching, layers, features_root, filter_size, pool_size, output_channel)
   
    return output, CoarseStitching



def Vgg19_simple_api(rgb, reuse):
    return vgg19.Vgg19_simple(rgb, reuse)
    
    