import tensorflow as tf
import os
import time
import numpy as np
import pickle
import cv2

from PIL import Image
from models import generator, H_estimator
from utils import InputLoader, LabelLoader, load, save, psnr_error
from constant import const
import matplotlib.pyplot as plt

slim = tf.contrib.slim

os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = const.GPU

test_folder = const.TEST_FOLDER

pretrained_model = '../checkpoints/stitch/model.ckpt-600000'

batch_size = 1

# define dataset
with tf.name_scope('dataset'):
    ##########testing###############
    test_inputs_clips_tensor = tf.placeholder(shape=[batch_size, 128, 128, 3 * 2], dtype=tf.float32)
    test_inputs = test_inputs_clips_tensor
    print('test inputs = {}'.format(test_inputs))



# define testing generator function and
with tf.variable_scope('generator', reuse=None):
    print('testing = {}'.format(tf.get_variable_scope().name))
    test_outputs, test_coarsestitching = generator(test_inputs, False) 


config = tf.ConfigProto()
config.gpu_options.allow_growth = True      
with tf.Session(config=config) as sess:
    # dataset
    input_loader = InputLoader(test_folder,128,128)

    # initialize weights
    sess.run(tf.global_variables_initializer())
    print('Init global successfully!')

    # tf saver
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=None)

    restore_var = [v for v in tf.global_variables()]
    loader = tf.train.Saver(var_list=restore_var)

    def inference_func(ckpt):
        print("============")
        print(ckpt)
        load(loader, sess, ckpt)
        print("============")
        length = 1000

        for i in range(0, length):
            input_clip = np.expand_dims(input_loader.get_video_clips(i), axis=0)
            stitch_result, coarse_result = sess.run([test_outputs, test_coarsestitching], feed_dict={test_inputs_clips_tensor: input_clip})
            
            stitch_result = (stitch_result+1) * 127.5    
            stitch_result = stitch_result[0]
            
            path = "../result/" + str(i+1).zfill(6) + ".jpg"
            cv2.imwrite(path, stitch_result)
            
            print('i = {} / {}'.format( i, length))
  
        print("===================DONE!==================")   
    inference_func(pretrained_model)

    

