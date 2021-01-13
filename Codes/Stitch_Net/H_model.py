import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from tensorDLT import solve_DLT
from tensorflow.contrib.layers import conv2d


def H_model(train_inputs, is_training, patch_size=128.):

    batch_size = tf.shape(train_inputs)[0]
    #prediect the shift
    pre_4pt_shift = build_model(train_inputs, is_training)
    
    return pre_4pt_shift

def _conv2d(is_training, x, num_out_layers, kernel_size, stride, activation_fn=tf.nn.relu, scope='',use_batch_norm = False):
    p = np.floor((kernel_size -1)/2).astype(np.int32)
    p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
    out_conv =  slim.conv2d(inputs=p_x, num_outputs=num_out_layers, kernel_size=kernel_size, stride=stride, padding="VALID", activation_fn=activation_fn, scope=scope)
    #if use_batch_norm:
    #  out_conv = slim.batch_norm(out_conv, is_training)
    return out_conv

def _conv_block(x, num_out_layers, kernel_sizes, strides, is_training):
    conv1 = _conv2d(is_training, x, num_out_layers[0], kernel_sizes[0], strides[0], scope='conv1')
    conv2 = _conv2d(is_training, conv1, num_out_layers[1], kernel_sizes[1], strides[1], scope='conv2')
    return conv2

def _maxpool2d(x, kernel_size, stride):
    p = np.floor((kernel_size -1)/2).astype(np.int32)
    p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
    return slim.max_pool2d(p_x, kernel_size, stride=stride)

def build_model(train_inputs, is_training):
    # Declare types of activation function, weight_initialization of conv layer. We can set for each conv by setting locally later
    with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training), \
              slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, padding='SAME'):
      with tf.variable_scope('model'):
        input1 = train_inputs[...,0:3]
        input2 = train_inputs[...,3:6]
        input1 = tf.expand_dims(tf.reduce_mean(input1, axis=3),[3])
        input2 = tf.expand_dims(tf.reduce_mean(input2, axis=3),[3])
        print("input1")
        print(input1.shape)
        inputs = tf.concat([input1, input2], axis=3)
        print("inputs")
        print(inputs.shape)
        pred_h4p = network(inputs, is_training)
        return pred_h4p


def feature_extractor(image_tf, is_training):
    image_tf = tf.expand_dims(image_tf, [3])
    with tf.variable_scope('conv_block1'): # H
      conv1 = _conv_block(image_tf, ([64, 64]), (3, 3), (1, 1),is_training)
      maxpool1 = _maxpool2d(conv1, 2, 2) # H/2
    with tf.variable_scope('conv_block2'):
      conv2 = _conv_block(maxpool1, ([64, 64]), (3, 3), (1, 1), is_training)
      maxpool2 = _maxpool2d(conv2, 2, 2) # H/4
    with tf.variable_scope('conv_block3'):
      conv3 = _conv_block(maxpool2, ([128, 128]), (3, 3), (1, 1), is_training)
      maxpool3 = _maxpool2d(conv3, 2, 2) # H/8
    with tf.variable_scope('conv_block4'):
      conv4 = _conv_block(maxpool3, ([128, 128]), (3, 3), (1, 1), is_training)
    
    return conv4

def cost_volume(c1, warp, search_range):
    """Build cost volume for associating a pixel from Image1 with its corresponding pixels in Image2.
    Args:
        c1: Level of the feature pyramid of Image1
        warp: Warped level of the feature pyramid of image22
        search_range: Search range (maximum displacement)
    """
    padded_lvl = tf.pad(warp, [[0, 0], [search_range, search_range], [search_range, search_range], [0, 0]])
    _, h, w, _ = tf.unstack(tf.shape(c1))
    max_offset = search_range * 2 + 1

    cost_vol = []
    for y in range(0, max_offset):
        for x in range(0, max_offset):
            slice = tf.slice(padded_lvl, [0, y, x, 0], [-1, h, w, -1])
            cost = tf.reduce_mean(c1 * slice, axis=3, keepdims=True)
            cost_vol.append(cost)
    cost_vol = tf.concat(cost_vol, axis=3)
    cost_vol = tf.nn.leaky_relu(cost_vol, alpha=0.1)

    return cost_vol


def network(train_inputs, is_training):
    with tf.variable_scope('feature_extract', reuse = None): 
      feature1 = feature_extractor(train_inputs[...,0], is_training)
    with tf.variable_scope('feature_extract', reuse = True): # H
      feature2 = feature_extractor(train_inputs[...,1], is_training)
      

    search_range = 16
    global_correlation = cost_volume(tf.nn.l2_normalize(feature1,axis=3), tf.nn.l2_normalize(feature2,axis=3), search_range)
    
      # Dropout
    keep_prob = 0.5 if is_training==True else 1.0
    #dropout_conv4 = slim.dropout(conv4, keep_prob)

    #3-convolution layers
    conv_1 = conv2d(inputs=global_correlation, num_outputs=512, kernel_size=3, activation_fn=tf.nn.relu)
    conv_2 = conv2d(inputs=conv_1, num_outputs=512, kernel_size=3, activation_fn=tf.nn.relu)
    conv_3 = conv2d(inputs=conv_2, num_outputs=512, kernel_size=3, activation_fn=tf.nn.relu)
    
    
    # Flatten dropout_conv4
    out_conv_flat = slim.flatten(conv_3)

    # Two fully-connected layers
    with tf.variable_scope('fc1'):
      fc1 = slim.fully_connected(out_conv_flat, 1024, scope='fc1')
      dropout_fc1 = slim.dropout(fc1, keep_prob)
    with tf.variable_scope('fc2'):
      fc2 = slim.fully_connected(dropout_fc1, 8, scope='fc2', activation_fn=None) #BATCH_SIZE x 8

    return fc2