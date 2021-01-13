import tensorflow as tf
import cv2
import numpy as np
from PIL import Image


def StructureStitchingLayer(inputs, H_tf):

    def _repeat(x, n_repeats):
        rep = tf.transpose(
            tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
        rep = tf.cast(rep, 'int32')
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        return tf.reshape(x, [-1])

    def _interpolate(im, x, y, out_size):
        # with tf.variable_scope('_interpolate'):
        # constants
        num_batch = tf.shape(im)[0]
        height = tf.shape(im)[1]
        width = tf.shape(im)[2]
        channels = tf.shape(im)[3]

        x = tf.cast(x, 'float32')
        y = tf.cast(y, 'float32')
        height_f = tf.cast(height, 'float32')
        width_f = tf.cast(width, 'float32')
        out_height = out_size[0]
        out_width = out_size[1]
        zero = tf.zeros([], dtype='int32')
        max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
        max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

        # scale indices from [-1, 1] to [0, width/height]
        # x = (x + 1.0) * (width_f) / 2.0
        # y = (y + 1.0) * (height_f) / 2.0

        # do sampling
        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1

        x0 = tf.clip_by_value(x0, zero, max_x)  
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)
        dim2 = width
        dim1 = width * height
        base = _repeat(tf.range(num_batch) * dim1, out_height * out_width)
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = tf.reshape(im, tf.stack([-1, channels]))
        im_flat = tf.cast(im_flat, 'float32')
        Ia = tf.gather(im_flat, idx_a)
        Ib = tf.gather(im_flat, idx_b)
        Ic = tf.gather(im_flat, idx_c)
        Id = tf.gather(im_flat, idx_d)

        # and finally calculate interpolated values
        x0_f = tf.cast(x0, 'float32')
        x1_f = tf.cast(x1, 'float32')
        y0_f = tf.cast(y0, 'float32')
        y1_f = tf.cast(y1, 'float32')
        wa = tf.expand_dims(((x1_f - x) * (y1_f - y)), 1)
        wb = tf.expand_dims(((x1_f - x) * (y - y0_f)), 1)
        wc = tf.expand_dims(((x - x0_f) * (y1_f - y)), 1)
        wd = tf.expand_dims(((x - x0_f) * (y - y0_f)), 1)
        output = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
        return output
        # return Ia

    def _meshgrid(height, width):
        # with tf.variable_scope('_meshgrid'):

        shift = (304. - 128.) / 2.
        x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                        tf.transpose(tf.expand_dims(tf.linspace(0. - shift, 1.0 * width - shift, width), 1), [1, 0]))
        y_t = tf.matmul(tf.expand_dims(tf.linspace(0. - shift, 1.0 * height - shift, height), 1),
                        tf.ones(shape=tf.stack([1, width])))

        # print(x_t.eval())

        x_t_flat = tf.reshape(x_t, (1, -1))
        y_t_flat = tf.reshape(y_t, (1, -1))

        ones = tf.ones_like(x_t_flat)
        grid = tf.concat([x_t_flat, y_t_flat, ones], 0)
        # sess = tf.get_default_session()
        # print '--grid: \n', grid.eval() # (session=sess.as_default())
        return grid

    def _transform(image_tf, H_tf):
        # with tf.variable_scope('_transform'):
        num_batch = tf.shape(image_tf)[0]
        num_channels = tf.shape(image_tf)[3]
        #  Changed
        # theta = tf.reshape(theta, (-1, 2, 3))
        H_tf = tf.reshape(H_tf, (-1, 3, 3))
        H_tf = tf.cast(H_tf, 'float32')

        #  Added: add two matrices M and B defined as follows in
        # order to perform the equation: H x M x [xs...;ys...;1s...] + H x [width/2...;height/2...;0...]
        H_tf_shape = H_tf.get_shape().as_list()
        # initial

        # grid of (x_t, y_t, 1), eq (1) in ref [1]
        out_height = 304
        out_width = 304
        grid = _meshgrid(out_height, out_width)
        grid = tf.expand_dims(grid, 0)
        grid = tf.reshape(grid, [-1])
        grid = tf.tile(grid, tf.stack([num_batch]))  # stack num_batch grids
        grid = tf.reshape(grid, tf.stack([num_batch, 3, -1]))

        # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
        T_g = tf.matmul(H_tf, grid)
        # T_g = grid
        x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
        # Ty changed
        # y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
        y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
        # Ty added
        t_s = tf.slice(T_g, [0, 2, 0], [-1, 1, -1])
        # The problem may be here as a general homo does not preserve the parallelism
        # while an affine transformation preserves it.
        t_s_flat = tf.reshape(t_s, [-1])

        # # Avoid zero division
        # zero = tf.constant(0, dtype=tf.float32)
        one = tf.constant(1, dtype=tf.float32)

        # smaller
        small = tf.constant(1e-7, dtype=tf.float32)
        smallers = 1e-6 * (one - tf.cast(tf.greater_equal(tf.abs(t_s_flat), small), tf.float32))

        t_s_flat = t_s_flat + smallers
        # condition = tf.reduce_sum(tf.cast(tf.greater(tf.abs(t_s_flat), small), tf.float32))

        # 一维： batchsize * width * height
        x_s_flat = tf.reshape(x_s, [-1]) / t_s_flat
        y_s_flat = tf.reshape(y_s, [-1]) / t_s_flat

        input_transformed = _interpolate(image_tf, x_s_flat, y_s_flat, (304, 304))

        output = tf.reshape(input_transformed, tf.stack([num_batch, out_height, out_width, num_channels]))
        return output

    #with tf.variable_scope(name):
    batch_size = tf.shape(H_tf)[0]
    mask_one = tf.ones_like(inputs[... , 0:3], dtype=tf.float32)
    #step 1. 
    H_one = tf.eye(3)
    H_one = tf.tile(tf.expand_dims(H_one, [0]), [batch_size, 1, 1])
    img1_tf = inputs[... , 0:3] + 1.
    img1_tf = _transform(img1_tf, H_one)
    
    #step 2.
    warp_tf = inputs[...,3:6] + 1.
    warp_tf = _transform(warp_tf, H_tf)
    
    #step 3.
    one = tf.ones_like(img1_tf, dtype=tf.float32)
    zero = tf.zeros_like(img1_tf, dtype=tf.float32)
    mask1 = tf.where(tf.abs(img1_tf) < 1e-6, x=zero, y=one)
    mask2 = tf.where(tf.abs(warp_tf) < 1e-6, x=zero, y=one)
    mask = tf.multiply(mask1, mask2)
    mask1 = mask1 - mask
    mask2 = mask2 - mask
    structureStitching = zero + tf.multiply(warp_tf, mask2) + tf.multiply(img1_tf, mask1) \
              + 0.5*tf.multiply(img1_tf, mask)+ 0.5*tf.multiply(warp_tf, mask)
    img1_tf = img1_tf - 1.
    warp_tf = warp_tf - 1.
    structureStitching = structureStitching - 1.
    
    #step 4.
    output = tf.concat([img1_tf, warp_tf, structureStitching], axis=3)
    
    return output






