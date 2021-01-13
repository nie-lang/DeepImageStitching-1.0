import tensorflow as tf
import os

from models import H_estimator
from utils import InputLoader, HLoader, load, save
from constant import const
from PIL import Image
import numpy as np


os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = const.GPU


train_folder = const.TRAIN_FOLDER
test_folder = const.TEST_FOLDER

batch_size = const.BATCH_SIZE
iterations = const.ITERATIONS
height, width = 128, 128


summary_dir = const.SUMMARY_DIR
snapshot_dir = const.SNAPSHOT_DIR


# define dataset
with tf.name_scope('dataset'):
    ##########training###############
    ###input###
    train_input_loader = InputLoader(train_folder, resize_height=height, resize_width=width)
    train_input_dataset = train_input_loader(batch_size=batch_size)
    train_input_it = train_input_dataset.make_one_shot_iterator()
    train_input_tensor = train_input_it.get_next()
    train_input_tensor.set_shape([batch_size, height, width, 3*2])
    train_inputs = train_input_tensor
    ###homography###
    train_h_loader = HLoader(train_folder)
    train_h_dataset = train_h_loader(batch_size=batch_size)
    train_h_it = train_h_dataset.make_one_shot_iterator()
    train_h_tensor = train_h_it.get_next()
    train_h_tensor.set_shape([batch_size, 8, 1])
    train_h = train_h_tensor
    print('train inputs = {}'.format(train_inputs))
    print('train prediction h = {}'.format(train_h))

    ##########testing###############
    ###input###
    test_input_loader = InputLoader(test_folder, resize_height=height, resize_width=width)
    test_input_dataset = test_input_loader(batch_size=batch_size)
    test_input_it = test_input_dataset.make_one_shot_iterator()
    test_input_tensor = test_input_it.get_next()
    test_input_tensor.set_shape([batch_size, height, width, 3*2])
    test_inputs = test_input_tensor
    ###homography###
    test_h_loader = HLoader(test_folder)
    test_h_dataset = test_h_loader(batch_size=batch_size)
    test_h_it = test_h_dataset.make_one_shot_iterator()
    test_h_tensor = test_h_it.get_next()
    test_h_tensor.set_shape([batch_size, 8, 1])
    test_h = test_h_tensor
    print('test inputs = {}'.format(test_inputs))
    print('test prediction h = {}'.format(test_h))




# define training generator function
with tf.variable_scope('generator', reuse=None):
    print('training = {}'.format(tf.get_variable_scope().name))
    train_outputs, train_warp_gt = H_estimator(train_inputs, train_h, True) 
    train_error = tf.reduce_mean(tf.abs((train_outputs - train_h) ** 2))

# define testing generator function
with tf.variable_scope('generator', reuse=True):
    print('testing = {}'.format(tf.get_variable_scope().name))
    test_outputs, test_warp_gt = H_estimator(test_inputs, test_h, False)
    test_error = tf.reduce_mean(tf.abs((test_outputs - test_h) ** 2))


# define intensity loss
lam_h = 1
if lam_h != 0:
    h_loss = tf.reduce_mean(tf.abs((train_outputs - train_h) ** 2))
else:
    h_loss = tf.constant(0.0, dtype=tf.float32)




with tf.name_scope('training'):
    g_loss = tf.add_n([h_loss * lam_h], name='g_loss')
    g_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='g_step')
    g_lrate = tf.train.piecewise_constant(g_step, boundaries=[500000], values=[0.0002, 0.00002])
    g_optimizer = tf.train.AdamOptimizer(learning_rate=g_lrate, name='g_optimizer')
    g_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

    #gradients clip to avoid NAN
    grads = g_optimizer.compute_gradients(g_loss, var_list=g_vars)
    for i, (g, v) in enumerate(grads):
      if g is not None:
        grads[i] = (tf.clip_by_norm(g, 5), v)  # clip gradients
    g_train_op = g_optimizer.apply_gradients(grads, global_step=g_step, name='g_train_op')


# add all to summaries
tf.summary.scalar(tensor=train_error, name='train_error')
tf.summary.scalar(tensor=test_error, name='test_error')
tf.summary.scalar(tensor=g_loss, name='g_loss')
tf.summary.image(tensor=train_inputs[...,0:3], name='train_inpu1')
tf.summary.image(tensor=train_inputs[...,3:6], name='train_inpu2')
tf.summary.image(tensor=train_warp_gt, name='train_warp_gt')
tf.summary.image(tensor=test_inputs[...,0:3], name='test_inpu1')
tf.summary.image(tensor=test_inputs[...,3:6], name='test_inpu2')
tf.summary.image(tensor=test_warp_gt, name='test_warp_gt')
summary_op = tf.summary.merge_all()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    # summaries
    summary_writer = tf.summary.FileWriter(summary_dir, graph=sess.graph)

    # initialize weights
    sess.run(tf.global_variables_initializer())
    print('Init successfully!')

    # tf saver
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=None)
    restore_var = [v for v in tf.global_variables()]
    loader = tf.train.Saver(var_list=restore_var)
    if os.path.isdir(snapshot_dir):
        ckpt = tf.train.get_checkpoint_state(snapshot_dir)
        if ckpt and ckpt.model_checkpoint_path:
            load(loader, sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found.')
    else:
        load(loader, sess, snapshot_dir)

    _step, _loss, _summaries = 0, None, None

    print("============starting training===========")
    while _step < iterations:
        try:
            print('Training generator...')
            _, _g_lr, _step, _h_loss, _g_loss, _summaries = sess.run(
                [g_train_op, g_lrate, g_step, h_loss,  g_loss, summary_op])

            if _step % 10 == 0:
                print('GeneratorModel : Step {}, lr = {:.6f}'.format(_step, _g_lr))
                print('                 Global      Loss : ', _g_loss)
                print('                 intensity   Loss : ({:.4f} * {:.4f} = {:.4f})'.format(_h_loss, lam_h, _h_loss * lam_h))
                #print('                 PSNR  Error      : ', _train_psnr)
            if _step % 100 == 0:
                summary_writer.add_summary(_summaries, global_step=_step)
                print('Save summaries...')

            if _step % 100000 == 0:
                save(saver, sess, snapshot_dir, _step)

        except tf.errors.OutOfRangeError:
            print('Finish successfully!')
            save(saver, sess, snapshot_dir, _step)
            break
