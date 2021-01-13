import tensorflow as tf
import os
import tensorlayer as tl

from models import generator,  H_estimator, Vgg19_simple_api
from utils import InputLoader, LabelLoader, load, save, psnr_error
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
height_label, width_label = 304, 304


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
    ###label###
    train_label_loader = LabelLoader(train_folder, resize_height=height_label, resize_width=width_label)
    train_label_dataset = train_label_loader(batch_size=batch_size)
    train_label_it = train_label_dataset.make_one_shot_iterator()
    train_label_tensor = train_label_it.get_next()
    train_label_tensor.set_shape([batch_size, height_label, width_label, 3])
    train_label = train_label_tensor 
    print('train inputs = {}'.format(train_inputs))
    print('train label = {}'.format(train_label))

    ##########testing###############
    ###input###
    test_input_loader = InputLoader(test_folder, resize_height=height, resize_width=width)
    test_input_dataset = test_input_loader(batch_size=batch_size)
    test_input_it = test_input_dataset.make_one_shot_iterator()
    test_input_tensor = test_input_it.get_next()
    test_input_tensor.set_shape([batch_size, height, width, 3*2])
    test_inputs = test_input_tensor
    ###label###
    test_label_loader = LabelLoader(test_folder, resize_height=height_label, resize_width=width_label)
    test_label_dataset = test_label_loader(batch_size=batch_size)
    test_label_it = test_label_dataset.make_one_shot_iterator()
    test_label_tensor = test_label_it.get_next()
    test_label_tensor.set_shape([batch_size, height_label, width_label, 3])
    test_label = test_label_tensor 
    print('test inputs = {}'.format(test_inputs))
    print('test label = {}'.format(test_label))




# define training generator function
with tf.variable_scope('generator', reuse=None):
    print('training = {}'.format(tf.get_variable_scope().name))
    train_outputs, train_coarsestitching = generator(train_inputs, True)
    train_psnr_error = psnr_error(train_outputs, train_label)

# define testing generator function
with tf.variable_scope('generator', reuse=True):
    print('testing = {}'.format(tf.get_variable_scope().name))
    test_outputs, test_coarsestitching = generator(test_inputs, False)
    test_psnr_error = psnr_error(test_outputs, test_label)


# define intensity loss
lam_lp = 1
if lam_lp != 0:
    lp_loss = tf.reduce_mean(tf.abs((train_outputs - train_label) ** 1))
else:
    lp_loss = tf.constant(0.0, dtype=tf.float32)


lam_percep =  2e-6
if lam_percep != 0:
    train_outputs_224 = tf.image.resize_images(train_outputs, size=[224, 224], method=0,align_corners=False)  
    train_label_224 = tf.image.resize_images(train_label, size=[224, 224], method=0, align_corners=False)     

    net_vgg, feature_fake = Vgg19_simple_api((train_outputs_224 + 1) / 2, reuse=False)
    _, feature_real = Vgg19_simple_api((train_label_224 + 1) / 2, reuse = True)
    percep_loss = tl.cost.mean_squared_error(feature_fake.outputs, feature_real.outputs, is_mean=True)
else:
    percep_loss = tf.constant(0.0, dtype=tf.float32)



with tf.name_scope('training'):
    g_loss = tf.add_n([lp_loss * lam_lp, percep_loss * lam_percep], name='g_loss')
    g_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='g_step')
    g_lrate = tf.train.piecewise_constant(g_step, boundaries=[400000], values=[0.0002, 0.00002])
    g_optimizer = tf.train.AdamOptimizer(learning_rate=g_lrate, name='g_optimizer')
    g_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    #g_train_op = g_optimizer.minimize(g_loss, global_step=g_step, var_list=g_vars, name='g_train_op')

    grads_and_vars = g_optimizer.compute_gradients(g_loss, g_vars)
    new_gradients = []
    for item in grads_and_vars:
        grad, var = item
        var_name = var.name
        if var_name.startswith('generator/model') and not('BatchNorm' in var_name):
            print(var_name)
            grad = grad*0
        new_gradients.append((grad, var))
    g_train_op = g_optimizer.apply_gradients(new_gradients, global_step=g_step)


# add all to summaries
tf.summary.scalar(tensor=train_psnr_error, name='train_psnr')
tf.summary.scalar(tensor=test_psnr_error, name='test_psnr')
tf.summary.scalar(tensor=g_loss, name='g_loss')
tf.summary.scalar(tensor=lp_loss*lam_lp, name='lp_loss')
tf.summary.scalar(tensor=percep_loss*lam_percep, name='percep_loss')
tf.summary.image(tensor=train_inputs[...,0:3], name='train_input1')
tf.summary.image(tensor=train_inputs[...,3:6], name='train_input2')
tf.summary.image(tensor=train_coarsestitching[..., -3: ], name='train_coarseStitch')
tf.summary.image(tensor=train_outputs, name='train_outputs')
tf.summary.image(tensor=train_label, name='train_gt')
summary_op = tf.summary.merge_all()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    # summaries
    summary_writer = tf.summary.FileWriter(summary_dir, graph=sess.graph)

    # initialize weights
    sess.run(tf.global_variables_initializer())
    print('Init successfully!')
    
    
    #initialize vgg19
    ###============================= LOAD VGG ===============================###
    print("load vgg19 pretrained model")
    params = []
    if lam_percep != 0:
        vgg19_npy_path = "../checkpoints/vgg19/vgg19.npy"
        #check path
        if not os.path.isfile(vgg19_npy_path):
            print("Please download vgg19.npz from : https://github.com/machrisaa/tensorflow-vgg")
            exit()
        else:
            print('checkpoint found')
        #load model
        npz = np.load(vgg19_npy_path, encoding='latin1', allow_pickle=True).item()
        count = 0
        for val in sorted(npz.items()):
            W = np.asarray(val[1][0])
            b = np.asarray(val[1][1])
            print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
            params.extend([W, b])
            count = count + 1
            if count >= 12:
               break
        tl.files.assign_params(sess, params, feature_fake)
        print("load vgg19 pretrained model done!")

    #initialize homography
    print("load homography pretrained model")
    variables = tf.contrib.slim.get_variables_to_restore()
    variables_to_restore_homography = [v for v in variables if v.name.split('/')[1] == 'model']
    saver_homography = tf.train.Saver(variables_to_restore_homography)
    saver_homography.restore(sess, "../checkpoints/homography/model.ckpt-600000")
    print("load homography pretrained model done!")

    # tf saver
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=None)
    restore_var = [v for v in tf.global_variables()]
    loader = tf.train.Saver(var_list=restore_var)
    print(snapshot_dir)
    if os.path.isdir(snapshot_dir):
        ckpt = tf.train.get_checkpoint_state(snapshot_dir)
        if ckpt and ckpt.model_checkpoint_path:
            load(loader, sess, ckpt.model_checkpoint_path)
            print('===========restart from===========')
            print(ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found.')
    else:
        load(loader, sess, snapshot_dir)

    _step, _loss, _summaries = 0, None, None


    print("============starting training===========")
    while _step < iterations:
        try:
            print('Training generator...')
            _, _g_lr, _step, _lp_loss, _percep_loss, _g_loss, _train_psnr, _summaries = sess.run(
                [g_train_op, g_lrate, g_step, lp_loss, percep_loss, g_loss, train_psnr_error, summary_op])
            
            
            if _step % 10 == 0:
                print('GeneratorModel : Step {}, lr = {:.6f}'.format(_step, _g_lr))
                print('                 Global      Loss : ', _g_loss)
                print('                 lp   Loss : ({:.4f} * {:.4f} = {:.4f})'.format(_lp_loss, lam_lp, _lp_loss * lam_lp))
                print('                 perceptual     Loss : ({:.4f} * {:.6f} = {:.4f})'.format(_percep_loss, lam_percep, _percep_loss * lam_percep))
                print('                 PSNR  Error      : ', _train_psnr)
            if _step % 200 == 0:
                summary_writer.add_summary(_summaries, global_step=_step)
                print('Save summaries...')

            if _step % 200000 == 0:
                save(saver, sess, snapshot_dir, _step)

        except tf.errors.OutOfRangeError:
            save(saver, sess, snapshot_dir, _step)
            break
