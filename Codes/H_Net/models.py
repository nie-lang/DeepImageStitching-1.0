import tensorflow as tf

import H_model


slim = tf.contrib.slim

def H_estimator(train_inputs, train_gt, is_training):
    return H_model.H_model(train_inputs, train_gt, is_training)




