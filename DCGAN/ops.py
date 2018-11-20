'''Helper operation definations'''

import tensorflow as tf

def init_weight(shape):
    return tf.Variable(tf.random_normal(shape=shape,stddev=0.2)) ## weight initialization

def init_bias(shape):
    return tf.Variable(tf.zeros(shape)) ## Bias Initialization

def loss(labels,logits):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,logits=logits))

def conv2d(x,filter,stride,padding="SAME"):
    return tf.nn.conv2d(input=x,filter=filter,strides=[1,stride,stride,1],padding=padding)

def deconv2d(x,filter,stride,padding="SAME",final_channel=None):
    x_shape=tf.shape(x)
    if not final_channel:
        output_shape=tf.stack([x_shape[0],x_shape[1]*2,x_shape[2]*2,x_shape[3]//2])
    else:
        output_shape=tf.stack([x_shape[0],x_shape[1]*2,x_shape[2]*2,final_channel])
    return tf.nn.conv2d_transpose(x,filter,output_shape,strides=[1,stride,stride,1],padding=padding)
