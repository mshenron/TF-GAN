import tensorflow as tf
from ops import init_bias,init_weight,conv2d,deconv2d

class discriminator():
    def __init__(self,img_shape):
        self.height,self.width,self.channels=img_shape
        with tf.variable_scope("DIS"):
            ## Variable initialization
            self.w1=init_weight([5,5,self.channels,32])
            self.w2=init_weight([3,3,32,64])
            self.w3=init_weight([2,2,64,128])
            self.w4=init_weight([7*7*128,1]) ## Fianl layer is a feed forward network thus needs flattened input
            self.b1=init_bias([32])
            self.b2=init_bias([64])
            self.b3=init_bias([128])
            self.b4=init_bias([1])

    def feed(self,X,momentum=0.3):
        ##Input
        x=tf.reshape(X,[-1,self.height,self.width,self.channels])
        # Layer 1
        x=conv2d(x,self.w1,2)
        x=tf.nn.bias_add(x,self.b1)
        x=tf.nn.leaky_relu(x)
        #Layer 2
        x=conv2d(x,self.w2,1)
        x=tf.nn.bias_add(x,self.b2)
        x=tf.layers.batch_normalization(x,momentum=momentum)
        x=tf.nn.leaky_relu(x)
        #Layer 3
        x=conv2d(x,self.w3,2)
        x=tf.nn.bias_add(x,self.b3)
        x=tf.layers.batch_normalization(x,momentum=momentum)
        x=tf.nn.leaky_relu(x)
        #Layer 4
        x=tf.reshape(x,[-1,7*7*128])
        logits=tf.matmul(x,self.w4)
        logits=tf.nn.bias_add(logits,self.b4)
        return logits

class genrator():
    def __init__(self,img_shape,z_shape,init_channels=256):
        self.height,self.width,self.channels=img_shape
        self.init_channels=init_channels
        with tf.variable_scope('GEN'):
            self.w1=init_weight([z_shape,7*7*init_channels])
            self.w2=init_weight([4,4,init_channels//2,init_channels])
            self.w3=init_weight([2,2,1,init_channels//2])
            self.b1=init_bias([7*7*init_channels])
            self.b2=init_bias([init_channels//2])

    def feed(self,x,momentum=0.5):
        # Layer 1 Fully Connected
        x=tf.matmul(x,self.w1)
        x=tf.nn.bias_add(x,self.b1)
        x=tf.nn.leaky_relu(x)
        x=tf.reshape(x,[-1,7,7,self.init_channels])
        # Layer 2 Deconv
        x=deconv2d(x,filter=self.w2,stride=2)
        x=tf.nn.bias_add(x,self.b2)
        x=tf.layers.batch_normalization(x,momentum=momentum)
        x=tf.nn.leaky_relu(x)
        #Layer 3 Deconv
        x=deconv2d(x,filter=self.w3,stride=2,final_channel=1)
        x=tf.nn.tanh(x)
        return x
