import tensorflow as tf
import numpy as np
#import ipdb

from crf_rnn_layer import crf_rnn_layer


#-------------------------------------------------------------------------------
def smooth_l1_loss(x):
    square_loss   = 0.5*x**2
    absolute_loss = tf.abs(x)
    return tf.where(tf.less(absolute_loss, 1.), square_loss, absolute_loss-0.5)

def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = x.op.name
    # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _variable_summaries(var):
    """Attach a lot of summaries to a Tensor."""
    if not tf.get_variable_scope().reuse:
        name = var.op.name
        #logging.debug("Creating Summary for: %s" % name)
        with tf.name_scope('summaries'):
            tf.summary.scalar(name, var)
            #mean = tf.reduce_mean(var)
            #tf.summary.scalar(name + '/mean', mean)
            #with tf.name_scope('stddev'):
            #    stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
            #tf.summary.scalar(name + '/sttdev', stddev)
            #tf.summary.scalar(name + '/max', tf.reduce_max(var))
            #tf.summary.scalar(name + '/min', tf.reduce_min(var))
            tf.summary.histogram(name, var)



#-------------------------------------------------------------------------------
class UnetCrfRnn:

    def __init__(self, batch_size, wd=0.000005, bd=0.0000005):

        self.batch_size = batch_size
        self.num_classes = 3
        self.wreg = tf.contrib.layers.l2_regularizer(wd)
        self.breg = tf.contrib.layers.l2_regularizer(bd)

        self.build()


    #-----------------------------------------------------------------------
    # Building the net
    #-----------------------------------------------------------------------
    def build(self):

        self.image = tf.placeholder(tf.float32, [self.batch_size, 512, 512, 1])
        self.gtSeg = tf.placeholder(tf.float32, [self.batch_size, 512, 512, self.num_classes])

        with tf.variable_scope("Unet"):

            def conv2d(inData, outChannels, kerSize, strid, layerName):
                if kerSize == 3:
                    inData = tf.pad(inData, [[0, 0], [1, 1], [1, 1], [0, 0]], 'CONSTANT')
                outData = tf.layers.conv2d(inData, outChannels, kerSize, strid, name=layerName, reuse=False, activation=tf.nn.elu, padding='valid', kernel_regularizer=self.wreg, bias_regularizer=self.breg)
                _activation_summary(outData)
                return outData

            def deconv2dAndConcat(inData, concatData, outChannels, kerSize, strid, layerName):
                outData = tf.layers.conv2d_transpose(inData, outChannels, kerSize, strid, name=layerName, reuse=False, activation=tf.nn.elu, padding='same', kernel_regularizer=self.wreg, bias_regularizer=self.breg)
                _activation_summary(outData)
                outData = tf.concat([outData, concatData], axis=-1)
                return outData

            ## Down part ##
            x =        conv2d(self.image ,  16 , 3 , 1 , 'conv1'     )
            conv1out = conv2d(x          ,   8 , 1 , 1 , 'conv1Redu' )
            x =        conv2d(conv1out   ,  16 , 3 , 1 , 'conv2'     )
            x =        conv2d(x          ,  32 , 3 , 2 , 'conv2a'    )
            conv2out = conv2d(x          ,  16 , 1 , 1 , 'conv2Redu' )
            x =        conv2d(conv2out   ,  32 , 3 , 1 , 'conv3'     )
            x =        conv2d(x          ,  64 , 3 , 2 , 'conv3a'    )
            conv3out = conv2d(x          ,  32 , 1 , 1 , 'conv3Redu' )
            x =        conv2d(conv3out   ,  64 , 3 , 1 , 'conv4'     )
            x =        conv2d(x          , 128 , 3 , 2 , 'conv4a'    )
            conv4out = conv2d(x          ,  64 , 1 , 1 , 'conv4Redu' )
            x =        conv2d(conv4out   , 128 , 3 , 1 , 'conv5'     )
            x =        conv2d(x          , 256 , 3 , 2 , 'conv5a'    )
            conv5out = conv2d(x          , 128 , 1 , 1 , 'conv5Redu' )
            ## Output at this point - batchx32x32x128

            ## Up part ##
            x = deconv2dAndConcat(conv5out, conv4out,  64 , 3 , 2 , 'deconv1'   )
            x =            conv2d(x                 , 128 , 3 , 1 , 'conv6'     )
            conv6out =     conv2d(x                 ,  64 , 1 , 1 , 'conv6Redu' )
            x = deconv2dAndConcat(conv6out, conv3out,  32 , 3 , 2 , 'deconv2'   )
            x =            conv2d(x                 ,  64 , 3 , 1 , 'conv7'     )
            conv7out =     conv2d(x                 ,  32 , 1 , 1 , 'conv7Redu' )
            x = deconv2dAndConcat(conv7out, conv2out,  16 , 3 , 2 , 'deconv3'   )
            x =            conv2d(x                 ,  32 , 3 , 1 , 'conv8'     )
            conv8out =     conv2d(x                 ,  16 , 1 , 1 , 'conv8Redu' )
            x = deconv2dAndConcat(conv8out, conv1out,   8 , 3 , 2 , 'deconv4'   )
            conv9out =     conv2d(x                 ,  16 , 3 , 1 , 'conv9'     )
            #x =            conv2d(x                 ,  16 , 3 , 1 , 'conv9'     )
            #conv9out =     conv2d(x                 ,   8 , 1 , 1 , 'conv9Redu' )

            self.unetOut = conv2d(conv9out , self.num_classes , 1 , 1 , 'unetOut' )
            ## Output at this point - batchx512x512x2

        with tf.variable_scope("CrfRnn"):

            theta_alpha = 50.
            theta_beta = 3.
            theta_gamma = 10.
            num_iterations = 5
            self.crfrnnOut = crf_rnn_layer(unaries = self.unetOut,
                                   reference_image = self.image,
                                   num_classes = self.num_classes,
                                   theta_alpha = theta_alpha,
                                   theta_beta = theta_beta,
                                   theta_gamma = theta_gamma,
                                   num_iterations = num_iterations)


        with tf.variable_scope("result"):

            self.segProbability = tf.nn.softmax(self.crfrnnOut)
            #self.segProbability = tf.nn.softmax(self.unetOut)
            self.segPrediction = tf.argmax(self.segProbability, axis=3)

            self.result = {
                    'segProb': self.segProbability,
                    'segPred': self.segPrediction
                    }


        #-----------------------------------------------------------------------
        # Compute loss
        #-----------------------------------------------------------------------
        with tf.variable_scope("loss"):

            logits = tf.reshape(self.crfrnnOut, (-1, self.num_classes))
            #logits = tf.reshape(self.unetOut, (-1, self.num_classes))
            labels = tf.reshape(self.gtSeg, (-1, self.num_classes))

            #classesWeights = tf.constant([[1., 1., 2.]])
            #weights = tf.reduce_sum(classesWeights * labels, axis=1)

            loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
            #loss = loss * weights
            self.loss = tf.reduce_sum(loss, name='CrossEntropyLoss')
            _variable_summaries(self.loss)


