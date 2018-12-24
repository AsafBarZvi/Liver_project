import cv2
import tensorflow as tf
import numpy as np
from inspect_checkpoint import getAllVariables

#import matplotlib as mpl
#mpl.use('Agg')
#import matplotlib.pyplot as plt

#import ipdb


#-------------------------------------------------------------------------------
def initialize_uninitialized_variables(sess):
    """
    Only initialize the weights that have not yet been initialized by other
    means, such as importing a metagraph and a checkpoint. It's useful when
    extending an existing model.
    """
    uninit_vars    = []
    uninit_tensors = []
    for var in tf.global_variables():
        uninit_vars.append(var)
        uninit_tensors.append(tf.is_variable_initialized(var))
    uninit_bools = sess.run(uninit_tensors)
    uninit = zip(uninit_bools, uninit_vars)
    for init , var in uninit:
        if not init:
            print "Init to {}".format(var)
    uninit = [var for init, var in uninit if not init]
    sess.run(tf.variables_initializer(uninit))

#-------------------------------------------------------------------------------
def initialize_variables_from_ckpt(sess, ckpt_path):
    """
    try to initialize var from given ckpt_path
    """

    try :
        gsaver = tf.train.Saver()
        gsaver.restore(sess, ckpt_path)
    except Exception as E:
        print "Unable to apply global varbiable restorer on {} attempting per var restore".format(ckpt_path)
        ckpt_vars = getAllVariables(ckpt_path)
        assign_ops = []
        for var in tf.global_variables():
            vname = var.name.replace(":0","")
            ngpu_var_name = vname.replace("gpus_loop/","")
            if   (vname in ckpt_vars) and var.shape == ckpt_vars[vname].shape:
                assign_ops.append(tf.assign(var , ckpt_vars[vname]))
                print "Found {}".format(var)
            elif (ngpu_var_name in ckpt_vars) and var.shape == ckpt_vars[ngpu_var_name].shape:
                assign_ops.append(tf.assign(var , ckpt_vars[ngpu_var_name]))
                print "Found {}".format(var)
            else:
                print "failed to Find {}".format(var)
        sess.run(assign_ops)

#-------------------------------------------------------------------------------
def load_data_source(data_source):
    """
    Load a data source given it's name
    """
    source_module = __import__('source_'+data_source)
    get_source    = getattr(source_module, 'get_source')
    return get_source()

#-------------------------------------------------------------------------------
def drawSeg(seg, segGT):

    FRAME_WIDTH = 512
    FRAME_HEIGHT = 512

    classesColors = [[0,0,0],[0,255,0],[255,0,0],[255,255,0],[0,0,255]]

    segColor = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    segColor[segGT[:,:,1] == 1] = classesColors[3]
    segColor[seg == 1] = classesColors[1]
    segColor[segGT[:,:,2] == 1] = classesColors[4]
    segColor[seg == 2] = classesColors[2]

    labels = ['BCG','LIV','LES','LIVgt','LESgt']
    cv2.rectangle(segColor, (5,FRAME_HEIGHT-2), (45,FRAME_HEIGHT-70), (255,255,255), -1)
    for shift in range(len(labels)):
        cv2.putText(segColor, labels[shift], (10,FRAME_HEIGHT-(shift+1)*12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, classesColors[shift], 1, cv2.LINE_AA, True)

    return np.copy(segColor)

#-------------------------------------------------------------------------------
class MetricsSummary:
    #---------------------------------------------------------------------------
    def __init__(self, session, writer, sample_name, restore=False):
        self.session = session
        self.writer = writer

        sess = session
        ph_name_dist = sample_name+'_dice_precision_ph'
        sum_name_dist = sample_name+'_dice_precision'
        ph_name_sem = sample_name+'_precision_precision_ph'
        sum_name_sem = sample_name+'_precision_precision'
        ph_name_total = sample_name+'_precision_precision_ph'
        sum_name_total = sample_name+'_recall_precision'

        self.dice_placeholder = tf.placeholder(tf.float32, name=ph_name_dist)
        self.dice_summary_op = tf.summary.scalar(sum_name_dist, self.dice_placeholder)
        self.precision_placeholder = tf.placeholder(tf.float32, name=ph_name_sem)
        self.precision_summary_op = tf.summary.scalar(sum_name_sem, self.precision_placeholder)
        self.recall_placeholder = tf.placeholder(tf.float32, name=ph_name_total)
        self.recall_summary_op = tf.summary.scalar(sum_name_total, self.recall_placeholder)

    #---------------------------------------------------------------------------
    def push(self, epoch, dice, precision, recall):

        feed = {self.dice_placeholder: dice}
        summary = self.session.run(self.dice_summary_op, feed_dict=feed)
        self.writer.add_summary(summary, epoch)

        feed = {self.precision_placeholder: precision}
        summary = self.session.run(self.precision_summary_op, feed_dict=feed)
        self.writer.add_summary(summary, epoch)

        feed = {self.recall_placeholder: recall}
        summary = self.session.run(self.recall_summary_op, feed_dict=feed)
        self.writer.add_summary(summary, epoch)

#-------------------------------------------------------------------------------
class ImageSummary:
    #---------------------------------------------------------------------------
    def __init__(self, session, writer, sample_name, restore=False):
        self.session = session
        self.writer = writer

        sess = session
        sum_name = sample_name+'_img'
        ph_name = sample_name+'_img_ph'
        self.img_placeholder = tf.placeholder(tf.float32, name=ph_name, shape=[None, None, None, 3])
        self.img_summary_op = tf.summary.image(sum_name, self.img_placeholder, max_outputs=10)

    #---------------------------------------------------------------------------
    def push(self, epoch, samples):

        FRAME_WIDTH = 512
        FRAME_HEIGHT = 512

        imgs = np.zeros((10, FRAME_HEIGHT, FRAME_WIDTH, 3))

        for i, sample in enumerate(samples):
            img = sample[0].astype(np.uint8)
            imgRGB = np.concatenate([img[:,:,np.newaxis], img[:,:,np.newaxis], img[:,:,np.newaxis]], axis=-1)
            seg = np.copy(sample[1])
            segGT = np.copy(sample[2])

            segColor = drawSeg(seg, segGT) #Predicted labels

            alpha = 0.3
            cv2.addWeighted(segColor, alpha, imgRGB, 1.-alpha, 0, imgRGB)
            imgs[i] = imgRGB[::-1,:,:]

        feed = {self.img_placeholder: imgs}
        summary = self.session.run(self.img_summary_op, feed_dict=feed)
        self.writer.add_summary(summary, epoch)

#-------------------------------------------------------------------------------
class LossSummary:
    #---------------------------------------------------------------------------
    def __init__(self, session, writer, sample_name, num_samples):
        self.session = session
        self.writer = writer
        self.num_samples = num_samples
        self.loss_names = ['L1']
        self.loss_values = {}
        self.placeholders = {}

        sess = session

        summary_ops = []
        for loss in self.loss_names:
            sum_name = sample_name+'_'+loss+'_loss'
            ph_name = sample_name+'_'+loss+'_loss_ph'

            placeholder = tf.placeholder(tf.float32, name=ph_name)
            summary_op = tf.summary.scalar(sum_name, placeholder)

            self.loss_values[loss] = float(0)
            self.placeholders[loss] = placeholder
            summary_ops.append(summary_op)

        self.summary_ops = tf.summary.merge(summary_ops)

    #---------------------------------------------------------------------------
    def add(self, value):
        for loss in self.loss_names:
            self.loss_values[loss] += value

    #---------------------------------------------------------------------------
    def push(self, epoch):
        feed = {}
        for loss in self.loss_names:
            feed[self.placeholders[loss]] = self.loss_values[loss]/self.num_samples

        summary = self.session.run(self.summary_ops, feed_dict=feed)
        self.writer.add_summary(summary, epoch)

        for loss in self.loss_names:
            self.loss_values[loss] = float(0)
