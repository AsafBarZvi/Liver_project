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
def rgb2bgr(tpl):
    """
    Convert RGB color tuple to BGR
    """
    return (tpl[2], tpl[1], tpl[0])

#-------------------------------------------------------------------------------
def drawFS(img, line, semantics, existence, isRE, gt=False):

    FRAME_WIDTH = 640
    FRAME_HEIGHT = 256+90
    NORM_FACTOR = 87.5
    ORIGINY = 176

    if gt:
        lineBottomNorm = line[:320]
        lineHeightNorm = line[320:]
    else:
        lineBottomNorm = line[:320]/8
        lineHeightNorm = line[320:]/16

    lineHeight = ((lineBottomNorm[::2]+lineHeightNorm)*NORM_FACTOR-NORM_FACTOR+ORIGINY).astype(int)+90
    lineBottom = (lineBottomNorm*NORM_FACTOR-NORM_FACTOR+ORIGINY).astype(int)+90

    x = np.linspace(0,638, 320).astype(int)
    x160 = np.linspace(0,638, 160).astype(int)
    y2 = np.full(160,2).astype(int)
    y8 = np.full(160,8).astype(int)
    colors = [[0,0,0],[0,255,0],[255,255,0],[255,0,0],[148,0,211],[255,127,0],[80, 208, 255],[102,51,0],[0,0,255]]

    existence = 1/(1+np.exp(-existence)) #Use sigmoid function to convert existence output to probability
    existenceColormap = (np.squeeze(cv2.applyColorMap((existence*255).astype(np.uint8), cv2.COLORMAP_JET))).astype(int)

    isRE = 1/(1+np.exp(-isRE)) #Use sigmoid function to convert isRE output to probability
    isREColormap = (np.squeeze(cv2.applyColorMap((isRE*255).astype(np.uint8), cv2.COLORMAP_JET))).astype(int)

    for xLoc in range(320):
        if np.argmax(semantics[xLoc/2,:]) == 0:
            lineBottom[xLoc] = 88
        cv2.drawMarker(img, (x[xLoc],lineBottom[xLoc]), color=colors[np.argmax(semantics[xLoc/2,:])], markerType=cv2.MARKER_SQUARE, markerSize=2, thickness=2)
    for xLoc in range(160):
        if np.argmax(semantics[xLoc,:]) == 0:
            lineHeight[xLoc] = 89
        cv2.drawMarker(img, (x160[xLoc],lineHeight[xLoc]), color=colors[np.argmax(semantics[xLoc,:])], markerType=cv2.MARKER_STAR, markerSize=4, thickness=1)
        cv2.drawMarker(img, (x160[xLoc],y2[xLoc]), color=existenceColormap[xLoc], markerType=cv2.MARKER_DIAMOND, markerSize=2, thickness=3)
        cv2.drawMarker(img, (x160[xLoc],y8[xLoc]), color=isREColormap[xLoc], markerType=cv2.MARKER_DIAMOND, markerSize=2, thickness=3)

    labels = ['noFS','GO','GR','CC','CU','FL','SN','IF','NS']
    cv2.rectangle(img, (5,FRAME_HEIGHT-2), (35,FRAME_HEIGHT-100), (0,0,0), -1)
    for shift in range(8):
        cv2.putText(img, labels[shift+1], (10,FRAME_HEIGHT-(shift+1)*12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[shift+1], 1, cv2.LINE_AA, True)

    return np.copy(img)

#-------------------------------------------------------------------------------
class PrecisionSummary:
    #---------------------------------------------------------------------------
    def __init__(self, session, writer, sample_name, restore=False):
        self.session = session
        self.writer = writer

        sess = session
        ph_name_dist = sample_name+'_distMetric_precision_ph'
        sum_name_dist = sample_name+'_distMetric_precision'
        ph_name_sem = sample_name+'_semMetric_precision_ph'
        sum_name_sem = sample_name+'_semMetric_precision'
        ph_name_total = sample_name+'_semMetric_precision_ph'
        sum_name_total = sample_name+'_totalMetric_precision'

        self.distMetric_placeholder = tf.placeholder(tf.float32, name=ph_name_dist)
        self.distMetric_summary_op = tf.summary.scalar(sum_name_dist, self.distMetric_placeholder)
        self.semMetric_placeholder = tf.placeholder(tf.float32, name=ph_name_sem)
        self.semMetric_summary_op = tf.summary.scalar(sum_name_sem, self.semMetric_placeholder)
        self.totalMetric_placeholder = tf.placeholder(tf.float32, name=ph_name_total)
        self.totalMetric_summary_op = tf.summary.scalar(sum_name_total, self.totalMetric_placeholder)

    #---------------------------------------------------------------------------
    def push(self, epoch, distMetric, semMetric, totalMetric):

        feed = {self.distMetric_placeholder: distMetric}
        summary = self.session.run(self.distMetric_summary_op, feed_dict=feed)
        self.writer.add_summary(summary, epoch)

        feed = {self.semMetric_placeholder: semMetric}
        summary = self.session.run(self.semMetric_summary_op, feed_dict=feed)
        self.writer.add_summary(summary, epoch)

        feed = {self.totalMetric_placeholder: totalMetric}
        summary = self.session.run(self.totalMetric_summary_op, feed_dict=feed)
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

        FRAME_WIDTH = 640
        FRAME_HEIGHT = 256+90
        imgs = np.zeros((10, FRAME_HEIGHT, FRAME_WIDTH, 3))

        for i, sample in enumerate(samples):
            img = np.concatenate((np.full((90,FRAME_WIDTH),255, dtype=np.uint8), sample[0]), axis=0)
            imgRGB = np.concatenate([img[:,:,np.newaxis], img[:,:,np.newaxis], img[:,:,np.newaxis]], axis=-1)
            imgGT = np.copy(imgRGB)
            imgPred = np.copy(imgRGB)

            imgGT = drawFS(imgGT, sample[1], sample[2], sample[5], sample[6], True) #GT
            imgPred = drawFS(imgPred, sample[3], sample[4], sample[5], sample[6]) #Predict

            alpha = 0.3
            cv2.addWeighted(imgGT, alpha, imgPred, 1.-alpha, 0, imgRGB)
            prints = sample[7]
            cv2.putText(imgRGB, prints, (10,45), cv2.FONT_HERSHEY_DUPLEX, 0.5, [0,0,0], 1, cv2.LINE_AA, True)
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
