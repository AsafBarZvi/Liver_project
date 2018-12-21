import numpy as np
import tensorflow as tf
import sys
import re
#import base6Net as net
import net
from mepy_algo.common.utils.pyplot_utils import ginput,show_im
from matplotlib import pyplot as plt

plt.ion()
fig = None


TFmodel = sys.argv[1]
print "TF model detected: {}".format(TFmodel)
tf.reset_default_graph()

with tf.Session() as sess:

    fsNet = net.FSNET(1)
    saver = tf.train.Saver()
    saver.restore(sess, TFmodel)

    graph = tf.get_default_graph()
    for varName in tf.trainable_variables():
        #print varName
        layerName = re.search("tf.Variable '(.+)'", str(varName))
        layerName = layerName.group(1)

    #while True:

        #key = raw_input("Select layer to plot:\n")
        #if key == 'q':
        #    sys.exit(0)

        weight = graph.get_tensor_by_name(layerName)
        weights = weight.eval(session=sess)

        weights_1 = np.reshape(weights, (-1))
        #fig = plt.figure()
        plt.figure(figsize=(12,7))
        n, bins, patches = plt.hist(weights_1, bins=5)
        plt.suptitle('layer name: {}\nbins: {}\nweights number: {}'.format(layerName, n, bins), fontsize=10)

        #mng = plt.get_current_fig_manager()
        #mng.window.showMaximized()
        #if fig == None:
        #    fig, ax = show_im(histPlot)
        #else:
        #    show_im(histPlot, fig.number, ax)

        #fig.canvas.draw()

        #key = ginput(1, fig.number)
        key = ginput(1, None)
        key = key[0]['key']

        if key == 'n':
            plt.close()
        elif key == 'q':
            sys.exit()








