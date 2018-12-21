import tensorflow as tf
import numpy as np
#import ipdb


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
class FSNET:

    def __init__(self, batch_size, wd=0.000005, bd=0.0000005):

        self.batch_size = batch_size
        self.wreg = tf.contrib.layers.l2_regularizer(wd)
        self.breg = tf.contrib.layers.l2_regularizer(bd)

        self.build()


    #-----------------------------------------------------------------------
    # Building the net
    #-----------------------------------------------------------------------
    def build(self):

        self.image = tf.placeholder(tf.float32, [None, 256, 640, 1])
        self.lineGT = tf.placeholder(tf.float32, [None, 480])
        self.semanticGT = tf.placeholder(tf.float32, [None, 160, 9])

        with tf.variable_scope("net"):

            def conv2d(inData, outChannels, kerSize, strid, layerName):
                if kerSize == 3:
                    inData = tf.pad(inData, [[0, 0], [1, 1], [1, 1], [0, 0]], 'CONSTANT')
                outData = tf.layers.conv2d(inData, outChannels, kerSize, strid, name=layerName, reuse=False, activation=tf.nn.relu, padding='valid', kernel_regularizer=self.wreg, bias_regularizer=self.breg)
                _activation_summary(outData)
                return outData

            def firstLayer(inData, outChannels, kerSize, strid, layerName):
                if kerSize == 3:
                    inData = tf.pad(inData, [[0, 0], [1, 1], [1, 1], [0, 0]], 'CONSTANT')
                outData = tf.layers.conv2d(inData, outChannels, kerSize, 1, name=layerName[0], reuse=False, activation=None, padding='valid', kernel_regularizer=self.wreg, bias_regularizer=self.breg)
                outData = tf.layers.max_pooling2d(outData, 2, strid, name=layerName[1])
                outData = tf.nn.relu(outData, name=layerName[2])
                _activation_summary(outData)
                return outData

            x = firstLayer(self.image , 8 , 3 , 2 , ['conv1', 'pool1', 'relu1'])

            x = conv2d(x ,  16 , 3 , 1 , 'conv2'     )
            x = conv2d(x ,  32 , 3 , 2 , 'conv2a'    )
            x = conv2d(x ,  16 , 1 , 1 , 'conv3Redu' )
            x = conv2d(x ,  32 , 3 , 1 , 'conv3'     )
            x = conv2d(x ,  64 , 3 , 2 , 'conv3a'    )
            x = conv2d(x ,  32 , 1 , 1 , 'conv4Redu' )
            x = conv2d(x ,  64 , 3 , 1 , 'conv4'     )
            x = conv2d(x ,  96 , 3 , 2 , 'conv4a'    )
            x = conv2d(x ,  64 , 1 , 1 , 'conv5Redu' )
            x = conv2d(x ,  96 , 3 , 1 , 'conv5'     )
            x = conv2d(x , 128 , 3 , 2 , 'conv5a'    )
            x = conv2d(x ,  32 , 1 , 1 , 'conv6Redu' )
            x = conv2d(x , 128 , 3 , 1 , 'conv6'     )
            x = conv2d(x ,  32 , 1 , 1 , 'conv7Redu' )
            x = conv2d(x , 128 , 3 , 1 , 'conv7'     )
            x = conv2d(x ,  32 , 1 , 1 , 'conv8Redu' )
            x = conv2d(x , 128 , 3 , 1 , 'conv8'     )
            x = conv2d(x ,  32 , 1 , 1 , 'conv9Redu' )
            x = conv2d(x , 128 , 3 , 1 , 'conv9'     )
            x = conv2d(x ,  32 , 1 , 1 , 'conv10Redu')
            x = conv2d(x , 128 , 3 , 1 , 'conv10'    )
            x = conv2d(x ,  16 , 1 , 1 , 'conv11Redu')
            ## Output of the CNN body - batchx20x8x16

            #x = tf.layers.flatten(x)
            x_shape = x.get_shape().as_list()
            x = tf.reshape(x, [-1, x_shape[1] * x_shape[2] * x_shape[3]])
            ## After flatten - batchx2560
            #xLength = x.get_shape().as_list()[1]
            #lineLength, semLength = xLength*1/3-1, xLength*2/3+2
            lineLength, semLength, netExistence = 800, 1600, 160
            self.fc_line, self.fc_sem, self.fc_exis = tf.split(x, [lineLength, semLength, netExistence], 1)
            ## After split - lineLength = batchx800, semLength = batchx1600, netExistence = batchx160

        with tf.variable_scope("result"):

            with tf.variable_scope("exis"):

                self.exis = tf.layers.dense(self.fc_exis, 160, name='fc_exis', activation=None, kernel_regularizer=self.wreg , bias_regularizer=self.breg)
                self.isRE = tf.layers.dense(self.fc_exis, 160, name='fc_isRE', activation=None, kernel_regularizer=self.wreg , bias_regularizer=self.breg)

            with tf.variable_scope("line"):

                li0, li1 = tf.split(self.fc_line, [540, 260], 1)

                lo0 = tf.layers.dense(li0, 320, name='fc_line_group0', activation=None, kernel_regularizer=self.wreg , bias_regularizer=self.breg)
                lo1 = tf.layers.dense(li1, 160, name='fc_line_group1', activation=None, kernel_regularizer=self.wreg , bias_regularizer=self.breg)
                self.line = tf.concat([lo0,lo1], 1)

            with tf.variable_scope("semantics"):

                si0, si1, si2, si3, si4, si5, si6, si7 = tf.split(self.fc_sem, num_or_size_splits=8, axis=1)

                so0 = tf.layers.dense(si0, 160, name='fc_sem_group0', activation=None, kernel_regularizer=self.wreg , bias_regularizer=self.breg)
                so1 = tf.layers.dense(si1, 160, name='fc_sem_group1', activation=None, kernel_regularizer=self.wreg , bias_regularizer=self.breg)
                so2 = tf.layers.dense(si2, 160, name='fc_sem_group2', activation=None, kernel_regularizer=self.wreg , bias_regularizer=self.breg)
                so3 = tf.layers.dense(si3, 160, name='fc_sem_group3', activation=None, kernel_regularizer=self.wreg , bias_regularizer=self.breg)
                so4 = tf.layers.dense(si4, 160, name='fc_sem_group4', activation=None, kernel_regularizer=self.wreg , bias_regularizer=self.breg)
                so5 = tf.layers.dense(si5, 160, name='fc_sem_group5', activation=None, kernel_regularizer=self.wreg , bias_regularizer=self.breg)
                so6 = tf.layers.dense(si6, 160, name='fc_sem_group6', activation=None, kernel_regularizer=self.wreg , bias_regularizer=self.breg)
                so7 = tf.layers.dense(si7, 160, name='fc_sem_group7', activation=None, kernel_regularizer=self.wreg , bias_regularizer=self.breg)
                soAll = tf.concat([so0,so1,so2,so3,so4,so5,so6,so7], 1)
                self.sem = tf.transpose(tf.reshape(soAll, [-1, 8, 160]), perm=[0,2,1])


            self.semSM = tf.nn.softmax(self.sem)
            self.result = {
                'line': self.line,
                'semantic': self.sem,
                'semanticSM': self.semSM,
                'existence': self.exis,
                'isRE': self.isRE
            }


        #-----------------------------------------------------------------------
        # Compute loss
        #-----------------------------------------------------------------------
        with tf.variable_scope("loss"):

            lineGT = self.lineGT
            line = self.line
            semGT = self.semanticGT
            sem = self.sem
            batchSize = self.batch_size
            exis = self.exis
            isRE = self.isRE

            lineBottomGT = lineGT[:,:320]*8
            lineHeightGT = lineGT[:,320:]*16
            lineBottom = line[:,:320]
            lineHeight = line[:,320:]

            #-----------------------------------------------------------------------
            # Compute bottom line loss
            #-----------------------------------------------------------------------
            # Create no FS pixels mask and calculate the number of valid pixels for later dividing
            semGT_flat = tf.reshape(semGT, [batchSize*160,-1])
            noFS_sem = tf.where(tf.equal(semGT_flat[:,0], 1.), semGT_flat[:,0], tf.zeros([batchSize*160,], dtype=tf.float32))
            infFS_sem = tf.where(tf.equal(semGT_flat[:,7], 1.), semGT_flat[:,7], tf.zeros([batchSize*160,], dtype=tf.float32))
            dcFS_sem = tf.where(tf.equal(semGT_flat[:,8], 1.), semGT_flat[:,8], tf.zeros([batchSize*160,], dtype=tf.float32))
            noBottom_sem = tf.add(noFS_sem, infFS_sem)
            noBottom_sem = tf.where(tf.equal(noBottom_sem, 2.), noFS_sem, noBottom_sem)
            #noBottom_sem = tf.add(dcFS_sem, noBottom_sem)
            #noBottom_sem = tf.where(tf.equal(noBottom_sem, 2.), dcFS_sem, noBottom_sem)
            #noBottom_sem = noFS_sem + infFS_sem + dcFS_sem
            #noBottom_sem = tf.where(tf.greater_equal(noBottom_sem, 2.), tf.ones([batchSize*160,], dtype=tf.float32), noBottom_sem)
            validX_count = tf.subtract(tf.convert_to_tensor(batchSize*160.), tf.reduce_sum(noBottom_sem))
            validX_count = tf.multiply(validX_count, 2.)
            # For statistic
            negMining = tf.reduce_sum(noFS_sem, name='negMiningAmount')
            _variable_summaries(negMining)
            noSem = tf.reduce_sum(dcFS_sem, name='noSemanticAmount')
            _variable_summaries(noSem)
            infSem = tf.reduce_sum(infFS_sem, name='infintyAmount')
            _variable_summaries(infSem)

            # Create vector in the size of 'line*batchSize' with zero value for inf/non FS type
            lineBottomGT_flat = tf.reshape(lineBottomGT, [batchSize*320,])
            lineBottom_flat = tf.reshape(lineBottom, [batchSize*320,])
            # Duplicate each element in 'noBottom_sem' to get a vector size in the size of 'line' vector
            noBottom_semDouble = tf.squeeze(noBottom_sem)
            noBottom_semDouble = tf.reshape(noBottom_semDouble, [-1,1])
            noBottom_semDouble = tf.tile(noBottom_semDouble, [1,2])
            noBottom_semDouble = tf.reshape(noBottom_semDouble, [-1])
            #noBottom_semDouble = tf.manip.roll(noBottom_semDouble, -1, axis=0)
            # Zero the values of non Bottom type, so loss isn't affected
            lineBottomGT_flat = tf.where(tf.equal(noBottom_semDouble, 1.), tf.zeros([batchSize*320,], dtype=tf.float32), lineBottomGT_flat)
            # Clear quantization errors in bottom line values due to resulotion diffs 160 semantics vs. 320 bottom line values
            reduceInvalidX = tf.where(tf.less(lineBottomGT_flat, -800), tf.ones([batchSize*320,], dtype=tf.float32), tf.zeros([batchSize*320,], dtype=tf.float32))
            reduceInvalidX = tf.reduce_sum(reduceInvalidX, name='reduceInvalidX')
            validX_count = tf.subtract(validX_count, reduceInvalidX, name='validX_count')
            _variable_summaries(validX_count)
            # For statistic
            _variable_summaries(reduceInvalidX)
            # Continue the dist loss calculation...
            lineBottom_flat = tf.where(tf.equal(noBottom_semDouble, 1.), tf.zeros([batchSize*320,], dtype=tf.float32), lineBottom_flat)
            lineBottom_flat = tf.where(tf.less(lineBottomGT_flat, -800), tf.zeros([batchSize*320,], dtype=tf.float32), lineBottom_flat)
            lineBottomGT_flat = tf.where(tf.less(lineBottomGT_flat, -800), tf.zeros([batchSize*320,], dtype=tf.float32), lineBottomGT_flat)
            # Regresses the bottem line
            distPredGT_line = tf.subtract(lineBottomGT_flat, lineBottom_flat)
            # For statistic
            bottomDistLossAbs = tf.abs(distPredGT_line)
            bottomDistLossAbs = tf.reduce_sum(bottomDistLossAbs)
            self.bottomDistLossAbs = tf.divide(bottomDistLossAbs, validX_count, name='bottomDistLossAbs')
            _variable_summaries(self.bottomDistLossAbs)
            # Smooth L1 loss on each distance values
            bottomDistLoss = smooth_l1_loss(distPredGT_line)
            # Sum all values
            bottomDistLoss = tf.reduce_sum(bottomDistLoss)
            # Mean bottom distance loss for the batch - distLoss=scalar
            self.bottomDistLoss = tf.divide(bottomDistLoss, validX_count, name='bottomDistLoss')
            _variable_summaries(self.bottomDistLoss)

            #-----------------------------------------------------------------------
            # Compute height line loss
            #-----------------------------------------------------------------------
            # Create flat pixels mask and calculate the number of valid non flat pixels for later dividing
            flatFS_sem = tf.where(tf.equal(semGT_flat[:,5],1.), semGT_flat[:,5], tf.zeros([batchSize*160,], dtype=tf.float32))
            snowFS_sem = tf.where(tf.equal(semGT_flat[:,6],1.), semGT_flat[:,6], tf.zeros([batchSize*160,], dtype=tf.float32))
            genobjFS_sem = tf.where(tf.equal(semGT_flat[:,1],1.), semGT_flat[:,1], tf.zeros([batchSize*160,], dtype=tf.float32))
            #noHeight_sem = tf.add(genobjFS_sem, flatFS_sem)
            #noHeight_sem = tf.where(tf.equal(noHeight_sem,2), genobjFS_sem, noHeight_sem)
            noHeight_sem = flatFS_sem + snowFS_sem + genobjFS_sem
            noHeight_sem = tf.where(tf.greater_equal(noHeight_sem, 2.), tf.ones([batchSize*160,], dtype=tf.float32), noHeight_sem)
            noHeightBottom_sem = tf.add(noBottom_sem, noHeight_sem)
            noHeightBottom_sem = tf.where(tf.equal(noHeightBottom_sem,2), noBottom_sem, noHeightBottom_sem)
            #validHeight_count = tf.subtract(tf.convert_to_tensor(batchSize*160.), tf.reduce_sum(noHeight_sem), name='validHeight_count')
            validHeight_count = tf.subtract(tf.convert_to_tensor(batchSize*160.), tf.reduce_sum(noBottom_sem), name='validHeight_count')
            _variable_summaries(validHeight_count)
            validHeight_count = tf.where(tf.equal(validHeight_count, 0.), 1., validHeight_count)
            # For statistic
            negMiningHeight = tf.reduce_sum(noHeight_sem, name='negMiningAmountHeight')
            _variable_summaries(negMiningHeight)

            # Create vector in the size of 'line*batchSize' with zero value for flat semantics
            lineHeightGT_flat = tf.reshape(lineHeightGT, [batchSize*160,])
            lineHeight_flat = tf.reshape(lineHeight, [batchSize*160,])
            # Zero the values of non FS type, so loss isn't affected
            lineHeightGT_flat = tf.where(tf.equal(noHeightBottom_sem, 1.), tf.zeros([batchSize*160,], dtype=tf.float32), lineHeightGT_flat)
            lineHeight_flat = tf.where(tf.equal(noBottom_sem, 1.), tf.zeros([batchSize*160,], dtype=tf.float32), lineHeight_flat)
            # Regresses the height line
            distPredGT_height = tf.subtract(lineHeightGT_flat, lineHeight_flat)
            distPredGT_height = tf.where(tf.equal(lineHeightGT_flat, 1600.), tf.zeros([batchSize*160,], dtype=tf.float32), distPredGT_height)
            # For statistic
            heightDistLossAbs = tf.abs(distPredGT_height)
            heightDistLossAbs = tf.reduce_sum(heightDistLossAbs)
            self.heightDistLossAbs = tf.divide(heightDistLossAbs, validHeight_count, name='heightDistLossAbs')
            _variable_summaries(self.heightDistLossAbs)
            # Smooth L1 loss on each distance values
            heightDistLoss = smooth_l1_loss(distPredGT_height)
            # Sum all values
            heightDistLoss = tf.reduce_sum(heightDistLoss)
            # Mean bottom distance loss for the batch - distLoss=scalar
            self.heightDistLoss = tf.divide(heightDistLoss, validHeight_count, name='heightDistLoss')
            _variable_summaries(self.heightDistLoss)

            #-----------------------------------------------------------------------
            # Compute multiclass semantic loss
            #-----------------------------------------------------------------------
            classesWeights = tf.constant([[1.2, 1., 1., 1., 1., 1., 1., 1.2, 1.]])
            weights = tf.reduce_sum(classesWeights * semGT_flat, axis=1)
            sem_flat = tf.reshape(sem, [batchSize*160,-1])
            sem_flat = tf.concat([sem_flat, tf.zeros([batchSize*160,1])], axis=1)
            # Make sure that no conf loss calculated for 'dontCare' labels
            sem_flat = tf.where(tf.equal(tf.tile(tf.reshape(dcFS_sem,[-1,1]),[1,9]), 1.), tf.concat([tf.zeros([batchSize*160,8]),tf.ones([batchSize*160,1])], axis=1), sem_flat)
            semLoss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=semGT_flat, logits=sem_flat)
            semLoss = semLoss * weights
            semLoss = tf.reduce_sum(semLoss)
            self.semLoss = tf.divide(semLoss, batchSize*160., name='semLoss')

            #-----------------------------------------------------------------------
            # Compute existence loss
            #-----------------------------------------------------------------------

            # Params
            focal=403
            camH=1.3
            dZ = 0.5
            cPos = 2 # Positive threshold in pixels
            cNeg = 3 # Negative threshold in pixels

            # Claculate the error between the predicted and GT lines in pixels
            lineBottomGT_pix = lineBottomGT_flat[::2]*5.5 # why 5.5? --> to pixel:(line-1)*normFactor+horizon --> relative to origin: ((line-1)*normFactor+horizon-horizon)/2 -->
            lineBottom_pix = lineBottom_flat[::2]*5.5     # --> remove -1 bias because the substraction: line*normFactor/2 --> divide by the line by the scale factor: line*(normFactor/(2*8))=line*5.5
            err = tf.subtract(lineBottomGT_pix, lineBottom_pix)
            err = tf.abs(err)

            thrPos = tf.where(tf.greater_equal(lineBottomGT_pix, 0.), cPos*tf.ones([batchSize*160,], dtype=tf.float32),
                    tf.maximum(lineBottomGT_pix*lineBottomGT_pix*dZ/focal/camH, cPos*tf.ones([batchSize*160,], dtype=tf.float32)))
            thrNeg = tf.where(tf.greater_equal(lineBottomGT_pix, 0.), cNeg*tf.ones([batchSize*160,], dtype=tf.float32),
                    tf.maximum(lineBottomGT_pix*lineBottomGT_pix*dZ/focal/camH, cNeg*tf.ones([batchSize*160,], dtype=tf.float32)))

            labels = tf.where(tf.equal(err < thrPos, True), tf.ones([batchSize*160,], dtype=tf.float32), tf.zeros([batchSize*160,], dtype=tf.float32))
            # Filter all ignore cases when the error is within the pos-neg band
            weights = tf.where(tf.equal(tf.logical_and(err > thrPos, err < thrNeg), True), tf.zeros([batchSize*160,], dtype=tf.float32), tf.ones([batchSize*160,], dtype=tf.float32))

            exis_flat = tf.reshape(exis, [batchSize*160,])
            exis_flat_weighted = tf.where(tf.equal(weights, 0.), tf.zeros([batchSize*160,], dtype=tf.float32), exis_flat)

            existenceLoss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=exis_flat_weighted)
            existenceLoss = tf.reduce_sum(existenceLoss)
            self.existenceLoss = tf.divide(existenceLoss, tf.reduce_sum(weights), name='existenceLoss')

            #-----------------------------------------------------------------------
            # Compute isRE loss
            #-----------------------------------------------------------------------

            semGT_re = tf.reduce_sum(semGT_flat[:,2:6], axis=1) + semGT_flat[:,8]
            labels = tf.where(tf.equal(semGT_re, 1.), tf.ones([batchSize*160,], dtype=tf.float32), tf.zeros([batchSize*160,], dtype=tf.float32))
            isRE_flat = tf.reshape(isRE, [batchSize*160,])
            isRELoss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=isRE_flat)
            isRELoss = tf.reduce_sum(isRELoss)
            self.isRELoss = tf.divide(isRELoss, batchSize*160., name='isRELoss')

            #self.checks = {'thrPos': thrPos, 'thrNeg': thrNeg, 'err': err, 'labels': labels, 'weights': weights, 'existenceLoss': existenceLoss}

        #-----------------------------------------------------------------------
        # Compute total loss
        #-----------------------------------------------------------------------
        with tf.variable_scope('total_loss'):
            self.distLoss = tf.add(self.heightDistLossAbs, self.bottomDistLossAbs, name='distLoss')
            self.loss_semDist = tf.add(self.semLoss, self.distLoss, name='loss_semDist')
            self.loss = self.loss_semDist + self.existenceLoss + self.isRELoss
            _variable_summaries(self.semLoss)
            _variable_summaries(self.existenceLoss)
            _variable_summaries(self.isRELoss)
            _variable_summaries(self.distLoss)
            _variable_summaries(self.loss_semDist)
            #self.regLoss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='regLoss')
            #self.regLoss = tf.losses.get_regularization_loss()
            #_variable_summaries(self.regLoss)
            #self.totalLoss = tf.add(self.loss, self.regLoss, name='totalLoss')
            #_variable_summaries(self.totalLoss)


        #-----------------------------------------------------------------------
        # Store the tensors
        #-----------------------------------------------------------------------
        self.losses = {
            'total': self.loss,
            'dist': self.distLoss,
            'conf': self.semLoss,
            'existence': self.existenceLoss,
            'isRE': self.isRELoss,
            'absBottomLoss': self.bottomDistLossAbs,
            'absHeightLoss': self.heightDistLossAbs
        }


