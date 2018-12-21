import tensorflow as tf
import os
import numpy as np
#import ipdb


class DataPrep:
    #---------------------------------------------------------------------------
    def __init__(self, data_dir):
        #-----------------------------------------------------------------------
        # Read the dataset info
        #-----------------------------------------------------------------------
        try:
            self.framesFileTrain = os.readlink(data_dir + '/000000.data_lm1')
            self.linesFileTrain = os.readlink(data_dir + '/000000.fsPrfFloat')
            self.semanticFileTrain = os.readlink(data_dir + '/000000.fsPrfIntOpt1')

            self.framesFileTest = os.readlink(data_dir + '/000000.data_lm1_test')
            self.linesFileTest = os.readlink(data_dir + '/000000.fsPrfFloat_test')
            self.semanticFileTest = os.readlink(data_dir + '/000000.fsPrfIntOpt1_test')

            self.trainSet_size = 1000000
            self.testSet_size = 200000

        except Exception as e:
            raise RuntimeError(str(e))


    def run(self, batch_size):

        def blob2mat(file_name, count, channels, width, height, data_type, to_squeeze=True, offset=0):
            """
            Read a numpy array of type data_type from a binary file.
            The array will have the shape (count, channels, width, height).
            Finally, unused dimensions are optionally squeezed out
            :param file_name: the file name
            :param count: number of samples to read. If count=-1, read entire file
            :param channels: channels in each sample (e.g, we usually look at Rects as samples with width and height 1, and 4
            channels)
            :param width: width of sample
            :param height: height of sample
            :param data_type: one of numpy's datatypes, e.g. np.float32
            :param to_squeeze: Set to true (default) if you want to discard unused dimensions
            :return: the resulting numpy array
            """
            if count >= 0:
                primitives_to_read = count * channels * width * height
            else:
                primitives_to_read = -1
            # read raw data from file
            with open(file_name, 'rb') as fid:
                fid.seek(offset * channels * width * height * np.dtype(data_type).itemsize)
                blob = np.fromfile(fid, data_type, primitives_to_read)
            # reshape to desired size
            blob.shape = (-1, channels, height, width)
            return blob.squeeze() if to_squeeze else blob


        def load_data_set(framesFile, linesFile, semanticFile, size):

            FRAME_WIDTH = 640
            FRAME_HEIGH = 256
            #NORM_FACTOR = 87.5
            #originY = 176
            #lines = blob2mat(linesFile, size, 1, 960, 1, np.float32, True, 0)
            #semantics = blob2mat(semanticFile, size, 1, 960, 1, np.int8, True, 0)
            first = True
            for dataIdx in range(size):
                frame = blob2mat(framesFile, 1, 1, FRAME_WIDTH, FRAME_HEIGH, np.uint8, True, dataIdx)
                line = blob2mat(linesFile, 1, 1, 960, 1, np.float32, True, dataIdx)
                semantic = blob2mat(semanticFile, 1, 1, 960, 1, np.int8, True, dataIdx)
                numOfNonZeroVal = sum(line[:320]>0)
                if numOfNonZeroVal > 50:
                    '''
                    FS multiclass represntation: 0-noFS, 1-genObj, 2-GR, 3-CC, 4-CU, 5-GR
                    '''
                    semanticMat = np.zeros((160,6), dtype=np.float32)
                    for xLoc in range(160):
                        if semantic[xLoc] == 1:
                            semanticMat[xLoc,1] = 1
                            continue
                        if semantic[320+xLoc] == 1:
                            semanticMat[xLoc,2] = 1
                            continue
                        if semantic[480+xLoc] == 1:
                            semanticMat[xLoc,3] = 1
                            continue
                        if semantic[640+xLoc] == 1:
                            semanticMat[xLoc,4] = 1
                            continue
                        if semantic[800+xLoc] == 1:
                            semanticMat[xLoc,5] = 1
                            continue
                        else:
                            semanticMat[xLoc,0] = 1

                    #lineBottom = line[:320]*NORM_FACTOR-NORM_FACTOR+originY
                    #lineHeight = (line[:320][::2]+line[640:800])*NORM_FACTOR-NORM_FACTOR+originY
                    lineBottom = line[:320]
                    lineHeightDelta = line[640:800]
                    line = np.concatenate((lineBottom, lineHeightDelta))
                    if first:
                        lines = line[np.newaxis].astype(np.float32)
                        frames = frame[np.newaxis].astype(np.float32)
                        semantics = semanticMat[np.newaxis]
                        first = False
                    else:
                        lines = np.append(lines, line[np.newaxis], axis=0)
                        frames = np.append(frames, frame[np.newaxis], axis=0)
                        semantics = np.append(semantics, semanticMat[np.newaxis], axis=0)

            return [frames, lines, semantics]


        def next_batch(input_queue, batchSize):
            min_queue_examples = 256
            num_threads = 8
            frameTensor = input_queue[0]
            lineTensor = input_queue[1]
            semTensor = input_queue[2]
            [frameBatch, lineBatch, semBatch] = tf.train.shuffle_batch(
                [frameTensor, lineTensor, semTensor],
                batch_size=batchSize,
                num_threads=num_threads,
                capacity=min_queue_examples+(num_threads+2)*batchSize,
                seed=88,
                min_after_dequeue=min_queue_examples)
            return [frameBatch, lineBatch, semBatch]



        [trainFrames, trainLines, trainSem] = load_data_set(self.framesFileTrain, self.linesFileTrain, self.semanticFileTrain, self.trainSet_size)
        [testFrames, testLines, testSem] = load_data_set(self.framesFileTest, self.linesFileTest, self.semanticFileTest, self.testSet_size)

        trainFramesTensors = tf.convert_to_tensor(trainFrames, dtype=tf.float32)
        trainLinesTensors = tf.convert_to_tensor(trainLines, dtype=tf.float32)
        trainSemTensors = tf.convert_to_tensor(trainSem, dtype=tf.float32)
        testFramesTensors = tf.convert_to_tensor(testFrames, dtype=tf.float32)
        testLinesTensors = tf.convert_to_tensor(testLines, dtype=tf.float32)
        testSemTensors = tf.convert_to_tensor(testSem, dtype=tf.float32)

        trainInputQueue = tf.train.slice_input_producer([trainFramesTensors, trainLinesTensors, trainSemTensors], shuffle=True)
        testInputQueue = tf.train.slice_input_producer([testFramesTensors, testLinesTensors, testSemTensors], shuffle=False)
        self.batch_queue = next_batch(trainInputQueue, batch_size)
        self.batch_test_queue = next_batch(testInputQueue, batch_size)

        #-----------------------------------------------------------------------
        # Set the attributes up
        #-----------------------------------------------------------------------
        self.num_train       = trainFrames.shape[0]
        self.num_valid       = testFrames.shape[0]




