import tensorflow as tf
import os
import numpy as np
import math
import cv2
import multiprocessing as mp
from random import shuffle
import Queue as q
from timer import timer_dict
from data_queue import DataQueue
import blob2mat
#import ipdb


class DataPrep:
    #---------------------------------------------------------------------------
    def __init__(self, dataDir, dataTypeRatio):
        #-----------------------------------------------------------------------
        # Prepare the dataset info
        #-----------------------------------------------------------------------
        try:

            framesFileTrain = []
            linesFileTrain = []
            semanticFileTrain = []
            clipIdxFileTrain = []
            originFileTrain = []

            framesFileTest = []
            linesFileTest = []
            semanticFileTest = []
            clipIdxFileTest = []
            originFileTest = []

            dataTypes = ['main', 'snow', 'UTAC_ALL', 'left_right', 'front_rear', 'front_corners', 'rear_corners']
            dataTypeRatio = dataTypeRatio.split(',')
            dataTypeRatio = [float(dt) for dt in dataTypeRatio]
            #dataTypeRatio = [1, 4, 1, 0, 0, 0, 0] # '1' means using the initial set size without boostig, '0' means data type is unused
            dataTypeAvail = np.ones((len(dataTypeRatio)), dtype=int)
            dataTypeAvail[np.array(dataTypeRatio) == 0.] = 0
            dataTypeAvailTest = np.array([1, 0, 0, 0, 0, 0, 0])
            print "Data types: 'main', 'snow', 'UTAC', 'left_right', 'front_rear', 'front_corners', 'rear_corners'"
            print "Data type in use for train: {}".format(dataTypeAvail)
            print "Data type in use for test: {}".format(dataTypeAvailTest)

            trainDataTypeSetSizes = []
            testDataTypeSetSizes = []
            trainSetSize = 0
            testSetSize = 0
            for idx, dataType in enumerate(dataTypes):

                framesFileTrain.append('./{}/train/{}/000000.data_lm1.bin'.format(dataDir, dataType))
                linesFileTrain.append('./{}/train/{}/000000.fsPrfFloat.bin'.format(dataDir, dataType))
                semanticFileTrain.append('./{}/train/{}/000000.fsPrfIntOpt1.bin'.format(dataDir, dataType))
                clipIdxFileTrain.append('./{}/train/{}/000000.clip_ind.bin'.format(dataDir, dataType))
                originFileTrain.append('./{}/train/{}/000000.origin_lm1.bin'.format(dataDir, dataType))

                framesFileTest.append('./{}/test/{}/000000.data_lm1.bin'.format(dataDir, dataType))
                linesFileTest.append('./{}/test/{}/000000.fsPrfFloat.bin'.format(dataDir, dataType))
                semanticFileTest.append('./{}/test/{}/000000.fsPrfIntOpt1.bin'.format(dataDir, dataType))
                clipIdxFileTest.append('./{}/test/{}/000000.clip_ind.bin'.format(dataDir, dataType))
                originFileTest.append('./{}/test/{}/000000.origin_lm1.bin'.format(dataDir, dataType))

                trainDataTypeSetSize = os.path.getsize(clipIdxFileTrain[idx])/4
                testDataTypeSetSize = os.path.getsize(clipIdxFileTest[idx])/4
                trainDataTypeSetSizes.append(int(trainDataTypeSetSize * dataTypeRatio[idx]))
                testDataTypeSetSizes.append(testDataTypeSetSize * dataTypeAvailTest[idx])
                trainSetSize += trainDataTypeSetSizes[idx]
                testSetSize += testDataTypeSetSizes[idx]

            trainDataTypeRate = np.array(trainDataTypeSetSizes, dtype=np.float32) / trainSetSize
            testDataTypeRate = np.array(testDataTypeSetSizes, dtype=np.float32) / testSetSize

            # Shuffle the indexs of the train data set, and create index availbility arrays
            idxsList = range(trainSetSize)
            idxsListTest = range(testSetSize)
            randIdxs = range(trainSetSize)
            shuffle(randIdxs)
            dataTypeRateAvailIdxsTrain = []
            dataTypeRateAvailIdxsTest = []
            for idx in range(len(dataTypes)):
                if dataTypeAvail[idx] == 0:
                    dataTypeRateAvailIdxsTrain.append(np.array([]))
                else:
                    availIdx = (np.array(idxsList)*trainDataTypeRate[idx]).astype(int)
                    availIdxRoll = np.roll(availIdx, 1)
                    availIdxRoll[0] = -1
                    availIdxRoll = availIdxRoll - availIdx
                    availIdx[availIdxRoll == 0] = -1
                    availIdx[availIdx > -1] = (availIdx[availIdx > -1]/dataTypeRatio[idx]).astype(int)
                    availIdx = availIdx[randIdxs]
                    dataTypeRateAvailIdxsTrain.append(availIdx)

                if dataTypeAvailTest[idx] == 0:
                    dataTypeRateAvailIdxsTest.append(np.array([]))
                else:
                    availIdx = (np.array(idxsListTest)*testDataTypeRate[idx]).astype(int)
                    availIdxRoll = np.roll(availIdx, 1)
                    availIdxRoll[0] = -1
                    availIdxRoll = availIdxRoll - availIdx
                    availIdx[availIdxRoll == 0] = -1
                    dataTypeRateAvailIdxsTest.append(availIdx)

            # Create unified array of data indexes for the multiprocess nbatch generator
            finalDataIdxTrain = np.full((2, len(idxsList)), -1, dtype=int)
            finalDataIdxTest = np.full((2, len(idxsListTest)), -1, dtype=int)
            relIdxTrain = 0
            relIdxTest = 0
            for idx in range(len(idxsList)):
                for dataTypeIdx in range(len(dataTypes)):
                    if dataTypeAvail[dataTypeIdx]:
                        if not dataTypeRateAvailIdxsTrain[dataTypeIdx][idx] == -1:
                            finalDataIdxTrain[0, relIdxTrain] = dataTypeRateAvailIdxsTrain[dataTypeIdx][idx]
                            finalDataIdxTrain[1, relIdxTrain] = dataTypeIdx
                            relIdxTrain += 1
                    if dataTypeAvailTest[dataTypeIdx]:
                        if idx < len(idxsListTest):
                            if not dataTypeRateAvailIdxsTest[dataTypeIdx][idx] == -1:
                                finalDataIdxTest[0, relIdxTest] = dataTypeRateAvailIdxsTest[dataTypeIdx][idx]
                                finalDataIdxTest[1, relIdxTest] = dataTypeIdx
                                relIdxTest += 1

            self.clipNamesMain = [line.rstrip()[:-6] for line in open(dataDir + '/main_clip_list_18_5_18.txt').readlines()]
            self.clipNamesSnow = [line.rstrip()[:-6] for line in open(dataDir + '/snow_clip_list_20_11_18.txt').readlines()]
            self.clipNamesUtac = [line.rstrip()[:-6] for line in open(dataDir + '/utac_clip_list_17_12_18.txt').readlines()]
            self.clipNamesParking = [line.rstrip()[:-6] for line in open(dataDir + '/parking_clip_list_21_10_18.txt').readlines()]

            self.blackList = [int(line.rstrip()) for line in open('./blackList.txt').readlines()]
            self.train_generator = self.__batch_generator(framesFileTrain, linesFileTrain, semanticFileTrain, clipIdxFileTrain, originFileTrain, trainSetSize, finalDataIdxTrain, True)
            self.valid_generator = self.__batch_generator(framesFileTest, linesFileTest, semanticFileTest, clipIdxFileTest, originFileTest, testSetSize, finalDataIdxTest)

            #-----------------------------------------------------------------------
            # Set the attributes up
            #-----------------------------------------------------------------------
            self.num_train       = trainSetSize
            self.num_valid       = testSetSize


        except Exception as e:
            raise RuntimeError(str(e))


    #---------------------------------------------------------------------------
    def __batch_generator(self, framesFile, linesFile, semanticFile, clipIdxFile, originFile, setSize, dataIdxsArray, train=False):

        FRAME_WIDTH = 640
        FRAME_HEIGH = 256
        LINES_SIZE = 480
        SEMANTICS_SIZE = [160,9]

        #-----------------------------------------------------------------------
        def process_samples(offset, batch_size):

            first = True
            validSample = True
            counter = 0
            dataIdx = offset
            while counter < batch_size:
                # If not enough samples to fill the batch or invalid sample
                if dataIdx >= setSize or not validSample:
                    dataIdx = int(np.random.rand()*setSize)

                idx = dataIdxsArray[0, dataIdx]
                dataTypeIdx = dataIdxsArray[1, dataIdx]

                frame = blob2mat.blob2mat(framesFile[dataTypeIdx], 1, 1, FRAME_WIDTH, FRAME_HEIGH, np.uint8, True, idx)
                line = blob2mat.blob2mat(linesFile[dataTypeIdx], 1, 1, 960, 1, np.float32, True, idx)
                if dataTypeIdx == 1 or dataTypeIdx == 2:
                    semantic = blob2mat.blob2mat(semanticFile[dataTypeIdx], 1, 1, 1120, 1, np.int8, True, idx)
                else:
                    semantic = blob2mat.blob2mat(semanticFile[dataTypeIdx], 1, 1, 960, 1, np.int8, True, idx)
                if not frame.any() or not line.any():
                    print "Invalid data reading!"
                    continue
                clipIdx = blob2mat.blob2mat(clipIdxFile[dataTypeIdx], 1, 1, 1, 1, np.int32, True, idx)
                origin = blob2mat.blob2mat(originFile[dataTypeIdx], 1, 2, 1, 1, np.int32, True, idx)

                # Check sample relevancy for train if the sample have enough valid pixels
                if dataTypeIdx > 2:
                    numOfNonZeroVal = sum(line[320:640] < 99)
                else:
                    numOfNonZeroVal = sum(line[:320] > -1)

                # Check data type black list
                blackListValid = True
                if dataTypeIdx == 0:
                    if train and idx in self.blackList:
                        blackListValid = False

                if numOfNonZeroVal > 50 and blackListValid:
                    validSample = True
                    lineBottom = line[:320]
                    lineBottomMax = line[320:640]
                    lineHeightMin = line[640:800]
                    lineHeightMax = line[800:960]
                    lineHeightDelta = line[640:800]
                    '''
                    FS multiclass represntation: 0-noFS, 1-genObj, 2-GR, 3-CC, 4-CU, 5-FL, 6-SN, 7-infFS, 8-NS(no semantics marked)
                    **Height value '100' means, height ignored during training**
                    '''
                    semanticMat = np.zeros((160,9), dtype=np.float32)
                    if dataTypeIdx > 2:
                        for xLoc in range(160):
                            if lineBottomMax[xLoc*2] < 99 and lineBottom[xLoc*2] > -5:
                                if semantic[xLoc] == 0:
                                    semanticMat[xLoc,0] = 1
                                    print "Invalid state occure, line bottom max have a valid value while the line got inf semantics..."
                                    continue
                                # Parking genobj
                                if semantic[xLoc] == 1:
                                    semanticMat[xLoc,1] = 1
                                    #if (lineBottom[xLoc] == lineBottom[xLoc+1] and semantic[xLoc+1] == 1) or (lineBottom[xLoc] == lineBottom[xLoc-1] and semantic[xLoc-1] == 1):
                                    #    semanticMat[xLoc,8] = 1
                                    #else:
                                    #    semanticMat[xLoc,1] = 1
                                    continue
                                # Parking re
                                if semantic[xLoc] == -1:
                                    semanticMat[xLoc,8] = 1
                                    lineHeightDelta[xLoc] = 100.
                            else:
                                semanticMat[xLoc,7] = 1

                    else:
                        for xLoc in range(160):
                            # If line bottom value is above -2, FS line exist with or without semantics
                            if lineBottom[xLoc*2] > -2:
                                if semantic[xLoc] == 1:
                                    semanticMat[xLoc,1] = 1
                                    continue
                                if semantic[320+xLoc] == 1:
                                    if lineHeightMax[xLoc] != lineHeightMin[xLoc]:
                                        lineHeightDelta[xLoc] = 100.
                                    semanticMat[xLoc,2] = 1
                                    continue
                                if semantic[480+xLoc] == 1:
                                    if lineHeightMax[xLoc] != lineHeightMin[xLoc]:
                                        lineHeightDelta[xLoc] = 100.
                                    semanticMat[xLoc,3] = 1
                                    continue
                                if semantic[640+xLoc] == 1:
                                    if lineHeightMax[xLoc] != lineHeightMin[xLoc]:
                                        lineHeightDelta[xLoc] = 100.
                                    semanticMat[xLoc,4] = 1
                                    continue
                                if semantic[800+xLoc] == 1:
                                    semanticMat[xLoc,5] = 1
                                    continue
                                if dataTypeIdx == 1 and semantic[960+xLoc] == 1:
                                    semanticMat[xLoc,6] = 1
                                    continue
                                semanticMat[xLoc,8] = 1
                                if lineHeightMax[xLoc] != lineHeightMin[xLoc] or lineHeightMax[xLoc] > 0.99:
                                    lineHeightDelta[xLoc] = 100.
                            else:
                                # If bottom max is -1 or less, no FS is available
                                if lineBottomMax[xLoc*2] <= -1.0:
                                    semanticMat[xLoc,0] = 1
                                    continue
                                # If bottom max is greater then 900 (999 most of the time), FS is infinity
                                if lineBottomMax[xLoc*2] > 900:
                                    semanticMat[xLoc,7] = 1
                                    continue
                                # For any other case (error in marking) use infFS as it ignore for regression
                                semanticMat[xLoc,7] = 1

                        # Fix semantics to 'inf' for large depth jumps
                        for xLoc in range(1,160):
                            if (max(semanticMat[xLoc,1:6]) > 0 and max(semanticMat[xLoc-1,1:6]) > 0) or (semanticMat[xLoc,8] == 1 and semanticMat[xLoc-1,8] == 1):
                                if (np.argmax(semanticMat[xLoc,1:6]) != np.argmax(semanticMat[xLoc-1,1:6]) or semanticMat[xLoc,8] == 1) and np.abs(lineBottom[xLoc*2]-lineBottom[xLoc*2-1]) > 0.1:
                                    semanticMat[xLoc-2:xLoc+2,:] = np.array([0,0,0,0,0,0,0,1,0])


                    # Extract the clip's name from the fits list
                    if dataTypeIdx == 0:
                        clipName = 'main-{}-{}'.format(clipIdx, self.clipNamesMain[clipIdx])
                    elif dataTypeIdx == 1:
                        clipName = 'snow-{}-{}'.format(clipIdx, self.clipNamesSnow[clipIdx])
                    elif dataTypeIdx == 2:
                        clipName = 'UTAC-{}-{}'.format(clipIdx, self.clipNamesUtac[clipIdx])
                    else:
                        clipName = 'parking-{}-{}'.format(clipIdx, self.clipNamesParking[clipIdx])

                    if first:
                        line = np.concatenate((lineBottom, lineHeightDelta))
                        lines = line[np.newaxis].astype(np.float32)
                        frames = frame[np.newaxis,:,:,np.newaxis].astype(np.float32)
                        semantics = semanticMat[np.newaxis]
                        clipNames = np.array(clipName)
                        origins = origin[np.newaxis]
                        first = False
                    else:
                        line = np.concatenate((lineBottom, lineHeightDelta))
                        lines = np.append(lines, line[np.newaxis], axis=0)
                        frames = np.append(frames, frame[np.newaxis,:,:,np.newaxis], axis=0)
                        semantics = np.append(semantics, semanticMat[np.newaxis], axis=0)
                        clipNames = np.append(clipNames, clipName)
                        origins = np.append(origins, origin[np.newaxis], axis=0)

                    counter += 1
                    dataIdx += 1

                else:
                    validSample = False


            return [frames, lines, semantics, clipNames, origins]

        #-----------------------------------------------------------------------
        def batch_producer(sample_queue, batch_queue, batch_size):
            while True:
                #---------------------------------------------------------------
                # Process the sample
                #---------------------------------------------------------------
                try:
                    offset = sample_queue.get(timeout=1)
                except q.Empty:
                    break

                frames, lines, semantics, clipNames, origins = process_samples(offset, batch_size)
                batch_queue.put(frames, lines, semantics, clipNames, origins)

        #-----------------------------------------------------------------------
        def gen_batch(batch_size, num_workers=0):

            #-------------------------------------------------------------------
            # Set up the parallel generator
            #-------------------------------------------------------------------
            if num_workers > 0:
                #---------------------------------------------------------------
                # Set up the queues
                #---------------------------------------------------------------
                frames_template = np.zeros((batch_size, FRAME_HEIGH, FRAME_WIDTH, 1), dtype=np.float32)
                lines_template = np.zeros((batch_size, LINES_SIZE), dtype=np.float32)
                semantics_template = np.zeros((batch_size, SEMANTICS_SIZE[0], SEMANTICS_SIZE[1]), dtype=np.float32)
                clipNames_template = np.empty((batch_size), dtype='S200')
                origins_template = np.zeros((batch_size, 2), dtype=np.int32)

                max_size = num_workers * 5
                n_batches = int(math.ceil(setSize/batch_size))
                sample_queue = mp.Queue(n_batches)
                batch_queue = DataQueue(frames_template, lines_template, semantics_template, clipNames_template, origins_template, max_size)

                #---------------------------------------------------------------
                # Set up the workers. Make sure we can fork safely even if
                # OpenCV has been compiled with CUDA and multi-threading
                # support.
                #---------------------------------------------------------------
                workers = []
                os.environ['CUDA_VISIBLE_DEVICES'] = ""
                cv2_num_threads = cv2.getNumThreads()
                cv2.setNumThreads(1)
                for i in range(num_workers):
                    args = (sample_queue, batch_queue, batch_size)
                    w = mp.Process(target=batch_producer, args=args)
                    workers.append(w)
                    w.start()

                del os.environ['CUDA_VISIBLE_DEVICES']
                cv2.setNumThreads(cv2_num_threads)

                #---------------------------------------------------------------
                # Fill the sample queue with data
                #---------------------------------------------------------------
                for offset in range(0, setSize, batch_size):
                    sample_queue.put(offset)

                #---------------------------------------------------------------
                # Return the data
                #---------------------------------------------------------------
                for offset in range(0, setSize, batch_size):
                    with timer_dict['batch_get']:
                        frames, lines, semantics, clipNames, origins = batch_queue.get()
                    yield frames, lines, semantics, clipNames, origins

                #---------------------------------------------------------------
                # Join the workers
                #---------------------------------------------------------------
                for w in workers:
                    w.join()

            #-------------------------------------------------------------------
            # Return a serial generator
            #-------------------------------------------------------------------
            else:
                for offset in range(0, setSize, batch_size):
                    frames, lines, semantics, clipNames, origins = process_samples(offset, batch_size)
                    yield frames, lines, semantics, clipNames, origins

        return gen_batch



