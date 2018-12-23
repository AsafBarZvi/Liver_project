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
from imgaug import augmenters as iaa
#import ipdb


class DataPrep:
    #---------------------------------------------------------------------------
    def __init__(self, dataDir):
        #-----------------------------------------------------------------------
        # Prepare the dataset info
        #-----------------------------------------------------------------------
        try:

            trainImagesDir = dataDir + '/train/ct/'
            trainSegsDir = dataDir + '/train/seg/'
            valImagesDir = dataDir + '/val/ct/'
            valSegsDir = dataDir + '/val/seg/'
            trainImagesList = [trainImagesDir + f for f in os.listdir(trainImagesDir) if os.path.isfile(trainImagesDir + f)]
            trainSegsList = [trainSegsDir + f for f in os.listdir(trainSegsDir) if os.path.isfile(trainSegsDir + f)]
            valImagesList = [valImagesDir + f for f in os.listdir(valImagesDir) if os.path.isfile(valImagesDir + f)]
            valSegsList = [valSegsDir + f for f in os.listdir(valSegsDir) if os.path.isfile(valSegsDir + f)]

            augSize = 7

            trainSetSize = len(trainImagesList)
            trainSetSize *= augSize
            valSetSize = len(valImagesList)

            # Shuffle the indexs of the train data set, and create index availbility arrays
            randIdxs = range(trainSetSize)
            shuffle(randIdxs)

            self.train_generator = self.__batch_generator(trainImagesList, trainSegsList, trainSetSize, randIdxs, augSize, True)
            self.valid_generator = self.__batch_generator(valImagesList, valSegsList, valSetSize)

            #-----------------------------------------------------------------------
            # Set the attributes up
            #-----------------------------------------------------------------------
            self.num_train       = trainSetSize
            self.num_valid       = valSetSize


        except Exception as e:
            raise RuntimeError(str(e))


    #---------------------------------------------------------------------------
    def __batch_generator(self, imagesList, segsList, setSize, randIdxs=[], augSize=1, train=False):

        IMAGE_HEIGH = 512
        IMAGE_WIDTH = 512
        NUM_OF_CLASSES = 2

        augTypes = ['noAug', 'flipLR', 'flipUD', 'blur', 'padding', 'brightness', 'contrast']

        #-----------------------------------------------------------------------
        def process_samples(offset, batch_size):

            first = True
            counter = 0
            dataIdx = offset
            while counter < batch_size:
                # If not enough samples to fill the batch
                if dataIdx >= setSize:
                    dataIdx = int(np.random.rand()*setSize)

                if train:
                    idx = randIdxs[dataIdx] % (setSize/augSize)
                    augName = augTypes[dataIdx % len(augTypes)]
                else:
                    idx = dataIdx

                image = cv2.imread(imagesList[idx])
                image = image[:,:,0]
                seg = cv2.imread(segsList[idx])
                seg = seg[:,:,0]

                ## Augmentation
                if augName == 'noAug':
                    imageAug = image
                    segAug = seg
                elif augName == 'flipLR':
                    flipper = iaa.Fliplr(1.0)
                    imageAug = flipper.augment_image(image)
                    segAug = flipper.augment_image(seg)
                elif augName == 'flipUD':
                    flipper = iaa.Flipud(1.0)
                    imageAug = flipper.augment_image(image)
                    segAug = flipper.augment_image(seg)
                elif augName == 'blur':
                    blurer = iaa.GaussianBlur(1.0)
                    imageAug = blurer.augment_image(image)
                    segAug = seg
                elif augName == 'padding':
                    translater = iaa.Affine(translate_percent={"x": (-0.1,0.1), "y": (-0.1,0.1)})
                    imageAug = translater.augment_image(image)
                    segAug = translater.augment_image(seg)
                elif augName == 'brightness':
                    brightnesser = iaa.Multiply((0.5, 2.0))
                    imageAug = brightnesser.augment_image(image)
                    segAug = seg
                elif augName == 'contrast':
                    contraster = iaa.ContrastNormalization((0.5, 2.0))
                    imageAug = contraster.augment_image(image)
                    segAug = seg

                labeledSeg = np.zeros((seg.shape[0], seg.shape[1], NUM_OF_CLASSES), dtype=np.float32)
                #labeledSeg[segAug == 255, 2] = 1
                labeledSeg[segAug == 127, 1] = 1
                labeledSeg[segAug == 0, 0] = 1

                if first:
                    images = imageAug[np.newaxis,:,:,np.newaxis].astype(np.float32)
                    segs = labeledSeg[np.newaxis]
                    first = False
                else:
                    images = np.append(images, imageAug[np.newaxis,:,:,np.newaxis], axis=0)
                    segs = np.append(segs, labeledSeg[np.newaxis], axis=0)

                counter += 1
                dataIdx += 1


            return [images, segs]

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

                images, segs = process_samples(offset, batch_size)
                batch_queue.put(images, segs)

        #-----------------------------------------------------------------------
        def gen_batch(batch_size, num_workers=0):

            #-------------------------------------------------------------------
            # Set up the parallel generator
            #-------------------------------------------------------------------
            if num_workers > 0:
                #---------------------------------------------------------------
                # Set up the queues
                #---------------------------------------------------------------
                images_template = np.zeros((batch_size, IMAGE_HEIGH, IMAGE_WIDTH, 1), dtype=np.float32)
                segs_template = np.zeros((batch_size, IMAGE_HEIGH, IMAGE_WIDTH, NUM_OF_CLASSES), dtype=np.float32)

                max_size = num_workers * 5
                n_batches = int(math.ceil(setSize/batch_size))
                sample_queue = mp.Queue(n_batches)
                batch_queue = DataQueue(images_template, segs_template, max_size)

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
                        images, segs = batch_queue.get()
                    yield images, segs

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
                    images, segs = process_samples(offset, batch_size)
                    yield images, segs

        return gen_batch



