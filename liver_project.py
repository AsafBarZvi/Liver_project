import math
import sys
import os
import shutil
import tensorflow as tf
import numpy as np
import random
from dataPrep import DataPrep
from utils import *
from tqdm import tqdm
from timer import timer_dict, timerStats
from default import config, printCfg
from metrics import lp_metrics
import net
#import net_onlyUnet as net
#import ipdb

config.__dict__.update()
config.name = sys.argv[1]
availableGPU = config.gpu
if availableGPU != 'CPU':
    if availableGPU == None:
        for gpuId in range(4):
            if int(os.popen("nvidia-smi -i {} -q --display=MEMORY | grep -m 1 Free | grep -o '[0-9]*'".format(gpuId)).readlines()[0]) < 1000:
                continue
            availableGPU = gpuId
            break
    if availableGPU == None:
        print "No available GPU device!"
        sys.exit(1)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(availableGPU)
print "{} \n {}".format(sys.argv[1],printCfg())
print('[i] Runing on GPU: {}'.format(availableGPU))

#-------------------------------------------------------------------------------
def compute_lr(lr_values, lr_boundaries):
    with tf.variable_scope('learning_rate'):
        global_step = tf.Variable(0, trainable=False, name='global_step')
        lr = tf.train.piecewise_constant(global_step, lr_boundaries, lr_values)
    return lr, global_step

#-------------------------------------------------------------------------------
def train(train_data_dir, val_data_dir, out_weight_dir):
    #---------------------------------------------------------------------------
    # Parse the commandline
    #---------------------------------------------------------------------------
    args = config

    snaps_path = os.path.join(out_weight_dir , config.name)
    #---------------------------------------------------------------------------
    # Find an existing checkpoint
    #---------------------------------------------------------------------------
    start_epoch = 0
    start_idx = 0
    checkpoint_file = args.checkpoint_file
    if args.continue_training:

        if checkpoint_file is None:
            print('[!] No checkpoints found, cannot continue!')
            return 1

        metagraph_file = checkpoint_file + '.meta'

        if not os.path.exists(metagraph_file):
            print('[!] Cannot find metagraph'.format(metagraph_file))
            return 1

        step = os.path.basename(checkpoint_file).split('.')[0]
        start_epoch = int(step.split('_')[0]) - 1
        start_idx = int(step.split('_')[1]) + 1

    #---------------------------------------------------------------------------
    # Create a project directory
    #---------------------------------------------------------------------------
    else:
        try:
            print('[i] Creating directory {}...'.format(snaps_path))
            os.makedirs(snaps_path)
        except Exception as e:
            print('[!] {}'.format( str(e)))
            #return 1

    print('[i] Starting at epoch: {}'.format( start_epoch+1))

    #---------------------------------------------------------------------------
    # Configure the training data
    #---------------------------------------------------------------------------
    print('[i] Configuring the training data...')
    try:
        dp = DataPrep(train_data_dir, val_data_dir)
        print('[i] # training samples: {}'.format(dp.num_train))
        print('[i] # validation samples: {}'.format(dp.num_valid))
        print('[i] # batch size train: {}'.format(args.batch_size))
    except (AttributeError, RuntimeError) as e:
        print('[!] Unable to load training data: ' + str(e))
        return 1

    #---------------------------------------------------------------------------
    # Create the network
    #---------------------------------------------------------------------------
    with tf.Session() as sess:
        print('[i] Creating the model...')
        n_train_batches = int(math.ceil(dp.num_train/args.batch_size))
        n_valid_batches = int(math.ceil(dp.num_valid/args.batch_size))

        lr_values = args.lr_values.split(';')
        try:
            lr_values = [float(x) for x in lr_values]
        except ValueError:
            print('[!] Learning rate values must be floats')
            sys.exit(1)

        lr_boundaries = args.lr_boundaries.split(';')
        try:
            lr_boundaries = [int(x)*n_train_batches for x in lr_boundaries]
        except ValueError:
            print('[!] Learning rate boundaries must be ints')
            sys.exit(1)

        ret = compute_lr(lr_values, lr_boundaries)
        learning_rate, global_step = ret

        ucrNet = net.UnetCrfRnn(args.batch_size, args.weight_decay, args.bias_decay)

        with tf.variable_scope('train_step'):
            train_step = tf.train.AdamOptimizer(learning_rate, args.momentum).minimize( \
                    ucrNet.loss, global_step=global_step, name='train_step')

        init = tf.global_variables_initializer()
        init_local = tf.local_variables_initializer()
        sess.run(init)
        sess.run(init_local)

        if (start_epoch != 0) or not checkpoint_file is None:
            try:
                initialize_variables_from_ckpt(sess, checkpoint_file)
                #saver = tf.train.Saver()
                #saver.restore(sess, checkpoint_file)
            except Exception as E:
                print E

        model_saver = tf.train.Saver(max_to_keep=args.max_snapshots_keep)

        if (not checkpoint_file is None) and not args.continue_training:
            sess.run([tf.assign(global_step,0)])

        initialize_uninitialized_variables(sess)

        #-----------------------------------------------------------------------
        # Create various helpers
        #-----------------------------------------------------------------------
        if not os.path.exists(args.logdir):
            os.mkdir(args.logdir)
        if os.path.exists(os.path.join(args.logdir , config.name)):
            shutil.rmtree(os.path.join(args.logdir , config.name))
            os.mkdir(os.path.join(args.logdir , config.name))
        else:
            os.mkdir(os.path.join(args.logdir , config.name))

        merged_summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(os.path.join(args.logdir , config.name), sess.graph)

        #-----------------------------------------------------------------------
        # Summaries
        #-----------------------------------------------------------------------
        restore = start_epoch != 0

        #training_metric = MetricsSummary(sess, summary_writer, ['dice', 'precision', 'recall', 'accuracy'], 1)
        validation_metric_liv = MetricsSummary(sess, summary_writer, ['diceLiv', 'precisionLiv', 'recallLiv', 'accuracyLiv'], 1)
        validation_metric_les = MetricsSummary(sess, summary_writer, ['diceLes', 'precisionLes', 'recallLes', 'accuracyLes'], 1)

        training_imgs = ImageSummary(sess, summary_writer, 'training', restore)
        validation_imgs = ImageSummary(sess, summary_writer, 'validation', restore)

        training_loss = LossSummary(sess, summary_writer, 'training', args.summary_interval)
        validation_loss = LossSummary(sess, summary_writer, 'validation', n_valid_batches)

        #---------------------------------------------------------------------------
        # metrics initilazing
        #---------------------------------------------------------------------------
        #metricsTrainLiver = lp_metrics()
        #metricsTrainLesion = lp_metrics()
        metricsTestLiver = lp_metrics()
        metricsTestLesion = lp_metrics()

        #-----------------------------------------------------------------------
        # Get the initial snapshot of the network
        #-----------------------------------------------------------------------
        #net_summary_ops = net.build_summaries(restore)
        #if start_epoch == 0:
        #    net_summary = sess.run(net_summary_ops)
        #    summary_writer.add_summary(net_summary, 0)
        #summary_writer.flush()

        #-----------------------------------------------------------------------
        # Cycle through the epoch
        #-----------------------------------------------------------------------
        print('[i] Training...')
        if start_idx == n_train_batches:
            start_idx = 0
            start_epoch += 1
        for e in range(start_epoch, args.epochs):
            training_imgs_samples = []
            validation_imgs_samples = []

            #-------------------------------------------------------------------
            # Train
            #-------------------------------------------------------------------
            randTrainBatchIdx = random.sample(range(args.summary_interval-1), 10)
            generator = dp.train_generator(args.batch_size, args.num_workers)
            description = '[i] Train {:>2}/{}'.format(e+1, args.epochs)
            for idx, (images, gtSegs) in enumerate(tqdm(generator, total=n_train_batches, initial=start_idx, desc=description, unit='batches', leave=False), start=start_idx):

                with timer_dict['train']:
                    [_, loss, segPrediction] = sess.run([train_step, ucrNet.loss, ucrNet.segPrediction], feed_dict={ucrNet.image: images, ucrNet.gtSeg: gtSegs})

                training_loss.add(loss)

                if idx in randTrainBatchIdx:
                    if args.batch_size > 1:
                        randTrainFrameIdx = np.random.randint(0,args.batch_size-1,1)[0]
                    else:
                        randTrainFrameIdx = 0
                    with timer_dict['summary']:
                        training_imgs_samples.append((np.copy(images[randTrainFrameIdx,:,:,0]), np.copy(segPrediction[randTrainFrameIdx,:,:]), np.copy(gtSegs[randTrainFrameIdx,:,:,:])))

                    #timerStats()

                iteration = int(tf.train.global_step(sess, global_step))
                if iteration == 0 or ((iteration % args.summary_interval) != 0 and (iteration % args.val_interval) != 0):
                    continue

                #-------------------------------------------------------------------
                # Write summaries
                #-------------------------------------------------------------------
                training_loss.push(iteration)

                summary = sess.run(merged_summary, feed_dict={ucrNet.image: images,
                                                              ucrNet.gtSeg: gtSegs})
                summary_writer.add_summary(summary, iteration)

                training_imgs.push(iteration, training_imgs_samples)
                training_imgs_samples = []

                summary_writer.flush()

                randTrainBatchIdx = random.sample(range(idx+1, idx+args.summary_interval-1), 10)

                #-------------------------------------------------------------------
                # Validate
                #-------------------------------------------------------------------
                if (iteration % args.val_interval) != 0:
                    continue

                randValidBatchIdx = random.sample(range(n_valid_batches-1), 10)
                generator = dp.valid_generator(args.batch_size, args.num_workers)
                description = '[i] Valid {:>2}/{}'.format(e+1, args.epochs)
                for idxTest, (images, gtSegs) in enumerate(tqdm(generator, total=n_valid_batches, desc=description, unit='batches', leave=False)):

                    [loss, segPrediction] = sess.run([ucrNet.loss, ucrNet.segPrediction], feed_dict={ucrNet.image: images, ucrNet.gtSeg: gtSegs})

                    validation_loss.add(loss)
                    liverPrediction = np.zeros((segPrediction.shape), dtype=np.float32)
                    liverPrediction[segPrediction == 1] = 1
                    metricsTestLiver.calc_metrics(gtSegs[:,:,:,1], liverPrediction)
                    lesionPrediction = np.zeros((segPrediction.shape), dtype=np.float32)
                    lesionPrediction[segPrediction == 2] = 1
                    metricsTestLesion.calc_metrics(gtSegs[:,:,:,2], lesionPrediction)

                    if idxTest in randValidBatchIdx:
                        if args.batch_size > 1:
                            randValidFrameIdx = np.random.randint(0,args.batch_size-1,1)[0]
                        else:
                            randValidFrameIdx = 0
                        with timer_dict['summary']:
                            validation_imgs_samples.append((np.copy(images[randValidFrameIdx,:,:,0]), np.copy(segPrediction[randValidFrameIdx,:,:]), np.copy(gtSegs[randValidFrameIdx,:,:,:])))

                        #timerStats()

                #-------------------------------------------------------------------
                # Write summaries
                #-------------------------------------------------------------------
                validation_loss.push(iteration)

                summary = sess.run(merged_summary, feed_dict={ucrNet.image: images,
                                                              ucrNet.gtSeg: gtSegs})
                summary_writer.add_summary(summary, iteration)

                metricsRes = metricsTestLiver.summarize_metrics()
                validation_metric_liv.add([metricsRes['dice'], metricsRes['precision'], metricsRes['recall'], metricsRes['accuracy']])
                validation_metric_liv.push(iteration)

                metricsRes = metricsTestLesion.summarize_metrics()
                validation_metric_les.add([metricsRes['dice'], metricsRes['precision'], metricsRes['recall'], metricsRes['accuracy']])
                validation_metric_les.push(iteration)

                validation_imgs.push(iteration, validation_imgs_samples)
                validation_imgs_samples = []

                summary_writer.flush()
                metricsTestLiver.__init__()
                metricsTestLesion.__init__()

                #-------------------------------------------------------------------
                # Save a checktpoint
                #-------------------------------------------------------------------
                checkpoint = '{}/{}_{}.ckpt'.format(snaps_path, e+1, idx)
                model_saver.save(sess, checkpoint)
                #print('[i] Checkpoint saved: ' + checkpoint)

            start_idx = 0

        #-------------------------------------------------------------------
        # Save final checktpoint
        #-------------------------------------------------------------------
        timerStats()
        checkpoint = '{}/weights_final.ckpt'.format(snaps_path)
        model_saver.save(sess, checkpoint)
        print('[i] Checkpoint saved:' + checkpoint)
        print('[i] Done training...')

    return 0


def predict(test_data_dir, weights_path, out_seg_dir):
    #---------------------------------------------------------------------------
    # Parse the commandline
    #---------------------------------------------------------------------------
    args = config

    #---------------------------------------------------------------------------
    # Find an existing checkpoint
    #---------------------------------------------------------------------------
    checkpoint_file = weights_path + '/weights_final.ckpt'

    if checkpoint_file is None:
        print('[!] No checkpoints found, cannot continue!')
        return 1

    #---------------------------------------------------------------------------
    # Configure the training data
    #---------------------------------------------------------------------------
    print('[i] Configuring the test data...')
    try:
        dp = DataPrep(test_data_dir, test_data_dir, args.testData)
        print('[i] # test samples: {}'.format(dp.num_valid))
        print('[i] # batch size train: {}'.format(args.batch_size))
    except (AttributeError, RuntimeError) as e:
        print('[!] Unable to load data: ' + str(e))
        return 1

    #---------------------------------------------------------------------------
    # Create the network
    #---------------------------------------------------------------------------
    with tf.Session() as sess:
        print('[i] Uploading the model...')
        n_valid_batches = int(math.ceil(dp.num_valid/args.batch_size))
        ucrNet = net.UnetCrfRnn(args.batch_size, args.weight_decay, args.bias_decay)

        init = tf.global_variables_initializer()
        init_local = tf.local_variables_initializer()
        sess.run(init)
        sess.run(init_local)

        if not checkpoint_file is None:
            try:
                initialize_variables_from_ckpt(sess, checkpoint_file)
                #saver = tf.train.Saver()
                #saver.restore(sess, checkpoint_file)
            except Exception as E:
                print E

        #-----------------------------------------------------------------------
        # Create various helpers
        #-----------------------------------------------------------------------
        if not os.path.exists(args.logdir):
            os.mkdir(args.logdir)
        if os.path.exists(os.path.join(args.logdir , config.name)):
            shutil.rmtree(os.path.join(args.logdir , config.name))
            os.mkdir(os.path.join(args.logdir , config.name))
        else:
            os.mkdir(os.path.join(args.logdir , config.name))

        if not os.path.exists(out_seg_dir):
            os.mkdir(out_seg_dir)

        merged_summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(os.path.join(args.logdir , config.name), sess.graph)

        #-----------------------------------------------------------------------
        # Summaries
        #-----------------------------------------------------------------------
        validation_metric_liv = MetricsSummary(sess, summary_writer, ['diceLiv', 'precisionLiv', 'recallLiv', 'accuracyLiv'], 1)
        validation_metric_les = MetricsSummary(sess, summary_writer, ['diceLes', 'precisionLes', 'recallLes', 'accuracyLes'], 1)
        validation_imgs = ImageSummary(sess, summary_writer, 'validation')
        validation_loss = LossSummary(sess, summary_writer, 'validation', n_valid_batches)

        #---------------------------------------------------------------------------
        # metrics initilazing
        #---------------------------------------------------------------------------
        #metricsTrainLiver = lp_metrics()
        #metricsTrainLesion = lp_metrics()
        metricsTestLiver = lp_metrics()
        metricsTestLesion = lp_metrics()

        #-------------------------------------------------------------------
        # Validate
        #-------------------------------------------------------------------
        iteration = 0
        validation_imgs_samples = []
        randValidBatchIdx = random.sample(range(n_valid_batches-1), 10)
        generator = dp.valid_generator(args.batch_size, args.num_workers)
        description = '[i] Valid '
        for idxTest, (images, gtSegs) in enumerate(tqdm(generator, total=n_valid_batches, desc=description, unit='batches', leave=False)):

            if args.testData:
                [segPrediction] = sess.run([ucrNet.segPrediction], feed_dict={ucrNet.image: images})
            else:
                [loss, segPrediction] = sess.run([ucrNet.loss, ucrNet.segPrediction], feed_dict={ucrNet.image: images, ucrNet.gtSeg: gtSegs})

                validation_loss.add(loss)
                liverPrediction = np.zeros((segPrediction.shape), dtype=np.float32)
                liverPrediction[segPrediction == 1] = 1
                metricsTestLiver.calc_metrics(gtSegs[:,:,:,1], liverPrediction)
                lesionPrediction = np.zeros((segPrediction.shape), dtype=np.float32)
                lesionPrediction[segPrediction == 2] = 1
                metricsTestLesion.calc_metrics(gtSegs[:,:,:,2], lesionPrediction)

                if idxTest in randValidBatchIdx:
                    if args.batch_size > 1:
                        randValidFrameIdx = np.random.randint(0,args.batch_size-1,1)[0]
                    else:
                        randValidFrameIdx = 0
                    with timer_dict['summary']:
                        validation_imgs_samples.append((np.copy(images[randValidFrameIdx,:,:,0]), np.copy(segPrediction[randValidFrameIdx,:,:]), np.copy(gtSegs[randValidFrameIdx,:,:,:])))

                    #timerStats()

            #-------------------------------------------------------------------
            # Dump segmented images
            #-------------------------------------------------------------------
            for segNum in range(gtSegs.shape[0]):
                segToSave = np.zeros((segPrediction.shape[1], segPrediction.shape[2]), dtype=np.uint8)
                segToSave[segPrediction[segNum] == 1] = 127
                segToSave[segPrediction[segNum] == 2] = 255
                cv2.imwrite('{}/{}.png'.format(out_seg_dir, idxTest*gtSegs.shape[0]+segNum), segToSave)

        #-------------------------------------------------------------------
        # Write summaries
        #-------------------------------------------------------------------
        if not args.testData:

            validation_loss.push(iteration)

            summary = sess.run(merged_summary, feed_dict={ucrNet.image: images,
                                                          ucrNet.gtSeg: gtSegs})
            summary_writer.add_summary(summary, iteration)

            metricsRes = metricsTestLiver.summarize_metrics()
            validation_metric_liv.add([metricsRes['dice'], metricsRes['precision'], metricsRes['recall'], metricsRes['accuracy']])
            validation_metric_liv.push(iteration)
            print "Liver segmentation results: {}".format(metricsRes)

            metricsRes = metricsTestLesion.summarize_metrics()
            validation_metric_les.add([metricsRes['dice'], metricsRes['precision'], metricsRes['recall'], metricsRes['accuracy']])
            validation_metric_les.push(iteration)
            print "Lesion segmentation results: {}".format(metricsRes)

            validation_imgs.push(iteration, validation_imgs_samples)
            validation_imgs_samples = []

            summary_writer.flush()

        print('[i] Done testing...')


    return 0



if __name__ == '__main__':

    key = int(raw_input("Please type: train only - 1, both train and predict - 2, predict only - 3\n"))
    if key < 3:
        train_data_dir = config.train_data_dir
        val_data_dir = config.val_data_dir
        out_weight_dir = config.snapdir

        tf.reset_default_graph()
        status = train(train_data_dir, val_data_dir, out_weight_dir)
        if status == 1:
            sys.exit(status)

    if key > 1:
        test_data_dir = config.val_data_dir
        if key == 2:
            weights_path = config.snapdir + '/' + config.name
        else:
            weights_path = config.weight_path_test
        out_seg_dir = config.out_seg_dir

        tf.reset_default_graph()
        status = predict(test_data_dir, weights_path, out_seg_dir)
        if status == 1:
            sys.exit(status)




