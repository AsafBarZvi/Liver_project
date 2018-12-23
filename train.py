import math
import sys
import os
import shutil
import tensorflow as tf
import numpy as np
from dataPrep import DataPrep
from utils import *
from tqdm import tqdm
from timer import timer_dict, timerStats
from default import config, printCfg
import net
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
def main():
    #---------------------------------------------------------------------------
    # Parse the commandline
    #---------------------------------------------------------------------------
    args = config

    snaps_path = os.path.join(config.snapdir , config.name)
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
        dp = DataPrep(args.data_dir)
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

        #validation_metric_calc = analyze_results.CalculateMetrics(args.batch_size)

        #-----------------------------------------------------------------------
        # Summaries
        #-----------------------------------------------------------------------
        restore = start_epoch != 0

        #training_metric = MetricsSummary(sess, summary_writer, 'training', restore)
        validation_metric = MetricsSummary(sess, summary_writer, 'validation', restore)

        training_imgs = ImageSummary(sess, summary_writer, 'training', restore)
        validation_imgs = ImageSummary(sess, summary_writer, 'validation', restore)

        training_loss = LossSummary(sess, summary_writer, 'training', args.summary_interval)
        validation_loss = LossSummary(sess, summary_writer, 'validation', n_valid_batches)

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
            randTrainBatchIdx = np.random.randint(0, args.summary_interval-2, 10)
            generator = dp.train_generator(args.batch_size, args.num_workers)
            description = '[i] Train {:>2}/{}'.format(e+1, args.epochs)
            for idx, (images, gtSegs) in enumerate(tqdm(generator, total=n_train_batches, initial=start_idx, desc=description, unit='batches', leave=False), start=start_idx):

                with timer_dict['train']:
                    [_, loss, segPrediction] = sess.run([train_step, ucrNet.loss, ucrNet.segPrediction], feed_dict={ucrNet.image: images, ucrNet.gtSeg: gtSegs})

                    #print loss

                training_loss.add(loss)

                if idx in randTrainBatchIdx:
                    if args.batch_size > 1:
                        randTrainFrameIdx = np.random.randint(0,args.batch_size-1,1)[0]
                    else:
                        randTrainFrameIdx = 0
                    with timer_dict['summary']:
                        training_imgs_samples.append((np.copy(images[randTrainFrameIdx,:,:,0]), np.copy(segPrediction[randTrainFrameIdx,:,:,:])))

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

                randTrainBatchIdx = np.random.randint(idx+1, idx+args.summary_interval-2, 10)

                #-------------------------------------------------------------------
                # Validate
                #-------------------------------------------------------------------
                if (iteration % args.val_interval) != 0:
                    continue

                randValidBatchIdx = np.random.randint(0, n_valid_batches-1, 10)
                generator = dp.valid_generator(args.batch_size, args.num_workers)
                description = '[i] Valid {:>2}/{}'.format(e+1, args.epochs)
                for idxTest, (images, gtSegs) in enumerate(tqdm(generator, total=n_valid_batches, desc=description, unit='batches', leave=False)):

                    [loss, segPrediction] = sess.run([ucrNet.loss, ucrNet.segPrediction], feed_dict={ucrNet.image: images, ucrNet.gtSeg: gtSegs})

                    validation_loss.add(loss)

                    if idxTest in randValidBatchIdx:
                        if args.batch_size > 1:
                            randValidFrameIdx = np.random.randint(0,args.batch_size-1,1)[0]
                        else:
                            randValidFrameIdx = 0
                        with timer_dict['summary']:
                            validation_imgs_samples.append((np.copy(images[randValidFrameIdx,:,:,0]), np.copy(segPrediction[randValidFrameIdx,:,:,:])))

                        #timerStats()

                #-------------------------------------------------------------------
                # Write summaries
                #-------------------------------------------------------------------
                validation_loss.push(iteration)

                summary = sess.run(merged_summary, feed_dict={ucrNet.image: images,
                                                              ucrNet.gtSeg: gtSegs})
                summary_writer.add_summary(summary, iteration)

                #validation_metric.push(iteration, precision, recall, Dice)

                validation_imgs.push(iteration, validation_imgs_samples)
                validation_imgs_samples = []

                summary_writer.flush()

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
        checkpoint = '{}/final.ckpt'.format(snaps_path)
        model_saver.save(sess, checkpoint)
        print('[i] Checkpoint saved:' + checkpoint)

    return 0

if __name__ == '__main__':
    sys.exit(main())
