from __future__ import print_function
import argparse
from datetime import datetime
import os
import sys
import time
import json
import time

import tensorflow as tf
from reader import create_inputs
from espcn import ESPCN

import pdb

BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.0001
LOGDIR_ROOT = './logdir'

def get_arguments():
    parser = argparse.ArgumentParser(description='EspcnNet example network')
    parser.add_argument('--checkpoint', type=str,
                        help='Which model checkpoint to load from', default=None)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='How many image files to process at once.')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help='Number of epochs.')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help='Learning rate for training.')
    parser.add_argument('--logdir_root', type=str, default=LOGDIR_ROOT,
                        help='Root directory to place the logging '
                        'output and generated model. These are stored '
                        'under the dated subdirectory of --logdir_root. '
                        'Cannot use with --logdir.')

    return parser.parse_args()

def check_params(args, params):
    if len(params['filters_size']) - len(params['channels']) != 1:
        print("The length of 'filters_size' must be greater then the length of 'channels' by 1.")
        return False
    return True

def train():
    args = get_arguments()

    with open("./params.json", 'r') as f:
        params = json.load(f)

    if check_params(args, params) == False:
        return

    logdir_root = args.logdir_root # ./logdir
    logdir = os.path.join(logdir_root, 'train') # ./logdir/train

    # Load training data as np arrays
    lr_images, hr_labels = create_inputs(params)

    net = ESPCN(filters_size=params['filters_size'],
                   channels=params['channels'],
                   ratio=params['ratio'],
                   batch_size=args.batch_size,
                   lr_size=params['lr_size'],
                   edge=params['edge'])

    loss, images, labels = net.build_model()
    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
    trainable = tf.trainable_variables()
    optim = optimizer.minimize(loss, var_list=trainable)

    # set up logging for tensorboard
    writer = tf.summary.FileWriter(logdir)
    writer.add_graph(tf.get_default_graph())
    summaries = tf.summary.merge_all()

    # set up session
    sess = tf.Session()

    # saver for storing/restoring checkpoints of the model
    saver = tf.train.Saver()

    init = tf.initialize_all_variables()
    sess.run(init)

    if net.load(sess, saver, logdir):
        print("[*] Checkpoint load success!")
    else:
        print("[*] Checkpoint load failed/no checkpoint found")

    try:
        steps, start_average, end_average = 0, 0, 0
        start_time = time.time()
        for ep in xrange(1, args.epochs + 1):
            batch_idxs = len(lr_images) // args.batch_size
            batch_average = 0
            for idx in xrange(0, batch_idxs):
                # On the fly batch generation instead of Queue to optimize GPU usage
                batch_images = lr_images[idx * args.batch_size : (idx + 1) * args.batch_size]
                batch_labels = hr_labels[idx * args.batch_size : (idx + 1) * args.batch_size]
                
                steps += 1
                summary, loss_value, _ = sess.run([summaries, loss, optim], feed_dict={images: batch_images, labels: batch_labels})
                writer.add_summary(summary, steps)
                batch_average += loss_value

            # Compare loss of first 20% and last 20%
            batch_average = float(batch_average) / batch_idxs
            if ep < (args.epochs * 0.2):
                start_average += batch_average
            elif ep >= (args.epochs * 0.8):
                end_average += batch_average

            duration = time.time() - start_time
            print('Epoch: {}, step: {:d}, loss: {:.9f}, ({:.3f} sec/epoch)'.format(ep, steps, batch_average, duration))
            start_time = time.time()
            net.save(sess, saver, logdir, steps)
    except KeyboardInterrupt:
        print()
    finally:
        start_average = float(start_average) / (args.epochs * 0.2)
        end_average = float(end_average) / (args.epochs * 0.2)
        print("Start Average: [%.6f], End Average: [%.6f], Improved: [%.2f%%]" \
          % (start_average, end_average, 100 - (100*end_average/start_average)))

if __name__ == '__main__':
    train()
