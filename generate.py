import argparse
import tensorflow as tf
from scipy import ndimage
from scipy import misc
import numpy as np
from prepare_data import *
from psnr import psnr
import json
import pdb

from espcn import ESPCN

def get_arguments():
    parser = argparse.ArgumentParser(description='EspcnNet generation script')
    parser.add_argument('--checkpoint', type=str,
                        help='Which model checkpoint to generate from')
    parser.add_argument('--lr_image', type=str,
                        help='The low-resolution image waiting for processed.')
    parser.add_argument('--hr_image', type=str,
                        help='The high-resolution image which is used to calculate PSNR.')
    parser.add_argument('--out_path', type=str,
                        help='The output path for the super-resolution image')
    return parser.parse_args()

def check_params(args, params):
    if len(params['filters_size']) - len(params['channels']) != 1:
        print("The length of 'filters_size' must be greater then the length of 'channels' by 1.")
        return False
    return True

def generate():
    args = get_arguments()

    with open("./params.json", 'r') as f:
        params = json.load(f)

    if check_params(args, params) == False:
        return

    sess = tf.Session()

    net = ESPCN(filters_size=params['filters_size'],
                   channels=params['channels'],
                   ratio=params['ratio'],
                   batch_size=1,
                   lr_size=params['lr_size'],
                   edge=params['edge'])

    loss, images, labels = net.build_model()

    lr_image = tf.placeholder(tf.uint8)
    lr_image_data = misc.imread(args.lr_image)
    lr_image_ycbcr_data = rgb2ycbcr(lr_image_data)
    lr_image_y_data = lr_image_ycbcr_data[:, :, 0:1]
    lr_image_cb_data = lr_image_ycbcr_data[:, :, 1:2]
    lr_image_cr_data = lr_image_ycbcr_data[:, :, 2:3]
    lr_image_batch = np.zeros((1,) + lr_image_y_data.shape)
    lr_image_batch[0] = lr_image_y_data

    sr_image = net.generate(lr_image)

    saver = tf.train.Saver()
    try:
        model_loaded = net.load(sess, saver, args.checkpoint)
    except:
        raise Exception("Failed to load model, does the ratio in params.json match the ratio you trained your checkpoint with?")

    if model_loaded:
        print("[*] Checkpoint load success!")
    else:
        print("[*] Checkpoint load failed/no checkpoint found")
        return

    sr_image_y_data = sess.run(sr_image, feed_dict={lr_image: lr_image_batch})

    sr_image_y_data = shuffle(sr_image_y_data[0], params['ratio'])
    sr_image_ycbcr_data = misc.imresize(lr_image_ycbcr_data,
                                    params['ratio'] * np.array(lr_image_data.shape[0:2]),
                                    'bicubic')

    edge = params['edge'] * params['ratio'] / 2

    sr_image_ycbcr_data = np.concatenate((sr_image_y_data, sr_image_ycbcr_data[edge:-edge,edge:-edge,1:3]), axis=2)
    sr_image_data = ycbcr2rgb(sr_image_ycbcr_data)

    misc.imsave(args.out_path + '.png', sr_image_data)

    if args.hr_image != None:
        hr_image_data = misc.imread(args.hr_image)
        model_psnr = psnr(hr_image_data, sr_image_data, edge)
        print('PSNR of the model: {:.2f}dB'.format(model_psnr))

        sr_image_bicubic_data = misc.imresize(lr_image_data,
                                        params['ratio'] * np.array(lr_image_data.shape[0:2]),
                                        'bicubic')
        misc.imsave(args.out_path + '_bicubic.png', sr_image_bicubic_data)
        bicubic_psnr = psnr(hr_image_data, sr_image_bicubic_data, 0)
        print('PSNR of Bicubic: {:.2f}dB'.format(bicubic_psnr))


if __name__ == '__main__':
    generate()