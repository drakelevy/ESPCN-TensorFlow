import argparse
import tensorflow as tf
from scipy import ndimage
from scipy import misc
import numpy as np
from prepare_data import *
from psnr import psnr
import json
import pdb
import os
from time import time

from espcn import ESPCN

RATIO = 2

def get_arguments():
    parser = argparse.ArgumentParser(description='EspcnNet generation script')
    parser.add_argument('--checkpoint', type=str,
                        help='Which model checkpoint to generate from')
    parser.add_argument('--ratio', type=int, default=RATIO,
                        help='The ratio for up-sampling, should be the same with the model.')
    parser.add_argument('--lr_image_dir', type=str,
                        help='The low-resolution image directory waiting for processed.')
    parser.add_argument('--hr_image_dir', type=str,
                        help='The high-resolution image directory which is used to calculate PSNR.')
    parser.add_argument('--out_path_dir', type=str,
                        help='The output directory for the super-resolution image')
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

    files = [f for f in os.listdir(args.lr_image_dir) if os.path.isfile(os.path.join(args.lr_image_dir, f))]

    saver = tf.train.Saver()
    if net.load(sess, saver, args.checkpoint):
        print("[*] Checkpoint load success!")
    else:
        print("[*] Checkpoint load failed/no checkpoint found")
        return

    frame_range = (87, 10000)

    for fileName in files:
        try:
            ts = time()
            frame_cnt = int(fileName[5:10])
            if frame_cnt < frame_range[0] or frame_cnt > frame_range[1]:
                print 'Ignoring frame ' + str(frame_cnt)
                continue
            else:
                print 'start sr for frame ' + str(frame_cnt)

            input_file = os.path.join(args.lr_image_dir, fileName)
            output_file = os.path.join(args.out_path_dir, fileName)

            lr_image = tf.placeholder(tf.uint8)
            lr_image_data = misc.imread(input_file) # pip install pillow
            lr_image_ycbcr_data = rgb2ycbcr(lr_image_data)
            lr_image_y_data = lr_image_ycbcr_data[:, :, 0:1]
            lr_image_cb_data = lr_image_ycbcr_data[:, :, 1:2]
            lr_image_cr_data = lr_image_ycbcr_data[:, :, 2:3]
            lr_image_batch = np.zeros((1,) + lr_image_y_data.shape)
            lr_image_batch[0] = lr_image_y_data
            print 'preprocessed %d ms' % ((time()-ts)*1000)
            ts = time()

            sr_image = net.generate(lr_image)
            print 'network generated %d ms' % ((time()-ts)*1000)
            ts = time()


            sr_image_y_data = sess.run(sr_image, feed_dict={lr_image: lr_image_batch})

            print 'run %d ms' % ((time()-ts)*1000)
            ts = time()

            sr_image_y_data = shuffle(sr_image_y_data[0], args.ratio)
            sr_image_ycbcr_data = misc.imresize(lr_image_ycbcr_data,
                                            params['ratio'] * np.array(lr_image_data.shape[0:2]),
                                            'bicubic')

            edge = params['edge'] * params['ratio'] / 2

            sr_image_ycbcr_data = np.concatenate((sr_image_y_data, sr_image_ycbcr_data[edge:-edge,edge:-edge,1:3]), axis=2)
            print 'mixed %d ms' % ((time()-ts)*1000)
            ts = time()
            sr_image_data = ycbcr2rgb(sr_image_ycbcr_data)
            #sr_image_data = sr_image_ycbcr_data
            print 'converted %d ms' % ((time()-ts)*1000)
            ts = time()

            misc.imsave(output_file, sr_image_data)
            print output_file + ' generated %d ms' % ((time()-ts)*1000)
            ts = time()

            if args.hr_image_dir != None:
                hr_image_path = os.path.join(args.hr_image_dir, fileName)
                hr_image_data = misc.imread(hr_image_path)
                model_psnr = psnr(hr_image_data, sr_image_data, edge)
                print('PSNR of the model: {:.2f}dB'.format(model_psnr))

                sr_image_bicubic_data = misc.imresize(lr_image_data,
                                                params['ratio'] * np.array(lr_image_data.shape[0:2]),
                                                'bicubic')
                bicubic_path = os.path.join(args.out_path_dir, fileName + '_bicubic.png')
                misc.imsave(bicubic_path, sr_image_bicubic_data)
                bicubic_psnr = psnr(hr_image_data, sr_image_bicubic_data, 0)
                print('PSNR of Bicubic: {:.2f}dB'.format(bicubic_psnr))
        except IndexError:
            print 'Index error caught'
        except IOError:
            print 'Cannot identify image file: ' + fileName
        except ValueError:
            print 'Cannot parse file name: ' + fileName



if __name__ == '__main__':
    generate()
