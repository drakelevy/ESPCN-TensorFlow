import tensorflow as tf
import numpy as np
import os
import pdb

def create_inputs(params):
    """
    Loads prepared training files and appends them as np arrays to a list.
    This approach is better because a FIFOQueue with a reader can't utilize
    the GPU while this approach can.
    """
    sess = tf.Session()

    lr_images, hr_labels = [], []
    training_dir = params['training_dir']
    lr_shape = (params['lr_size'], params['lr_size'], 3)
    hr_shape = output_shape = (params['lr_size'] - params['edge'], params['lr_size'] - params['edge'], 3 * params['ratio']**2)
    for file in os.listdir(training_dir):
        train_file = open("{}/{}".format(training_dir, file), "rb")
        train_data = np.fromfile(train_file, dtype=np.uint8)

        lr_image = train_data[:17 * 17 * 3].reshape(lr_shape)
        lr_images.append(lr_image)

        hr_label = train_data[17 * 17 * 3:].reshape(hr_shape)
        hr_labels.append(hr_label)

    return lr_images, hr_labels
