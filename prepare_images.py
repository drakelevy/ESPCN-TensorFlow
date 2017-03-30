from scipy import misc
import numpy as np
import imghdr
import shutil
import os
import json

def prepare_images():
    with open("./params.json", 'r') as f:
        params = json.load(f)

    ratio, training_num, lr_stride, lr_size = params['ratio'], params['training_num'], params['lr_stride'], params['lr_size']
    hr_stride = lr_stride * ratio
    hr_size = lr_size * ratio

    # first clear old images and create new directories
    for ele in ['training', 'validation', 'test']:
        if os.path.isdir(params[ele + '_image_dir']):
            shutil.rmtree(params[ele + '_image_dir'])
        for sub_dir in ['/hr', 'lr', '/hr_full', 'lr_full']:
            os.makedirs(params[ele + '_image_dir'] + sub_dir)

    image_num = 0
    folder = params['training_image_dir']
    for root, dirnames, filenames in os.walk(params['image_dir']):
        for filename in filenames:
            path = os.path.join(root, filename)
            if imghdr.what(path) != 'jpeg':
                continue
                
            hr_image = misc.imread(path)
            height = hr_image.shape[0]
            new_height = height - height % ratio
            width = hr_image.shape[1]
            new_width = width - width % ratio
            hr_image = hr_image[0:new_height,0:new_width]
            lr_image = blurred[::ratio,::ratio,:]

            height = hr_image.shape[0]
            width = hr_image.shape[1]
            vertical_number = height / hr_stride - 1
            horizontal_number = width / hr_stride - 1
            image_num = image_num + 1
            if image_num % 10 == 0:
                print image_num
            if image_num > training_num and image_num <= training_num + params['validation_num']:
                folder = params['validation_image_dir']
            elif image_num > training_num + params['validation_num']:
                folder = params['test_image_dir']
            misc.imsave(folder + 'hr_full/' + filename[0:-4] + '.png', hr_image)
            misc.imsave(folder + 'lr_full/' + filename[0:-4] + '.png', lr_image)
            for x in range(0, horizontal_number):
                for y in range(0, vertical_number):
                    hr_sub_image = hr_image[y * hr_stride : y * hr_stride + hr_size, x * hr_stride : x * hr_stride + hr_size]
                    lr_sub_image = lr_image[y * lr_stride : y * lr_stride + lr_size, x * lr_stride : x * lr_stride + lr_size]
                    misc.imsave("{}hr/{}_{}_{}.png".format(folder, filename[0:-4], y, x), hr_sub_image)
                    misc.imsave("{}lr/{}_{}_{}.png".format(folder, filename[0:-4], y, x), lr_sub_image)
            if image_num >= training_num + params['validation_num'] + params['test_num']:
                break
        else:
            continue
        break


if __name__ == '__main__':
    prepare_images()
