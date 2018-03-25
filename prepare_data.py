from scipy import misc, ndimage
import numpy as np
import imghdr
import shutil
import os
import json

mat = np.array(
    [[ 65.481, 128.553, 24.966 ],
     [-37.797, -74.203, 112.0  ],
     [  112.0, -93.786, -18.214]])
mat_inv = np.linalg.inv(mat)
offset = np.array([16, 128, 128])

def rgb2ycbcr(rgb_img):
    ycbcr_img = np.zeros(rgb_img.shape, dtype=np.uint8)
    for x in range(rgb_img.shape[0]):
        for y in range(rgb_img.shape[1]):
            ycbcr_img[x, y, :] = np.round(np.dot(mat, rgb_img[x, y, :] * 1.0 / 255) + offset)
    return ycbcr_img

def ycbcr2rgb(ycbcr_img):
    rgb_img = np.zeros(ycbcr_img.shape, dtype=np.uint8)
    for x in range(ycbcr_img.shape[0]):
        for y in range(ycbcr_img.shape[1]):
            [r, g, b] = ycbcr_img[x,y,:]
            rgb_img[x, y, :] = np.maximum(0, np.minimum(255, np.round(np.dot(mat_inv, ycbcr_img[x, y, :] - offset) * 255.0)))
    return rgb_img

def my_anti_shuffle(input_image, ratio):
    shape = input_image.shape
    ori_height = int(shape[0])
    ori_width = int(shape[1])
    ori_channels = int(shape[2])
    if ori_height % ratio != 0 or ori_width % ratio != 0:
        print("Error! Height and width must be divided by ratio!")
        return
    height = ori_height / ratio
    width = ori_width / ratio
    channels = ori_channels * ratio * ratio
    anti_shuffle = np.zeros((height, width, channels), dtype=np.uint8)
    for c in range(0, ori_channels):
        for x in range(0, ratio):
            for y in range(0, ratio):
                anti_shuffle[:,:,c * ratio * ratio + x * ratio + y] = input_image[x::ratio, y::ratio, c]
    return anti_shuffle

def shuffle(input_image, ratio):
    shape = input_image.shape
    height = int(shape[0]) * ratio
    width = int(shape[1]) * ratio
    channels = int(shape[2]) / ratio / ratio
    shuffled = np.zeros((height, width, channels), dtype=np.uint8)
    for i in range(0, height):
        for j in range(0, width):
            for k in range(0, channels):
                shuffled[i,j,k] = input_image[i / ratio, j / ratio, k * ratio * ratio + (i % ratio) * ratio + (j % ratio)]
    return shuffled

def prepare_images(params):
    ratio, training_num, lr_stride, lr_size = params['ratio'], params['training_num'], params['lr_stride'], params['lr_size']
    hr_stride = lr_stride * ratio
    hr_size = lr_size * ratio

    # first clear old images and create new directories
    for ele in ['training', 'validation', 'test']:
        new_dir = params[ele + '_image_dir'].format(ratio)
        if os.path.isdir(new_dir):
            shutil.rmtree(new_dir)
        for sub_dir in ['/hr', 'lr']:
            os.makedirs(new_dir + sub_dir)

    image_num = 0
    folder = params['training_image_dir'].format(ratio)
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
            blurred = ndimage.gaussian_filter(hr_image, sigma=(1, 1, 0))
            lr_image = blurred[::ratio,::ratio,:]

            height = hr_image.shape[0]
            width = hr_image.shape[1]
            vertical_number = height / hr_stride - 1
            horizontal_number = width / hr_stride - 1
            image_num = image_num + 1
            if image_num % 10 == 0:
                print "Finished image: {}".format(image_num)
            if image_num > training_num and image_num <= training_num + params['validation_num']:
                folder = params['validation_image_dir'].format(ratio)
            elif image_num > training_num + params['validation_num']:
                folder = params['test_image_dir'].format(ratio)
            #misc.imsave(folder + 'hr_full/' + filename[0:-4] + '.png', hr_image)
            #misc.imsave(folder + 'lr_full/' + filename[0:-4] + '.png', lr_image)
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

def prepare_data(params):
    ratio = params['ratio']
    params['hr_stride'] = params['lr_stride'] * ratio
    params['hr_size'] = params['lr_size'] * ratio

    for ele in ['training', 'validation', 'test']:
        new_dir = params[ele + '_dir'].format(ratio)
        if os.path.isdir(new_dir):
            shutil.rmtree(new_dir)
        os.makedirs(new_dir)

    ratio, lr_size, edge = params['ratio'], params['lr_size'], params['edge']
    image_dirs = [d.format(ratio) for d in [params['training_image_dir'], params['validation_image_dir'], params['test_image_dir']]]
    data_dirs = [d.format(ratio) for d in [params['training_dir'], params['validation_dir'], params['test_dir']]]
    hr_start_idx = ratio * edge / 2
    hr_end_idx = hr_start_idx + (lr_size - edge) * ratio
    sub_hr_size = (lr_size - edge) * ratio
    for dir_idx, image_dir in enumerate(image_dirs):
        data_dir = data_dirs[dir_idx]
        print "Creating {}".format(data_dir)
        for root, dirnames, filenames in os.walk(image_dir + "/lr"):
            for filename in filenames:
                lr_path = os.path.join(root, filename)
                hr_path = image_dir + "/hr/" + filename
                lr_image = misc.imread(lr_path)
                hr_image = misc.imread(hr_path)
                # convert to Ycbcr color space
                lr_image_y = rgb2ycbcr(lr_image)
                hr_image_y = rgb2ycbcr(hr_image)
                lr_data = lr_image_y.reshape((lr_size * lr_size * 3))
                sub_hr_image_y = hr_image_y[hr_start_idx:hr_end_idx:1,hr_start_idx:hr_end_idx:1]
                hr_data = my_anti_shuffle(sub_hr_image_y, ratio).reshape(sub_hr_size * sub_hr_size * 3)
                data = np.concatenate([lr_data, hr_data])
                data.astype('uint8').tofile(data_dir + "/" + filename[0:-4])

def remove_images(params):
    # Don't need old image folders
    for ele in ['training', 'validation', 'test']:
        rm_dir = params[ele + '_image_dir'].format(params['ratio'])
        if os.path.isdir(rm_dir):
            shutil.rmtree(rm_dir)


if __name__ == '__main__':
    with open("./params.json", 'r') as f:
        params = json.load(f)

    print "Preparing images with scaling ratio: {}".format(params['ratio'])
    print "If you want a different ratio change 'ratio' in params.json"
    print "Splitting images (1/3)"
    prepare_images(params)

    print "Preparing data, this may take a while (2/3)"
    prepare_data(params)

    print "Cleaning up split images (3/3)"
    remove_images(params)
    print "Done, you can now train the model!"
