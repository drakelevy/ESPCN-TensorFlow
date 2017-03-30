import os
from scipy import misc
import numpy as np
import json
import shutil

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

def prepare_data():
    with open("./params.json", 'r') as f:
        params = json.load(f)
    params['hr_stride'] = params['lr_stride'] * ratio
    params['hr_size'] = lr_size * ratio

    for ele in ['training', 'validation', 'test']:
        if os.path.isdir(params[ele + '_dir']):
            shutil.rmtree(params[ele + '_dir'])
        os.makedirs(params[ele + '_dir'])

    ratio, lr_size, edge = params['ratio'], params['lr_size'], params['edge']
    image_dirs = [params['training_image_dir'], params['validation_image_dir'], params['test_image_dir']]
    data_dirs = [params['training_dir'], params['validation_dir'], params['test_dir']]
    hr_start_idx = ratio * edge / 2
    hr_end_idx = hr_start_idx + (lr_size - edge) * ratio
    sub_hr_size = (lr_size - edge) * ratio
    for dir_idx, image_dir in enumerate(image_dirs):
        data_dir = data_dirs[dir_idx]
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


if __name__ == '__main__':
    prepare_data()

