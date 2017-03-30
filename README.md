# ESPCN-TensorFlow
TensorFlow implementation of the Efficient Sub-Pixel Convolutional Neural Network in TensorFlow (ESPCN). Network based on this [paper](https://arxiv.org/pdf/1609.05158.pdf) and code adapted from this [repo](https://github.com/JesseYang/Espcn).

## Prerequisites
 * Python 2.7
 * TensorFlow
 * Numpy
 * Scipy version > 0.18

## Usage
For training: `python train.py`
<br>
Can specify epochs, learning rate, batch size etc:
<br>
`python train.py --epochs 10 --learning_rate 0.0001 --batch_size 32`
<br>

For generating: `python generate.py`
<br>
Must specify checkpoint, low-resolution image, and output path

`python generate.py --checkpoint logdir/train --lr_image images/butterfly_lr.png --out_path results/butterfly_hr.png`

Run `prepare_images.py` and then `prepare_data.py` to format the training and validation data

## Result

<br>
Original butterfly image:
<br>
![orig](https://github.com/drakelevy/ESPCN-TensorFlow/blob/master/result/original.jpg)<br>
Bicubic interpolated image:
<br>
![bicubic](https://github.com/drakelevy/ESPCN-Tensorflow/blob/master/result/bicubic.jpg)<br>
Super-resolved image:
<br>
![espcn](https://github.com/drakelevy/ESPCN-Tensorflow/blob/master/result/espcn.png)

## References
* [JesseYang/Espcn](https://github.com/JesseYang/Espcn)
