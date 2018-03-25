# ESPCN-TensorFlow
TensorFlow implementation of the Efficient Sub-Pixel Convolutional Neural Network in TensorFlow (ESPCN). Network based on this [paper](https://arxiv.org/pdf/1609.05158.pdf) and code adapted from this [repo](https://github.com/JesseYang/Espcn).
<br>
This network can achieve the real-time performance of the [FSRCNN](https://arxiv.org/abs/1608.00367) while also surpassing the quality of the [SRCNN](https://arxiv.org/abs/1501.00092).

## Prerequisites
 * Python 2.7
 * TensorFlow
 * Numpy
 * Scipy version > 0.18

## Usage
Run `prepare_data.py` to format the training and validation data before training each new model
<br>
For training: `python train.py`
<br>
Can specify epochs, learning rate, batch size etc:
<br>
`python train.py --epochs 10 --learning_rate 0.0001 --batch_size 32`
<br>

For generating: `python generate.py`
<br>
Must specify checkpoint, low-resolution image, and output path
<br>
`python generate.py --checkpoint logdir_2x/train --lr_image images/butterfly_GT.png --out_path result/butterfly_HR`

Check `params.json` for parameter values and to change the upscaling ratio (2x, 3x, ...) the model is operating on.

## Result

Original butterfly image:
<br>
![orig](https://github.com/drakelevy/ESPCN-TensorFlow/blob/master/result/original.jpg)
<br>
Bicubic interpolated image:
<br>
![bicubic](https://github.com/drakelevy/ESPCN-Tensorflow/blob/master/result/bicubic.jpg)
<br>
Super-resolved image:
<br>
![espcn](https://github.com/drakelevy/ESPCN-Tensorflow/blob/master/result/espcn.png)

## References
* [JesseYang/Espcn](https://github.com/JesseYang/Espcn)
