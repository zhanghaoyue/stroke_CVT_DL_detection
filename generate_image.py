import argparse
import multiprocessing as mp
import os

import cv2
import numpy as np
import torch
from grad_cam import GradCAM, GradCamPlusPlus
from skimage import io
from torch import nn


def get_last_conv_name(net):
    """
    :param net:
    :return: last layer name 
    """
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    return layer_name


def norm_image(image):
    """
    normalize image
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)


def gen_cam(image, mask):
    """
    generate grad cam image
    :param image: [H,W,C], original input image
    :param mask: [H,W],range (0,1)
    :return: tuple(cam,heatmap)
    """
    # mask transform to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]  # gbr to rgb

    # merge heatmap back to original image
    cam = heatmap + np.float32(image)
    return norm_image(cam), heatmap


def save_image(image_dicts, input_image_name, network='frcnn', output_dir='./results'):
    prefix = os.path.splitext(input_image_name)[0]
    for key, image in image_dicts.items():
        io.imsave(os.path.join(output_dir, '{}-{}-{}.jpg'.format(prefix, network, key)), image)