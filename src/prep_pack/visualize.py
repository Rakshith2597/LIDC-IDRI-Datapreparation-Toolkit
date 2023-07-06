#!/usr/bin/env python
# coding: utf-8
#for review
import json
import numpy as np
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt

def visualize_data(series_uid, slice_num):
    """
    To visualize the image and nodule masks of the dataset

    Parameters:
    - series_uid (str): Series_instance_uid or filename of the image to visualize
    - slice_num (int): Slice number to visualize
    """
    assert isinstance(series_uid, str), "series_uid should be a string"
    assert isinstance(slice_num, int), "slice_num should be an integer"
    assert slice_num >= 0, "slice_num should be a non-negative integer"

    current_dir = os.path.dirname(os.path.realpath(__file__))
    target_location = os.path.sep.join(current_dir.split(os.path.sep)[:-3])
    datapath = os.path.join(target_location, 'data/')
    savepath = os.path.join(target_location, 'results/visualizations/')

    if not os.path.isdir(savepath):
        os.makedirs(savepath)

    img_name = series_uid + '_slice' + str(slice_num) + '.npy'
    mask = np.load(datapath + 'mask/' + img_name)
    img = np.load(datapath + 'img/' + img_name)
    lungseg = np.load(datapath + 'lungseg/' + img_name)

    plt.figure()
    plt.subplot(131)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.subplot(132)
    plt.imshow(lungseg, cmap='gray')
    plt.title('Ground truth (Lung)')
    plt.subplot(133)
    plt.imshow(mask, cmap='gray')
    plt.title('Ground truth (Nodule)')
    plt.savefig(savepath + 'visualization' + series_uid + '_slice' + str(slice_num) + '.png')
    plt.show()
    plt.close()