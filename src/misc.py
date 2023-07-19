# file to visualize images and json load function
import os
import numpy as np
import matplotlib.pyplot as plt



def data_clean_hist(img_dir):
    """
    Plot the histogram of an image file.

    Args:
        img_dir (str): Directory path of the image file.

    Returns:
        None
    """
    img = np.load(img_dir)
    plt.hist(img)
    plt.save()



