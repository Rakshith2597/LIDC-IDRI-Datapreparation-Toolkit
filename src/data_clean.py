import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import cv2
import math
from skimage.feature import match_template


def process_json_file(input_file, output_file):
    """
    Process the JSON file and create a new JSON file with modified data.

    Parameters:
    - input_file (str): Path to the input JSON file.
    - output_file (str): Path to the output JSON file.

    Returns:
    - None
    """
    input_file = os.path.abspath(input_file)
    output_file = os.path.abspath(output_file)

    with open(input_file) as f1:
        json_dict = json.load(f1)

    id_nod_rel = defaultdict(lambda: '', {})  # Initializing an empty defaultdict

    series_uid_list = json_dict.keys()

    for series_id in tqdm(series_uid_list):
        nod_list = []
        slice_list = list(json_dict[series_id].keys())

        for slice_id in slice_list:
            nod_list.append(json_dict[series_id][slice_id])

        uq_nod_ids = np.unique(nod_list)
        nod_dict = {}

        for nod_id in uq_nod_ids:
            idx = np.where(nod_list == nod_id)
            sl_no = [int(slice_list[i]) for i in idx[0]]
            sl_no = sorted(sl_no)
            middle_index = int((len(sl_no) - 1) / 2)
            middle_slice = sl_no[middle_index]
            nod_dict[str(middle_slice)] = int(nod_id)

        id_nod_rel[series_id] = nod_dict

    with open(output_file, 'w') as f1:
        json.dump(id_nod_rel, f1)


# Call the function to execute the code
input_file_path = 'LIDC-IDRI/nodule_segmentation_2022/jsons/seriesuid_nodule_map.json'
output_file_path = 'nodule_segmentation_2022/jsons/seriesuid_nodule_map_new.json'
process_json_file(input_file_path, output_file_path)


def calculate_mean_std(imgdir):
    """
    Calculate the mean and standard deviation of pixel intensities in a directory of images.

    Parameters:
    - imgdir (str): Path to the directory containing the images.

    Returns:
    - mean (float): Mean of pixel intensities.
    - std (float): Standard deviation of pixel intensities.
    """
    # Load the first image to calculate mean
    img_path = os.path.join(imgdir, os.listdir(imgdir)[0])
    img = np.load(img_path).astype(np.float32)
    mean = np.mean(img)

    # Calculate the standard deviation
    numerator = 0

    for i in tqdm(os.listdir(imgdir)):
        img_path = os.path.join(imgdir, i)
        img = np.load(img_path).astype(np.float32)
        numerator += np.sum((img - mean) ** 2)

    std = math.sqrt(numerator / (len(os.listdir(imgdir)) * 64 * 64))

    return mean, std

# Call the function and print the results
imgdir = 'storage/rakshith/lidc_data/patches/img'
mean, std = calculate_mean_std(imgdir)
print(mean)
print(std)

class ImageUtils:
    def __init__(self, imgdir):
        """
        Utility class for image operations.

        Parameters:
        - imgdir (str): Path to the directory containing the images.

        """
        self.imgdir = imgdir

    def load_image(self, filename):
        """
        Load an image from the specified directory.

        Parameters:
        - filename (str): Name of the image file.

        Returns:
        - img (numpy.ndarray): Loaded image as a NumPy array.

        """
        img_path = os.path.join(self.imgdir, filename)
        img = np.load(img_path)
        return img

    def plot_histogram(self, img):
        """
        Plot the histogram of pixel values in the image.

        Parameters:
        - img (numpy.ndarray): Image as a NumPy array.

        Returns:
        - None

        """
        plt.hist(img)
        plt.show()

# Create an instance of ImageUtils
imgdir = 'storage/rakshith/lidc_data/patches/img'
image_utils = ImageUtils(imgdir)

# Load the first image
img_files = os.listdir(imgdir)
img = image_utils.load_image(img_files[0])

# Plot the histogram
image_utils.plot_histogram(img)


class MaskUtils:
    def __init__(self, mask_dir):
        """
        Utility class for mask operations.

        Parameters:
        - mask_dir (str): Path to the directory containing the masks.

        """
        self.mask_dir = mask_dir

    def load_masks(self):
        """
        Load the list of masks from the specified directory.

        Returns:
        - mask_list (list): List of mask filenames.

        """
        mask_list = os.listdir(self.mask_dir)
        return mask_list

    def generate_cross_masks(self):
        """
        Generate cross-shaped masks for pattern matching.

        Returns:
        - cross1 (numpy.ndarray): Cross-shaped mask 1.
        - cross2 (numpy.ndarray): Cross-shaped mask 2.

        """
        cross1 = np.zeros((5, 5), dtype=np.uint8)
        cross2 = np.zeros((5, 5), dtype=np.uint8)

        for i in range(1, 4):
            cross1[i, 2] = 1
            cross1[2, i] = 1
            cross2[i, i] = 1

        cross2[3, 1] = 1
        cross2[1, 3] = 1

        return cross1, cross2

    def find_cross_masks(self, cross1, cross2):
        """
        Find masks in the given directory that match the specified cross-shaped masks.

        Parameters:
        - cross1 (numpy.ndarray): Cross-shaped mask 1.
        - cross2 (numpy.ndarray): Cross-shaped mask 2.

        Returns:
        - cross1_list (list): List of filenames matching cross-shaped mask 1.
        - cross2_list (list): List of filenames matching cross-shaped mask 2.

        """
        mask_list = self.load_masks()
        cross1_list = []
        cross2_list = []

        for filename in tqdm(mask_list):
            mask = np.load(os.path.join(self.mask_dir, filename))
            res1 = match_template(mask, cross1, pad_input=True)
            res2 = match_template(mask, cross2, pad_input=True)
            x1 = np.where(res1 > 0.99)
            x2 = np.where(res2 > 0.99)
            if len(x1[0]):
                cross1_list.append(filename)
            if len(x2[0]):
                cross2_list.append(filename)

        return cross1_list, cross2_list

    def save_cross_list(self, cross_list, output_file):
        """
        Save the list of filenames to a JSON file.

        Parameters:
        - cross_list (list): List of filenames.
        - output_file (str): Path to the output JSON file.

        Returns:
        - None

        """
        with open(output_file, 'w') as f:
            json.dump(cross_list, f)

    def plot_masks(self, mask_list):
        """
        Plot the masks from the given list.

        Parameters:
        - mask_list (list): List of mask filenames.

        Returns:
        - None

        """
        for file in mask_list:
            mask = np.load(os.path.join(self.mask_dir, file))
            plt.figure()
            plt.imshow(mask)
            plt.title(file)
            plt.show()
            break


# Create an instance of MaskUtils
mask_dir = '/storage/rakshith/lidc_data/patches/masks'
mask_utils = Mask

