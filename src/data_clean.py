#for review

import json
import math
import os
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import match_template
from tqdm import tqdm


def process_json_file(input_file, output_file):
    """
    Process the JSON file and create a new JSON file with modified data.

    Parameters:
    - input_file (str): Path to the input JSON file.
    - output_file (str): Path to the output JSON file.

    Returns:
    - None
    """
    assert isinstance(input_file, str), "input_file should be a string"
    assert isinstance(output_file, str), "output_file should be a string"

    input_file = os.path.abspath(input_file)
    output_file = os.path.abspath(output_file)

    with open(input_file) as f1:
        json_dict = json.load(f1)

    assert isinstance(json_dict, dict), "json_dict should be a dictionary"

    id_nod_rel = defaultdict(lambda: '', {})  # Initializing an empty defaultdict

    series_uid_list = json_dict.keys()

    assert isinstance(series_uid_list, list), "series_uid_list should be a list"

    for series_id in tqdm(series_uid_list):
        assert isinstance(series_id, str), "series_id should be a string"

        nod_list = []
        slice_list = list(json_dict[series_id].keys())

        assert isinstance(slice_list, list), "slice_list should be a list"

        for slice_id in slice_list:
            assert isinstance(slice_id, str), "slice_id should be a string"
            assert slice_id.isdigit(), "slice_id should be a numeric string"

            nod_list.append(json_dict[series_id][slice_id])

        assert all(isinstance(nod_id, int) for nod_id in nod_list), "nod_list should contain integers"

        uq_nod_ids = np.unique(nod_list)
        nod_dict = {}

        for nod_id in uq_nod_ids:
            assert isinstance(nod_id, int), "nod_id should be an integer"

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
# Static paths needs to be changed

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
    assert isinstance(imgdir, str), "imgdir should be a string"
    assert os.path.isdir(imgdir), "imgdir should be a valid directory"

    # Load the first image to calculate mean
    img_path = os.path.join(imgdir, os.listdir(imgdir)[0])
    img = np.load(img_path).astype(np.float32)

    assert isinstance(img, np.ndarray), "img should be a NumPy array"
    assert img.dtype == np.float32, "img should have dtype float32"
    assert img.ndim == 2, "img should be a 2D array"

    mean = np.mean(img)

    # Calculate the standard deviation
    numerator = 0

    file_list = os.listdir(imgdir)
    assert len(file_list) > 0, "imgdir should contain at least one image file"

    for i in tqdm(file_list):
        img_path = os.path.join(imgdir, i)
        img = np.load(img_path).astype(np.float32)

        assert isinstance(img, np.ndarray), "img should be a NumPy array"
        assert img.dtype == np.float32, "img should have dtype float32"
        assert img.ndim == 2, "img should be a 2D array"
        assert img.shape == (64, 64), "img should have shape (64, 64)"

        numerator += np.sum((img - mean) ** 2)

    std = math.sqrt(numerator / (len(file_list) * 64 * 64))

    return mean, std

# Call the function and print the results
imgdir = 'storage/rakshith/lidc_data/patches/img'
mean, std = calculate_mean_std(imgdir)
print(mean)
print(std)

class ImageUtils:
    """
    Add docstring here
    
    """
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
        assert isinstance(filename, str), "filename should be a string"

        img_path = os.path.join(self.imgdir, filename)
        assert os.path.isfile(img_path), "Invalid image file"

        img = np.load(img_path)
        assert isinstance(img, np.ndarray), "img should be a NumPy array"

        return img

    def plot_histogram(self, img):
        """
        Plot the histogram of pixel values in the image.

        Parameters:
        - img (numpy.ndarray): Image as a NumPy array.

        Returns:
        - None

        """
        assert isinstance(img, np.ndarray), "img should be a NumPy array"
        assert img.ndim == 2, "img should be a 2D array"

        plt.hist(img)
        plt.show()

# Create an instance of ImageUtils
#static path to be removed
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
        assert isinstance(mask_dir, str), "mask_dir should be a string"
        assert os.path.isdir(mask_dir), "Invalid mask directory"

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
        assert isinstance(cross1, np.ndarray), "cross1 should be a NumPy array"
        assert isinstance(cross2, np.ndarray), "cross2 should be a NumPy array"

        mask_list = self.load_masks()
        cross1_list = []
        cross2_list = []

        for filename in tqdm(mask_list):
            mask = np.load(os.path.join(self.mask_dir, filename))
            assert isinstance(mask, np.ndarray), f"Invalid mask file: {filename}"

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
        assert isinstance(cross_list, list), "cross_list should be a list"
        assert isinstance(output_file, str), "output_file should be a string"

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
        assert isinstance(mask_list, list), "mask_list should be a list"

        for file in mask_list:
            assert isinstance(file, str), "Invalid mask filename"
            mask = np.load(os.path.join(self.mask_dir, file))
            assert isinstance(mask, np.ndarray), f"Invalid mask file: {file}"

            plt.figure()
            plt.imshow(mask)
            plt.title(file)
            plt.show()
            break


# Create an instance of MaskUtils
mask_dir = '/storage/rakshith/lidc_data/patches/masks'
mask_utils = Mask

