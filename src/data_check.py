import pylidc as pl
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import numpy as np


def get_hu_image(vol):
    """
    Convert the input volume to a Hounsfield Unit (HU) image.

    Args:
        vol (ndarray): The input volume.

    Returns:
        ndarray: The HU image.
    """
    assert isinstance(vol, np.ndarray), "vol should be a NumPy array"

    image = vol.copy()
    image[image > 400] = -1000
    image[image < -1000] = -1000
    return image


def get_hu_image_new(vol):
    """
    Convert the input volume to a Hounsfield Unit (HU) image.

    Args:
        vol (ndarray): The input volume.

    Returns:
        ndarray: The HU image.
    """
    assert isinstance(vol, np.ndarray), "vol should be a NumPy array"

    image = vol.copy()
    image = image - 1024
    image[image > 1000] = 1000
    image[image < -2000] = -2000
    return image


class LIDCProcessor:
    def __init__(self):
        self.id_list = []
        self.patch_list = []
        self.remove_list = []

    def load_id_list(self, file_path):
        """
        Load a list of IDs from a file.

        Args:
            file_path (str): The path to the file containing IDs.

        Returns:
            None
        """
        with open(file_path) as f:
            self.id_list = json.load(f)

    def load_patch_list(self, directory):
        """
        Load a list of patches from a directory.

        Args:
            directory (str): The path to the directory containing patches.

        Returns:
            None
        """
        self.patch_list = os.listdir(directory)

    def load_remove_list(self, file_path):
        """
        Load a list of IDs to be removed from a file.

        Args:
            file_path (str): The path to the file containing IDs to remove.

        Returns:
            None
        """
        with open(file_path) as f:
            self.remove_list = json.load(f)

    def process_nodules(self, series_uid):
        """
        Process nodules based on the series UID.

        Args:
            series_uid (str): The series UID of the nodules to process.

        Returns:
            None
        """
        mask_loadpath = os.path.join('storage', 'rakshith', 'lidc_data', 'patches', 'masks')
        image_loadpath = os.path.join('storage', 'rakshith', 'lidc_data', 'patches', 'images')
        mask_list = [mask for mask in self.patch_list if series_uid in mask]
        image_list = [image for image in self.patch_list if series_uid in image]
        nod_1 = [mask for mask in mask_list if mask.split('_')[2].split('.')[0] == str(1)]
        nod_1_img = [image for image in image_list if image.split('_')[2].split('.')[0] == str(1)]
        print(len(nod_1), nod_1[0])
        nod_1 = sorted(nod_1)
        nodule_mask = np.load(os.path.join(mask_loadpath, nod_1[0]))
        nodule_image = np.load(os.path.join(image_loadpath, nod_1_img[0]))
        print(f'Initial shape: {nodule_mask.shape}')
        for i in range(1, len(nod_1)):
            temp = np.load(os.path.join(mask_loadpath, nod_1[i]))
            temp_image = np.load(os.path.join(image_loadpath, nod_1_img[i]))
            nodule_mask = np.dstack((nodule_mask, temp))
            nodule_image = np.dstack((nodule_image, temp_image))
        print(f'Final shape: {nodule_mask.shape}')

    def check_imperfect_data(self):
        """
        Check for imperfect data based on the ID list.

        Args:
            None

        Returns:
            None
        """
        file_list = os.listdir(os.path.join('storage', 'rakshith', 'lidc_data', 'patches', 'images'))
        print(len(file_list))
        for i, f_name in enumerate(tqdm(file_list)):
            series_uid = file_list[i].split('_')[0]
            if series_uid in self.id_list:
                print(f'Imperfect data detected: {f_name}')

    def visualize_annotation(self):
        """
        Visualize annotations.

        Args:
            None

        Returns:
            None
        """
        ann = pl.query(pl.Annotation).first()
        vol = ann.scan.to_volume()
        padding = [(30, 10), (10, 25), (0, 0)]
        mask = ann.boolean_mask(pad=padding)
        bbox = ann.bbox(pad=padding)
        fig, ax = plt.subplots(1, 2, figsize=(5, 3))
        ax[0].imshow(vol[bbox][:, :, 2], cmap=plt.cm.gray)
        ax[0].axis('off')
        ax[1].imshow(mask[:, :, 2], cmap=plt.cm.gray)
        ax[1].axis('off')
        plt.tight_layout()
        plt.show()

    def visualize_annotation_in_3d(self, series_uid):
        """
        Visualize annotations in 3D based on the series UID.

        Args:
            series_uid (str): The series UID of the annotations to visualize.

        Returns:
            None
        """
        ann = pl.query(pl.Annotation)
        for x in ann:
            if x.scan.series_instance_uid == series_uid:
                x.visualize_in_3d()
                break


# Example usage
#lidc_processor = LIDCProcessor()
#lidc_processor.load_id_list('./jsons/seriesid_remove_list.json')
#lidc_processor.load_patch_list(os.path.join('storage', 'rakshith', 'lidc_data', 'patches', 'masks'))
#lidc_processor.load_remove_list('./jsons/seriesid_remove_list.json')
#lidc_processor.process_nodules('1.3.6.1.4.1.14519.5.2.1.6279.6001.138080888843357047811238713686')
#lidc_processor.check_imperfect_data()
#lidc_processor.visualize_annotation()
#lidc_processor.visualize_annotation_in_3d('1.3.6.1.4.1.14519.5.2.1.6279.6001.138080888843357047811238713686')
