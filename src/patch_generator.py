import numpy as np
import os
import cv2
import json
from tqdm import tqdm as tq
import pickle
from skimage.util.shape import view_as_windows
import random
from collections import defaultdict

class PatchExtractor:
    """
    A class for extracting patches from image volumes and corresponding masks.

    Parameters:
        data_path (str): The path to the data directory.

    Attributes:
        data_path (str): The path to the data directory.
        imgpath (str): The path to the images directory.
        consensus_maskpath (str): The path to the consensus masks directory.
        intersection_maskpath (str): The path to the intersection masks directory.
        union_maskpath (str): The path to the union masks directory.
        savepath (str): The path to the patches directory.
        file_list (list): A list of filenames in the images directory.
        consensus_list (list): A list of filenames in the consensus masks directory.
        union_list (list): A list of filenames in the union masks directory.
        intersection_list (list): A list of filenames in the intersection masks directory.
        series_id_list (list): A list of series IDs.
        id_list_2 (list): A list of IDs extracted from filenames in the images directory.
        series_id_new_list (list): A filtered list of series IDs that exist in both series_id_list and id_list_2.
    """

    def __init__(self, data_path):

        assert isinstance(data_path, str), "Data path should be a string."

        self.data_path = data_path
        self.imgpath = os.path.join(data_path, 'images')
        self.consensus_maskpath = os.path.join(data_path, 'consensus_masks')
        self.intersection_maskpath = os.path.join(data_path, 'intersection_masks')
        self.union_maskpath = os.path.join(data_path, 'union_masks')
        self.savepath = os.path.join(data_path, 'patches')
        
        assert os.path.exists(self.savepath) or os.makedirs(self.savepath), "Failed to create the save path directory."

        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)
        
        with open('/home/rakshith/MIRIAD_WP4/stage3_segmentation_classification/jsons/seriesuid_nodule_map.json') as f:
            self.seriesuid_nod = json.load(f)
        
    def setup_file_paths(self):
          """
        Set up the file paths and create lists of filenames.

        Returns:
            None
        """
        assert isinstance(self.imgpath, str), "imgpath should be a string."
        assert isinstance(self.consensus_maskpath, str), "consensus_maskpath should be a string."
        assert isinstance(self.union_maskpath, str), "union_maskpath should be a string."
        assert isinstance(self.intersection_maskpath, str), "intersection_maskpath should be a string."
        assert isinstance(self.seriesuid_nod, dict), "seriesuid_nod should be a dictionary."

        self.file_list = os.listdir(self.imgpath)
        self.consensus_list = os.listdir(self.consensus_maskpath)
        self.union_list = os.listdir(self.union_maskpath)
        self.intersection_list = os.listdir(self.intersection_maskpath)

        self.series_id_list = list(self.seriesuid_nod.keys())  # Obtain the keys of self.seriesuid_nod as a list
        assert isinstance(self.series_id_list, list), "series_id_list should be a list."
        assert all(isinstance(id, str) for id in self.series_id_list), "series_id_list should contain strings only."
        self.id_list_2 = [id.split('.npy')[0] for id in self.file_list]

        assert isinstance(self.series_id_new_list, list), "series_id_new_list should be a list."
        assert all(isinstance(id, str) for id in self.series_id_new_list), "series_id_new_list should contain strings only."
        self.series_id_new_list = [id for id in self.series_id_list if id in self.id_list_2]

        
    def process_patches(self):
        """
        Process the image volumes and corresponding masks to extract patches.

        Returns:
            None
        """
        img_savepath = os.path.join(self.savepath, 'img')
        mask_savepath = os.path.join(self.savepath, 'c_mask')
        u_mask_savepath = os.path.join(self.savepath, 'u_mask')
        i_mask_savepath = os.path.join(self.savepath, 'i_mask')

        if not os.path.exists(img_savepath):
            os.makedirs(img_savepath)

        if not os.path.exists(mask_savepath):
            os.makedirs(mask_savepath)

        if not os.path.exists(u_mask_savepath):
            os.makedirs(u_mask_savepath)

        if not os.path.exists(i_mask_savepath):
            os.makedirs(i_mask_savepath)

        for series_id in tq(self.series_id_new_list):
            size = 64
            index = 0
            filename = series_id + '.npy'
            img_vol = np.load(os.path.join(self.imgpath, filename))
            mask_vol = np.load(os.path.join(self.consensus_maskpath, filename))
            union_vol = np.load(os.path.join(self.union_maskpath, filename))
            intersection_vol = np.load(os.path.join(self.intersection_maskpath, filename))
            img_shape = img_vol.shape
            seriesid = filename.split('.npy')[0]
            missing_list = []
            slice_list = self.seriesuid_nod[series_id].keys()

            for sl in slice_list:
                sl = int(sl)
                mask = mask_vol[:, :, sl]
                mask = mask.astype(np.uint8)

                if np.any(mask):
                    img = img_vol[:, :, sl]
                    u_mask = union_vol[:, :, sl]
                    u_mask = u_mask.astype(np.uint8)
                    i_mask = intersection_vol[:, :, sl]
                    i_mask = i_mask.astype(np.uint8)
                    keys_in_json = self.seriesuid_nod[seriesid].keys()
                    noduleno = self.seriesuid_nod[seriesid][str(sl)]
                    _, th_mask = cv2.threshold(mask, 0.5, 1, 0, cv2.THRESH_BINARY)

                    contours, hierarchy = cv2.findContours(th_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    contours = sorted(contours, key=lambda x: cv2.contourArea(x))

                    _, uth_mask = cv2.threshold(u_mask, 0.5, 1, 0, cv2.THRESH_BINARY)
                    u_contours, u_hierarchy = cv2.findContours(uth_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    u_contours = sorted(u_contours, key=lambda x: cv2.contourArea(x))

                    _, ith_mask = cv2.threshold(i_mask, 0.5, 1, 0, cv2.THRESH_BINARY)
                    i_contours, i_hierarchy = cv2.findContours(ith_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    i_contours = sorted(i_contours, key=lambda x: cv2.contourArea(x))

                    for cntr in contours:
                        xr, yr, wr, hr = cv2.boundingRect(cntr)
                        xc, yc = int(xr + wr / 2), int(yr + hr / 2)

                        if int(yc - size / 2) < 0 or int(xc - size / 2) < 0:
                            if int(yc - size / 2) < 0 and int(xc - size / 2) < 0:
                                patch_img1 = img[0:size, 0:size].copy().astype(np.float16)
                                patch_mask1 = mask[0:size, 0:size].copy().astype(np.float16)
                                patch_mask2 = u_mask[0:size, 0:size].copy().astype(np.float16)
                                patch_mask3 = i_mask[0:size, 0:size].copy().astype(np.float16)

                            elif int(yc - size / 2) > 0 and int(xc - size / 2) < 0:
                                patch_img1 = img[int(yc - size / 2):int(yc + size / 2), 0:size].copy().astype(np.float16)
                                patch_mask1 = mask[int(yc - size / 2):int(yc + size / 2), 0:size].copy().astype(np.float16)
                                patch_mask2 = u_mask[0:size, 0:size].copy().astype(np.float16)
                                patch_mask3 = i_mask[0:size, 0:size].copy().astype(np.float16)

                            elif int(yc - size / 2) < 0 and int(xc - size / 2) > 0:
                                patch_img1 = img[0:size, int(xc - size / 2):int(xc + size / 2)].copy().astype(np.float16)
                                patch_mask1 = mask[0:size, int(xc - size / 2):int(xc + size / 2)].copy().astype(np.float16)
                                patch_mask2 = u_mask[0:size, 0:size].copy().astype(np.float16)
                                patch_mask3 = i_mask[0:size, 0:size].copy().astype(np.float16)

                        elif int(yc + size / 2) > 512 or int(xc + size / 2) > 512:
                            if int(yc + size / 2) > 512 and int(xc + size / 2) > 512:
                                m = yc + size - 512
                                n = xc + size - 512
                                patch_img1 = img[int(yc - m):512, int(xc - n):512].copy().astype(np.float16)
                                patch_mask1 = mask[int(yc - m):512, int(xc - n):512].copy().astype(np.float16)
                                patch_mask2 = u_mask[0:size, 0:size].copy().astype(np.float16)
                                patch_mask3 = i_mask[0:size, 0:size].copy().astype(np.float16)

                            elif int(yc + size / 2) > 512 and int(xc + size / 2) < 512:
                                m = yc + size - 512
                                patch_img1 = img[int(yc - m):512, int(xc - size / 2):int(xc + size / 2)].copy().astype(np.float16)
                                patch_mask1 = mask[int(yc - m):512, int(xc - size / 2):int(xc + size / 2)].copy().astype(np.float16)
                                patch_mask2 = u_mask[int(yc - m):512, int(xc - size / 2):int(xc + size / 2)].copy().astype(np.float16)
                                patch_mask3 = i_mask[int(yc - m):512, int(xc - size / 2):int(xc + size / 2)].copy().astype(np.float16)

                            elif int(yc + size / 2) < 512 and int(xc + size / 2) > 512:
                                n = xc + size - 512
                                patch_img1 = img[int(yc - size / 2):int(yc + size / 2), int(xc - n):512].copy().astype(np.float16)
                                patch_mask1 = mask[int(yc - size / 2):int(yc + size / 2), int(xc - n):512].copy().astype(np.float16)
                                patch_mask2 = u_mask[int(yc - size / 2):int(yc + size / 2), int(xc - n):512].copy().astype(np.float16)
                                patch_mask3 = i_mask[int(yc - size / 2):int(yc + size / 2), int(xc - n):512].copy().astype(np.float16)

                        elif int(yc - size / 2) >= 0 and int(yc + size / 2) <= 512:
                            if int(xc - size / 2) >= 0 and int(xc + size / 2) <= 512:
                                patch_img1 = img[int(yc - size / 2):int(yc + size / 2), int(xc - size / 2):int(xc + size / 2)].copy().astype(np.float16)
                                patch_mask1 = mask[int(yc - size / 2):int(yc + size / 2), int(xc - size / 2):int(xc + size / 2)].copy().astype(np.float16)
                                patch_mask2 = u_mask[int(yc - size / 2):int(yc + size / 2), int(xc - size / 2):int(xc + size / 2)].copy().astype(np.float16)
                                patch_mask3 = i_mask[int(yc - size / 2):int(yc + size / 2), int(xc - size / 2):int(xc + size / 2)].copy().astype(np.float16)

                        if np.shape(patch_img1) != (64, 64):
                            print('shape', np.shape(patch_img1))
                            print('coordinate of BBox', xr, yr, wr, hr)

                        if not os.path.isdir(img_savepath):
                            os.makedirs(img_savepath)

                        if not os.path.isdir(mask_savepath):
                            os.makedirs(mask_savepath)

                        if not os.path.isdir(u_mask_savepath):
                            os.makedirs(u_mask_savepath)

                        if not os.path.isdir(i_mask_savepath):
                            os.makedirs(i_mask_savepath)

                        if np.shape(patch_img1) != (64, 64):
                            missing_list.append((filename, sl))

                        patch_img1 = patch_img1 - np.mean(patch_img1)
                        patch_img1 = patch_img1 / np.std(patch_img1)
                        np.save(os.path.join(img_savepath, str(seriesid) + '_' + str(sl) + '_' + str(index) + '.npy'), patch_img1)
                        np.save(os.path.join(mask_savepath, str(seriesid) + '_' + str(sl) + '_' + str(index) + '.npy'), patch_mask1)
                        np.save(os.path.join(u_mask_savepath, str(seriesid) + '_' + str(sl) + '_' + str(index) + '.npy'), patch_mask2)
                        np.save(os.path.join(i_mask_savepath, str(seriesid) + '_' + str(sl) + '_' + str(index) + '.npy'), patch_mask3)
                        index += 1

        if len(missing_list) > 0:
            print('Missing List:', missing_list)
        else:
            print('All Patches Extracted')


#to execute:
#extractor = PatchExtractor('/path/to/data')
#extractor.setup_file_paths()
#extractor.process_patches()


class ImagePatchSplitter:
    def __init__(self, current_dir, target_location):
        """
        Initializes an instance of the ImagePatchSplitter class.

        Parameters:
        - current_dir (str): The current directory path.
        - target_location (str): The target location directory path.
        """

        assert isinstance(current_dir, str), "current_dir must be a string"
        assert isinstance(target_location, str), "target_location must be a string"
        assert os.path.isdir(current_dir), "current_dir must be a valid directory"
        assert os.path.isdir(target_location), "target_location must be a valid directory"

        self.current_dir = current_dir
        self.target_location = target_location
        self.data_path = os.path.join(target_location, 'isbi_data2')
        self.imgpath = os.path.join(self.data_path, 'images')
        self.maskpath = os.path.join(self.data_path, 'masks')
        self.savepath = self.data_path
        self.img_dir = self.imgpath
        self.mask_dir = self.maskpath
        self.seriesuid_nod = {}

    def load_seriesuid_nodule_map(self, filepath):
        """
        Loads the seriesuid_nodule_map from the specified filepath.

        Parameters:
        - filepath (str): The filepath of the seriesuid_nodule_map JSON file.
        """

        assert isinstance(filepath, str), "filepath must be a string"
        assert os.path.isfile(filepath), "filepath must be a valid file path"

        with open(filepath) as f:
            self.seriesuid_nod = json.load(f)

    def process_image_patches(self, size=64):
        """
        Processes image patches from the input images and masks.

        Parameters:
        - size (int): The size of the image patch. Default is 64.

        Prints:
        - shape: Shape of the image patch if it is not (64, 64).
        - coordinate of patch: x, y coordinates of the patch.
        - coordinate of BBox: x, y, width, height of the bounding box.

        Saves:
        - Patches: Image patches and corresponding masks as numpy arrays in the 'patches/img' and 'patches/mask' directories.
        """
        file_list = os.listdir(self.imgpath)
        for idx, filename in enumerate(tq(file_list)):
            index = 0
            img_vol = np.load(os.path.join(self.img_dir, filename))
            mask_vol = np.load(os.path.join(self.mask_dir, filename))
            img_shape = img_vol.shape
            seriesid = filename.split('.npy')[0]

            for sl in range(img_shape[2]):
                img = img_vol[:, :, sl]
                mask = mask_vol[:, :, sl]
                mask = mask.astype(np.uint8)

                if np.any(mask):
                    noduleno = self.seriesuid_nod[seriesid][str(sl)]
                    _, th_mask = cv2.threshold(mask, 0.5, 1, 0, cv2.THRESH_BINARY)
                    contours, hierarchy = cv2.findContours(th_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    contours = sorted(contours, key=lambda x: cv2.contourArea(x))

                    for cntr in contours:
                        xr, yr, wr, hr = cv2.boundingRect(cntr)
                        xc, yc = int(xr + wr / 2), int(yr + hr / 2)

                        if int(yc - size / 2) < 0 or int(xc - size / 2) < 0:
                            if int(yc - size / 2) < 0 and int(xc - size / 2) < 0:
                                patch_img1 = img[0:size, 0:size].copy().astype(np.float16)
                                patch_mask1 = mask[0:size, 0:size].copy().astype(np.float16)

                            elif int(yc - size / 2) > 0 and int(xc - size / 2) < 0:
                                patch_img1 = img[int(yc - size / 2):int(yc + size / 2), 0:size].copy().astype(np.float16)
                                patch_mask1 = mask[int(yc - size / 2):int(yc + size / 2), 0:size].copy().astype(np.float16)

                            elif int(yc - size / 2) < 0 and int(xc - size / 2) > 0:
                                patch_img1 = img[0:size, int(xc - size / 2):int(xc + size / 2)].copy().astype(np.float16)
                                patch_mask1 = mask[0:size, int(xc - size / 2):int(xc + size / 2)].copy().astype(np.float16)

                        elif int(yc + size / 2) > 512 or int(xc + size / 2) > 512:
                            if int(yc + size / 2) > 512 and int(xc + size / 2) > 512:
                                m = yc + size - 512
                                n = xc + size - 512
                                patch_img1 = img[int(yc - m):512, int(xc - n):512].copy().astype(np.float16)
                                patch_mask1 = mask[int(yc - m):512, int(xc - n):512].copy().astype(np.float16)

                            elif int(yc + size / 2) > 512 and int(xc + size / 2) < 512:
                                m = yc + size - 512
                                patch_img1 = img[int(yc - m):512, int(xc - size / 2):int(xc + size / 2)].copy().astype(np.float16)
                                patch_mask1 = mask[int(yc - m):512, int(xc - size / 2):int(xc + size / 2)].copy().astype(np.float16)

                            elif int(yc + size / 2) < 512 and int(xc + size / 2) > 512:
                                n = xc + size - 512
                                patch_img1 = img[int(yc - size / 2):int(yc + size / 2), int(xc - n):512].copy().astype(np.float16)
                                patch_mask1 = mask[int(yc - size / 2):int(yc + size / 2), int(xc - n):512].copy().astype(np.float16)

                        elif (int(yc - size / 2) >= 0 and int(yc + size / 2) <= 512):
                            if (int(xc - size / 2) >= 0 and int(xc + size / 2) <= 512):
                                patch_img1 = img[int(yc - size / 2):int(yc + size / 2), int(xc - size / 2):int(xc + size / 2)].copy().astype(np.float16)
                                patch_mask1 = mask[int(yc - size / 2):int(yc + size / 2), int(xc - size / 2):int(xc + size / 2)].copy().astype(np.float16)

                        if np.shape(patch_img1) != (64, 64):
                            print('shape', np.shape(patch_img1))
                            print('coordinate of patch', x, x + size, y, y + size)
                            print('coordinate of BBox', xr, yr, wr, hr)

                        img_savepath = self.savepath + '/patches/' + '/img/'
                        mask_savepath = self.savepath + '/patches/' + '/mask/'
                        if not os.path.isdir(img_savepath):
                            os.makedirs(self.savepath + '/patches/' + '/img/')
                            np.save(img_savepath + seriesid + '_' + str(sl) + '_' + str(noduleno) + '.npy', patch_img1)
                        else:
                            np.save(img_savepath + seriesid + '_' + str(sl) + '_' + str(noduleno) + '.npy', patch_img1)

                        if not os.path.isdir(mask_savepath):
                            os.makedirs(self.savepath + '/patches/' + '/mask/')
                            np.save(mask_savepath + seriesid + '_' + str(sl) + '_' + str(noduleno) + '.npy', patch_mask1)
                        else:
                            np.save(mask_savepath + seriesid + '_' + str(sl) + '_' + str(noduleno) + '.npy', patch_mask1)

    def split_data(self):
        """
        Split the data into train, validation, and test sets.

        This method splits the image data into train, validation, and test sets based on a random sampling of series IDs. The split ratio is 70% for training, 20% for validation, and 10% for testing. The split information is saved in a JSON file named 'patch_split_new.json'.

        Returns:
            None
        """
        image_list = os.listdir(self.imgpath)
        series_list = []
        for file in image_list:
            seriesid = file.split('_')[0]
            if seriesid not in series_list:
                series_list.append(seriesid)

        test_list = random.sample(series_list, 70)
        valid_list = []
        train_list = []
        count = 0
        for file in series_list:
            if file not in test_list:
                valid_list.append(file)
                count += 1

            if count == 70:
                break

        for file in series_list:
            if file not in test_list:
                if file not in valid_list:
                    train_list.append(file)

        train_image_list = []
        valid_image_list = []
        test_image_list = []

        for idx in train_list:
            for file in image_list:
                if idx in file:
                    train_image_list.append(file)

        for idx in valid_list:
            for file in image_list:
                if idx in file:
                    valid_image_list.append(file)

        for idx in test_list:
            for file in image_list:
                if idx in file:
                    test_image_list.append(file)

        patch_split = {'train_set': train_image_list, 'valid_set': valid_image_list, 'test_set': test_image_list}
        with open('patch_split_new.json', 'w') as f:
            json.dump(patch_split, f)
