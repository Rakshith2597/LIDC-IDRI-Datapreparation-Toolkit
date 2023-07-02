import pylidc as pl
from tqdm.notebook import tqdm as tq
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import cv2

class NoduleMaskGenerator:
    def __init__(self, savepath):
        """
        Initializes an instance of NoduleMaskGenerator.

        Parameters:
        - savepath (str): Path to save the generated masks and images.
        """
        self.savepath = savepath
        self.failed_cases = []
        self.series_nod = {}

    def get_consensus_mask(self, vol_shape, nodule_dict):
        """
        Generates the consensus mask for a given nodule.

        Parameters:
        - vol_shape (tuple): Shape of the volume (3D array).
        - nodule_dict (dict): Dictionary containing the contours of the nodule.

        Returns:
        - mask_output (3D array): Consensus mask for the nodule.
        - flag (bool): Flag indicating whether the consensus mask was successfully generated.
        - nod_sl (dict): Dictionary mapping slice numbers to nodules.
        """
        assert isinstance(vol_shape, tuple), "vol_shape must be a tuple"
        assert len(vol_shape) == 3, "vol_shape must have length 3"
        assert all(isinstance(dim, int) and dim > 0 for dim in vol_shape), "vol_shape dimensions must be positive integers"

        assert isinstance(nodule_dict, dict), "nodule_dict must be a dictionary"

        temp_mask = np.zeros((512, 512))
        mask_output = np.zeros(vol_shape)
        flag = True
        nod_sl = {}

        for i in nodule_dict.keys():
            num_radio = len(nodule_dict[i].keys())
            slice_nums = nodule_dict[i][0].keys()

            for sl in slice_nums:
                nod_sl[int(sl)] = i
                if num_radio == 4:
                    try:
                        A1 = nodule_dict[i][0][sl]
                        merged_list = np.array([[A1[1][i], A1[0][i]] for i in range(0, len(A1[0]))])
                        a1_mask = temp_mask.copy()
                        cv2.fillPoly(a1_mask, pts=np.int32([merged_list]), color=(1, 1, 1))
                    except:
                        a1_mask = temp_mask.copy()

                    try:
                        A2 = nodule_dict[i][1][sl]
                        merged_list = np.array([[A2[1][i], A2[0][i]] for i in range(0, len(A2[0]))])
                        a2_mask = temp_mask.copy()
                        cv2.fillPoly(a2_mask, pts=np.int32([merged_list]), color=(1, 1, 1))
                    except:
                        a2_mask = temp_mask.copy()

                    try:
                        A3 = nodule_dict[i][2][sl]
                        merged_list = np.array([[A3[1][i], A3[0][i]] for i in range(0, len(A3[0]))])
                        a3_mask = temp_mask.copy()
                        cv2.fillPoly(a3_mask, pts=np.int32([merged_list]), color=(1, 1, 1))
                    except:
                        a3_mask = temp_mask.copy()

                    try:
                        A4 = nodule_dict[i][3][sl]
                        merged_list = np.array([[A4[1][i], A4[0][i]] for i in range(0, len(A4[0]))])
                        a4_mask = temp_mask.copy()
                        cv2.fillPoly(a4_mask, pts=np.int32([merged_list]), color=(1, 1, 1))
                    except:
                        a4_mask = temp_mask.copy()

                    combined_mask = np.array((a1_mask, a2_mask, a3_mask, a4_mask))
                    mask = np.mean(combined_mask, axis=0) >= 0.5
                    mask_output[:, :, sl] = mask

                elif num_radio == 3:
                    try:
                        A1 = nodule_dict[i][0][sl]
                        merged_list = np.array([[A1[1][i], A1[0][i]] for i in range(0, len(A1[0]))])
                        a1_mask = temp_mask.copy()
                        cv2.fillPoly(a1_mask, pts=np.int32([merged_list]), color=(1, 1, 1))
                    except:
                        a1_mask = temp_mask.copy()

                    try:
                        A2 = nodule_dict[i][1][sl]
                        merged_list = np.array([[A2[1][i], A2[0][i]] for i in range(0, len(A2[0]))])
                        a2_mask = temp_mask.copy()
                        cv2.fillPoly(a2_mask, pts=np.int32([merged_list]), color=(1, 1, 1))
                    except:
                        a2_mask = temp_mask.copy()

                    try:
                        A3 = nodule_dict[i][2][sl]
                        merged_list = np.array([[A3[1][i], A3[0][i]] for i in range(0, len(A3[0]))])
                        a3_mask = temp_mask.copy()
                        cv2.fillPoly(a3_mask, pts=np.int32([merged_list]), color=(1, 1, 1))
                    except:
                        a3_mask = temp_mask.copy()

                    combined_mask = np.array((a1_mask, a2_mask, a3_mask))
                    mask = np.mean(combined_mask, axis=0) >= 0.5
                    mask_output[:, :, sl] = mask

                elif num_radio == 2:
                    try:
                        A1 = nodule_dict[i][0][sl]
                        merged_list = np.array([[A1[1][i], A1[0][i]] for i in range(0, len(A1[0]))])
                        a1_mask = temp_mask.copy()
                        cv2.fillPoly(a1_mask, pts=np.int32([merged_list]), color=(1, 1, 1))
                    except:
                        a1_mask = temp_mask.copy()

                    try:
                        A2 = nodule_dict[i][1][sl]
                        merged_list = np.array([[A2[1][i], A2[0][i]] for i in range(0, len(A2[0]))])
                        a2_mask = temp_mask.copy()
                        cv2.fillPoly(a2_mask, pts=np.int32([merged_list]), color=(1, 1, 1))
                    except:
                        a2_mask = temp_mask.copy()

                    combined_mask = np.array((a1_mask, a2_mask))
                    mask = np.mean(combined_mask, axis=0) >= 0.5
                    mask_output[:, :, sl] = mask

                elif num_radio == 1:
                    try:
                        A1 = nodule_dict[i][0][sl]
                        merged_list = np.array([[A1[1][i], A1[0][i]] for i in range(0, len(A1[0]))])
                        a1_mask = temp_mask.copy()
                        cv2.fillPoly(a1_mask, pts=np.int32([merged_list]), color=(1, 1, 1))
                    except:
                        a1_mask = temp_mask.copy()

                    combined_mask = np.array((a1_mask))
                    mask = combined_mask
                    mask_output[:, :, sl] = mask

                else:
                    flag = False

        return mask_output, flag, nod_sl

    def generate_masks(self):
        """
    Generates masks for nodules in CT scans and saves them as numpy arrays.
    
    Parameters:
    - savepath (str): The path to save the generated masks and images.
    
    Returns:
    None
    """
    ct_lidc = pl.query(pl.Scan)

    for idx, scan in enumerate(tq(ct_lidc)):
        series_uid = scan.series_instance_uid
        vol = scan.to_volume(verbose=False)
        vol_shape = vol.shape
        nods = scan.cluster_annotations()

        assert isinstance(vol_shape, tuple), "vol_shape should be a tuple"
        assert len(vol_shape) == 3, "vol_shape should have three dimensions"
        assert isinstance(nodule_dict, dict), "nodule_dict should be a dictionary"

        for k, nod in enumerate(nods):
            num_annotation = len(nod)
            ann_dict = {}

            for i in range(num_annotation):
                ann = nod[i]
                kvals = ann.contour_slice_indices
                slice_list.append(kvals)
                num_slices = len(kvals)
                slice_dict = {}

                for j in range(num_slices):
                    if ann.contours[j]:
                        con = ann.contours[j]
                        k_slice = con.image_k_position
                        ii, jj = ann.contours[j].to_matrix(include_k=False).T
                        slice_dict[k_slice] = (ii, jj)
                ann_dict[i] = slice_dict
            nodule_dict[k] = ann_dict

        consensus_mask, flag, nod_sl = self.get_consensus_mask(vol_shape, nodule_dict)
        self.series_nod[str(series_uid)] = nod_sl

        assert isinstance(consensus_mask, np.ndarray), "consensus_mask should be a numpy array"
        assert consensus_mask.shape == vol_shape, "consensus_mask shape should match vol_shape"
        assert isinstance(ct_image, np.ndarray), "ct_image should be a numpy array"
        assert ct_image.shape == vol_shape, "ct_image shape should match vol_shape"

        if flag:
            np.save(os.path.join(self.savepath, 'masks', series_uid + '.npy'), consensus_mask)
            np.save(os.path.join(self.savepath, 'images', series_uid + '.npy'), ct_image)
        else:
            print(series_uid)
            self.failed_cases.append(series_uid)

    print('Total failed cases', len(self.failed_cases))

    with open('failed_cases.json', 'w') as f:
        json.dump(self.failed_cases, f)

    with open('seriesuid_nodule_map.json', 'w') as f2:
        json.dump(self.series_nod, f2)