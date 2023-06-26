import pylidc as pl
from tqdm import tqdm as tq
import os
import numpy as np
from .prep_utils import CASE_LIST, get_consensus_mask, get_hu_image
import json
import os.path

def process_data():
    """
    Process the data from the LIDC dataset and generate consensus masks and CT images for each series.

    Returns:
    - None

    Parameters:
    - None

    Saves:
    - Consensus masks: Numpy arrays representing the consensus masks for each series. Saved in the 'consensus_masks' directory.
    - Intersection masks: Numpy arrays representing the intersection masks for each series. Saved in the 'intersection_masks' directory.
    - Union masks: Numpy arrays representing the union masks for each series. Saved in the 'union_masks' directory.
    - CT images: Numpy arrays representing the CT images for each series. Saved in the 'images' directory.
    - Series UID to nodule map: A JSON file mapping the series UIDs to the nodules present in each series. Saved as 'seriesuid_nodule_map.json'.
    """
    savepath = 'lidc_data'  # Relative path to the save directory

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    ct_lidc = pl.query(pl.Scan)
    failed_cases = []
    series_nod = {}

    for scan in tq(ct_lidc, total=ct_lidc.count()):
        series_uid = scan.series_instance_uid
        patient_id = scan.patient_id

        if patient_id not in CASE_LIST and series_uid not in failed_cases:
            vol = scan.to_volume(verbose=False)
            vol_shape = vol.shape
            nods = scan.cluster_annotations()

            if len(scan.annotations) > 0:
                slice_list = []  # List to store the slices which have nodules marked
                nodule_dict = {}
                num_nodules = len(nods)

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
                                con = ann.contours[j]  # All contours for a specific nodule collected
                                k_slice = con.image_k_position  # Returns the slice number/index which has the nodule
                                ii, jj = ann.contours[j].to_matrix(include_k=False).T
                                slice_dict[k_slice] = (ii, jj)

                        ann_dict[i] = slice_dict

                    nodule_dict[k] = ann_dict

                consensus_mask, intersection_mask, union_mask, uncertainty_mask, flag, nod_sl = get_consensus_mask(
                    vol_shape, nodule_dict)
                nod_sl = get_consensus_mask(vol_shape, nodule_dict)

                series_nod[str(series_uid)] = nod_sl

                if flag == True:
                    ct_image = get_hu_image(vol)
                    consensus_path = os.path.join(savepath, 'consensus_masks')
                    intersection_path = os.path.join(savepath, 'intersection_masks')
                    union_path = os.path.join(savepath, 'union_masks')
                    ct_path = os.path.join(savepath, 'images')

                    if not os.path.exists(consensus_path):
                        os.makedirs(consensus_path)
                    if not os.path.exists(intersection_path):
                        os.makedirs(intersection_path)
                    if not os.path.exists(union_path):
                        os.makedirs(union_path)
                    if not os.path.exists(ct_path):
                        os.makedirs(ct_path)

                    np.save(os.path.join(consensus_path, f'{series_uid}.npy'), consensus_mask)
                    np.save(os.path.join(intersection_path, f'{series_uid}.npy'), intersection_mask)
                    np.save(os.path.join(union_path, f'{series_uid}.npy'), union_mask)
                    np.save(os.path.join(ct_path, f'{series_uid}.npy'), ct_image)
                else:
                    print(series_uid)
                    failed_cases.append(series_uid)
            else:
                consensus_mask = np.zeros(vol_shape)
                ct_image = get_hu_image(vol)
                ct_path = os.path.join(savepath, 'images')
                consensus_path = os.path.join(savepath, 'consensus_masks')

                if not os.path.exists(consensus_path):
                    os.makedirs(consensus_path)
                if not os.path.exists(ct_path):
                    os.makedirs(ct_path)

                np.save(os.path.join(consensus_path, f'{series_uid}.npy'), consensus_mask)
                np.save(os.path.join(ct_path, f'{series_uid}.npy'), ct_image)

    with open('seriesuid_nodule_map.json', 'w') as f2:
        json.dump(series_nod, f2)

# Call the function to execute the code
process_data()
