import numpy as np
import cv2

def get_hu_image(vol, threshold):
    """
    Generate a Hounsfield Unit (HU) image from a volume by applying a threshold.

    Parameters:
        - vol (ndarray): The input volume to be processed. It should be a NumPy ndarray representing a 3D image.
        - threshold (float): The threshold value to apply. Pixels with intensity values greater than the threshold will be clamped to the threshold value,
          and pixels with intensity values less than the negative threshold will be clamped to the negative threshold value.

    Returns:
        ndarray: The HU image obtained from the volume after applying the threshold. It has the same shape and data type as the input volume.
    """
    image = vol.copy()
    image[image > threshold] = threshold
    image[image < -threshold] = -threshold
    return image


def get_consensus_mask(vol_shape, nodule_dict):
    """
    Generate a consensus mask based on radiologist annotations for each nodule.

    Parameters:
        - vol_shape (tuple): The shape of the volume for which the consensus mask is generated.
        - nodule_dict (dict): A dictionary containing radiologist annotations for each nodule. The keys are the nodule IDs, and the values are dictionaries
          containing the annotations for each radiologist. The structure of the dictionary is as follows:
          {
              nodule_id: {
                  radiologist_id: {
                      slice_number: [(x1, y1), (x2, y2), ...]  # List of contour points
                  },
                  ...
              },
              ...
          }

    Returns:
        tuple: A tuple containing the following elements:
        - nod_sl (dict): A dictionary mapping slice numbers to their corresponding nodule IDs.
        - mask_output (ndarray): The consensus mask obtained by combining the radiologist annotations. It has the same shape as the input volume.
        - intersection_of_mask_output (ndarray): An intermediate output representing the intersection of masks for each nodule and slice. It has the same shape as the input volume.
        - union_of_mask_output (ndarray): An intermediate output representing the union of masks for each nodule and slice. It has the same shape as the input volume.
        - uncertainty_output (ndarray): An intermediate output representing the uncertainty of the consensus mask for each nodule and slice. It has the same shape as the input volume.
        - flag (bool): A flag indicating whether the consensus mask generation was successful or not. If any nodule has an invalid number of radiologist annotations, the flag will be False.
    """
    temp_mask = np.zeros((512, 512))
    mask_output = np.zeros(vol_shape)
    intersection_of_mask_output = np.zeros(vol_shape)
    uncertainty_output = np.zeros(vol_shape)
    union_of_mask_output = np.zeros(vol_shape)
    flag = True
    nod_sl = {}

    for i in nodule_dict.keys():
        num_radio = len(nodule_dict[i].keys())  # Gets the number of radiologist annotations for that nodule
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

                union_of_mask = a1_mask + a2_mask + a3_mask + a4_mask
                union_of_mask[union_of_mask > 1] = 1
                intersection_of_mask_a = np.multiply(a1_mask, a2_mask)
                intersection_of_mask_b = np.multiply(a3_mask, a4_mask)
                intersection_of_mask = np.multiply(intersection_of_mask_a, intersection_of_mask_b)
                uncertainty = union_of_mask - intersection_of_mask

                intersection_of_mask_output[:, :, sl] = intersection_of_mask
                uncertainty_output[:, :, sl] = uncertainty
                union_of_mask_output[:, :, sl] = union_of_mask

                combined_mask = np.array((a1_mask, a2_mask, a3_mask, a4_mask))
                mask = np.mean(combined_mask, axis=0) >= 0.5
                mask_output[:, :, sl] = mask
                print(combined_mask.shape)
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

                union_of_mask = a1_mask + a2_mask + a3_mask
                union_of_mask[union_of_mask > 1] = 1
                intersection_of_mask_a = np.multiply(a1_mask, a2_mask)
                intersection_of_mask_b = np.multiply(a3_mask, a4_mask)
                intersection_of_mask = np.multiply(intersection_of_mask_a, a3_mask)
                uncertainty = union_of_mask - intersection_of_mask

                intersection_of_mask_output[:, :, sl] = intersection_of_mask
                uncertainty_output[:, :, sl] = uncertainty
                union_of_mask_output[:, :, sl] = union_of_mask

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

                union_of_mask = a1_mask + a2_mask
                union_of_mask[union_of_mask > 1] = 1
                intersection_of_mask_a = np.multiply(a1_mask, a2_mask)
                intersection_of_mask_b = np.multiply(a3_mask, a4_mask)
                intersection_of_mask = np.multiply(intersection_of_mask_a, a3_mask)
                uncertainty = union_of_mask - intersection_of_mask

                intersection_of_mask_output[:, :, sl] = intersection_of_mask
                uncertainty_output[:, :, sl] = uncertainty
                union_of_mask_output[:, :, sl] = union_of_mask

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

    return nod_sl, mask_output, intersection_of_mask_output, union_of_mask_output, uncertainty_output, flag

CASE_LIST = ['LIDC-IDRI-0028']
''''LIDC-IDRI-0032',
'LIDC-IDRI-0062',
'LIDC-IDRI-0071',
'LIDC-IDRI-0100',
'LIDC-IDRI-0107',
'LIDC-IDRI-0143',
'LIDC-IDRI-0174',
'LIDC-IDRI-0189',
'LIDC-IDRI-0197',
'LIDC-IDRI-0205',
'LIDC-IDRI-0214',
'LIDC-IDRI-0218',
'LIDC-IDRI-0224',
'LIDC-IDRI-0225',
'LIDC-IDRI-0226',
'LIDC-IDRI-0239',
'LIDC-IDRI-0253',
'LIDC-IDRI-0261',
'LIDC-IDRI-0279',
'LIDC-IDRI-0293',
'LIDC-IDRI-0295',
'LIDC-IDRI-0306',
'LIDC-IDRI-0307',
'LIDC-IDRI-0316',
'LIDC-IDRI-0322',
'LIDC-IDRI-0327',
'LIDC-IDRI-0330',
'LIDC-IDRI-0331',
'LIDC-IDRI-0333',
'LIDC-IDRI-0336',
'LIDC-IDRI-0342',
'LIDC-IDRI-0349',
'LIDC-IDRI-0361',
'LIDC-IDRI-0364',
'LIDC-IDRI-0382',
'LIDC-IDRI-0383',
'LIDC-IDRI-0389',
'LIDC-IDRI-0391',
'LIDC-IDRI-0401',
'LIDC-IDRI-0410',
'LIDC-IDRI-0417',
'LIDC-IDRI-0418',
'LIDC-IDRI-0422',
'LIDC-IDRI-0425',
'LIDC-IDRI-0428',
'LIDC-IDRI-0441',
'LIDC-IDRI-0446',
'LIDC-IDRI-0455',
'LIDC-IDRI-0465',
'LIDC-IDRI-0472',
'LIDC-IDRI-0482',
'LIDC-IDRI-0506',
'LIDC-IDRI-0511',
'LIDC-IDRI-0512',
'LIDC-IDRI-0513',
'LIDC-IDRI-0514',
'LIDC-IDRI-0519',
'LIDC-IDRI-0528',
'LIDC-IDRI-0531',
'LIDC-IDRI-0536',
'LIDC-IDRI-0540',
'LIDC-IDRI-0544',
'LIDC-IDRI-0548',
'LIDC-IDRI-0561',
'LIDC-IDRI-0564',
'LIDC-IDRI-0571',
'LIDC-IDRI-0572',
'LIDC-IDRI-0573',
'LIDC-IDRI-0589',
'LIDC-IDRI-0600',
'LIDC-IDRI-0603',
'LIDC-IDRI-0610',
'LIDC-IDRI-0612',
'LIDC-IDRI-0616',
'LIDC-IDRI-0622',
'LIDC-IDRI-0623',
'LIDC-IDRI-0627',
'LIDC-IDRI-0632',
'LIDC-IDRI-0646',
'LIDC-IDRI-0652',
'LIDC-IDRI-0653',
'LIDC-IDRI-0665',
'LIDC-IDRI-0666',
'LIDC-IDRI-0667',
'LIDC-IDRI-0668',
'LIDC-IDRI-0679',
'LIDC-IDRI-0683',
'LIDC-IDRI-0685',
'LIDC-IDRI-0689',
'LIDC-IDRI-0690',
'LIDC-IDRI-0691',
'LIDC-IDRI-0710',
'LIDC-IDRI-0711',
'LIDC-IDRI-0716',
'LIDC-IDRI-0718',
'LIDC-IDRI-0731',
'LIDC-IDRI-0737',
'LIDC-IDRI-0738',
'LIDC-IDRI-0745',
'LIDC-IDRI-0746',
'LIDC-IDRI-0755',
'LIDC-IDRI-0760',
'LIDC-IDRI-0764',
'LIDC-IDRI-0774',
'LIDC-IDRI-0784',
'LIDC-IDRI-0804',
'LIDC-IDRI-0808',
'LIDC-IDRI-0839',
'LIDC-IDRI-0853',
'LIDC-IDRI-0862',
'LIDC-IDRI-0876',
'LIDC-IDRI-0877',
'LIDC-IDRI-0878',
'LIDC-IDRI-0881',
'LIDC-IDRI-0885',
'LIDC-IDRI-0887',
'LIDC-IDRI-0889',
'LIDC-IDRI-0891',
'LIDC-IDRI-0897',
'LIDC-IDRI-0900',
'LIDC-IDRI-0901',
'LIDC-IDRI-0903',
'LIDC-IDRI-0906',
'LIDC-IDRI-0918',
'LIDC-IDRI-0927',
'LIDC-IDRI-0930',
'LIDC-IDRI-0931',
'LIDC-IDRI-0934',
'LIDC-IDRI-0937',
'LIDC-IDRI-0948',
'LIDC-IDRI-0952',
'LIDC-IDRI-0954',
'LIDC-IDRI-0960',
'LIDC-IDRI-0964',
'LIDC-IDRI-0967',
'LIDC-IDRI-0970',
'LIDC-IDRI-0975',
'LIDC-IDRI-0988',
'LIDC-IDRI-0992',
'LIDC-IDRI-0995',
'LIDC-IDRI-0979']'''





    