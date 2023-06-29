import os
import json
from tqdm.notebook import tqdm as tq
import numpy as np

def extract_nodule_attributes(annotations_file, patches_directory):
    """
    Extracts nodule attributes from patch filenames based on the provided annotations file.

    Args:
        annotations_file (str): The path to the JSON file containing the nodule annotations.
        patches_directory (str): The path to the directory containing the patch files.

    Returns:
        tuple: A tuple containing the unique values and their counts for each nodule attribute
            (subtlety, texture, sphericity, margin, lobulation, spiculation, malignancy).
    """
    with open(annotations_file) as f:
        dataset_annotation = json.load(f)

    file_list = os.listdir(patches_directory)
    sub_list = []
    text_list = []
    sp_list = []
    margin_list = []
    lob_list = []
    spic_list = []
    mal_list = []

    for filename in tq(file_list):
        series_uid = filename.split('_')[0]
        slice_num = filename.split('_')[1]
        nod_temp = filename.split('_')[2]
        nodule_num = nod_temp.split('.')[0]
        nodule_attribute = dataset_annotation[series_uid][nodule_num]
        sub_list.append(nodule_attribute['attributes']['subtlety'])
        text_list.append(nodule_attribute['attributes']['texture'])
        sp_list.append(nodule_attribute['attributes']['sphericity'])
        margin_list.append(nodule_attribute['attributes']['margin'])
        lob_list.append(nodule_attribute['attributes']['lobulation'])
        spic_list.append(nodule_attribute['attributes']['spiculation'])
        mal_list.append(nodule_attribute['attributes']['malignancy'])

    unique_sub, counts_sub = np.unique(sub_list, return_counts=True)
    unique_text, counts_text = np.unique(text_list, return_counts=True)
    unique_sp, counts_sp = np.unique(sp_list, return_counts=True)
    unique_margin, counts_margin = np.unique(margin_list, return_counts=True)
    unique_lob, counts_lob = np.unique(lob_list, return_counts=True)
    unique_spic, counts_spic = np.unique(spic_list, return_counts=True)
    unique_mal, counts_mal = np.unique(mal_list, return_counts=True)

    return (unique_sub, counts_sub), (unique_text, counts_text), (unique_sp, counts_sp), \
           (unique_margin, counts_margin), (unique_lob, counts_lob), (unique_spic, counts_spic), \
           (unique_mal, counts_mal)

# Example usage

#annotations_file = '/home/sankarshanaa/jsons/nodule_attributes_binary.json'
#patches_directory = '/home/rakshith/isbi_data2/patches2/img'
#attribute_counts = extract_nodule_attributes(annotations_file, patches_directory)

# Access the attribute counts
#(unique_sub, counts_sub), (unique_text, counts_text), (unique_sp, counts_sp), \
#(unique_margin, counts_margin), (unique_lob, counts_lob), (unique_spic, counts_spic), \
#(unique_mal, counts_mal) = attribute_counts

# Print the unique values and their counts for each attribute
#print("Subtlety:", unique_sub, counts_sub)
#print("Texture:", unique_text, counts_text)
#print("Sphericity:", unique_sp, counts_sp)
#print("Margin:", unique_margin, counts_margin)
#print("Lobulation:", unique_lob, counts_lob)
#print("Spiculation:", unique_spic, counts_spic)
#print("Malignancy:", unique_mal, counts_mal)
