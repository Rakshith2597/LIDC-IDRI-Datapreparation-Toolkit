import pylidc as pl
from tqdm.notebook import tqdm as tq
import json
import os
import numpy as np
import plotly.graph_objects as go
from collections import Counter, defaultdict

def highest_rank(lst):
    """
    Returns the most common element in a list.

    Args:
        lst (list): The list of elements.

    Returns:
        object: The most common element in the list.
    """
    data = Counter(lst)
    return data.most_common(1)[0][0]

class NoduleAttributesGenerator:
    def __init__(self):
        """
        Initializes an instance of the NoduleAttributesGenerator class.
        """
        self.dataset_annotation = defaultdict(lambda: '')
        self.dataset_annotation_unsure = defaultdict(lambda: '')
        self.series_uid_list = []

    def generate_nodule_attributes(self, num_scans=4):
        """
        Generates nodule attributes from the LIDC dataset.

        Args:
            num_scans (int): The number of scans to process.

        Returns:
            None
        """
        ct_lidc = pl.query(pl.Scan)
        print('Total CT volumes in LIDC:', ct_lidc.count())

        count_clean = 0
        count_nodule = 0
        count_nodule_Rad34 = 0
        diam_list_total = []

        for idx, scan in enumerate(tq(ct_lidc)):
            series_uid = scan.series_instance_uid

            if len(scan.annotations) > 0:
                nodules = scan.cluster_annotations()
                count_nodule += len(nodules)
                nodule_anno_dict = {}
                nodule_anno_dict_unsure = {}

                for nodule_no, annotations in enumerate(nodules):
                    if len(annotations) >= 3:
                        count_nodule_Rad34 += 1

                        diameter_list = [a for a in range(len(annotations)) if annotations[a].diameter > 3]
                        malignancy_list = np.array([annotations[c].malignancy for c in diameter_list])
                        subtlety_list = np.array([annotations[c].subtlety for c in diameter_list])
                        sphericity_list = np.array([annotations[c].sphericity for c in diameter_list])
                        margin_list = np.array([annotations[c].margin for c in diameter_list])
                        lobulation_list = np.array([annotations[c].lobulation for c in diameter_list])
                        spiculation_list = np.array([annotations[c].spiculation for c in diameter_list])
                        texture_list = np.array([annotations[c].texture for c in diameter_list])

                        nodule_attributes = {
                            'subtlety': 0,
                            'iStructure': 0,
                            'calcification': 0,
                            'sphericity': 0,
                            'margin': 0,
                            'lobulation': 0,
                            'spiculation': 0,
                            'texture': 0,
                            'malignancy': 0
                        }
                        nodule_attributes_unsure = {
                            'subtlety': 0,
                            'iStructure': 0,
                            'calcification': 0,
                            'sphericity': 0,
                            'margin': 0,
                            'lobulation': 0,
                            'spiculation': 0,
                            'texture': 0,
                            'malignancy': 0
                        }

                        malignancy_list[malignancy_list > 3] = 1
                        malignancy_list[malignancy_list < 3] = 0
                        malignancy_list[malignancy_list == 3] = -1
                        subtlety_list[subtlety_list > 3] = 1
                        subtlety_list[subtlety_list < 3] = 0
                        subtlety_list[subtlety_list == 3] = -1
                        sphericity_list[sphericity_list > 3] = 1
                        sphericity_list[sphericity_list < 3] = 0
                        sphericity_list[sphericity_list == 3] = -1
                        margin_list[margin_list > 3] = 1
                        margin_list[margin_list < 3] = 0
                        margin_list[margin_list == 3] = -1
                        lobulation_list[lobulation_list > 3] = 1
                        lobulation_list[lobulation_list < 3] = 0
                        lobulation_list[lobulation_list == 3] = -1
                        spiculation_list[spiculation_list > 3] = 1
                        spiculation_list[spiculation_list < 3] = 0
                        spiculation_list[spiculation_list == 3] = -1
                        texture_list[texture_list > 3] = 1
                        texture_list[texture_list < 3] = 0
                        texture_list[texture_list == 3] = -1

                        if (len(np.unique(malignancy_list)) != len(malignancy_list)) \
                                and (len(np.unique(malignancy_list)) != len(malignancy_list) / 2):
                            nodule_attributes['malignancy'] = highest_rank(malignancy_list)
                        else:
                            nodule_attributes_unsure['malignancy'] = -1

                        if (len(np.unique(subtlety_list)) != len(subtlety_list)) \
                                and (len(np.unique(subtlety_list)) != len(subtlety_list) / 2):
                            nodule_attributes['subtlety'] = highest_rank(subtlety_list)
                        else:
                            nodule_attributes_unsure['subtlety'] = -1

                        if (len(np.unique(sphericity_list)) != len(sphericity_list)) \
                                and (len(np.unique(sphericity_list)) != len(sphericity_list) / 2):
                            nodule_attributes['sphericity'] = highest_rank(sphericity_list)
                        else:
                            nodule_attributes_unsure['sphericity'] = -1

                        if (len(np.unique(margin_list)) != len(margin_list)) \
                                and (len(np.unique(margin_list)) != len(margin_list) / 2):
                            nodule_attributes['margin'] = highest_rank(margin_list)
                        else:
                            nodule_attributes_unsure['margin'] = -1

                        if (len(np.unique(lobulation_list)) != len(lobulation_list)) \
                                and (len(np.unique(lobulation_list)) != len(lobulation_list) / 2):
                            nodule_attributes['lobulation'] = highest_rank(lobulation_list)
                        else:
                            nodule_attributes_unsure['lobulation'] = -1

                        if (len(np.unique(spiculation_list)) != len(spiculation_list)) \
                                and (len(np.unique(spiculation_list)) != len(spiculation_list) / 2):
                            nodule_attributes['spiculation'] = highest_rank(spiculation_list)
                        else:
                            nodule_attributes_unsure['spiculation'] = -1

                        if (len(np.unique(texture_list)) != len(texture_list)) \
                                and (len(np.unique(texture_list)) != len(texture_list) / 2):
                            nodule_attributes['spiculation'] = highest_rank(texture_list)
                        else:
                            nodule_attributes_unsure['spiculation'] = -1

                        count_clean += 1

                        nodule_anno_dict[nodule_no] = {'attributes': nodule_attributes}
                        nodule_anno_dict_unsure[nodule_no] = {'attributes': nodule_attributes}

                self.dataset_annotation[str(series_uid)] = nodule_anno_dict
                self.dataset_annotation_unsure[str(series_uid)] = nodule_anno_dict_unsure

                if idx > 3:
                    break

        print('Total nodules:', count_nodule)
        print('Nodules with Annotation of 3-4 Rad:', count_nodule_Rad34)
        print(len(self.dataset_annotation))
        print(len(self.dataset_annotation_unsure))

    def save_annotations_to_json(self, sure_json_path, unsure_json_path):
        """
        Save the generated nodule annotations to JSON files.

        Parameters:
        - sure_json_path (str): The path to save the JSON file containing the sure annotations.
        - unsure_json_path (str): The path to save the JSON file containing the unsure annotations.

        Returns:
        None
        """
        sure_json_path = os.path.relpath(sure_json_path)
        unsure_json_path = os.path.relpath(unsure_json_path)

        with open(sure_json_path, 'w') as f:
            json.dump(self.dataset_annotation, f)

        with open(unsure_json_path, 'w') as f:
            json.dump(self.dataset_annotation_unsure, f)

    def load_annotations_from_json(self, json_path):
        """
        Load the nodule annotations from a JSON file.

        Parameters:
        - json_path (str): The path to the JSON file containing the annotations.

        Returns:
        None
        """
        json_path = os.path.relpath(json_path)

        with open(json_path) as f:
            self.dataset_annotation = json.load(f)

    def get_series_uid_list(self):
        """
        Get the list of series instance UIDs.

        Returns:
        - series_uid_list (list): The list of series instance UIDs.
        """
        return list(self.dataset_annotation.keys())

    def get_attribute_list(self, attribute):
        """
        Get the list of attribute values for a specific attribute.

        Parameters:
        - attribute (str): The attribute name.

        Returns:
        - attribute_list (list): The list of attribute values.
        """
        return [self.dataset_annotation[uid]['attributes'][attribute] for uid in self.get_series_uid_list()]
