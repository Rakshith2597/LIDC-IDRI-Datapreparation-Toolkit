import json

def load_json_file(file_path):
    """
    Load data from a JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        list: The data loaded from the JSON file.
    """
    with open(file_path) as f:
        data = json.load(f)
    return data


def merge_json_lists(*json_files):
    """
    Merge multiple JSON lists into a single list.

    Args:
        json_files (str): The paths to the JSON files.

    Returns:
        list: The merged list from the JSON files.
    """
    merged_list = []
    for file_path in json_files:
        data = load_json_file(file_path)
        merged_list.extend(data)
    return merged_list


class NoduleSegmentationStats:
    def __init__(self):
        self.case0_list = []
        self.case1_list = []
        self.case2_list = []
        self.case0_count = 0
        self.case1_count = 0
        self.case2_count = 0

    def load_data(self, case0_files, case1_files, case2_files):
        """
        Load data from JSON files into different case lists.

        Args:
            case0_files (list): List of paths to case 0 JSON files.
            case1_files (list): List of paths to case 1 JSON files.
            case2_files (list): List of paths to case 2 JSON files.

        Returns:
            None
        """
        self.case0_list = merge_json_lists(*case0_files)
        self.case1_list = merge_json_lists(*case1_files)
        self.case2_list = merge_json_lists(*case2_files)
        self.case0_count = len(self.case0_list)
        self.case1_count = len(self.case1_list)
        self.case2_count = len(self.case2_list)


# Example usage:
#nodule_stats = NoduleSegmentationStats()
#case0_files = [
#   '/home/rakshith/nodule_seg_nas/jsons/confident_case0_list.json',
#   '/home/rakshith/nodule_seg_nas/jsons/low_confidence_case0_list.json',
#   '/home/rakshith/nodule_seg_nas/jsons/uncertain_case0_list.json'
#
#case1_files = [
#    '/home/rakshith/nodule_seg_nas/jsons/confident_case1_list.json',
 #   '/home/rakshith/nodule_seg_nas/jsons/low_confidence_case1_list.json',
  #  '/home/rakshith/nodule_seg_nas/jsons/uncertain_case1_list.json'
#]
#case2_files = [
 #   '/home/rakshith/nodule_seg_nas/jsons/confident_case2_list.json',
  #  '/home/rakshith/nodule_seg_nas/jsons/low_confidence_case2_list.json',
   # '/home/rakshith/nodule_seg_nas/jsons/uncertain_case2_list.json'
#]

#nodule_stats.load_data(case0_files, case1_files, case2_files)
#print(f'Case 0: {nodule_stats.case0_count}')
#print(f'Case 1: {nodule_stats.case1_count}')
#print(f'Case 2: {nodule_stats.case2_count}')
