import numpy as np
import os
from natsort import natsorted
from tqdm import tqdm as tq
from collections import defaultdict
import json
import argparse

def positive_negative_classifier():
    """Classifies slices as positive and negative slices.

    If any non-zero value is present in the mask/GT of
    the specified slice then it is classified as positive else negative.

    Returns
    -------
    None

    """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    target_location = os.path.sep.join(current_dir.split(os.path.sep)[:-4])
    data_path = os.path.join(target_location, 'isbi_data2')
    mask_path = os.path.join(data_path, 'masks/')
    save_path = os.path.join(data_path, 'jsons/')
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    file_list = natsorted(os.listdir(mask_path))
    positive_list, negative_list = [], []

    for file in tq(file_list):
        try:
            assert isinstance(file, str), "File name should be a string."

            mask = np.load(mask_path + file)
            assert isinstance(mask, np.ndarray), "Mask should be a NumPy array."

            if np.any(mask):
                positive_list.append(file)
            else:
                negative_list.append(file)
        except AssertionError as e:
            print(f"Assertion Error: {e}")
            continue
        except Exception as e:
            print(f"Error occurred: {e}")
            continue

    with open(save_path + 'positive_slices.json', 'w') as f:
        json.dump(positive_list, f)
    with open(save_path + 'negative_slices.json', 'w') as g:
        json.dump(negative_list, g) 



def subset_classifier():
	""" Classifies the slices according the subset of origin

	Returns
	-------
	dict
		A dictionary consisting of filename according to their subset.
	"""
	current_dir = os.path.dirname(os.path.realpath(__file__))
	target_location = os.path.sep.join(current_dir.split(os.path.sep)[:-3])
	dataset_path = os.path.join(target_location,'dataset/')
	data_path = os.path.join(target_location,'data')
	save_path = os.path.join(data_path,'jsons/')

	dict_subset={}
	dict_subset = defaultdict(lambda:[],dict_subset)
	for i in range(10): 
		#Since 10 subsets are provided in the dataset.
		for file in tq(os.listdir(dataset_path+'subset'+str(i))):
			file_name=os.path.basename(file)
			if file_name.endswith(".mhd"):
				dict_subset['subset'+str(i)].append(file_name)

	with open(save_path+'subset_classification.json', 'w') as h:
		json.dump(dict_subset, h)
		
	return dict_subset  



def assign_folds(dict_subset):
	""" Divides subsets into train,validation and testing sets of corresponding folds 

	Parameters
	----------
	dict_subset: dict
		Dictionary which has the files and its corresponding subset

	Returns
	-------
	None
	"""
	current_dir = os.path.dirname(os.path.realpath(__file__))
	target_location = os.path.sep.join(current_dir.split(os.path.sep)[:-3])
	save_path = os.path.join(target_location,'data/jsons/')

	dataset_list=[dict_subset['subset0'],dict_subset['subset1'],
					dict_subset['subset2'],dict_subset['subset3'],
					dict_subset['subset4'],dict_subset['subset5'],
					dict_subset['subset6'],dict_subset['subset7'],
					dict_subset['subset8'],dict_subset['subset9']]

	for i in tq(range(10)): #10 Subsets in the dataset

		fold={}
		fold = defaultdict(lambda:0,fold)
		fold['train_set']=dataset_list[0-i]+dataset_list[1-i]+dataset_list[2-i]+dataset_list[3-i]+dataset_list[4-i]+dataset_list[5-i]+dataset_list[6-i]+dataset_list[7-i]
		fold['valid_set']=dataset_list[8-i]
		fold['test_set']=dataset_list[9-i]
		
		
		fold_name='fold'+str(i)+'_mhd.json'
		with open(save_path+fold_name, 'w') as j:
			json.dump(fold, j)
			 

def add_additional_slices(series_uid_npylist,fold_npy,series_uid_train,series_uid_val,series_uid_test):

	"""Adds additional negative slices to the prepared datalist

	Parameters
	----------
	series_uid_npylist: nparray
	fold_npy: dict
	series_uid_train: list
	sereies_uid_val: list
	sereies_uid_test: list

	Returns
	-------
	dict

	"""
	count = {}
	current_dir = os.path.dirname(os.path.realpath(__file__))
	target_location = os.path.sep.join(current_dir.split(os.path.sep)[:-3])
	save_path = os.path.join(target_location,'data/jsons/')
	with open(save_path+'positive_slices.json') as c:
		pos_slices_json=json.load(c)



	pos_list=[x.split('.mhd')[0] for x in pos_slices_json]
	pos_list_uq=np.unique(np.array(pos_list))
		
	for i in (series_uid_train):
		c = series_uid_npylist.count(i)
		count[i] = c
		
		if i in pos_list:

			for j in range(5):
				file = str(i)+'_slice'+str(j)+'.npy'
				fold_npy['train_set'].append(file)
			for j in range(count[i]-5,count[i]):
				file = str(i)+'_slice'+str(j)+'.npy'
				fold_npy['train_set'].append(file)    


	for i in (series_uid_val):
		c = series_uid_npylist.count(i)
		count[i] = c
		
		if i in pos_list:

			for j in range(5):
				file = str(i)+'_slice'+str(j)+'.npy'
				fold_npy['valid_set'].append(file)
			for j in range(count[i]-5,count[i]):
				file = str(i)+'_slice'+str(j)+'.npy'
				fold_npy['valid_set'].append(file)  


	for i in (series_uid_test):
		c = series_uid_npylist.count(i)
		count[i] = c
		
		if i in pos_list:

			for j in range(5):
				file = str(i)+'_slice'+str(j)+'.npy'
				fold_npy['test_set'].append(file)
			for j in range(count[i]-5,count[i]):
				file = str(i)+'_slice'+str(j)+'.npy'
				fold_npy['test_set'].append(file) 

	return fold_npy        

def create_balanced_dataset(additional=False):

	"""Creates balanced dataset with equal positive and negative slices

	Parameters
	----------
	additional: Boolean,Optional
		If True add additonal negative slices 

	Returns
	-------
	dict
		Returns dict with equal positive and negative slices.
	"""
	current_dir = os.path.dirname(os.path.realpath(__file__))
	target_location = os.path.sep.join(current_dir.split(os.path.sep)[:-3])
	data_path = os.path.join(target_location,'data')
	save_path = os.path.join(data_path,'jsons/')
	img_path = os.path.join(data_path,'img')
	npy_list=natsorted(os.listdir(img_path))

	with open(save_path+'positive_slices.json') as c:
		pos_slices_json=json.load(c)


	pos_list=[x.split('.mhd')[0] for x in pos_slices_json]
	pos_list_uq=np.unique(np.array(pos_list))

		
	print('Sorting entire image set. Will take time.')
	sorted_list=natsorted(os.listdir(img_path))

	print('Sorting completed')


	for i in tq(range(10)):
		with open(save_path+'fold'+str(i)+'_mhd.json') as f:
			j_data=json.load(f)

		pos_count=0
		neg_count=0
		
		fold_npy={}
		fold_npy = defaultdict(lambda:[],fold_npy)
		series_uid_train=[x.split('.mhd')[0] for x in j_data['train_set']]
		series_uid_val=[x.split('.mhd')[0] for x in j_data['valid_set']]
		series_uid_test=[x.split('.mhd')[0] for x in j_data['test_set']]
		fold_npy_name='fold'+str(i)+'_pos_neg_eq.json'
		# npy_list=natsorted(os.listdir(img_path))
		series_uid_npylist=[x.split('_')[0] for x in npy_list]
		series_uid_npylist_uq=np.unique(np.array(series_uid_npylist))

		for f,name in enumerate(sorted_list):

			for q in series_uid_train:
				if q in name :
					if name in pos_slices_json: 
						#pos_slices_json contains the list of all positive slices.
						pos_count += 1
						fold_npy['train_set'].append(name)
					elif pos_count>neg_count:
						# Here the slice will be negative since 'name' not in pos_slices 
						neg_count += 1
						fold_npy['train_set'].append(name)
					else:
						continue
				else:
					continue


			for q in series_uid_val:

				if q in name :
					if name in pos_slices_json:

						pos_count += 1
						fold_npy['valid_set'].append(name)
					elif pos_count>neg_count:

						neg_count += 1
						fold_npy['valid_set'].append(name)
					else:
						continue
				else:
					continue

			for q in series_uid_test:
				if q in name :
					if name in pos_slices_json:
						pos_count += 1
						fold_npy['test_set'].append(name)
					elif pos_count>neg_count:
						neg_count += 1
						fold_npy['test_set'].append(name)
					else:
						continue
				else:
					continue
		with open(save_path+fold_npy_name, 'w') as z:
			json.dump(fold_npy,z)

	if additional == True:
		fold_npy = add_additional_slices(series_uid_npylist,fold_npy,series_uid_train,series_uid_val,series_uid_test)

	

	print('Balanced dataset identified and json saved')

	return fold_npy
