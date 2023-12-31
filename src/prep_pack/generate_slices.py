#!/usr/bin/env python
# coding: utf-8
#for review

import SimpleITK as sitk
import pylidc as pl
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os
import glob
import cv2
from tqdm import tqdm as tq


def make_mask(height, width, slice_list, *args, **kwargs):
    """
    Creates masks from the annotations given.

    Parameters:
    - height (int): Height of the mask to be created.
    - width (int): Width of the mask image to be created.
    - slice_list (list): List of slices.
    - **kwargs: Additional keyword arguments.

    Returns:
    - np.ndarray: The created mask.
    """
    assert isinstance(height, int) and height > 0, "height should be a positive integer"
    assert isinstance(width, int) and width > 0, "width should be a positive integer"
    assert isinstance(slice_list, list), "slice_list should be a list"

    mask = np.zeros((height, width))
    n = kwargs.get('n', None)
    point_dictx = kwargs.get('ii', None)
    point_dicty = kwargs.get('jj', None)

    if n in slice_list:
        assert isinstance(point_dictx, dict), "ii should be a dictionary"
        assert isinstance(point_dicty, dict), "jj should be a dictionary"

        temp_listx = point_dictx[n]
        temp_listy = point_dicty[n]
        plot_listx = [sum(x) / len(point_dictx[n]) for x in zip(*temp_listx)]
        plot_listy = [sum(y) / len(point_dicty[n]) for y in zip(*temp_listy)]
        merged_list = np.array([[plot_listy[i], plot_listx[i]] for i in range(len(plot_listx))])

        cv2.fillPoly(mask, pts=np.int32([merged_list]), color=(255, 255, 255))

    return mask

#def extract_slices():
    """Extracts induvidual slices from the CT volumes given 
    in the dataset, clips the max-min values and stores them
    as numpy arrays.

    Returns
    -------

    None
    """


    file_list=[]
    current_dir = os.path.dirname(os.path.realpath(__file__))
    target_location = os.path.sep.join(current_dir.split(os.path.sep)[:-3])
    dataset_path = os.path.join(target_location,'dataset')
    save_path = os.path.join(target_location,'data')


    for tr in tq(range(10)):
        subset_path=dataset_path+"/subset"+str(tr)+"/"
        for file in os.listdir(subset_path):
            if file.endswith(".mhd"):
                file_list.append(os.path.join(subset_path, file))


    for file in tq(file_list):
        file_name=os.path.basename(file)
        series_instance_uid=os.path.splitext(file_name)[0]
        img_file=file
        
        itk_img = sitk.ReadImage(img_file) 
        img_array = sitk.GetArrayFromImage(itk_img)
        num_slice, height, width = img_array.shape
        #Has the image data

        scan = pl.query(pl.Scan).filter(pl.Scan.series_instance_uid== series_instance_uid).first()
      
        #Maped the image data with annotation using series id

        nods = scan.cluster_annotations() #Function used to determine which annotation belongs to which nodule



        nodule_dict={} #Dict to store number of contour markings for that nodule
        slice_list=[] # List to store the slices which has nodules marked
        points_dictx={} # These dicts are to store the points to be plotted (key=slice_index, )
        points_dicty={}
        points_dictx = defaultdict(lambda:[],points_dictx)
        points_dicty = defaultdict(lambda:[],points_dicty)
        for i,nod in enumerate(nods):
            nodule_dict[i]=len(nods[i])  #Stores a dict which has count of annotation for each nodule

        for key,value in nodule_dict.items():
            #if value>=3 :    #Taking annotations provided by 3 or more annotator
                for i in range(value):
                    ann=nods[key][i] #-1 to keep index correct   
                    con=ann.contours[0] #All coutours for specific nodule collected

                    k = con.image_k_position # Returns the slice number/index which has the nodule
                    slice_list.append(k)
                    ii,jj = ann.contours[0].to_matrix(include_k=False).T
                    points_dictx[k].append(ii)
                    points_dicty[k].append(jj)


        '''
        !!Note!! The pylidc package gives cordinates for single slices, If more than one annotaions are give then
        Sum(x)/total no: of annotation for all provided pixel is given as input

        '''    


        for n in range(1,num_slice):

            image=(img_array[n].copy()).astype(np.float32) 
            im_max = np.max(image)
            im_min = np.min(image)
            if im_max != 0:
                image[image>400]=400
                image[image<-1000]=-1000 
                mask=make_mask(height,width,slice_list,ii=points_dictx,jj=points_dicty,n=n)
                mask = np.array(mask, dtype=np.float32)
                image = image - image.min()
                image = image/image.max()

                if not os.path.isdir(save_path):
                    os.makedirs(save_path)

                if not os.path.isdir(save_path+'/img'):
                    os.makedirs(save_path+'/img')
                    np.save(save_path+'/img/'+series_instance_uid+'_slice'+str(n)+'.npy',image)
                else:
                    np.save(save_path+'/img/'+series_instance_uid+'_slice'+str(n)+'.npy',image)

                if not os.path.isdir(save_path+'/mask'):
                    os.makedirs(save_path+'/mask')
                    np.save(save_path+'/mask/'+series_instance_uid+'_slice'+str(n)+'.npy',mask)
                else:
                    np.save(save_path+'/mask/'+series_instance_uid+'_slice'+str(n)+'.npy',mask)

def extract_slices():
    """
    Extracts individual slices from the CT volumes given in the dataset, clips the max-min values, and stores them
    as numpy arrays.

    Returns:
    - None
    """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    target_location = os.path.sep.join(current_dir.split(os.path.sep)[:-3])
    dataset_path = os.path.join(target_location, 'dataset')
    save_path = os.path.join(target_location, 'data')

    assert os.path.isdir(dataset_path), "dataset path does not exist"
    assert os.path.isdir(save_path), "save path does not exist"

    file_list = []
    for tr in tq(range(10)):
        subset_path = os.path.join(dataset_path, "subset" + str(tr))
        assert os.path.isdir(subset_path), f"subset {tr} path does not exist"
        for file in os.listdir(subset_path):
            if file.endswith(".mhd"):
                file_list.append(os.path.join(subset_path, file))

    assert file_list, "no .mhd files found in the dataset"

    for file in tq(file_list):
        file_name = os.path.basename(file)
        series_instance_uid = os.path.splitext(file_name)[0]
        img_file = file

        itk_img = sitk.ReadImage(img_file)
        img_array = sitk.GetArrayFromImage(itk_img)
        num_slice, height, width = img_array.shape

        assert num_slice > 0, "invalid number of slices"

        scan = pl.query(pl.Scan).filter(pl.Scan.series_instance_uid == series_instance_uid).first()

        assert scan is not None, "scan not found for the series instance UID"

        nods = scan.cluster_annotations()

        nodule_dict = {}
        slice_list = []
        points_dictx = defaultdict(list)
        points_dicty = defaultdict(list)

        for i, nod in enumerate(nods):
            nodule_dict[i] = len(nods[i])

        for key, value in nodule_dict.items():
            for i in range(value):
                ann = nods[key][i]
                con = ann.contours[0]
                k = con.image_k_position
                slice_list.append(k)
                ii, jj = ann.contours[0].to_matrix(include_k=False).T
                points_dictx[k].append(ii)
                points_dicty[k].append(jj)

        for n in range(1, num_slice):
            image = img_array[n].copy().astype(np.float32)
            im_max = np.max(image)
            im_min = np.min(image)
            
            assert im_max is not None, "maximum value not found in image"
            assert im_min is not None, "minimum value not found in image"
            
            if im_max != 0:
                image[image > 400] = 400
                image[image < -1000] = -1000
                mask = make_mask(height, width, slice_list, ii=points_dictx, jj=points_dicty, n=n)
                mask = np.array(mask, dtype=np.float32)
                image = image - image.min()
                image = image / image.max()

                img_save_path = os.path.join(save_path, 'img')
                assert os.path.isdir(img_save_path), f"img save path does not exist"
                np.save(os.path.join(img_save_path, series_instance_uid + '_slice' + str(n) + '.npy'), image)

                mask_save_path = os.path.join(save_path, 'mask')
                assert os.path.isdir(mask_save_path), f"mask save path does not exist"
                np.save(os.path.join(mask_save_path, series_instance_uid + '_slice' + str(n) + '.npy'), mask)


#def generate_lungseg():
    """Generates lung masks for each slice.


    """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    target_location = os.path.sep.join(current_dir.split(os.path.sep)[:-3])
    # print(target_location)
    dataset_path = os.path.join(target_location,'dataset/seg-lungs-LUNA16')
    save_path = os.path.join(target_location,'data/')

    volume_list = os.listdir(dataset_path)
    file_list = []

    for file in os.listdir(dataset_path):
        if file.endswith(".mhd"):
            file_list.append(os.path.join(dataset_path, file))

    for img_file in tq(file_list):
        file_name=os.path.basename(img_file)
        series_instance_uid=os.path.splitext(file_name)[0]
        itk_img = sitk.ReadImage(img_file) 
        img_array = sitk.GetArrayFromImage(itk_img)
        num_slice, height, width = img_array.shape
        img_array[img_array>0] = 1
        for n in range(1,num_slice):
            if not os.path.isdir(save_path+'lungseg'):
                os.makedirs(save_path+'lungseg')
                np.save(save_path+'lungseg/'+series_instance_uid+'_slice'+str(n)+'.npy',img_array[n])
            else:
                np.save(save_path+'lungseg/'+series_instance_uid+'_slice'+str(n)+'.npy',img_array[n])

def generate_lungseg():
    """Generates lung masks for each slice.

    Raises:
    - AssertionError: If the parameter types or ranges are not as expected.

    Returns:
    - None
    """

    current_dir = os.path.dirname(os.path.realpath(__file__))
    target_location = os.path.sep.join(current_dir.split(os.path.sep)[:-3])
    dataset_path = os.path.join(target_location, 'dataset/seg-lungs-LUNA16')
    save_path = os.path.join(target_location, 'data/')

    assert isinstance(dataset_path, str), "dataset_path should be a string"
    assert isinstance(save_path, str), "save_path should be a string"

    assert os.path.isdir(dataset_path), "dataset_path does not exist"
    assert os.path.isdir(save_path), "save_path does not exist"

    volume_list = os.listdir(dataset_path)
    file_list = []

    assert isinstance(volume_list, list), "volume_list should be a list"

    for file in volume_list:
        if file.endswith(".mhd"):
            file_list.append(os.path.join(dataset_path, file))

    for img_file in file_list:
        file_name = os.path.basename(img_file)
        series_instance_uid = os.path.splitext(file_name)[0]

        assert isinstance(series_instance_uid, str), "series_instance_uid should be a string"

        itk_img = sitk.ReadImage(img_file)
        img_array = sitk.GetArrayFromImage(itk_img)

        assert isinstance(img_array, np.ndarray), "img_array should be a NumPy array"

        num_slice, height, width = img_array.shape

        assert isinstance(num_slice, int), "num_slice should be an integer"
        assert num_slice > 0, "num_slice should be greater than 0"

        assert isinstance(height, int), "height should be an integer"
        assert height > 0, "height should be greater than 0"

        assert isinstance(width, int), "width should be an integer"
        assert width > 0, "width should be greater than 0"

        img_array[img_array > 0] = 1

        for n in range(1, num_slice):
            if not os.path.isdir(save_path + 'lungseg'):
                os.makedirs(save_path + 'lungseg')

            assert isinstance(n, int), "n should be an integer"
            assert 1 <= n < num_slice, "n should be in the range [1, num_slice)"

            np.save(save_path + 'lungseg/' + series_instance_uid + '_slice' + str(n) + '.npy', img_array[n])
