import pickle
import numpy as np
import os
import cv2
from skimage.util.shape import view_as_windows
import json
from tqdm import tqdm as tq 
import matplotlib.pyplot as plt
import random
from collections import defaultdict

def generate_patchlist(patchtype):
    """Generates positive slices in each fold

    Parameters:
    - patchtype (str): Positive/negative

    Raises:
    - AssertionError: If the parameter types or ranges are not as expected.

    Returns:
    - None
    """

    current_dir = os.path.dirname(os.path.realpath(__file__))
    target_location = os.path.sep.join(current_dir.split(os.path.sep)[:-3])
    data_path = os.path.join(target_location, 'data')
    save_path = os.path.join(data_path, 'jsons/')

    assert isinstance(patchtype, str), "patchtype should be a string"

    for fold_no in tq(range(10)):
        with open(save_path + 'fold' + str(fold_no) + '_pos_neg_eq.json') as file:
            j_data = json.load(file)
        with open(save_path + patchtype + '_slices.json') as c:
            pos_slices_json = json.load(c)

        assert isinstance(j_data, dict), "j_data should be a dictionary"
        assert isinstance(pos_slices_json, dict), "pos_slices_json should be a dictionary"

        train_set = j_data['train_set']
        valid_set = j_data['valid_set']
        test_set = j_data['test_set']
        train_seg_list = []
        val_seg_list = []
        test_seg_list = []

        for i in train_set:
            if i in pos_slices_json:
                train_seg_list.append(i)

        for i in valid_set:
            if i in pos_slices_json:
                val_seg_list.append(i)

        for i in test_set:
            if i in pos_slices_json:
                test_seg_list.append(i)

        patch_npy = {}
        patch_npy = defaultdict(lambda: [], patch_npy)
        patch_npy['train_set'] = train_seg_list
        patch_npy['valid_set'] = val_seg_list
        patch_npy['test_set'] = test_seg_list

        with open(save_path + patchtype + '_patchlist_f' + str(fold_no) + '.json', 'w') as z:
            json.dump(patch_npy, z)



def generate_negative_patch():
    """Generates patches which don't have nodules

    Raises:
    - AssertionError: If the parameter types or ranges are not as expected.

    Returns:
    - None
    """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    target_location = os.path.sep.join(current_dir.split(os.path.sep)[:-3])
    data_path = os.path.join(target_location, 'data')
    jsonpath = os.path.join(data_path, 'jsons/')
    imgpath = os.path.join(data_path, 'img')
    lung_segpath = os.path.join(data_path, 'lungseg')
    savepath = os.path.join(data_path, 'data')
    category_list = ['train_set', 'valid_set', 'test_set']

    assert isinstance(jsonpath, str), "jsonpath should be a string"
    assert isinstance(imgpath, str), "imgpath should be a string"
    assert isinstance(lung_segpath, str), "lung_segpath should be a string"
    assert isinstance(savepath, str), "savepath should be a string"

    for fold in tq(range(10)):
        with open(jsonpath + 'negative_patchlist_f' + str(fold) + '.json') as file:
            j_data = json.load(file)

        assert isinstance(j_data, dict), "j_data should be a dictionary"

        for category in category_list:
            img_dir = imgpath
            mask_dir = lung_segpath
            nm_list = j_data[category]

            assert isinstance(nm_list, list), "nm_list should be a list"

            size = 64
            index = 0
            for img_name in nm_list:
                img = np.load(os.path.join(img_dir, img_name)).astype(np.float32)
                mask = np.load(os.path.join(mask_dir, img_name)).astype(np.uint8)

                assert isinstance(img, np.ndarray), "img should be a NumPy array"
                assert isinstance(mask, np.ndarray), "mask should be a NumPy array"

                assert img.ndim == 2, "img should be a 2-dimensional array"
                assert mask.ndim == 2, "mask should be a 2-dimensional array"

                assert img.shape[0] == 512, "img should have a shape of (512, 512)"
                assert img.shape[1] == 512, "img should have a shape of (512, 512)"
                assert mask.shape[0] == 512, "mask should have a shape of (512, 512)"
                assert mask.shape[1] == 512, "mask should have a shape of (512, 512)"

                if np.any(mask):
                    _, th_mask = cv2.threshold(mask, 0.5, 1, 0, cv2.THRESH_BINARY)
                    contours, hierarchy = cv2.findContours(th_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    contours = sorted(contours, key=lambda x: cv2.contourArea(x))
                    
                    #In certain cases there could be more than 2 contour, hence taking the largest 2 which will be lung
                    contours = contours[1:]

                    
                    for cntr in contours:
                        patch_count = 2
                        for i in range(patch_count):
                            xr, yr, wr, hr = cv2.boundingRect(cntr)
                            xc, yc = xr + wr / 2, yr + hr / 2

                            try:
                                x = random.randrange(xr, xr + wr - size / 2)
                                y = random.randrange(yr, yr + hr - size / 2)
                            except:
                                prob = random.randrange(0, 1)
                                if prob > 0.5:
                                    x = random.randrange(xr, xr + wr // 2)
                                    y = random.randrange(yr, yr + hr // 2)
                                else:
                                    x = random.randrange(int(xr + wr / 2), xr + wr)
                                    y = random.randrange(int(yr + hr / 2), yr + hr)

                            assert x >= 0 and x + size < 512, "x coordinate is out of range"
                            assert y >= 0 and y + size < 512, "y coordinate is out of range"

                            if x + size < 512 and y + size < 512:
                                patch_img = img[y: y + size, x: x + size].copy().astype(np.float16)
                                patch_mask = np.zeros((size, size)).astype(np.float16)
                            else:
                                if x - size <= 0 and y - size <= 0:
                                    patch_img = img[0: size, 0: size].copy().astype(np.float16)
                                    patch_mask = np.zeros((size, size)).astype(np.float16)
                                elif x - size <= 0 and y - size > 0:
                                    patch_img = img[y - size: y, 0: size].copy().astype(np.float16)
                                    patch_mask = np.zeros((size, size)).astype(np.float16)
                                elif x - size > 0 and y - size <= 0:
                                    patch_img = img[0: size, x - size: x].copy().astype(np.float16)
                                    patch_mask = np.zeros((size, size)).astype(np.float16)
                                else:
                                    patch_img = img[y - size: y, x - size: x].copy().astype(np.float16)
                                    patch_mask = np.zeros((size, size)).astype(np.float16)

                            assert patch_img.shape == (64, 64), "patch_img has an unexpected shape"
                            assert patch_mask.shape == (64, 64), "patch_mask has an unexpected shape"

                            index += 1
                            img_savepath = savepath + '/patches/' + '/img/'
                            mask_savepath = savepath + '/patches/' + '/mask/'

                            if not os.path.isdir(img_savepath):
                                os.makedirs(savepath + '/patches/' + '/img/')
                                np.save(img_savepath + 'patch_' + str(fold) + '_' + str(index) + '.npy', patch_img)
                            else:
                                np.save(img_savepath + 'patch_' + str(fold) + '_' + str(index) + '.npy', patch_img)

                            if not os.path.isdir(mask_savepath):
                                os.makedirs(savepath + '/patches/' + '/mask/')
                                np.save(mask_savepath + 'patch_' + str(fold) + '_' + str(index) + '.npy', patch_mask)
                            else:
                                np.save(mask_savepath + 'patch_' + str(fold) + '_' + str(index) + '.npy', patch_mask)


def generate_positive_patch():
    """Generate patches with nodules

    Returns
    -------
    None    

    """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    target_location = os.path.sep.join(current_dir.split(os.path.sep)[:-3])
    data_path = os.path.join(target_location,'data')
    jsonpath = os.path.join(data_path,'jsons/')
    imgpath = os.path.join(data_path,'img')
    maskpath = os.path.join(data_path,'mask')
    lung_segpath = os.path.join(data_path,'lungseg')
    savepath = data_path
    category_list = ['train_set','valid_set','test_set']

    for fold in tq(range(10)):
    
        with open(jsonpath+'positive_patchlist_f'+str(fold)+'.json') as file:
            j_data = json.load(file)

        for category in category_list:
            img_dir = imgpath
            mask_dir = maskpath 
            nm_list = j_data[category] 

            size = 64
            index = 0
            for img_name in nm_list:
                #Loading the masks as uint8 as threshold function accepts 8bit image as parameter.
                img = np.load(os.path.join(img_dir, img_name)).astype(np.float16)
                mask = np.load(os.path.join(mask_dir, img_name))/255
                mask = mask.astype(np.uint8)

                if np.any(mask):
                    #Convert grayscale image to binary
                    _, th_mask = cv2.threshold(mask, 0.5, 1, 0,cv2.THRESH_BINARY) #parameters are ip_img,threshold,max_value
                    contours, hierarchy = cv2.findContours(th_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    contours = sorted(contours, key=lambda x: cv2.contourArea(x))
                    
                    for cntr in contours:
                        patch_count = 4
                        
                        xr,yr,wr,hr = cv2.boundingRect(cntr) #Gives X,Y cordinate of BBox origin,height and width
                        xc,yc = int(xr+wr/2),int(yr+hr/2)

                        assert isinstance(xc, int), "xc should be an integer."
                        assert isinstance(yc, int), "yc should be an integer."

                        assert 0 <= xc < 512, "xc should be within the range [0, 512)."
                        assert 0 <= yc < 512, "yc should be within the range [0, 512)."


                        if int(yc-size/2) <0 or int(xc-size/2)<0:
                            if int(yc-size/2) <0 and int(xc-size/2)<0:
                                patch_img1 = img[0:size , 0:size].copy().astype(np.float16)
                                patch_mask1 = mask[0:size , 0:size].copy().astype(np.float16)

                            elif int(yc-size/2) >0 and int(xc-size/2)<0:
                                patch_img1 = img[int(yc-size/2):int(yc+size/2) , 0:size].copy().astype(np.float16)
                                patch_mask1 = mask[int(yc-size/2):int(yc+size/2) , 0:size].copy().astype(np.float16)
                               
                            elif int(yc-size/2) <0 and int(xc-size/2)>0:
                                patch_img1 = img[0:size ,int(xc-size/2):int(xc+size/2)].copy().astype(np.float16)
                                patch_mask1 = mask[0:size ,int(xc-size/2):int(xc+size/2)].copy().astype(np.float16)              
                            
                            
                        elif int(yc+size/2)>512 or int(xc+size/2)>512:
                            if int(yc+size/2)>512 and int(xc+size/2)>512:
                                m = yc+size - 512
                                n = xc + size - 512
                                patch_img1 = img[int(yc-m):512,int(xc-n):512].copy().astype(np.float16)
                                patch_mask1 = mask[int(yc-m):512,int(xc-n):512].copy().astype(np.float16)                  
                                
                            elif int(yc+size/2)>512 and int(xc+size/2)<512:
                                m = yc+size - 512
                                patch_img1 = img[int(yc-m):512,int(xc-size/2):int(xc+size/2)].copy().astype(np.float16)
                                patch_mask1 = mask[int(yc-m):512,int(xc-size/2):int(xc+size/2)].copy().astype(np.float16)  
                            
                            elif int(yc+size/2)<512 and int(xc+size/2)>512:
                                n = xc+size - 512
                                patch_img1 = img[int(yc-size/2):int(yc+size/2),int(xc-n):512].copy().astype(np.float16)
                                patch_mask1 = mask[int(yc-size/2):int(yc+size/2),int(xc-n):512].copy().astype(np.float16)

                        elif (int(yc-size/2)>=0 and int(yc+size/2)<=512) :
                             if(int(xc-size/2)>=0 and int(xc+size/2)<=512):
                                patch_img1 = img[int(yc-size/2):int(yc+size/2) , int(xc-size/2):int(xc+size/2)].copy().astype(np.float16)
                                patch_mask1 = mask[int(yc-size/2):int(yc+size/2) , int(xc-size/2):int(xc+size/2)].copy().astype(np.float16)

                        if np.shape(patch_img1) != (64,64):
                            print('shape',np.shape(patch_img1))
                            print('cordinate of patch',x,x+size,y,y+size)
                            print('cordinate of BBox',xr,yr,wr,hr) 

                        img_savepath = savepath+'/patches/'+category+'/img/'
                        mask_savepath = savepath+'/patches/'+category+'/mask/'
                        if not os.path.isdir(img_savepath):
                            os.makedirs(savepath+'/patches/'+category+'/img/')
                            np.save(img_savepath+'patch_'+str(fold)+'_'+str(index)+'.npy',patch_img1)
                        else:
                            np.save(img_savepath+'patch_'+str(fold)+'_'+str(index)+'.npy',patch_img1)

                        if not os.path.isdir(mask_savepath):
                            os.makedirs(savepath+'/patches/'+category+'/mask/')
                            np.save(mask_savepath+'patch_'+str(fold)+'_'+str(index)+'.npy',patch_mask1)
                        else:
                            np.save(mask_savepath+'patch_'+str(fold)+'_'+str(index)+'.npy',patch_mask1)

                        index += 1
                        for i in range(patch_count):
                            xc,yc = xr,yr
                            xc,yc = xr+wr,yr+hr
                            
                            if i == 0:
                            
                                if xc+size<512 and yc+size<512:
                                    patch_img = img[yc:yc+size,xc:xc+size].copy().astype(np.float16)
                                    patch_mask = mask[yc:yc+size,xc:xc+size].copy().astype(np.float16)

                                elif xc+size>512 and yc+size<512:
                                    m = xc+size-512
                                    patch_img = img[yc:yc+size,xc-m:xc+size-m].copy().astype(np.float16)
                                    patch_mask = mask[yc:yc+size,xc-m:xc+size-m].copy().astype(np.float16)

                                elif xc+size<512 and yc+size>512:
                                    n = yc+size-512
                                    patch_img = img[yc-n:yc+size-n,xc:xc+size].copy().astype(np.float16)
                                    patch_mask = mask[yc-n:yc+size-n,xc:xc+size].copy().astype(np.float16) 
                                else:
                                    m = xc+size-512                    
                                    n = yc+size-512
                                    patch_img = img[yc-n:yc+size-n,xc-m:xc+size-m].copy().astype(np.float16)
                                    patch_mask = mask[yc-n:yc+size-n,xc-m:xc+size-m].copy().astype(np.float16)
                            elif i ==1:
                                
                                if xc-size>0 and yc+size<512:
                                    patch_img = img[yc:yc+size,xc-size:xc].copy().astype(np.float16)
                                    patch_mask = mask[yc:yc+size,xc-size:xc].copy().astype(np.float16)
                                    
                                elif xc-size<0 and yc+size<512:
                                    
                                    patch_img = img[yc:yc+size,0:size].copy().astype(np.float16)
                                    patch_mask = mask[yc:yc+size,0:size].copy().astype(np.float16) 
                                    
                                elif xc-size>0 and yc+size>512:
                                    n = yc+size-512

                                    patch_img = img[yc-n:yc+size-n,xc-size:xc].copy().astype(np.float16)
                                    patch_mask = mask[yc-n:yc+size-n,xc-size:xc].copy().astype(np.float16) 
                                    
                                else:
                                    n = yc+size-512

                                    patch_img = img[yc-n:yc+size-n,0:size].copy().astype(np.float16)
                                    patch_mask = mask[yc-n:yc+size-n,0:size].copy().astype(np.float16) 
                            elif i ==2:
                                
                                if xc+size<512 and yc-size>0:
                                    patch_img = img[yc-size:yc,xc:xc+size].copy().astype(np.float16)
                                    patch_mask = mask[yc-size:yc,xc:xc+size].copy().astype(np.float16)                        

                                elif xc+size>512 and yc-size>0:
                                    m = xc+size-512
                                    patch_img = img[yc-size:yc,xc-m:xc+size-m].copy().astype(np.float16)
                                    patch_mask = mask[yc-size:yc,xc-m:xc+size-m].copy().astype(np.float16)
                                    
                                elif xc+size<512 and yc-size<0:
                                    patch_img = img[0:size,xc:xc+size].copy().astype(np.float16)
                                    patch_mask = mask[0:size,xc:xc+size].copy().astype(np.float16)
                                    
                                else:
                                    m = xc+size-512
                                    patch_img = img[yc-size:yc,xc-m:xc+size-m].copy().astype(np.float16)
                                    patch_mask = mask[yc-size:yc,xc-m:xc+size-m].copy().astype(np.float16)  
                                    
                            elif i==3:
                                
                                if xc-size>0 and yc-size>0:
                                    patch_img = img[yc-size:yc,xc-size:xc].copy().astype(np.float16)
                                    patch_mask = mask[yc-size:yc,xc-size:xc].copy().astype(np.float16)                        

                                elif xc-size<0 and yc-size>0:
                                    m = xc+size-512
                                    patch_img = img[yc-size:yc,0:size].copy().astype(np.float16)
                                    patch_mask = mask[yc-size:yc,0:size].copy().astype(np.float16)
                                    
                                elif xc-size>0 and yc-size<0:
                                    patch_img = img[0:size,xc-size:xc].copy().astype(np.float16)
                                    patch_mask = mask[0:size,xc-size:xc].copy().astype(np.float16)
                                    
                                else:
                                    patch_img = img[0:size,0:size].copy().astype(np.float16)
                                    patch_mask = mask[0:size,0:size].copy().astype(np.float16)  
                                    
                                    
                            if np.shape(patch_img) != (64,64):
                                print('shape',np.shape(patch_img))
                                
                            img_savepath = savepath+'/patches/'+category+'/img/'
                            mask_savepath = savepath+'/patches/'+category+'/mask/'
                            if not os.path.isdir(img_savepath):
                                os.makedirs(savepath+'/patches/'+category+'/img/')
                                np.save(img_savepath+'patch_'+str(fold)+'_'+str(index)+'.npy',patch_img)
                            else:
                                np.save(img_savepath+'patch_'+str(fold)+'_'+str(index)+'.npy',patch_img)

                            if not os.path.isdir(mask_savepath):
                                os.makedirs(savepath+'/patches/'+category+'/mask/')
                                np.save(mask_savepath+'patch_'+str(fold)+'_'+str(index)+'.npy',patch_mask)
                            else:
                                np.save(mask_savepath+'patch_'+str(fold)+'_'+str(index)+'.npy',patch_mask)

                            index += 1
                    

