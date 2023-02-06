from pprint import pprint as pp
import os
import numpy as np
import tifffile as tiff
import cv2
import matplotlib.pyplot as plt
import h5py
import re
class Tiffconvert:
    def __init__(self,basepath:str,
                 resize:bool = False,
                 resize_size:tuple =(200,200,3)):

        if os.path.exists(basepath):
                self.basepath:str = basepath
        else:
            raise FileNotFoundError
        self.resize = resize
        self.resize_size = resize_size #(width ,height, 3)
        self.img_size:tuple = (1000,1000, 3) 
        '''
        Dataset structure should be like this

        |-dataset
            |-train
                |-image
                |-mask
            |-test
                |-image
                |-mask
        You just only need to input path to dataset

        This class convert tiff images to np array
        the shape of images change (width ,height) to (width , height, 3) 
        '''
    def get_root(self):
        for (root,dirs,files) in os.walk(self.basepath):
            if (len(dirs) == 0) & (len(files) > 0):
                yield root

    def get_files(self)->dict:
        gt_r = self.get_root
        dict_files = dict(((roots,sorted((os.path.join(roots,filename) for filename in os.listdir(roots)))) for roots in gt_r()))
        return dict_files

    def preprocess_image(self,img_arr)->np.ndarray:
        '''
        normalize img ndarray [0 , 255] if (array.max() > 255)
        '''
        normalized = 255*((img_arr-np.min(img_arr))/(np.max(img_arr)-np.min(img_arr)))
        normalized = normalized.astype('int')
        return normalized

    def preprocess_or_not(self,img_arr)->np.ndarray:
        norm255 = self.preprocess_image
        if img_arr.shape[-1] != 3:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)
        if img_arr.max() > 255: # if img_rgb values not normed do norm
            img_arr = norm255(img_arr)
        if self.resize == True:
            img_arr = cv2.resize(img_arr,self.resize_size)
        self.img_size = img_arr.shape
        return img_arr      

# 왜인지는 모르지만 정규화 후에는 gray2rgb가 안먹힌다.
# I dont't know why but after normalize, gray2rgb does not work. 
    def data_generator(self,
                       save_path:str, 
                       save_type:str = 'npz'):
        # save_type = 'npz', h5py' 
        gt_f = self.get_files
        pon = self.preprocess_or_not
        d:dict = gt_f()
        
        for root,files in d.items():
            '''
            for file_path in files:
                tmp_img = pon(cv2.imread(file_path,cv2.IMREAD_UNCHANGED))
                plt.imshow(tmp_img)
                break
             above code for showing img for test
            '''
            root =   re.sub(self.basepath,'',root)
            root =  '/data'+ root #make filterd directory
            for file_path in files:
                tmp = np.array([pon(cv2.imread(file_path,cv2.IMREAD_UNCHANGED)) for file_path in files])
            
            if save_type == 'npz':
                savez_dict = dict()
                savez_dict[root] = tmp
            elif save_type == 'h5py':
                if not os.path.isfile(save_path+'/dataset.hdf5'): 
                    f = h5py.File(save_path+'/dataset.hdf5', 'w')
                f.create_dataset(root,data = tmp)
            else:
                print('You input wrong type to save')
        if save_type == 'npz':
            np.savez(save_path+"/dataset.npz", **savez_dict)
