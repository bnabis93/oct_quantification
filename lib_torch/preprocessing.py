import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import cv2

class InhaRawDataPreprocessing():
    '''
    Inha hospital's OCT raw data preprocessing
    just select 5,6,7,8 oct scan data
    
    class init var
        root
            
        save_dir
        
        crop_points
            tuple lists (y1, y2, x1, x2)
            
        add_data
    
    
    why I use opencv instead of PIL
        opencv vs PIL : https://www.kaggle.com/vfdev5/pil-vs-opencv
    
    Functions
        load_raw_data
            load raw data from directory.
            
        size_standardization
        
        save_raw_data
    
    '''
    def __init__(self,root, save_dir,crop_points = (0,500,500,1260) ,add_mode = False):
        '''
        get dir name list
        
        '''
        self.root = root
        self.full_file_path = []
        self.num_raw_data = 0
        self.raw_height = crop_points[1] - crop_points[0]
        self.raw_width = crop_points[3] - crop_points[2]

        # load 5~8 oct raw data
        self.load_raw_data()
        self.raw_data_lists = np.zeros((self.num_raw_data,self.raw_height,self.raw_width,3))

        # crop the data (crop points = x = 500 ~ 1260 / y = 0 ~ 500
        self.size_standardization(crop_points)
        #save the file
        self.save_raw_data(save_dir,add_mode)
            
                        
    def load_raw_data(self):
        for root, dirs, files in os.walk(self.root):
            for file in files:
                if file.startswith(".ipynb")==False and (file.endswith("5.bmp")==True or file.endswith("6.bmp")==True or file.endswith("7.bmp")==True or file.endswith("8.bmp")==True):
                    self.full_file_path.append(root+'/'+file)
                    self.num_raw_data += 1
        
    def size_standardization(self,crop_points):
        for idx,file_path in enumerate(self.full_file_path):
            temp_img = cv2.imread(file_path)
            temp_img = temp_img[crop_points[0]:crop_points[1], crop_points[2] : crop_points[3]]
            self.raw_data_lists[idx] = temp_img
        
        
    def save_raw_data(self,save_dir,add_mode = False):
        if os.path.isdir(save_dir) == False:
            os.mkdir(save_dir)
        else:
            print('already exist the folder in this path : {}'.format(save_dir))
            
            
        if add_mode == True:
            
            cnt = 0
            for count, filename in enumerate(sorted(os.listdir(save_dir)), start=0):
                if filename.startswith(".ipynb")==False:
                    cnt +=1
            
            #print(cnt)
            for idx in range(len(self.raw_data_lists)):
                cv2.imwrite(save_dir + 'raw_data_' + str(cnt)+'.png', self.raw_data_lists[idx])
                cnt +=1
            
                
        else:
            for idx in range(len(self.raw_data_lists)):
                if idx >= 10:
                    cv2.imwrite(save_dir + 'raw_data_' + str(idx)+'.png', self.raw_data_lists[idx])
                else:
                    cv2.imwrite(save_dir + 'raw_data_' + '0'+str(idx)+'.png', self.raw_data_lists[idx])