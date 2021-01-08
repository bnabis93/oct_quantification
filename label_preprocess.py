from PIL import Image

import numpy as np
import os
import cv2
from keras.utils.np_utils import to_categorical

img = cv2.imread('./data/inha_train/training/before_process_label/00.png')
img_shape = np.shape(img)
num_imgs = 40
print(np.shape(img))
mapping = {
            (0,0,0) : 0,
            (255,0,0) : 1,
            (0,0,255) : 2}

if len(img_shape) == 3:
        #ch = 3 or 1
    channels = img_shape[2]
    img_height = img_shape[0]
    img_width = img_shape[1]
        
elif len(img_shape) == 2:
    # ch = none
    img_height = img_shape[0]
    img_width = img_shape[1]

groundTruth_dir = './data/inha_train/training/before_process_label/'
groundTruth = np.zeros((num_imgs,img_height,img_width,channels))

for count, filename in enumerate(sorted(os.listdir(groundTruth_dir)), start=0):
    groundTruth_name = filename
    print ("ground truth name: " + groundTruth_name)
    g_truth = Image.open(groundTruth_dir + groundTruth_name)
    if len(np.shape(g_truth)) !=2:
        g_truth = cv2.imread(groundTruth_dir+groundTruth_name)
        masks = np.zeros((g_truth.shape))
        print(np.shape(g_truth))
        print(np.shape(masks))
        #h,w,c = g_truth.shape
        #masks = np.zeros((h,w))
        
        class0_idx_lists= np.where(np.all(g_truth == [0,0,0], axis=-1))
        class1_idx_lists= np.where(np.all(g_truth == [255,0,0], axis=-1))
        class2_idx_lists= np.where(np.all(g_truth == [0,0,255], axis=-1))
        
        masks[class0_idx_lists] = [0,0,0]
        masks[class1_idx_lists] = [255,0,0]
        masks[class2_idx_lists] = [0,0,255]
    
    groundTruth[count] = np.asarray(masks)
    cv2.imwrite('./data/inha_train/training/label/'+str(count)+'.png',groundTruth[count])
    
