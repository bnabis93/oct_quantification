import os, shutil
import sys
import h5py
import numpy as np
import cv2
from PIL import Image
sys.path.insert(0, './lib_keras/')
from help_functions import write_hdf5
import configparser


config = configparser.RawConfigParser()
config.read('pre_configuration.txt')
what_data = config.get('setting','data')
size_mode = config.get('setting', 'size_mode')
resize_constant = config.get('setting','resize_constant')
resize_constant = float(resize_constant)

mode = config.get('setting', 'mode')
save_dir_path = config.get('path', 'save_path')
save_dir_path = save_dir_path + mode + '/'

if mode =='train':
    original_img_train_path = config.get('path','original')
    ground_truth_img_train_path = config.get('path','ground')
    border_masks_imgs_train_path = config.get('path','mask')
    print('data : {} \ntrain img : {} \nlabel img : {} \nsave path : {}'.format(what_data, original_img_train_path,\
                                                                            ground_truth_img_train_path, save_dir_path))
elif mode =='test':
    original_img_train_path = config.get('path','test')
    print('data : {} \ntest img : {} \nsave path : {}'.format(what_data, original_img_train_path,save_dir_path))


print('mode : ',mode)

if os.path.isdir('./'+save_dir_path) == False:
    os.mkdir('./'+save_dir_path)
else:
    print('already exist the folder in this path : {}'.format('./'+save_dir_path))
    
    
if what_data == 'inha_oct' :
    path, dirs, files = next(os.walk(original_img_train_path))
    num_imgs = len(files)
    img = cv2.imread(original_img_train_path + files[0])
    #img = Image.open(original_img_train_path + files[0])
    img = np.asarray(img)
    img_shape = np.shape(img)
    
    if len(img_shape) == 3:
        #ch = 3 or 1
        channels = img_shape[2]
        img_height = img_shape[0]
        img_width = img_shape[1]
        
        if size_mode == 'resize':
            channels = img_shape[2]
            resize_height = int(img_height * resize_constant) 
            resize_width = int(img_width * resize_constant)
            
        
    elif len(img_shape) == 2:
        # ch = none
        img_height = img_shape[0]
        img_width = img_shape[1]
        
        if size_mode == 'resize':
            resize_height = int(img_height * resize_constant) 
            resize_width = int(img_width * resize_constant)
    
    print('img shape : {} \nnumber of imgs : {} \ndirs : {}\n'.format(img_shape, num_imgs, files))
    
elif what_data == 'inha_oct_5classes' :
    path, dirs, files = next(os.walk(original_img_train_path))
    num_imgs = len(files)
    img = cv2.imread(original_img_train_path + files[0])
    #img = Image.open(original_img_train_path + files[0])
    #img = np.asarray(img)
    img_shape = np.shape(img)
    
    if len(img_shape) == 3:
        #ch = 3 or 1
        channels = img_shape[2]
        img_height = img_shape[0]
        img_width = img_shape[1]
        if size_mode == 'resize':
            resize_height = int(img_height * resize_constant) 
            resize_width = int(img_width * resize_constant)
        
    elif len(img_shape) == 2:
        # ch = none
        img_height = img_shape[0]
        img_width = img_shape[1]
        if size_mode == 'resize':
            resize_height = int(img_height * resize_constant) 
            resize_width = int(img_width * resize_constant)
    
    print('img shape : {} \nnumber of imgs : {} \ndirs : {}\n'.format(img_shape, num_imgs, files))
    
    
def get_datasets(imgs_dir,groundTruth_dir,borderMasks_dir = False ,train_test="train"):
    
    if train_test =='train':
        if size_mode =='original':
            imgs = np.empty((num_imgs,img_height,img_width,channels))
            groundTruth = np.empty((num_imgs,img_height,img_width,3)) # RGB ch labels (multi segmentation)
        elif size_mode == 'resize':
            resized_imgs = np.empty((num_imgs, resize_height, resize_width, channels))
            resized_groundTruth = np.empty((num_imgs,resize_height,resize_width,3))
        
        print('img dir : ',imgs_dir)
                                           
        for count, filename in enumerate(sorted(os.listdir(imgs_dir)), start=0):
            print ("original image: " +filename)
            if size_mode == 'original':
                img = Image.open(imgs_dir+filename)
                #print('img shape : ',np.shape(img))
                imgs[count] = np.asarray(img)

            elif size_mode == 'resize':
                img = Image.open(imgs_dir+filename)
                img = np.asarray(img)
                resized_img = cv2.resize(img, (resize_width, resize_height), interpolation = cv2.INTER_LANCZOS4)
                resized_imgs[count] = resized_img
                  

        print('ground truth dir : ', groundTruth_dir)
        for count, filename in enumerate(sorted(os.listdir(groundTruth_dir)), start=0):
            groundTruth_name = filename
            print ("ground truth name: " + groundTruth_name)
            if size_mode =='original':
                g_truth = Image.open(groundTruth_dir + groundTruth_name)
                groundTruth[count] = np.asarray(g_truth)

            elif size_mode == 'resize':
                g_truth = Image.open(groundTruth_dir + groundTruth_name)
                g_truth = np.asarray(g_truth)
                resized_gTruth = cv2.resize(g_truth, (resize_width, resize_height), interpolation = cv2.INTER_LANCZOS4)
                resized_groundTruth[count] = resized_gTruth
                                           

        if size_mode == 'original':
            print ("imgs max: " +str(np.max(imgs)))
            print ("imgs min: " +str(np.min(imgs)))
            #assert(np.max(groundTruth)==255)
            #assert(np.min(groundTruth)==0)
            #reshaping for my standard tensors
            imgs = np.transpose(imgs,(0,3,1,2)) #num c h w
            groundTruth = np.transpose(groundTruth, (0,3,1,2))
            #assert(imgs.shape == (num_imgs,channels,img_height,img_width))
            #groundTruth = np.reshape(groundTruth,(num_imgs,3,img_height,img_width))
            assert(groundTruth.shape == (num_imgs,3,img_height,img_width))
            print('Done!')

            return imgs, groundTruth
                                           
        elif size_mode == 'resize':
            print('resized \n')
            print ("imgs max: " +str(np.max(resized_imgs)))
            print ("imgs min: " +str(np.min(resized_imgs)))
            #assert(np.max(groundTruth)==255)
            #assert(np.min(groundTruth)==0)
            #reshaping for my standard tensors
            #print('resied imgs shape : ', np.shape(resized_imgs))
            
            resized_imgs = np.transpose(resized_imgs,(0,3,1,2)) #num c h w
            resized_groundTruth= np.transpose(resized_groundTruth, (0,3,1,2))
            #assert(imgs.shape == (num_imgs,channels,img_height,img_width))
            #groundTruth = np.reshape(groundTruth,(num_imgs,3,img_height,img_width))
            assert(resized_groundTruth.shape == (num_imgs,3,resize_height,resize_width))
            print('Done!')

            return resized_imgs, resized_groundTruth
                                           
                                           
                                           
    elif train_test =='test':
        if size_mode =='original':
            imgs = np.empty((num_imgs,img_height,img_width,channels))
        elif size_mode =='resize':
            resized_imgs = np.empty((num_imgs,resize_height,resize_width,channels))
                                           
        print('img dir : ',imgs_dir)
        for count, filename in enumerate(sorted(os.listdir(imgs_dir)), start=0):
            print ("original image: " +filename)
            
            if size_mode == 'original':
                #img = Image.open(imgs_dir+filename)
                img = cv2.imread(imgs_dir+ filename)
                img = np.asarray(img)
                print('test img shape : ', np.shape(img))
                print('test imgs shape : ', np.shape(imgs))
                #if len(np.shape(img)) >2 :
                #    img = img[:,:,1]
                    
                imgs[count] = img
            elif size_mode == 'resize':
                img = Image.open(imgs_dir+filename)
                img = np.asarray(img)
                resized_img = cv2.resize(img, (resize_width, resize_height),interpolation = cv2.INTER_LANCZOS4 )
                resized_imgs[count] = resized_img
        
        if size_mode =='original':
            print ("imgs max: " +str(np.max(imgs)))
            print ("imgs min: " +str(np.min(imgs)))

            imgs = np.transpose(imgs,(0,3,1,2)) #num c h w

            print('Done!')

            return imgs
        elif size_mode =='resize':
            print ("imgs max: " +str(np.max(resized_imgs)))
            print ("imgs min: " +str(np.min(resized_imgs)))

            resize_imgs = np.transpose(resized_imgs,(0,3,1,2)) #num c h w

            print('Done!')

            return resized_imgs

    
    
    
if what_data =='inha_oct':
    imgs_train,groundTruth_train =get_datasets(original_img_train_path,ground_truth_img_train_path,train_test='train')
    write_hdf5(imgs_train, save_dir_path + what_data+"_train.hdf5")
    write_hdf5(groundTruth_train, save_dir_path+what_data + "_groundTruth_train.hdf5")
    #write_hdf5(border_masks_train,dataset_dir_path+what_data + "_borderMasks_train.hdf5")
elif what_data =='inha_oct_5classes':
    if mode == 'train':
        if size_mode == 'original':
            imgs_train,groundTruth_train =get_datasets(original_img_train_path,ground_truth_img_train_path,train_test='train')
            write_hdf5(imgs_train, save_dir_path + what_data+"_train.hdf5")
            write_hdf5(groundTruth_train, save_dir_path+what_data + "_groundTruth_train.hdf5")
            #write_hdf5(border_masks_train,dataset_dir_path+what_data + "_borderMasks_train.hdf5")    
        elif size_mode == 'resize':
            imgs_train,groundTruth_train =get_datasets(original_img_train_path,ground_truth_img_train_path,train_test='train')
            write_hdf5(imgs_train, save_dir_path + what_data+"_resized_train.hdf5")
            write_hdf5(groundTruth_train, save_dir_path+what_data + "_resized_groundTruth_train.hdf5")
            #write_hdf5(border_masks_train,dataset_dir_path+what_data + "_borderMasks_train.hdf5")    
                                           
    elif mode =='test':
        if size_mode == 'original':
            imgs_test =get_datasets(original_img_train_path,None,train_test='test')
            write_hdf5(imgs_test, save_dir_path + what_data+"_test.hdf5")
        elif size_mode == 'resize':
            imgs_test =get_datasets(original_img_train_path,None,train_test='test')
            write_hdf5(imgs_test, save_dir_path + what_data+"_resized_test.hdf5")    