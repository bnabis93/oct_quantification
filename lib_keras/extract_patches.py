import random
import numpy as np
import random
import configparser
import os
import cv2
import pandas as pd

import imgaug as ia
import imgaug.augmenters as iaa

from help_functions import load_hdf5
from help_functions import visualize
from help_functions import group_images
from pre_processing import my_preprocessing
from pre_processing import augmentations
from collections import OrderedDict
from tqdm import tqdm

from PIL import Image


    

def get_data_training(train_imgs_original,train_groudTruth,
                      patch_height,patch_width,
                      num_subimgs,label_mapping,inside_FOV, save_path):
    
    if os.path.isdir(save_path) == False:
        os.mkdir(save_path)
    else:
        print('already exist the folder in this path : {}'.format(save_path))
      

    print('number of subimages : ',num_subimgs)
    train_imgs_original = load_hdf5(train_imgs_original) #Pillow (RGB format)
    train_masks = load_hdf5(train_groudTruth)
    print('[DEBUG] shape of train_imgs_original : ',np.shape(train_imgs_original))
    print('[DEBUG] shape of train_imgs_label : ',np.shape(train_masks))

    
    visualize(group_images(train_imgs_original[:,:,:,:],5),'./'+save_path+'/imgs_train')  #check original imgs train
    visualize(group_images(train_masks[:,:,:,:],5),'./'+save_path+'/imgs_labels')
    
    train_imgs = my_preprocessing(train_imgs_original)
    
    visualize(group_images(train_imgs[:,:,:,:],5),'./'+ save_path+'/preprocessed')
    print('\n\n[get_data_training] preprocessed image shape : ',train_imgs.shape)
    print('\n[get_data_training] preprocessed mask shape : ',train_masks.shape)
    print('mask maximum val : ',np.max(train_masks))

    #train_masks = train_masks/255.
    '''
    train_masks mapper 
    5classes (RGB format)
    
    example) class01 : 0, class02 : 1, class03 : 2
    '''
    
    # convert images shape to square, input = 565 * 565
    
    print('[get_data_training] preprocessed2 image shape : ',train_imgs.shape)
    
    #zero padding
    train_imgs, train_masks = zero_padding(train_imgs, train_masks,patch_height)
    
    '''
    print(np.shape(train_masks))
    temp_path = './data/inha_oct_20_01_09/pad_label/'
    temp = train_masks
    temp = np.transpose(temp,(0,2,3,1))
    for i in range(40):
        temp2 = Image.fromarray(temp[i].astype('uint8'))
        temp2.save(temp_path+str(i)+'.png')
        '''
    

    print ('\n\n[get_data_training] train images/masks shape : {}'.format(train_imgs.shape))
    print ('[get_data_training] train images range (min-max) [{} , {}] '.format(str(np.min(train_imgs)),str(np.max(train_imgs))))
    print ('[get_data_training] train masks are within 0-1\n')
    
    patches_imgs_train, patches_masks_train,class_freq_tabel = extract_random(train_imgs,train_masks,patch_height,patch_width,num_subimgs,label_mapping,inside_FOV)
    
    print('[After patch] mask shape : ', patches_masks_train.shape)
    
    patches_imgs_train,patches_masks_train = augmentations(patches_imgs_train,patches_masks_train,20,0.3)
    #visualize(group_images(patches_masks_train[:,:,:,:],5),'./'+save_path+'/imgs_labels_aug')
    mode= None
    if mode == 'dynamic':
        #extract the TRAINING patches from the full images
        #extraact the random patches for data augmentation
        
        patches_imgs_train, patches_masks_train = extract_dynamic_random(train_imgs,train_masks,patch_height,patch_width,num_subimgs,inside_FOV)
        temp3_totimg = visualize(group_images(patches_imgs_train[0:50,:,:,:],5),'./'+save_path+'/train_patch_img')
        patches_imgs_train = augmentations(patches_imgs_train,0.2)
        print('[augmentation] patches_imgs_train : {}'.format(patches_imgs_train.shape))
        temp3_totimg = visualize(group_images(patches_imgs_train[0:50,:,:,:],5),'./'+save_path+'/augmented_patch_img')
        data_consistency_check(patches_imgs_train, patches_masks_train)
        return patches_imgs_train, patches_masks_train
    
    
    print ('\n\n[get_data_training] train PATCHES images/masks shape : {}'.format(patches_imgs_train.shape))
    print ('[get_data_training] train PATCHES images range (min-max): ' +str(np.min(patches_imgs_train)) +' - '+str(np.max(patches_imgs_train)))
    
    print('[get_data_training] patches_imgs_train : {}'.format(patches_imgs_train.shape))
    temp3_totimg = visualize(group_images(patches_imgs_train[0:50,:,:,:],5),'./'+save_path+'/train_patch_img')
    #data_consistency_check(patches_imgs_train, patches_masks_train)
    
    return patches_imgs_train, patches_masks_train, class_freq_tabel


#compare two data 
def data_consistency_check(imgs,masks):
    assert(len(imgs.shape)==len(masks.shape))
    assert(imgs.shape[0]==masks.shape[0])
    assert(imgs.shape[2]==masks.shape[2])
    assert(imgs.shape[3]==masks.shape[3])
    assert(masks.shape[1]==3)
    assert(imgs.shape[1]==1 or imgs.shape[1]==3)
    
#check if the patch is fully contained in the FOV (field of view)
def is_patch_inside_FOV(x,y,img_w,img_h,patch_h):
    x_ = x - int(img_w/2) # origin (0,0) shifted to image center
    y_ = y - int(img_h/2)  # origin (0,0) shifted to image center
    R_inside = 270 - int(patch_h * np.sqrt(2.0) / 2.0) #radius is 270 (from DRIVE db docs), minus the patch diagonal (assumed it is a square #this is the limit to contain the full patch in the FOV
    radius = np.sqrt((x_*x_)+(y_*y_))
    print('[is_patch_inside_FOV func] R_inside : {}, radius : {} '.format(R_inside, radius))
    if radius < R_inside:
        return True
    else:
        return False
    
def is_patch_avaliable_size(x,y,img_size, patch_size):
    '''
    hypothtsis)
        the img size and patch size is square format.
    
    
    parameter
    ---------
        x
        
        y
        
        img_size
        
        patch_size
    
    '''
    
def zero_padding(imgs, labels,patch_size):
    '''
    parameters 
        imgs
        labels
        patch_size
    
    return 
        padding_imgs
        padding_labels
    '''
    
    image_h, image_w = np.shape(imgs)[2],np.shape(imgs)[3]
    patch_h, patch_w = patch_size, patch_size
    
    padding_size_h = patch_h - (image_h % patch_h)
    padding_size_w = patch_w - (image_w % patch_w)
    
    print('\n\n[padding] pad h size : {}\t pad w size : {}'.format(padding_size_h, padding_size_w))
    
    imgs = np.transpose(imgs,(0,2,3,1))
    labels = np.transpose(labels,(0,2,3,1))
    print('\n\n[padding] imgs shape : {}\t labels shape : {}'.format(np.shape(imgs), np.shape(labels)))
    
    padding_images = np.zeros((np.shape(imgs)[0],image_h + padding_size_h, image_w+ padding_size_w,np.shape(imgs)[3] ))
    padding_labels = np.zeros((np.shape(labels)[0],image_h + padding_size_h, image_w+ padding_size_w,np.shape(labels)[3] ))
    
    print('\n\n[padding] pad imgs shape : {}\t pad labels shape : {}'.format(np.shape(padding_images), np.shape(padding_labels)))
    
    
    for idx in range(np.shape(imgs)[0]):
        if padding_size_h % 2 ==1:
            #print('shape of padding images : ',np.shape(padding_images[idx,int(padding_size_h/2)-1 : image_h + int(padding_size_h/2),int(padding_size_w/2) :image_w - int(padding_size_w/2),:]) )
            
            padding_images[idx,int(padding_size_h/2)-1 : image_h + int(padding_size_h/2),\
                          int(padding_size_w/2) :image_w + int(padding_size_w/2),:] = imgs[idx]
            padding_labels[idx,int(padding_size_h/2)-1 : image_h + int(padding_size_h/2),\
                          int(padding_size_w/2) :image_w + int(padding_size_w/2),:] = labels[idx]
            
        elif padding_size_w % 2 ==1:
            #print('shape of padding images : ',np.shape(padding_images[idx,int(padding_size_h/2) : image_h + int(padding_size_h/2),int(padding_size_w/2)-1 :image_w - int(padding_size_w/2),:]) )
            
            padding_images[idx,int(padding_size_h/2) : image_h + int(padding_size_h/2),\
                          int(padding_size_w/2)-1 :image_w + int(padding_size_w/2),:] = imgs[idx]
            padding_labels[idx,int(padding_size_h/2) : image_h + int(padding_size_h/2),\
                          int(padding_size_w/2)-1 :image_w + int(padding_size_w/2),:] = labels[idx]
            

        elif (padding_size_h %2 ==1) and (padding_size_w %2 ==1):
            #print('shape of padding images : ',np.shape(padding_images[idx,int(padding_size_h/2)-1 : image_h + int(padding_size_h/2),int(padding_size_w/2)-1 :image_w + int(padding_size_w/2),:]) )
            
            padding_images[idx,int(padding_size_h/2)-1 : image_h + int(padding_size_h/2),\
                          int(padding_size_w/2)-1 :image_w + int(padding_size_w/2),:] = imgs[idx]
            padding_labels[idx,int(padding_size_h/2)-1 : image_h + int(padding_size_h/2),\
                          int(padding_size_w/2)-1 :image_w + int(padding_size_w/2),:] = labels[idx]
            
        else:
            #print('shape of padding images : ',np.shape(padding_images[idx,int(padding_size_h/2) : image_h + int(padding_size_h/2),int(padding_size_w/2) :image_w + int(padding_size_w/2),:]) )
            
            padding_images[idx,int(padding_size_h/2) : image_h + int(padding_size_h/2),\
                          int(padding_size_w/2) :image_w + int(padding_size_w/2),:] = imgs[idx]
            padding_labels[idx,int(padding_size_h/2) : image_h + int(padding_size_h/2),\
                          int(padding_size_w/2) :image_w + int(padding_size_w/2),:] = labels[idx]
            
    #padding_images = np.transpose(padding_images,(0,3,1,2))
    #padding_labels = np.transpose(padding_labels,(0,3,1,2))
    print('\n\n[final pad] imgs shape : {}\t labels shape : {}'.format(np.shape(padding_images), np.shape(padding_labels)))
    
    padding_images = np.transpose(padding_images,(0,3,1,2))
    padding_labels = np.transpose(padding_labels,(0,3,1,2))
    
    print('\n\n[final transpose] imgs shape : {}\t labels shape : {}'.format(np.shape(padding_images), np.shape(padding_labels)))
    return padding_images, padding_labels
   
####################################################################################    
'''
Label processing function for patches

'''
    
def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a

def make_ordered_mapping(img):
    '''
    parameter 
        img (ch-last format img)
        
    return
        ordered mapping (unique value)
    '''
    unique_val = np.unique(img.reshape(-1, img.shape[2]), axis=0)
    print('unique val : ', unique_val)
    
    ordered_mapping =OrderedDict()
    num_class =0 
    for k in unique_val:
        ordered_mapping[totuple(k)] = num_class
        num_class = num_class+1
        
    
    return ordered_mapping
    

def ch_wise_one_hot_encoding(labels,class_num, mapping):
    '''
    input : patchs (ch first images)
            (ch, H, W)
    
    mapping example : 
        mapping = {
            (0,0,0) : 0,
            (255,0,0) : 1,
            (0,0,255) : 2
            }
    
    return patches (num, class-1,H, W)
    '''
    #print('[ch-wise one hot encoding] label shape : ', np.shape(labels))
    c, h, w = labels.shape
    encoded_labels = np.zeros((class_num, h, w ))
    
    ch_cnt =0 
    temp_label = np.transpose(labels, (1,2,0))
    #print(np.shape(temp_label))

    '''
    mapping 순서 이슈
    '''
    #print('[ch wise one hot] shape label : ', np.shape(temp_label))
    #print('[ch wise one hot] shape encoded label : ',np.shape(encoded_labels))
    for k in mapping:

        row,col = np.where(np.all(temp_label == k, axis = -1))
        encoded_labels[ch_cnt, row,col] = 1
        ch_cnt = ch_cnt + 1
        
        if k==(0,0,0):
            class0_pixels = len(row)
        elif k== (255,0,0):
            class1_pixels = len(row)
        elif k==(0,255,0):
            class2_pixels = len(row)
        elif k==(0,0,255):
            class3_pixels = len(row)
        elif k==(255,255,0):
            class4_pixels = len(row)
            
        # per image calculate class frequency

            
    #print('[ch-wise one hot encoding] encoded label shape : ', np.shape(encoded_labels))
    
    return encoded_labels,class0_pixels,class1_pixels,class2_pixels,class3_pixels,class4_pixels
    


####################################################################################
    
    
def extract_random(full_imgs,full_masks, patch_h,patch_w, num_patches,mapping, inside=True, ratio =1):
    # want to get each classes ratio
    class_frequency_table = pd.DataFrame(columns=['class_0', 'class_1', 'class_2','class_3','class_4', 'frequency_0','frequency_1','frequency_2','frequency_3','frequency_4'])
    
    
    
    num_patches = int(num_patches * ratio)
    '''
    if (num_patches % full_imgs.shape[0] != 0): #나머지가 0
        print ("[extract random] num_patches: please enter a multiple of 20")
        print('[extract randon] num_patches : {} , full imgs shape {}'.format(num_patches, full_imgs.shape))
        exit()'''
    #why..?
    assert (len(full_imgs.shape)==4 and len(full_masks.shape)==4)  #4D arrays (Tensor)
    assert (full_imgs.shape[1]==1 or full_imgs.shape[1]==3)  #check the channel is 1 or 3
    assert (full_masks.shape[1]==3)   #masks only black and white
    assert (full_imgs.shape[2] == full_masks.shape[2] and full_imgs.shape[3] == full_masks.shape[3]) #두 data의 사이즈는 같아야 한다.
    
    # each patches
    num_class = len(mapping)
    print('\n\n[extract random] num of class : ', num_class) 
   
    patches = np.empty((num_patches,full_imgs.shape[1], patch_h, patch_w))
    patches_masks = np.empty((num_patches, num_class, patch_h, patch_w)) #num, class-1 h, w
    
    
    print('[extract random] full image shape : {}'.format(full_imgs.shape))
    print('[extract random] full masks shape : {}'.format(full_masks.shape))
    print('[extract random] patches shape : {}'.format(patches.shape))
    print('[extract random] patches masks shape : {}'.format(patches_masks.shape))
    #print('[extract random] patch h : {} patch w : {}'.format(patch_h, patch_w))
    
    
    img_h = full_imgs.shape[2]
    img_w = full_imgs.shape[3]
    
    patch_per_img =int(num_patches / full_imgs.shape[0]) #ex, 48 / 20
    print('[extract random] patches per full image : {}'.format(patch_per_img))
    iter_total =0 #iter over the total number of patches (num_patches)
    
    for i in tqdm(range(full_imgs.shape[0]), desc = 'extract patches', mininterval = 10):  # loop, number of image samples
        k =0
        
        class_frequency_table.loc[i,'class_0' ] = 0
        class_frequency_table.loc[i,'class_1' ] = 0
        class_frequency_table.loc[i,'class_2' ] = 0
        class_frequency_table.loc[i,'class_3' ] = 0
        class_frequency_table.loc[i,'class_4' ] = 0
        
        # samping 
        temp_thr_label = full_masks[i].copy()
        temp_thr_label = np.transpose(temp_thr_label,(1,2,0)).astype(np.uint8)
        #print('temp_thr label shape : ', np.shape(temp_thr_label))
        temp_thr_label = cv2.cvtColor(temp_thr_label, cv2.COLOR_BGR2GRAY)
        temp_row, temp_col = np.nonzero(temp_thr_label)
        temp_thr_label[temp_row, temp_col]=255
        
        x, y, w, h = cv2.boundingRect(temp_thr_label)           #  Replaced code
        left = (x, np.argmax(temp_thr_label[:, x]))             #   
        right = (x+w-1, np.argmax(temp_thr_label[:, x+w-1]))    # 
        top = (np.argmax(temp_thr_label[y, :]), y)              # 
        bottom = (np.argmax(temp_thr_label[y+h-1, :]), y+h-1)   #

        while k<patch_per_img: #iter, number of patch per img
            # check sampling range
            
            
            x_center = random.randint(0+int(patch_w/2) , img_w-int(patch_w/2))
            #print('[extract random] x_center : {}'.format(x_center))
            y_center = random.randint(0+int(patch_w/2) , img_h-int(patch_w/2))
            #if y_center > bottom[1] or y_center < top[1]:
            #    pass
            
            
            #if inside ==True:
            #    if is_patch_inside_FOV(x_center,y_center,img_w,img_h,patch_h)==False:
            #        continue
                    
            patch = full_imgs[i,:,y_center-int(patch_h/2):y_center+int(patch_h/2),x_center-int(patch_w/2):x_center+int(patch_w/2)] #cropping
            patch_mask,c0,c1,c2,c3,c4 = ch_wise_one_hot_encoding(full_masks[i,:,y_center-int(patch_h/2):y_center+int(patch_h/2),x_center-int(patch_w/2):x_center+int(patch_w/2)],num_class,mapping)
            
            
            patches[iter_total]=patch
            patches_masks[iter_total]=patch_mask
            iter_total +=1   #total
            k+=1  #per full_img
            
            
            class_frequency_table.loc[i,'class_0' ] = class_frequency_table.loc[i,'class_0' ] + c0
            class_frequency_table.loc[i,'class_1' ] = class_frequency_table.loc[i,'class_1' ] + c1
            class_frequency_table.loc[i,'class_2' ] = class_frequency_table.loc[i,'class_2' ] + c2
            class_frequency_table.loc[i,'class_3' ] = class_frequency_table.loc[i,'class_3' ] + c3
            class_frequency_table.loc[i,'class_4' ] = class_frequency_table.loc[i,'class_4' ] + c4
            
        class_frequency_table.loc[i,'frequency_0' ] = (class_frequency_table.loc[i,'class_0']) / (class_frequency_table.loc[i,'class_0'] +class_frequency_table.loc[i,'class_1']+ class_frequency_table.loc[i,'class_2']+ class_frequency_table.loc[i,'class_3'] + class_frequency_table.loc[i,'class_4'])
        class_frequency_table.loc[i,'frequency_1' ] = (class_frequency_table.loc[i,'class_1']) / (class_frequency_table.loc[i,'class_0'] +class_frequency_table.loc[i,'class_1']+ class_frequency_table.loc[i,'class_2']+ class_frequency_table.loc[i,'class_3'] + class_frequency_table.loc[i,'class_4'])
        class_frequency_table.loc[i,'frequency_2' ] = (class_frequency_table.loc[i,'class_2']) / (class_frequency_table.loc[i,'class_0'] +class_frequency_table.loc[i,'class_1']+ class_frequency_table.loc[i,'class_2']+ class_frequency_table.loc[i,'class_3'] + class_frequency_table.loc[i,'class_4'])
        class_frequency_table.loc[i,'frequency_3' ] = (class_frequency_table.loc[i,'class_3']) / (class_frequency_table.loc[i,'class_0'] +class_frequency_table.loc[i,'class_1']+ class_frequency_table.loc[i,'class_2']+ class_frequency_table.loc[i,'class_3'] + class_frequency_table.loc[i,'class_4'])
        class_frequency_table.loc[i,'frequency_4' ] = (class_frequency_table.loc[i,'class_4']) / (class_frequency_table.loc[i,'class_0'] +class_frequency_table.loc[i,'class_1']+ class_frequency_table.loc[i,'class_2']+ class_frequency_table.loc[i,'class_3'] + class_frequency_table.loc[i,'class_4'])
            
    return patches, patches_masks, class_frequency_table


# no overlap mode
def recompose_img(data, num_h, num_w): # data = patch
    '''
    num_h, num_w -> Data를 몇분할 하느냐
    '''
    assert (len(data.shape) ==4)
    #assert (data.shape[1] ==1 or data.shape[1] ==3)    
    
    print('[DEBUG] data shape : ',np.shape(data)) 
    num_patch_per_img = num_h * num_w
    print('[DEBUG] data.shape[0] % num_patch_per_img : ',data.shape[0] % num_patch_per_img) 

    assert (data.shape[0] % num_patch_per_img ==0)
    
    num_full_imgs = int(data.shape[0]/num_patch_per_img)
    patch_h = data.shape[2]
    patch_w = data.shape[3]
    
    full_recomp = np.empty((num_full_imgs, data.shape[1], patch_h * num_h, patch_w*num_w))
    print('[DEBUG] full recomp shape : ',np.shape(full_recomp)) 
    iter_full_img = 0
    iter_single_patch =0
    
    while(iter_single_patch < data.shape[0]):
        #recompose
        single_recomp = np.empty((data.shape[1], patch_h * num_h, patch_w*num_w))
        
        for h in range(num_h):
            for w in range(num_w):
                single_recomp[:, h* patch_h : (h*patch_h) + patch_h, w*patch_w:(w*patch_w)+patch_w] = data[iter_single_patch]
                iter_single_patch = iter_single_patch +1
                
        full_recomp[iter_full_img]= single_recomp
        iter_full_img = iter_full_img+1
    
    assert (iter_full_img == num_full_imgs)
    
    print('[DEBUG] full result shape : ',np.shape(full_recomp))
    return full_recomp

def paint_border(data, patch_h, patch_w):
    # the data shapes are 4D arrays
    # 그리고 우리는 그것을 보장해줘야함
    assert (len(data.shape) ==4)
    # 여기서는 RGB or gray ch 
    assert (data.shape[1] ==1 or data.shape[1] ==3)
    
    img_h = data.shape[2]
    img_w = data.shape[3]
    
    new_img_h = 0
    new_img_w = 0
    #patch size가 정수로 딱 안나눠떨어질수도 있다.
    #이러한 상황을 보장해주어야 한다.
    if (img_h % patch_h)==0:
        new_img_h = img_h
    else:
        new_img_h = ((int(img_h/patch_h)) +1) * patch_h
        
    if (img_w % patch_w)==0:
        new_img_w = img_w
    else:
        new_img_w = ((int(img_w/patch_w)) +1) * patch_w
    
    #new data는 patch size를 위한 data frame이라 볼 수 있다.
    print('this ? : ', 'data shape : ',data.shape[0], data.shape[1], new_img_h, new_img_w)
    new_data = np.zeros((data.shape[0], data.shape[1], new_img_h, new_img_w))
    new_data[:,:,0:img_h, 0:img_w] = data[:,:,:,:]
    
    return new_data


def extract_ordered(full_imgs, patch_h, patch_w):
    assert(len(full_imgs.shape) ==4)
    assert(full_imgs.shape[1] ==1 or full_imgs.shape[1] ==3)
    #setting img's height, width
    img_h = full_imgs.shape[2]
    img_w = full_imgs.shape[3]
    
    num_patches_h = int(img_h / patch_h)
    num_patches_w = int(img_w / patch_w)
    print('[extract order func] patches h : {} patches w : {}'.format(num_patches_h, num_patches_w))
    #warning state
    if (img_h % patch_h !=0):
        print('[extract order func] warning ! {} patches in height, {} pixels left over!'.format(num_patches_h, img_h % patch_h))
    if (img_w & patch_w !=0):
        print('[extract order func] warning !  {} patches in height, {} pixels left over!'.format(num_patches_w, img_w % patch_w))
    
    num_patches_total = (num_patches_h * num_patches_w) * full_imgs.shape[0]
    patches = np.empty((num_patches_total, full_imgs.shape[1], patch_h, patch_w))
    
    print('[extract order func] number of patches : {}'.format(num_patches_total))

    iter_total =0
    #extract patches !
    for i in range(full_imgs.shape[0]):
        for h in range(num_patches_h):
            for w in range(num_patches_w):
                patch = full_imgs[i,:,h*patch_h : (h*patch_h)+patch_h, w*patch_w:(w*patch_w) + patch_w]
                patches[iter_total] = patch
                iter_total = iter_total+1
                
    assert (iter_total == num_patches_total)
    return patches
    
def zero_padding_test(imgs,patch_size):
    '''
    parameters 
        imgs
        patch_size
    
    return 
        padding_imgs
    '''
    
    image_h, image_w = np.shape(imgs)[2],np.shape(imgs)[3]
    patch_h, patch_w = patch_size, patch_size
    
    padding_size_h = patch_h - (image_h % patch_h)
    padding_size_w = patch_w - (image_w % patch_w)
    
    print('\n\n[padding] pad h size : {}\t pad w size : {}'.format(padding_size_h, padding_size_w))
    
    imgs = np.transpose(imgs,(0,2,3,1))
    print('\n\n[padding] imgs shape : {}'.format(np.shape(imgs)))
    
    padding_images = np.zeros((np.shape(imgs)[0],image_h + padding_size_h, image_w+ padding_size_w,np.shape(imgs)[3] ))
    
    print('\n\n[padding] pad imgs shape : {}'.format(np.shape(padding_images)))
    
    
    for idx in range(np.shape(imgs)[0]):
        if padding_size_h % 2 ==1:         
            padding_images[idx,int(padding_size_h/2)-1 : image_h + int(padding_size_h/2),\
                          int(padding_size_w/2) :image_w + int(padding_size_w/2),:] = imgs[idx]
         
        elif padding_size_w % 2 ==1:
            padding_images[idx,int(padding_size_h/2) : image_h + int(padding_size_h/2),\
                          int(padding_size_w/2)-1 :image_w + int(padding_size_w/2),:] = imgs[idx]

        elif (padding_size_h %2 ==1) and (padding_size_w %2 ==1):
            padding_images[idx,int(padding_size_h/2)-1 : image_h + int(padding_size_h/2),\
                          int(padding_size_w/2)-1 :image_w + int(padding_size_w/2),:] = imgs[idx]
            
        else:
            padding_images[idx,int(padding_size_h/2) : image_h + int(padding_size_h/2),\
                          int(padding_size_w/2) :image_w + int(padding_size_w/2),:] = imgs[idx]

    padding_images = np.transpose(padding_images,(0,3,1,2))
    
    return padding_images
   
    
def get_data_testing(test_img_ori_path, num_test_img,
                             patch_h, patch_w):
    
    # get img data
    test_img_ori = load_hdf5(test_img_ori_path)
    print('test img shape : ',np.shape(test_img_ori))
    test_imgs = my_preprocessing(test_img_ori)
    
    # extend both images and masks so they can be divided exactly by the patches dimensions
    # make tensor data.
    
    test_imgs = test_imgs[0:num_test_img,:,:,:]
    print('type : ',type(test_imgs))
    print('[get data testing func] prev test img shape : {}'.format(test_imgs.shape))
    
    test_imgs = zero_padding_test(test_imgs, patch_h)
    #paint_border(test_imgs,patch_h, patch_w)
    print('[get data testing func] after test img shape : {} '.format(test_imgs.shape))
    
    print ("[get_data_testing_func] test images range (min-max): " +str(np.min(test_imgs)) +' - '+str(np.max(test_imgs)))
    print ("[get_data_testing_fucn] test masks are within 0-1\n")
    
    patches_imgs_test = extract_ordered(test_imgs,patch_h,patch_w)
    #patches_grds_test = extract_ordered(test_grds,patch_h,patch_w)
    #data_consistency_check(test_imgs, test_grds)

    print ("\n[get_data_testing_fucn] test PATCHES images/grds shape:")
    print (patches_imgs_test.shape)
    print ("[get_data_testing_fucn] test PATCHES images range (min-max): {} - {}".format(str(np.min(patches_imgs_test)), str(np.max(patches_imgs_test))))

    return patches_imgs_test
    


# overlape mode
def get_data_testing_overlap(DRIVE_test_img_ori_path, num_test_img,
                             patch_h, patch_w, stride_h, stride_w):
    
    # get img data
    test_img_ori = load_hdf5(DRIVE_test_img_ori_path)
    print(np.shape(test_img_ori))
    test_imgs = my_preprocessing(test_img_ori)
    
    # extend both images and masks so they can be divided exactly by the patches dimensions
    # make tensor data.
    
    test_imgs = test_imgs[0:num_test_img,:,:,:]
    print('type : ',type(test_imgs))
    print('[get data testing overlap] prev test img shape : {} '.format(test_imgs.shape))
    
    # 여기서부터 달라진다.
    test_imgs = paint_border_overlap(test_imgs,patch_h, patch_w,stride_h, stride_w)
    print('[get data testing overlap] after test img shape : {} '.format(test_imgs.shape))
    
    print ("[get_data_testing_overlap func] test images range (min-max): " +str(np.min(test_imgs)) +' - '+str(np.max(test_imgs)))
    print ("[get_data_testing_overlap fucn] test masks are within 0-1\n")
    
    patches_imgs_test = extract_ordered_overlap(test_imgs,patch_h,patch_w,stride_h,stride_w)
    
    print ("\n[get_data_testing_overlap func] test PATCHES images shape:")
    print (patches_imgs_test.shape)
    print ("[get_data_testing_overlap func] test PATCHES images range (min-max): " +str(np.min(patches_imgs_test)) +' - '+str(np.max(patches_imgs_test)))

    return patches_imgs_test, test_imgs.shape[2], test_imgs.shape[3]


def extract_ordered_overlap(full_imgs, patch_h, patch_w, stride_h, stride_w):
    assert (len(full_imgs.shape) == 4)
    assert (full_imgs.shape[1] ==1 or full_imgs.shape[1] ==3)
    img_h = full_imgs.shape[2]
    img_w = full_imgs.shape[3]
    
    print('extract , : ',(img_h - patch_h),stride_h, (img_w - patch_w),stride_w)
    assert ((img_h - patch_h) % stride_h ==0 and (img_w - patch_w) % stride_w ==0)

    num_patch_img = ((img_h - patch_h)//stride_h +1) * ((img_w - patch_w)//stride_w +1)
    num_patch_total = num_patch_img * full_imgs.shape[0]
    
    print("[extrct_order_overlap func] Number of patches on h : " +str(((img_h-patch_h)//stride_h+1)))
    print("[extrct_order_overlap func] Number of patches on w : " +str(((img_w-patch_w)//stride_w+1)))
    print("[extrct_order_overlap func] number of patches per image: {} totally for this dataset: {}".format(str(num_patch_img), str(num_patch_total)))
    
    patches = np.empty((num_patch_total, full_imgs.shape[1], patch_h, patch_w))
    
    iter_total = 0
    for i in range(full_imgs.shape[0]):
        for h in range((img_h - patch_h)//stride_h +1):
            for w in range((img_w - patch_w)//stride_w +1): 
                patch = full_imgs[i,:,h*stride_h : (h*stride_h)+patch_h, w*stride_w : (w*stride_w)+patch_w ]
                patches[iter_total] = patch
                iter_total = iter_total+1
                
    assert (iter_total ==num_patch_total)
    return patches
    
def paint_border_overlap(data, patch_h, patch_w,stride_h, stride_w):
    # the data shapes are 4D arrays
    # 그리고 우리는 그것을 보장해줘야함
    assert (len(data.shape) ==4)
    # 여기서는 RGB or gray ch 
    assert (data.shape[1] ==1 or data.shape[1] ==3)
    
    img_h = data.shape[2]
    img_w = data.shape[3]
    
    leftover_h = (img_h - patch_h) % stride_h
    leftover_w = (img_w - patch_w) % stride_w
    
    if(leftover_h != 0):
        print("\nthe side H is not compatible with the selected stride of " +str(stride_h))
        print("img_h " +str(img_h) + ", patch_h " +str(patch_h) + ", stride_h " +str(stride_h))
        print("(img_h - patch_h) MOD stride_h: " +str(leftover_h))
        print("So the H dim will be padded with additional " +str(stride_h - leftover_h) + " pixels")
        
        temp_full_imgs = np.zeros((data.shape[0],data.shape[1],img_h+(stride_h-leftover_h),img_w))
        temp_full_imgs[0:data.shape[0],0:data.shape[1],0:img_h,0:img_w] = data
        data = temp_full_imgs
        
    if(leftover_w != 0):
        print("\nthe side H is not compatible with the selected stride of " +str(stride_w))
        print("img_h " +str(img_w) + ", patch_h " +str(patch_w) + ", stride_h " +str(stride_w))
        print("(img_h - patch_h) MOD stride_h: " +str(leftover_w))
        print("So the H dim will be padded with additional " +str(stride_w - leftover_w) + " pixels")
        
        temp_full_imgs = np.zeros((data.shape[0],data.shape[1],data.shape[2],img_w+(stride_w - leftover_w)))
        temp_full_imgs[0:data.shape[0],0:data.shape[1],0:data.shape[2],0:img_w] = data
        data = temp_full_imgs
    print ("new full images shape: \n" +str(data.shape))
    return data


def recompose_overlap_img(preds, img_h, img_w, stride_h, stride_w):
    assert (len(preds.shape)==4)  #4D arrays
    #assert (preds.shape[1]==1 or preds.shape[1]==3)  #check the channel is 1 or 3
    patch_h = preds.shape[2]
    patch_w = preds.shape[3]
    N_patches_h = (img_h-patch_h)//stride_h+1
    N_patches_w = (img_w-patch_w)//stride_w+1
    N_patches_img = N_patches_h * N_patches_w
    print ("N_patches_h: " +str(N_patches_h))
    print ("N_patches_w: " +str(N_patches_w))
    print ("N_patches_img: " +str(N_patches_img))
    assert (preds.shape[0]%N_patches_img==0)
    N_full_imgs = preds.shape[0]//N_patches_img
    print ("According to the dimension inserted, there are " +str(N_full_imgs) +" full images (of " +str(img_h)+"x" +str(img_w) +" each)")
    full_prob = np.zeros((N_full_imgs,preds.shape[1],img_h,img_w))  #itialize to zero mega array with sum of Probabilities
    full_sum = np.zeros((N_full_imgs,preds.shape[1],img_h,img_w))

    k = 0 #iterator over all the patches
    for i in range(N_full_imgs):
        for h in range((img_h-patch_h)//stride_h+1):
            for w in range((img_w-patch_w)//stride_w+1):
                full_prob[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]+=preds[k]
                full_sum[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]+=1
                k+=1

    assert(k==preds.shape[0])
    assert(np.min(full_sum)>=1.0)  #at least one
    final_avg = full_prob/full_sum
    print (final_avg.shape)
    assert(np.max(final_avg)<=1.0) #max value for a pixel is 1.0
    assert(np.min(final_avg)>=0.0) #min value for a pixel is 0.0
    return final_avg


# etc.
def inside_FOV_DRIVE(i, x, y,DRIVE_masks):
    assert(len(DRIVE_masks.shape) == 4)
    assert(DRIVE_masks.shape[1] == 1)
    # ONLY FOR DRIVE DATABASE
    if (x >= DRIVE_masks.shape[3] or y >= DRIVE_masks.shape[2]): #my image bigger than the original
        return False

    if (DRIVE_masks[i,0,y,x]>0):  #0==black pixels
        # print DRIVE_masks[i,0,y,x]  #verify it is working right
        return True
    
    
# remove data's border 
def kill_border(data, data_border_masks):
    assert(len(data.shape)==4)
    assert(data.shape[1] ==1 or data.shape[1] ==3)
    
    img_h = data.shape[2]
    img_w = data.shape[3]
    
    for i in range(data.shape[0]):
        for x in range(img_w):
            for y in range(img_h):
                if inside_FOV_DRIVE(i,x,y,data_border_masks) ==False:
                    data[i,:,y,x] = 0.0
    return data

def pred_only_FOV(target_img, target_img_mask, ori_img_border_mask):
    assert (len(target_img.shape) ==4 and len(target_img_mask.shape) ==4)
    assert (target_img.shape[0] == target_img_mask.shape[0])
    assert (target_img.shape[2] == target_img_mask.shape[2])
    assert (target_img.shape[3] == target_img_mask.shape[3])
    assert (target_img.shape[1]==1 and target_img_mask.shape[1] ==1)
    
    height = target_img.shape[2]
    width = target_img.shape[3]
    
    new_pred_img = []
    new_pred_mask = []
    
    for i in range(target_img.shape[0]): #number of image
        for x in range(width):
            for y in range(height):
                if inside_FOV_DRIVE(i,x,y,ori_img_border_mask) == True:
                    new_pred_img.append(target_img[i,:,y,x])
                    new_pred_mask.append(target_img_mask[i,:,y,x])
                    
    new_pred_img = np.asarray(new_pred_img)
    new_pred_mask = np.asarray(new_pred_mask)
    return new_pred_img, new_pred_mask



#### Decoding


