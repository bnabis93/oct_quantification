#help function


import h5py
import numpy as np
from PIL import Image
import subprocess
import os


def write_hdf5(arr,outfile):
    with h5py.File(outfile,"w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)
    
        
def load_hdf5(infile): #just load hdf5 format file.
    with h5py.File(infile,"r") as f:  #"with" close the file after its nested commands
        return f["image"][()]
    
def visualize(data,filename): # plot image
    assert (len(data.shape)==3) #height*width*channels
    img = None
    #print('data shape : ',data.shape)
    if data.shape[2]==1:  #in case it is black and white
        data = np.reshape(data,(data.shape[0],data.shape[1]))
    if np.max(data)>1:
        img = Image.fromarray(data.astype(np.uint8))   #the image is already 0-255
    else:
        img = Image.fromarray((data*255).astype(np.uint8))  #the image is between 0-1
    #print(img)
    img.save(filename + '.png')
    #print('file name : ',filename)
    return img


#group a set of images row per columns
def group_images(data,per_row):
    #print('[group images func] prev data shape  :', data.shape)
    assert (data.shape[0]%per_row==0)
    assert (data.shape[1]==1 or data.shape[1]==3)
    data = np.transpose(data,(0,2,3,1))  #corect format for imshow, make tensor, tensorflow backend
    #print('[group images func] after data shape : ', data.shape)

    all_stripe = []
    for i in range(int(data.shape[0]/per_row)):
        stripe = data[i*per_row]
        for k in range(i*per_row+1, i*per_row+per_row):
            stripe = np.concatenate((stripe,data[k]),axis=1)
        all_stripe.append(stripe)
    totimg = all_stripe[0]
    #print('[group images func] first total image : ', totimg.shape)

    for i in range(0,len(all_stripe)):
        totimg = np.concatenate((totimg,all_stripe[i]),axis=0)
        
    #print('[group images func] final total image : ', totimg.shape)

    return totimg

def rgb2gray(rgb):
    assert (len(rgb.shape)==4)  #4D arrays
    assert (rgb.shape[1]==3)
    bn_imgs = rgb[:,0,:,:]*0.299 + rgb[:,1,:,:]*0.587 + rgb[:,2,:,:]*0.114
    bn_imgs = np.reshape(bn_imgs,(rgb.shape[0],1,rgb.shape[2],rgb.shape[3]))
    return bn_imgs


#prepare the mask in the right shape for the Unet, Tensor
def masks_Unet(masks):
    '''
    make pixel-wise masks for semantic segmentation
    
    parameter
        masks (label)
        example)
            (50000, 3, 128, 128) -> (50000, 128 * 128, 3)
        
    return
        new masks (pixel-wise label)
    
    '''
    assert (len(masks.shape)==4)  #4D arrays
    assert (masks.shape[1]==3 )  #check the channel is 1
    
    im_ch = masks.shape[1]
    im_h = masks.shape[2]
    im_w = masks.shape[3]
    masks = np.reshape(masks,(masks.shape[0],im_ch,im_h*im_w))
    print('masks reshape : ', np.shape(masks))
    masks = np.transpose(masks,(0,2,1))
    print('masks transpose : ', np.shape(masks))
    
    new_masks = np.empty((masks.shape[0],im_h*im_w,im_ch))
    
    for i in range(masks.shape[0]):
        for ch in range(im_ch):
            for j in range(im_h*im_w):
                if  masks[i,j,0] == 0:
                    new_masks[i,j,0]=1
                    new_masks[i,j,1]=0
                    new_masks[i,j,2]=0
                elif masks[i,j,1] == 0:
                    print('pass')
                else:
                    print('pass')
    return new_masks

def pred_to_imgs(pred, patch_h, patch_w, mode = 'original'):
    assert(len(pred.shape) ==3)
    #assert(pred.shape[2] ==2) #binary?
    
    # (num, flatten img, class-1), remove background
    pred_imgs = np.empty((pred.shape[0], pred.shape[1], pred.shape[2] - 1))
    print('pred imgs shape : ',np.shape(pred_imgs))
    
    if mode == 'original':
        for i in range(pred.shape[0]): #patch num
            for pix in range(pred.shape[1]): #all pixels
                for ch in range(pred.shape[2]): # background는 빼겠다는 거지 (기존 0 = background 1 = foreground)
                    if ch !=0:
                        pred_imgs[i,pix, ch-1] = pred[i,pix,ch]
                
    elif mode == 'threshold':
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                for ch in range(pred.shape[2]):
                    if ch != 0:
                        if pred[i,pix,ch] >= 0.40:
                            pred_imgs[i,pix,ch-1] =1
                        else:
                            pred_imgs[i,pix,ch-1] =0
    else:
        print ("mode " +str(mode) +" not recognized, it can be 'original' or 'threshold'")
        exit(-1)
        
    # pred img -> (num, H, W ,ch)
    pred_imgs = np.reshape(pred_imgs,(pred_imgs.shape[0], patch_h, patch_w,pred.shape[2] - 1))
    pred_imgs = np.transpose(pred_imgs,(0,3,1,2))
    return pred_imgs

def show_on_jupyter(img,color= None,title=None, fig_size = (9,9)):
    import matplotlib.pyplot as plt
    import cv2
    """Show img on jupyter notebook. No return Value
    
    You should check the img's color space.
    I just consider about RGB color space & 1 ch color space(like green ch, gray space, ...)
    
    using matplotlib
    
    Parameters
    ----------
    img : 2-D Array
        numpy 2-D array
        opencv / sklearn / plt are avaliable.
        float / uint8 data type.
        
    color : string
        'gray' or 'None'
        'gray' means that img has a 1 ch.
        'None' means that img has a RGB ch.
        (default: None)
        
    title : string
        decide img's title
        (default : None)
        
    Returns
    -------
        No return value.
    
    Example
    -------
    >>> img = cv2.imread(img_path)
    >>> show_on_jupyter(img)
    
    img has a 1 ch
    >>> img = cv2.imread(img_path)
    >>> show_on_jupyter(img,'gray')
    """
    if color == 'gray':
        plt.axis("off")
        plt.title(title)
        plt.figure(figsize=fig_size)
        plt.imshow(img,cmap=color)
        plt.show()
    elif color == None:
        plt.axis("off")
        plt.title(title)
        plt.figure(figsize=fig_size)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()
    else:
        print("Gray or None")
        
def show_histogram(img,color = None, dtype = 'int', background_checker = False):
    import cv2
    import matplotlib.pyplot as plt
    """Show histogram on jupyter notebook
    
    I consider about img's color space and data type.
    
    color space : RGB / 1ch color space(also gray)
    data type : uint8 or float64
    
    so, you should check your img's color space and data type.
    
    Parameters
    ----------
    img : 2-D Array
        numpy 2-D array
        opencv / sklearn / plt are avaliable.
        float / uint8 data type.
        
    color : string
        'gray' or 'None'
        'gray' means that img has a 1 ch.
        'None' means that img has a RGB ch.
        (default: None)
        
    dtype :  string
        'int' or 'float'
        img's data type
        float : [min,max] and divided 256 values
        int = [0,256]
        (default : int)
        
    Returns
    -------
        No return value.
    
    Example
    -------
    >>> img = cv2.imread(img_path)
    >>> show_histogram(img)
    
    img has a 1 ch
    >>> img = cv2.imread(img_path)
    >>> show_histogram(img,'gray')
    
    1ch img & float img
    >>> img = cv2.imread(img_path)
    >>> show_histogram(img,'gray','float')
    """
    if background_checker == False:
        if (color == None) and (dtype =='int'):
            color = ('b','g','r')
            for i,col in enumerate(color):
                histr = cv2.calcHist([img],[i],None,[256],[0,256])
                plt.plot(histr,color = col)
                plt.xlim([0,256])
            plt.show()
        if (color == None) and (dtype =='float'):
            color = ('b','g','r')
            for i,col in enumerate(color):
                histr = cv2.calcHist([img],[i],None,[256],[0,1])
                plt.plot(histr,color = col)
                plt.xlim([0,256])
            plt.show()
        elif (color == 'gray')and (dtype=='float') :
            plt.hist(img.ravel(),256,[img.min(),img.max()])
            plt.title('Histogram for gray scale picture')
            plt.show()
        elif (color == 'gray') and (dtype == 'int'):
            plt.hist(img.ravel(),256,[0,256])
            plt.title('Histogram for gray scale picture')
            plt.show()
        else:
            print('check your parameter')
    else:
        if (color == None) and (dtype =='int'):
            color = ('b','g','r')
            for i,col in enumerate(color):
                histr = cv2.calcHist([img],[i],None,[256],[1,256])
                plt.plot(histr,color = col)
                plt.xlim([0,256])
            plt.show()
            
        if (color == None) and (dtype =='float'):
            color = ('b','g','r')
            for i,col in enumerate(color):
                histr = cv2.calcHist([img],[i],None,[256],[0,1])
                plt.plot(histr,color = col)
                plt.xlim([0,256])
            plt.show()
        elif (color == 'gray')and (dtype=='float') :
            plt.hist(img.ravel(),256,[img.min(),img.max()])
            plt.title('Histogram for gray scale picture')
            plt.show()
        elif (color == 'gray') and (dtype == 'int'):
            plt.hist(img.ravel(),256,[1,256])
            plt.title('Histogram for gray scale picture')
            plt.show()
        else:
            print('check your parameter')
            
            

def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ])
    result = result.decode('utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

def dir_checker(dir_name):
    if os.path.isdir(dir_name) == False:
        os.mkdir(dir_name)
    else:
        print('already exist the folder in this path : {}'.format(dir_name)) 