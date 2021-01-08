from PIL import Image
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split



class OctTrainData(Dataset):
    '''
    data loader class
    
    '''
    def __init__(self, data_path, transform = None):
        super(OctTrainData, self).__init__()
        self.train_data_path = data_path['train']
        self.train_label_path = data_path['label']
        self.train_data = []
        self.train_lables = []
        self.transform = transform
        self.mapping = {
            (0,0,0) : 0,
            (255,0,0) : 1,
            (0,0,255) : 2
        }
        self.num_class = len(self.mapping) 
        
        for count, filename in enumerate(sorted(os.listdir(self.train_data_path)), start=0):
            if filename.startswith('.ipynb') ==False:
                self.train_data.append(os.path.join(self.train_data_path,filename))
                
        for count, filename in enumerate(sorted(os.listdir(self.train_label_path)), start=0):
            if filename.startswith('.ipynb') ==False:
                self.train_lables.append(os.path.join(self.train_label_path,filename))
    
    
    def mask_to_class(self,label):
        
        for k in self.mapping:
            label[(label == k).all(axis=2)] = self.mapping[k]

        return label
    
    def __len__(self):
        return len(self.train_data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        imgs = Image.open(self.train_data[idx]).convert('L')
        labels  = Image.open(self.train_lables[idx])
        
        if self.transform != None:
            imgs= self.transform(imgs)
            #labels = self.transform(labels)
        
        labels = np.array(labels)
        masks = self.mask_to_class(labels)
        masks = self.transform(masks)
        
        #print('masks shape : ',np.shape(masks))
        #print('imgs  shape : ', np.shape(imgs))
        
        return imgs, masks
