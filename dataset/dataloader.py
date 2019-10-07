from torch.utils.data import Dataset
from torchvision import transforms as T 
from config import config
from PIL import Image 
from itertools import chain 
from glob import glob
from tqdm import tqdm
from .augmentations import get_train_transform,get_test_transform
import random 
import numpy as np 
import pandas as pd 
import os 
import cv2
import torch 

#1.set random seed
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)

#2.define dataset
class ChaojieDataset(Dataset):
    def __init__(self,label_list,train=True,test=False):
        self.test = test 
        self.train = train 
        imgs = []
        if self.test:
            for index,row in label_list.iterrows():
                imgs.append((row["filename"]))
            self.imgs = imgs 
        else:
            for index,row in label_list.iterrows():
                imgs.append((row["filename"],row["label"]))
            self.imgs = imgs

    def __getitem__(self,index):
        if self.test:
            filename = self.imgs[index]
            img = cv2.imread(filename)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = get_test_transform(img.shape)(img)["image"]
            return img,filename
        else:
            filename,label = self.imgs[index] 
            img = cv2.imread(filename)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = get_train_transform(img.shape,augmentation=config.augmen_level)(img)["image"]
            return img,label
    def __len__(self):
        return len(self.imgs)

def collate_fn(batch):
    imgs = []
    label = []
    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[1])

    return torch.stack(imgs, 0), \
           label

def get_files(root,mode):
    #for test
    if mode == "test":
        files = []
        for img in os.listdir(root):
            files.append(root + img)
        files = pd.DataFrame({"filename":files})
        return files
    elif mode != "test": 
        #for train and val       
        all_data_path,labels = [],[]
        image_folders = list(map(lambda x:root+x,os.listdir(root)))
        all_images = list(chain.from_iterable(list(map(lambda x:glob(x+"/*"),image_folders))))
        print("loading train dataset")
        for file in tqdm(all_images):
            all_data_path.append(file)
            labels.append(int(file.split("/")[-2]))
        all_files = pd.DataFrame({"filename":all_data_path,"label":labels})
        return all_files
    else:
        print("check the mode please!")
    
