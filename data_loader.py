# -*- coding: utf-8 -*-
# @Time    : 2018/7/31 09:41
# @Author  : Spytensor
# @File    : data_loader.py
# @Email   : zhuchaojie@buaa.edu.cn
#====================================================
#               定义数据加载的部分                     
#====================================================
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from itertools import chain
from glob import glob
import numpy as np 
import os
class cloth_data(Dataset):
    def __init__(self,root,transforms=None,train=True,test=False):
        #1.定义状态
        self.transforms = transforms
        self.test = test
        #2.以路径的格式获取文件，区分测试/训练/验证
        #2.1单独对测试集进行处理
        if self.test:
            self.file_names = list(map(lambda x :root+x,os.listdir(root)))
        #2.2针对非测试集
        if not self.test:
            image_folders = list(map(lambda x :root+x,os.listdir(root)))
            file_names = list(chain.from_iterable(list(map(lambda x: glob(x+"/"+"*.png"), image_folders)))) 
            np.random.shuffle(file_names)
            train_files,val_files = train_test_split(file_names,test_size=0.2)
            #2.3返回划分好的数据集
            if train:
                self.file_names = train_files
            else:
                self.file_names = val_files

    def __getitem__(self,index):
        #一次返回一个处理好的图片
        file_name = self.file_names[index]
        if self.test:
            label = file_name
        else:
            label = int(file_name.split("/")[-2])
        img_data = Image.open(file_name).convert("RGB")
        data = self.transforms(img_data)
        return data,label
    def __len__(self):
        return len(self.file_names)