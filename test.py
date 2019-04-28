import os 
import random 
import time
import json
import torch
import torchvision
import numpy as np 
import pandas as pd 
import warnings
from datetime import datetime
from torch import nn,optim
from config import config 
from collections import OrderedDict
from torch.autograd import Variable 
from torch.utils.data import DataLoader
from dataset.dataloader import *
from sklearn.model_selection import train_test_split,StratifiedKFold
from timeit import default_timer as timer
from models.model import *
from utils import *
from IPython import embed
#1. set random.seed and cudnn performance
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')

#3. test model on public dataset and save the probability matrix
def test(test_loader,model,folds):
    #3.1 confirm the model converted to cuda
    test_labels = open("%s.csv"%config.model_name,"w")
    model.cuda()
    model.eval()
    tta = False
    for i,(input,filepath) in enumerate(tqdm(test_loader)):
        #3.2 change everything to cuda and get only basename
        filepath = [os.path.basename(x) for x in filepath]
        with torch.no_grad():
            image_var = Variable(input).cuda()
            #3.3.output
            #print(filepath)
            #print(input,input.shape)
            if tta == False:
                y_pred = model(image_var)
                smax = nn.Softmax(1)
                smax_out = smax(y_pred)
                label = np.argmax(smax_out.cpu().data.numpy())
                test_labels.write(filepath[0]+","+str(label)+"\n")

#4. more details to build main function    
def main():
    fold = 0
    #4.1 mkdirs
    if not os.path.exists(config.submit):
        os.mkdir(config.submit)
    if not os.path.exists(config.weights):
        os.mkdir(config.weights)
    if not os.path.exists(config.best_models):
        os.mkdir(config.best_models)
    if not os.path.exists(config.logs):
        os.mkdir(config.logs)
    if not os.path.exists(config.weights + config.model_name + os.sep +str(fold) + os.sep):
        os.makedirs(config.weights + config.model_name + os.sep +str(fold) + os.sep)
    if not os.path.exists(config.best_models + config.model_name + os.sep +str(fold) + os.sep):
        os.makedirs(config.best_models + config.model_name + os.sep +str(fold) + os.sep)       
    #4.2 get model and optimizer
    model = get_net()
    #model = torch.nn.DataParallel(model)
    model.cuda()
    #4.5 get files and split for K-fold dataset
    test_files = get_files(config.test_data,"test")
    #4.5.4 load dataset
    test_dataloader = DataLoader(ChaojieDataset(test_files,test=True),batch_size=1,shuffle=False,pin_memory=False)
    best_model = torch.load("checkpoints/best_model/%s/0/model_best.pth.tar"%config.model_name)
    model.load_state_dict(best_model["state_dict"])
    test(test_dataloader,model,fold)

if __name__ =="__main__":
    main()





















