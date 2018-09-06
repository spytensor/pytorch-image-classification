# -*- coding: utf-8 -*-
# @Time    : 2018/7/31 09:41
# @Author  : Spytensor
# @File    : config.py
# @Email   : zhuchaojie@buaa.edu.cn
#====================================================
#               定义所需要的一些参数                     
#====================================================
from datetime import datetime
class DefaultConfigs(object):
    #1.str类参数
    data_path = "/data2/dockspace_zcj/traffic-sign/train/"
    test_path = "/data2/dockspace_zcj/traffic-sign/test/"
    model_name = "resnet152"
    weights_path = "./logs/{}_params.pkl".format(model_name)
    predict_path = "./results/{}".format("%s-lgb_two_three_features"%(datetime.now().strftime("%Y%m%d-%H%M%S")))
    #2.数字类特征
    img_size = 224
    channels = 3
    epochs = 120
    batch_size = 32
    lr = 1e-4
    weight_decay = 1e-4
    print_freq = 20
    num_classes = 62
    gpu = "3"
config = DefaultConfigs()

