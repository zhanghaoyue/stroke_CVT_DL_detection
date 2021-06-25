import os
import sys
import json
import random
import time
from collections import OrderedDict
import collections

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from torchvision.ops import box_iou

from tensorboardX import SummaryWriter
from torchsummary import summary

from config import opt
from utils import non_model
from make_dataset import train_Dataset, val_Dataset
from net import model_tools

import numpy as np

from tqdm import tqdm
from pprint import pprint
from copy import deepcopy
import shutil
import cv2
import warnings
warnings.filterwarnings("ignore")

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2000, rlimit[1]))

# train 应该是一个不受任务类型，数据类型，网络结构影响的通用文件
# 所有与上述内容相关的部分，都应该封装为模块
def train(**kwargs):
    # stage 1 - 参数设定
    # 基于输入参数进行配置修改
    kwargs, data_info_dict = non_model.read_kwargs(kwargs)
    # 参数全局化
    opt.load_config('../config/all.txt')
    config_dict = opt._spec(kwargs)

    # stage 2 - 路径设定
    # 模型存储路径
    save_model_folder = '../model/%s/' % (opt.path_key) + str(opt.net_idx) + '/'
    # info 存储路径
    save_info_folder = '../info/%s/' % (opt.path_key) + str(opt.net_idx) + '/'
    # 生成路径
    non_model.make_path_folder(save_model_folder)
    non_model.make_path_folder(save_info_folder)
    # 参数配置存储
    with open(save_info_folder + 'config.json', 'w', encoding='utf-8') as json_file:
        json.dump(config_dict, json_file, ensure_ascii=False, indent=4)

    # 读取用于 training 的数据
    fold_list = data_info_dict['Train']
    test_list = data_info_dict['Test']

    for k in opt.kidx:
        ###### 设置随机数种子 ######
        GLOBAL_SEED = 2021
        random.seed(GLOBAL_SEED)
        np.random.seed(GLOBAL_SEED)
        torch.manual_seed(GLOBAL_SEED)
        torch.cuda.manual_seed(GLOBAL_SEED)
        torch.cuda.manual_seed_all(GLOBAL_SEED)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        ###### GPU 环境 ######
        # 训练时用固定环境的方式选择 GPU，因为需要 GPU 定义的参数很多
        # 外部测试时通过 cuda(gpu) 的方式选择 GPU，因为需要 GPU 定义的参数相对较少
        data_gpu = opt.gpu_idx
        torch.cuda.set_device(data_gpu)

        ###### 初始化网络 ######
        # 网络结构 & 加载 cuda
        net = model_tools.get_model()
        net = net.cuda()

        ###### 设置优化器相关参数 ######
        # 优化器
        lr = opt.lr
        if opt.optim == 'SGD':
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                                        lr=lr, weight_decay=opt.wd, momentum=0.9)
            print('================== SGD lr = %.6f ==================' % lr)

        elif opt.optim == 'AdamW':
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()),
                                          lr=lr, weight_decay=opt.wd)
            print('================== AdamW lr = %.6f ==================' % lr)
        # 学习率策略
        if opt.cos_lr:
            cheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.Tmax, \
                                                                  eta_min=opt.lr / opt.lr_gap)
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=opt.patience)

        ###### Dataloader Setting ######
        # 定义 worker
        def set_seed(seed):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        GLOBAL_WORKER_ID = None

        def worker_init_fn(worker_id):
            global GLOBAL_WORKER_ID
            GLOBAL_WORKER_ID = worker_id
            set_seed(GLOBAL_SEED + worker_id)

        # 定义 train set
        train_slice_list = fold_list[str(k)]['train']
        val_slice_list = fold_list[str(k)]['val']
        count_slice_list = train_slice_list + val_slice_list + test_list
        train_set = train_Dataset(count_slice_list)
        train_data_num = len(train_set.img_list)
        train_batch = Data.DataLoader(dataset=train_set, batch_size=opt.train_bs, shuffle=False, \
                                      num_workers=opt.num_workers, worker_init_fn=worker_init_fn, \
                                      drop_last=True, collate_fn = non_model.num_collate)
        print('load train data done, num =', train_data_num)

        anno_num = 0
        det_anno_num = 0
        miss_obj = []

        for i, return_list in tqdm(enumerate(train_batch)):
            case_name, x, y = return_list
            im = Variable(x.type(torch.FloatTensor).cuda())
            label = Variable(y.type(torch.FloatTensor).cuda())

            if i == 0:
                anchors = net.anchors(im)
                anchor = anchors[0, :, :]
                print(anchor.shape)

            batch_size = im.shape[0]
            annotations = label

            for j in range(batch_size):
                bbox_annotation = annotations[j, :, :]
                bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

                if bbox_annotation.shape[0] == 0:
                    continue

                anno_num += bbox_annotation.shape[0]
                IoU = box_iou(anchor, bbox_annotation[:, :4])
                IoU_max, IoU_argmax = torch.max(IoU, dim=0)  # num_anchors x 1

                for idx in range(IoU_max.shape[0]):
                    tmp_iou = IoU_max[idx]
                    if tmp_iou >= opt.hth:
                        det_anno_num += 1
                    else:
                        tmp_bbox = bbox_annotation[idx]
                        miss_obj.append(tmp_bbox)

        print(anno_num, det_anno_num)
        #pprint(miss_obj)


if __name__ == '__main__':
    import fire

    fire.Fire()
    

