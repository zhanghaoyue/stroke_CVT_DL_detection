import numpy as np
import cv2
import os
import torch
import random
from utils import in_model
import sys
from config import opt
import json
from config import opt
from copy import deepcopy
import process
import pywt
import warnings
warnings.filterwarnings("ignore")

class train_Dataset:
    def __init__(self, img_list):
        self.img_path = opt.path_img
        self.img_list = img_list

        return

    def __getitem__(self, idx):
        path_name = self.img_list[idx]

        # for img
        tmp_data = in_model.get_img(self.img_path, path_name)

        img_dict = process.train_preprocess(tmp_data)
        img_list = [img_dict['T1'], img_dict['T2'], img_dict['FL']]
        img = np.array(img_list)
        img = img.astype('float32')
        img = torch.tensor(img)

        bbox_array = img_dict['BBOX']
        annot = in_model.get_bbox(bbox_array)
        annot = annot.astype('float32')

        num_objs = annot.shape[0]
        boxes = []
        labels = []
        target = {}
        if num_objs == 0:
            boxes = torch.zeros((0,4), dtype=torch.float32)
            labels = torch.zeros((1,1), dtype=torch.int64)
        else:
            for i in range(num_objs):
                boxes.append(annot[i][:4])
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.ones((num_objs,), dtype=torch.int64)
        target["boxes"] = boxes
        target["labels"] = labels

        image_id = torch.tensor([idx])
        area = (boxes[:,3]-boxes[:,1])*(boxes[:,2]-boxes[:,0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return_list = [path_name, img, target]

        return return_list

    def __len__(self):
        return len(self.img_list)

class val_Dataset:
    def __init__(self, img_list):
        self.img_path = opt.path_img
        self.img_list = img_list
        return

    def __getitem__(self, idx):
        path_name = self.img_list[idx]

        # for img
        tmp_data = in_model.get_img(self.img_path, path_name)

        img_dict = process.val_preprocess(tmp_data)
        img_list = [img_dict['T1'], img_dict['T2'], img_dict['FL']]
        img = np.array(img_list)
        img = img.astype('float32')
        img = torch.tensor(img)

        bbox_array = img_dict['BBOX']
        annot = in_model.get_bbox(bbox_array)
        annot = annot.astype('float32')

        num_objs = annot.shape[0]
        boxes = []
        labels = []
        target = {}

        if num_objs == 0:
            boxes = torch.zeros((0,4), dtype=torch.float32)
            labels = torch.zeros((1,1), dtype=torch.int64)
        else:
            for i in range(num_objs):
                boxes.append(annot[i][:4])
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.ones((num_objs,), dtype=torch.int64)
        target["boxes"] = boxes
        target["labels"] = labels
            
        image_id = torch.tensor([idx])
        area = (boxes[:,3]-boxes[:,1])*(boxes[:,2]-boxes[:,0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return_list = [path_name, img, target]

        return return_list

    def __len__(self):
        return len(self.img_list)