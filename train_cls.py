
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
import sklearn.metrics as skmetrics

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
	GLOBAL_SEED = 2021
	###### 设置随机数种子 ######
	for k in opt.kidx:
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

		# print(net)
		# return

		# #summary(net, (1,192,192))
		# #return
		#
		# if opt.eval == True:
		#     net = net.eval()
		#     a = torch.ones([8, 3, 448, 448])
		#     a = Variable(a.type(torch.FloatTensor).cuda())
		#
		#     print(a.shape)
		#     output = net(a)
		#     print('in forward')
		#     return
		# # print(output.shape)
		# return
		#
		# if isinstance(output, tuple) == False:
		#     print(output.shape)
		# else:
		#     y_pre = output[0]
		#     M_list = output[1]
		#     for M in M_list:
		#         print(M.shape)
		# #print(.shape)
		# return


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
			scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.Tmax, \
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
		train_set = train_Dataset(train_slice_list)
		train_data_num = len(train_set.img_list)
		train_batch = Data.DataLoader(dataset=train_set, batch_size=opt.train_bs, shuffle=True, \
									num_workers=opt.num_workers, worker_init_fn=worker_init_fn, \
									drop_last=True, collate_fn = non_model.num_collate)
		print('load train data done, num =', train_data_num)

		# 定义 val set
		val_slice_list = fold_list[str(k)]['val']
		val_set = val_Dataset(val_slice_list)
		val_data_num = len(val_set.img_list)
		val_batch = Data.DataLoader(dataset=val_set, batch_size=opt.val_bs, shuffle=False,
									num_workers=opt.test_num_workers, worker_init_fn=worker_init_fn)
		print('load val data done, num =', val_data_num)

		# return

		###### Task based metric ######
		best_net = None
		best_auc = 0.0
		epoch_save = 0
		best_metric = 0
		lr_change = 0

		loss_hist = collections.deque(maxlen=500)

		###### Start Training ######
		for e in range(opt.epoch):
			tmp_epoch = e+opt.start_epoch
			print('====================== Folder %s Epoch %s ========================' % (k, tmp_epoch))

			# 当前 epoch 的 lr
			tmp_lr = optimizer.__getstate__()['param_groups'][0]['lr']

			# 如果使用 cycle_save，在每个 cycle 开始时重置保存条件
			if opt.cycle_r > 0:
				if e % (2*opt.Tmax) == 0:
					best_net = None
					best_metric_list = np.zeros((opt.label_length - 1))
					best_metric = 0
					min_loss = 10
			# 否则使用 early stop
			else:
				if tmp_epoch > epoch_save + opt.gap_epoch:
					break
				if lr_change == 2:
					break

			net.training = True

			for i, return_list in tqdm(enumerate(train_batch)):
				case_name, x, y, z = return_list

				im = Variable(x.type(torch.FloatTensor).cuda())
				label = Variable(z.type(torch.FloatTensor).cuda())

				if e == 0 and i == 0:
					print('input size:',im.shape)

				# forward
				outputs = net(im)
				if label.size() != outputs.size():
					label = label.unsqueeze(1)
				loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, label)

				# return
				if bool(loss == 0):
					continue

				optimizer.zero_grad()
				loss.backward()
				torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
				optimizer.step()
				loss_hist.append(float(loss))

				if i % 50 == 0:
					print(
					'Ep: {} | Iter: {} | Running loss: {:1.4f}'.format(
						tmp_epoch, i, np.mean(loss_hist)))

			# 清除缓存，减少训练中内存占用
			torch.cuda.empty_cache()

			net = net.eval()
			val_loss = 0
			truth = []
			pred = []
			data_length = val_data_num

			with torch.no_grad():
				for i, return_list in tqdm(enumerate(val_batch)):
					case_name, x, y, z = return_list

					##################### Get detections ######################
					im = Variable(x.type(torch.FloatTensor).cuda())
					label = Variable(z.type(torch.FloatTensor).cuda())
					if e == 0 and i == 0:
						print('input size:',im.shape)

					# forward
					outputs = net(im)
					outputs = torch.sigmoid(outputs)
					if label.size() != outputs.size():
						label = label.unsqueeze(1)
					loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, label)
					truth.append(label.cpu().detach().numpy())
					pred.append(outputs.cpu().detach().numpy())
					batch_size = im.shape[0]
					val_loss += loss.item()*batch_size

			preds = np.concatenate(np.array(pred), axis=0)
			preds = torch.FloatTensor(preds).cpu().detach().numpy()
			pred_value = preds.flatten()
			truth_c = np.concatenate(truth, axis=0)

			roc_auc = skmetrics.roc_auc_score(truth_c, pred_value)
			threshold = non_model.Find_Optimal_Cutoff(truth_c, pred_value)
			acc = skmetrics.accuracy_score(truth_c, (pred_value>=threshold))
			precision = skmetrics.precision_score(truth_c, (pred_value>=threshold))
			recall = skmetrics.recall_score(truth_c, (pred_value>=threshold))
			tn, fp, fn, tp = skmetrics.confusion_matrix(truth_c, (pred_value>=threshold)).ravel()
			neg_recall = tn/(tn+fp)

			print('auc: %s' % roc_auc)
			print('accuracy: %s' % acc)
			print("precision: %s" % precision)
			print("sensitivity: %s" % recall)
			print("specificity: %s" % neg_recall)
			if roc_auc > best_auc:
				best_auc = roc_auc
				epoch_save = tmp_epoch
				save_dict = {}
				save_dict['net'] = net
				save_dict['config_dict'] = config_dict
				torch.save(save_dict, save_model_folder + 'K%s_%s_AUC_%.4f_sens_%.4f_spec_%.4f.pkl' %
							   (k, str(epoch_save).rjust(3,'0'), best_auc, best_auc, neg_recall))

				info_dict = {
					'predicted_prob': pred_value.tolist(),
					'pred': ((pred_value>=threshold)*1).tolist(),
					'truth': truth_c.tolist(),
				}
				with open(save_info_folder + 'K%s_%s_AUC_%.4f_sens_%.4f_spec_%.4f.json' %
							   (k, str(epoch_save).rjust(3,'0'), best_auc, best_auc, neg_recall), 'w') as f:
					json.dump(info_dict, f, indent=2)

				del save_dict
				del info_dict
				print('====================== model save ========================')

			if opt.cos_lr == True:
				scheduler.step()
			else:
				scheduler.step(best_auc)

			# 经过学习率策略后的 lr
			# 如果有变化，记录变化情况
			before_lr = optimizer.__getstate__()['param_groups'][0]['lr']

			if before_lr != tmp_lr:
				epoch_save = tmp_epoch
				lr_change += 1
				print('================== lr change to %.6f =================='%before_lr)

			# 清除缓存，减少训练中内存占用
			torch.cuda.empty_cache()

if __name__ == '__main__':
	import fire

	fire.Fire()
	

