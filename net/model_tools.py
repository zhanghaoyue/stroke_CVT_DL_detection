from config import opt
import torch
import math
from copy import deepcopy
import types
from torch import nn
import torchvision
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from .block_utils import *


def get_model():
    global net
    if opt.model == 'mr':
        net = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=opt.pre_train,
                                                                 pretrained_backbone=True,
                                                                 min_size=opt.in_size,
                                                                 max_size=opt.in_size)

    elif opt.model == 'mr_cls':
        net = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=opt.pre_train,
                                                                 pretrained_backbone=True,
                                                                 min_size=opt.in_size,
                                                                 max_size=opt.in_size)
        net.mode = 'det'
        net.backbone.forward = types.MethodType(mr_cls_backbone_pre, net.backbone)
        for dv in opt.dv_num:
            in_c = 256 * (2 ** dv)
            if opt.cls_block == 's':
                exec('''net.cls_branch_%s = nn.Sequential(
                        nn.Conv2d(%s, %s, 1),
                        nn.AdaptiveAvgPool2d((1,1)),
                        nn.Flatten(),
                        nn.Linear(%s, %s)
                    )'''%(dv, in_c, in_c, in_c, opt.cls_output))
            elif opt.cls_block == 'p':
                exec('''net.cls_branch_%s = nn.Sequential(
                    nn.Conv2d(%s, %s, 3, 1),
                    nn.BatchNorm2d(%s),
                    nn.ReLU(),
                    nn.Conv2d(%s, 256, 1, 1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((4, 4)),
                    nn.Flatten(),
                    nn.Linear(4096, %s)
                    )'''%(dv, in_c, in_c, in_c, in_c, opt.cls_output))

        for name, param in net.backbone.named_parameters():
            if "cls_branch" in name and "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
        net.forward = types.MethodType(mr_cls_forward, net)
        if opt.cls_loss == 'ce':
            net.cls_loss = nn.CrossEntropyLoss()
        elif opt.cls_loss == 'bce':
            net.cls_loss = nn.BCEWithLogitsLoss()

    elif opt.model == 'mr_SA':
        net = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=opt.pre_train,
                                                                 pretrained_backbone=True,
                                                                 min_size=opt.in_size,
                                                                 max_size=opt.in_size)
        for dv in opt.dv_num:
            exec('''net.backbone.SA_branch_%s = nn.Sequential(
                                    nn.Conv2d(2, 1, 3, 1, 1),
                                    nn.Sigmoid()
                        )''' % (dv))

        net.backbone.forward = types.MethodType(mrSA_forward_pre, net.backbone)
        for name, param in net.backbone.named_parameters():
            if "SA" in name and "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    elif opt.model == 'MC':
        net = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=opt.pre_train,
                                                                 pretrained_backbone=True,
                                                                 min_size=opt.in_size,
                                                                 max_size=opt.in_size)
        if opt.MC_share == False:
            net.backbone.body_1 = deepcopy(net.backbone.body)
            net.backbone.body_2 = deepcopy(net.backbone.body)


        for dv in opt.dv_num:
            in_c = 256 * (2 ** dv)
            if opt.MC_block == 'CO':
                exec('''net.backbone.block_%s = nn.Conv2d(in_c * 3, in_c, 1)'''%(dv))
            elif opt.MC_block == 'SE':
                exec('''net.backbone.block_%s = MC_se_block(in_c)'''%(dv))
            elif opt.MC_block == 'NL':
                exec('''net.backbone.block_%s = NLBlockND(in_c, mode=opt.NL_mode)''' % (dv))

        net.backbone.forward = types.MethodType(MC_forward, net.backbone)

        for name, param in net.backbone.named_parameters():
            if "concat" in name and "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    elif opt.model == 'MP':
        net = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=opt.pre_train,
                                                                 pretrained_backbone=True,
                                                                 min_size=opt.in_size,
                                                                 max_size=opt.in_size)

        backbone = net.backbone.body
        layer_0 = nn.Sequential(backbone.conv1,
                                backbone.bn1,
                                backbone.relu,
                                backbone.maxpool)

        layer_1 = backbone.layer1
        layer_2 = backbone.layer2
        layer_3 = backbone.layer3
        layer_4 = backbone.layer4

        for b in range(5):
            if b in opt.share_b:
                exec('''net.backbone.main_%s = layer_%s''' % (b, b))
            else:
                exec('''net.backbone.branch_%s_0 = deepcopy(layer_%s)''' % (b, b))
                exec('''net.backbone.branch_%s_1 = deepcopy(layer_%s)''' % (b, b))
                exec('''net.backbone.branch_%s_2 = deepcopy(layer_%s)''' % (b, b))

        for dv in opt.dv_num:
            dv = int(dv)
            in_c = 256 * (2 ** dv)
            if opt.MC_block == 'CO':
                exec('''net.backbone.block_%s = nn.Conv2d(in_c * 3, in_c, 1)''' % (dv))
            elif opt.MC_block == 'SE':
                exec('''net.backbone.block_%s = MC_se_block(in_c)''' % (dv))
            elif opt.MC_block == 'NL':
                exec('''net.backbone.block_%s = NLBlockND(in_c, mode=opt.NL_mode)''' % (dv))

        net.backbone.forward = types.MethodType(MP_forward, net.backbone)

        for name, param in net.backbone.named_parameters():
            if "concat" in name and "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    elif opt.model == 'MC_cls':
        net = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=opt.pre_train,
                                                                 pretrained_backbone=True,
                                                                 min_size=opt.in_size,
                                                                 max_size=opt.in_size)
        net.mode = 'det'
        net.backbone.body_1 = deepcopy(net.backbone.body)
        net.backbone.body_2 = deepcopy(net.backbone.body)

        net.backbone.concat_0 = nn.Conv2d(256 * 3, 256, 1)
        net.backbone.concat_1 = nn.Conv2d(256 * 2 * 3, 256 * 2, 1)
        net.backbone.concat_2 = nn.Conv2d(256 * 4 * 3, 256 * 4, 1)
        net.backbone.concat_3 = nn.Conv2d(256 * 8 * 3, 256 * 8, 1)
        net.backbone.forward = types.MethodType(MC_cls_backbone_pre, net.backbone)

        for name, param in net.backbone.named_parameters():
            if "concat" in name and "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

        net.roi_heads.mask_roi_pool = None

        anchor_size = [8, 16, 32, 64, 128]
        anchor_ratio = opt.in_size / 448
        anchor_size = [int(each * anchor_ratio) for each in anchor_size]

        anchor_generator = AnchorGenerator(
            sizes=tuple([anchor_size for _ in range(5)]),
            aspect_ratios=tuple([(0.25, 0.5, 1.0, 2.0, 4.0) for _ in range(5)]))

        net.rpn.anchor_generator = anchor_generator
        net.rpn.head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])

        in_features = net.roi_heads.box_predictor.cls_score.in_features
        net.roi_heads.box_predictor = FastRCNNPredictor(in_features, opt.label_length)

        for dv in opt.dv_num:
            if dv == 0:
                net.cls_branch_0 = nn.Sequential(
                    nn.Conv2d(256, 256, 1),
                    nn.AdaptiveAvgPool2d((1,1))
                )
                net.cls_fc_0 = nn.Linear(256, 2)
            elif dv == 1:
                net.cls_branch_1 = nn.Sequential(
                    nn.Conv2d(256 * 2, 256 * 2, 1),
                    nn.AdaptiveAvgPool2d((1, 1)),
                )
                net.cls_fc_1 = nn.Linear(256 * 2, 2)
            elif dv == 2:
                net.cls_branch_2 = nn.Sequential(
                    nn.Conv2d(256 * 4, 256 * 4, 1),
                    nn.AdaptiveAvgPool2d((1, 1)),
                )
                net.cls_fc_2 = nn.Linear(256 * 4, 2)
            elif dv == 3:
                net.cls_branch_3 = nn.Sequential(
                    nn.Conv2d(256 * 8, 256 * 4, 1),
                    nn.AdaptiveAvgPool2d((1, 1)),
                )
                net.cls_fc_3 = nn.Linear(256 * 4, 2)

        for name, param in net.backbone.named_parameters():
            if "cls_branch" in name and "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

        net.forward = types.MethodType(MC_cls_forward, net)

        return net


    anchor_size = [8, 16, 32, 64, 128]
    anchor_ratio = opt.in_size / 448
    anchor_size = [int(each * anchor_ratio) for each in anchor_size]

    anchor_generator = AnchorGenerator(
        sizes=tuple([anchor_size for _ in range(5)]),
        aspect_ratios=tuple([(0.25, 0.5, 1.0, 2.0, 4.0) for _ in range(5)]))

    net.rpn.anchor_generator = anchor_generator
    net.rpn.head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])

    net.roi_heads.mask_roi_pool = None

    in_features = net.roi_heads.box_predictor.cls_score.in_features
    net.roi_heads.box_predictor = FastRCNNPredictor(in_features, opt.label_length)


    return net

def get_result():
    return
