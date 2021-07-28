from config import opt
import torch
import math
import torchvision
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from . import model_resnet, model_resnet_new, model_timm_net


def get_model():
    global net
    if opt.model == 'retinanet':
        net = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=opt.pre_train, 
            pretrained_backbone=True)
        anchor_generator = AnchorGenerator(
                sizes=((8,16,32,64,128),),
                aspect_ratios=((0.25,0.5,1.0,2.0,4.0),))

        net.head = RetinaNetHead(256, 9, opt.label_length)
    elif opt.model == 'ssd':
        net = torchvision.models.detection.ssd300_vgg16(pretrained=opt.pre_train, pretrained_backbone=True)
        in_features = 512
        num_anchors = 4
        net.head.classification_head.num_classes = opt.label_length

        cls_logits = torch.nn.Conv2d(in_features, num_anchors*opt.label_length, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(cls_logits.weight, std=0.01)
        torch.nn.init.constant_(cls_logits.bias, -math.log((1-0.01)/0.01))
        net.head.classification_head.cls_logits = cls_logits
    elif opt.model[-4:] == 'rcnn':
        if opt.model == 'faster_rcnn':
            net = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=opt.pre_train, pretrained_backbone=True)
            anchor_generator = AnchorGenerator(
                sizes=tuple([(8,16,32,64,128) for _ in range(5)]),
                aspect_ratios=tuple([(0.25,0.5,1.0,2.0,4.0) for _ in range(5)]))
            net.rpn.anchor_generator = anchor_generator
            net.rpn.head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])
            in_features = net.roi_heads.box_predictor.cls_score.in_features
            net.roi_heads.box_predictor = FastRCNNPredictor(in_features, opt.label_length)
        elif opt.model == 'mask_rcnn':
            net = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=opt.pre_train, pretrained_backbone=True)
            net.roi_heads.mask_roi_pool = None
            anchor_generator = AnchorGenerator(
                sizes=tuple([(8,16,32,64,128) for _ in range(5)]),
                aspect_ratios=tuple([(0.25,0.5,1.0,2.0,4.0) for _ in range(5)]))
            net.rpn.anchor_generator = anchor_generator
            net.rpn.head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])
            in_features = net.roi_heads.box_predictor.cls_score.in_features
            net.roi_heads.box_predictor = FastRCNNPredictor(in_features, opt.label_length)
    return net


def get_result():
    return
