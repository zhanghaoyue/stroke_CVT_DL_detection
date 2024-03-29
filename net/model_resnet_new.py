import torch.nn as nn
import torch
import math
from torchvision.models import *
import torch.utils.model_zoo as model_zoo
from torchvision.ops import nms
from retinanet.utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
from retinanet.anchors import Anchors
from retinanet import losses
from config import opt
from torchvision.ops import box_iou

import sys
sys.path.append("..")
from .block_utils import *
from config import opt

r'''
'alexnet', 'densenet', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 
'detection', 'googlenet', 'inception', 'inception_v3', 
'mnasnet', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3', 
'mobilenet', 'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small', 'mobilenetv2', 'mobilenetv3', 
'quantization', 
'resnet', 'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50', 
'resnext101_32x8d', 'resnext50_32x4d', 
'segmentation', 
'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'shufflenetv2', 
'squeezenet', 'squeezenet1_0', 'squeezenet1_1', 
'utils', 
'vgg', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 
'video', 
'wide_resnet101_2', 'wide_resnet50_2']
'''

def encoder(name, pre_train):
    model_dict = {
        'nrx50': resnext50_32x4d,
        'nrx101': resnext101_32x8d,
        'nr18': resnet18,
        'nr34': resnet34,
        'nr50': resnet50,
        'nr101': resnet101,
        'nrw50': wide_resnet50_2,
        'nrw101': wide_resnet101_2,
    }
    if opt.pre_train == False:
        zero_init_residual = True
    else:
        zero_init_residual = False

    model = model_dict[name](pretrained=pre_train)
    if opt.pre_train:
        print('use pretrained model')
    return model

class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        FE = encoder(opt.model, opt.pre_train)

        if opt.dim == 3:
            self.conv1 = FE.conv1
        else:
            self.conv1 = nn.Conv2d(opt.dim, 64, (7, 7), (2, 2), (3, 3), bias=False)
            conv_weights = FE.conv1.state_dict()['weight'][:, 1, :, :].unsqueeze(dim=1).repeat((1, opt.dim, 1, 1))
            self.conv1.state_dict()['weight'] = conv_weights

        self.bn1 = FE.bn1
        self.relu = FE.relu
        self.maxpool = FE.maxpool
        self.layer1 = FE.layer1
        self.layer2 = FE.layer2
        self.layer3 = FE.layer3
        self.layer4 = FE.layer4

        layers = [len(self.layer1), len(self.layer2), len(self.layer3), len(self.layer4)]

        del FE

        if opt.model in ['nr18','nr34']:
            fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels,
                         self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        else:
            fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels,
                         self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])

        self.anchors = Anchors()
        num_anchors = self.anchors.ratios.shape[0] * self.anchors.scales.shape[0]

        self.regressionModel = RegressionModel(256, num_anchors=num_anchors)
        self.classificationModel = ClassificationModel(256, num_anchors=num_anchors, num_classes=num_classes)

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()

        self.focalLoss = losses.FocalLoss(alpha=opt.alpha, lth=opt.lth, hth=opt.hth)

        prior = 0.01
        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

        if opt.fix_BN:
            self.freeze_bn()

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs):
        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs

        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        features = self.fpn([x2, x3, x4])

        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)

        classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)

        anchors = self.anchors(img_batch)

        if self.training:
            return self.focalLoss(classification, regression, anchors, annotations)
        else:
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            finalScores = torch.Tensor([])
            finalAnchorBoxesIndexes = torch.Tensor([]).long()
            finalAnchorBoxesCoordinates = torch.Tensor([])

            if torch.cuda.is_available():
                finalScores = finalScores.cuda()
                finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.cuda()
                finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.cuda()

            for i in range(classification.shape[2]):
                scores = torch.squeeze(classification[:, :, i])
                scores_over_thresh = (scores > opt.cls_th)
                if scores_over_thresh.sum() == 0:
                    # no boxes to NMS, just continue
                    continue

                scores = scores[scores_over_thresh]
                anchorBoxes = torch.squeeze(transformed_anchors)
                anchorBoxes = anchorBoxes[scores_over_thresh]
                anchors_nms_idx = nms(anchorBoxes, scores, opt.nms_th)

                finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
                finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0])
                if torch.cuda.is_available():
                    finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()

                finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
                finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))

            return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates]