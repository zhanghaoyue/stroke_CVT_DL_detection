from config import opt
import torchvision
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from . import model_resnet, model_resnet_new, model_timm_net


def get_model():
    # Res Net 系列
    global net
    if opt.model[:2] == 'nr':
        net = model_resnet_new.ResNet(num_classes=opt.label_length)

    # Timm 系列
    elif opt.model[0] == 't':
        net = model_timm_net.Timm(num_classes=opt.label_length)

    elif opt.model[0] == 'r':
        # Create the model
        if opt.model == 'r18':
            net = model_resnet.resnet18(num_classes=opt.label_length, pretrained=opt.pre_train)
        elif opt.model == 'r34':
            net = model_resnet.resnet34(num_classes=opt.label_length, pretrained=opt.pre_train)
        elif opt.model == 'r50':
            net = model_resnet.resnet50(num_classes=opt.label_length, pretrained=opt.pre_train)
        elif opt.model == 'r101':
            net = model_resnet.resnet101(num_classes=opt.label_length, pretrained=opt.pre_train)
        elif opt.model == 'r152':
            net = model_resnet.resnet152(num_classes=opt.label_length, pretrained=opt.pre_train)

    elif opt.model[-4:] == 'rcnn':
        if opt.model == 'faster_rcnn':
            net = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=opt.pre_train, pretrained_backbone=True)
            anchor_generator = AnchorGenerator(
                sizes=tuple([(16,32,64,128,256,512) for _ in range(5)]),
                aspect_ratios=tuple([(0.25,0.5,1.0,2.0) for _ in range(5)]))
            net.rpn.anchor_generator = anchor_generator
            net.rpn.head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])
            in_features = net.roi_heads.box_predictor.cls_score.in_features
            net.roi_heads.box_predictor = FastRCNNPredictor(in_features, opt.label_length)
        elif opt.model == 'mask_rcnn':
            net = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=opt.pre_train, pretrained_backbone=True)
            net.roi_heads.mask_roi_pool = None
            anchor_generator = AnchorGenerator(
                sizes=tuple([(16,32,64,128,256,512) for _ in range(5)]),
                aspect_ratios=tuple([(0.25,0.5,1.0,2.0) for _ in range(5)]))
            net.rpn.anchor_generator = anchor_generator
            net.rpn.head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])
            in_features = net.roi_heads.box_predictor.cls_score.in_features
            net.roi_heads.box_predictor = FastRCNNPredictor(in_features, opt.label_length)
    return net


def get_result():
    return
