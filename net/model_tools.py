from config import opt
import torchvision.models as tmodels
from . import model_resnet, model_resnet_new, model_timm_net

def get_model():
    # Res Net 系列
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
    elif opt.model[0] == 'c':
        if opt.model == 'c18':
            net = tmodels.resnet18(num_classes=1, pretrained=opt.pre_train)
        elif opt.model == 'c34':
            net = tmodels.resnet34(num_classes=1, pretrained=opt.pre_train)
        elif opt.model == 'c50':
            net = tmodels.resnet50(num_classes=1, pretrained=opt.pre_train)
        elif opt.model == 'c101':
            net = tmodels.resnet101(num_classes=1, pretrained=opt.pre_train)
        elif opt.model == 'c152':
            net = tmodels.resnet152(num_classes=1, pretrained=opt.pre_train)
    return net

def get_result():
    return