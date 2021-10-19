import numpy as np
import torch
import torch.nn as nn

def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU

class FocalLoss(nn.Module):
    def __init__(self, alpha, lth, hth):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.lth = lth
        self.hth = hth

    def forward(self, classifications, regressions, anchors, annotations):
        alpha = self.alpha
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]

        # anchor 的 height 和 width
        anchor_widths  = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        # anchor 的中心点
        anchor_ctr_x   = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y   = anchor[:, 1] + 0.5 * anchor_heights

        # batch 中的每个 case 单独计算损失
        for j in range(batch_size):
            # 当前 case 的 classification 和 regression 输出
            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            # ground turth bbox
            # 每个 GT bbox list 中 GT bbox 的数量由 batch 中具有最多 GT bbox 的 case 确定
            # 不够这个数量的用 [-1, -1, -1, -1, -1] 填补
            bbox_annotation = annotations[j, :, :]
            # 去掉填补项
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            ########################## classification Loss ##########################
            # 对预测值进行截断
            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            # 整个 case 中没有 positive
            # classification_losses 正常使用 Focal Loss 进行计算
            # 但可以确定的是 cls 对应的 label 一定是 0
            # loss = - (1-a) y'^gamma * log(1 - y')
            #      = (1-a) * y'^gamma * (-1*log(1-y'))
            # regression_losses 全为 0
            if bbox_annotation.shape[0] == 0:
                # Focal Loss
                if torch.cuda.is_available():
                    alpha_factor = torch.ones(classification.shape).cuda() * alpha

                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(torch.tensor(0).float().cuda())

                else:
                    alpha_factor = torch.ones(classification.shape) * alpha

                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(torch.tensor(0).float())

                continue

            # num_anchors x num_annotations
            # 计算每个 anchor 与每个 GT bbox 的 IoU
            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4])

            # 对每个 anchor 确定一个与所有 GT bbox 的最大 IoU
            IoU_max, IoU_argmax = torch.max(IoU, dim=1) # num_anchors x 1

            # 设定计算 cls loss 需要的 y_true
            # 初始化所有位置的 y_true 为 -1
            targets = torch.ones(classification.shape) * -1

            if torch.cuda.is_available():
                targets = targets.cuda()

            # 对于每个 anchor，如果其与任意 bbox 的 iou < lth，则视为负样本
            # 将其对应的 y_true 设定为 0
            targets[torch.lt(IoU_max, self.lth), :] = 0

            # 对于每个 anchor，如果其与任意 bbox 的 iou > hth，则视为正样本
            # 先找到 positive anchor 对应的 index
            positive_indices = torch.ge(IoU_max, self.hth)

            # 正样本数量
            # 统计 positive anchor 数量
            num_positive_anchors = positive_indices.sum()

            # 对于多类检测，将 target 对应类的 label 设为 1，其它类的 label 设为 0
            # IoU_argmax 对应于 positive 的类别，如果是 binary detection，那 positive anchor 对应的就是 1
            assigned_annotations = bbox_annotation[IoU_argmax, :]
            # 先设置 positive anchor 的 y_true 都为 0
            targets[positive_indices, :] = 0
            # 然后将 positive anchor 对应 class 的 y_true 设为 1
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            # 还有一部分 anchor 的 IoU 在 0.4~0.5 之间
            # 这些 anchor 的 y_true 为 -1, 在之后的 loss 计算中被忽略

            if torch.cuda.is_available():
                alpha_factor = torch.ones(targets.shape).cuda() * alpha
            else:
                alpha_factor = torch.ones(targets.shape) * alpha

            # 用 torch.where 为 pos 和 neg case 分配 loss weight
            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            # 计算 Focal loss
            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))
            cls_loss = focal_weight * bce

            # 将 y_true == -1 的 anchor 的 loss 设置为 0
            if torch.cuda.is_available():
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
            else:
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape))

            # 计算 mean loss = (pos loss + neg loss) / pos num
            classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))
            #####################################################################


            ########################## regression Loss ##########################
            # 仅在存在 positive anchor 的情况下计算
            if positive_indices.sum() > 0:
                # 对应于 positive anchor 的 GT bbox
                assigned_annotations = assigned_annotations[positive_indices, :]

                # positive anchors 的 height 和 width
                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                # positive anchors 的中心点
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                # GT bbox 的 height 和 width
                gt_widths  = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                # GT bbox 的中心点
                gt_ctr_x   = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y   = assigned_annotations[:, 1] + 0.5 * gt_heights

                # GT bbox 的 height 和 width 不能小于 1
                gt_widths  = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                # regression 的 target value
                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

                if torch.cuda.is_available():
                    targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()
                else:
                    targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]])

                regression_diff = torch.abs(targets - regression[positive_indices, :])

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
            else:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).float().cuda())
                else:
                    regression_losses.append(torch.tensor(0).float())
            #####################################################################

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True)


