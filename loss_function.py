import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
# import torchsnooper

# @torchsnooper.snoop()
def compute_loss_no_ts(cfg, cls_logit, loc_logit, gt_cls, gt_loc, iou_score, negative_ratio=3., alpha=1.):
    """
    计算loss
    :param cfg: 参数类的对象
    :param cls_logit: 分类预测分数
    :param loc_logit: 定位预测分数
    :param gt_cls: gt类别
    :param gt_loc: gt位置坐标
    :param iou_score: gt与anchor的IoU
    :param negative_ratio: 负样本与正样本的比例为 negative_ratio:1
    :param alpha: 定位损失权重
    :return:
    """

    #flatten 所有的向量
    loc_flat = []
    gt_loc_flat = []
    cls_flat = []
    gt_cls_flat = []
    iou_score_flat = []
    for i in range(len(cls_logit)): #5个时间维度
        loc_flat.append(torch.reshape(loc_logit[i],(-1,2)))
        cls_flat.append(torch.reshape(cls_logit[i],(-1,cfg.classes_num)))
        gt_loc_flat.append(torch.reshape(gt_loc[i],(-1,2)))
        gt_cls_flat.append(torch.reshape(gt_cls[i],(-1,)))
        iou_score_flat.append(torch.reshape(iou_score[i],(-1,)))



    #拼接batch维度，转为一个tensor
    loc = torch.cat(loc_flat,0)
    gt_loc = torch.cat(gt_loc_flat,0)
    cls = torch.cat(cls_flat,0)
    gt_cls = torch.cat(gt_cls_flat,0)
    iou_score = torch.cat(iou_score_flat,0)


    #计算所有anchor的iou值，计算正负例
    pos_mask = iou_score >= cfg.pos_thr #正例掩码
    pos_mask_float = pos_mask.type_as(cls)
    pos_num = torch.sum(pos_mask_float) #正例个数

    #负样本挖掘
    no_cls = pos_mask.int()
    predictions = nn.Softmax(dim=1).forward(cls)
    neg_mask = np.logical_and(np.logical_not(pos_mask),iou_score > - 0.5) > 0#逻辑去反得到neg，并与iou>-0.5取交
    neg_mask_float = neg_mask.float()
    #从背景概率中从小到大取3倍的负例个数（ neg_values 有反向传播）
    neg_values = torch.where(neg_mask, predictions[:, 0], 1. - neg_mask_float) #True时为背景概率，False是为1.0
    neg_values_flat = torch.reshape(neg_values,(-1,))
    all_neg_sum = torch.sum(neg_mask_float).int()
    #负例数
    neg_num = (negative_ratio*pos_num).int()+cfg.batch
    neg_num = torch.min(neg_num,all_neg_sum)
    #取负概率最大的top-k个负例
    value,idx = torch.topk(-neg_values_flat,neg_num)
    max_neg_value = -value[-1]

    #最后的负例mask
    neg_mask =np.logical_and(neg_mask,neg_values < max_neg_value) >0
    neg_mask_float = neg_mask.float()
    neg_num = torch.sum(neg_mask_float)

    #计算loss
    #将gt_cls,no_cls 转成one_hot
    gt_one_hot = []
    no_one_hot = []
    for i in gt_cls:
        cnt = [0 for i in range(21)]
        cnt[i] = 1
        gt_one_hot.append(cnt)
    gt_one_hot = torch.Tensor(gt_one_hot)
    for i in no_cls:
        cnt = [0 for i in range(21)]
        cnt[i] = 1
        no_one_hot.append(cnt)
    no_one_hot = torch.Tensor(no_one_hot)
    #然后计算交叉熵损失 loss [1224,1] pos_loss和neg_loss都需要有梯度
    loss = -torch.sum(Variable(gt_one_hot) * F.log_softmax(cls, dim=1), dim=1)
    pos_loss = torch.div(torch.sum(loss * pos_mask_float), pos_num)
    loss = -torch.sum(Variable(no_one_hot) * F.log_softmax(cls, dim=1), dim=1)
    neg_loss = torch.div(torch.sum(loss * neg_mask_float), neg_num)

    #计算定位损失 loc_loss也需要有梯度
    loss = smooth_L1(loc-gt_loc)
    weight = (alpha * pos_mask_float).unsqueeze(1)
    loc_loss = torch.div(torch.sum(loss * weight),pos_num)

    loss = torch.add(loc_loss,torch.add(pos_loss,neg_loss))

    return loss,pos_loss,neg_loss,loc_loss,pos_num,neg_num


def compute_detector_loss(cfg, cls_logit, loc_logit, gt_cls, gt_loc, iou_score, device,negative_ratio=3., alpha=1.):
    """
    计算 detector loss
    :param cfg: 参数类的对象
    :param cls_logit: 分类预测分数
    :param loc_logit: 定位预测分数
    :param gt_cls: gt类别
    :param gt_loc: gt位置坐标
    :param iou_score: gt与anchor的IoU
    :param negative_ratio: 负样本与正样本的比例为 negative_ratio:1
    :param alpha: 定位损失权重
    :return:
    """

    #flatten 所有的向量
    loc_flat = []
    gt_loc_flat = []
    cls_flat = []
    gt_cls_flat = []
    iou_score_flat = []
    for i in range(len(cls_logit)): #5个时间维度
        loc_flat.append(torch.reshape(loc_logit[i],(-1,2)))
        cls_flat.append(torch.reshape(cls_logit[i],(-1,cfg.classes_num)))
        gt_loc_flat.append(torch.reshape(gt_loc[i],(-1,2)))
        gt_cls_flat.append(torch.reshape(gt_cls[i],(-1,)))
        iou_score_flat.append(torch.reshape(iou_score[i],(-1,)))



    #拼接batch维度，转为一个tensor
    loc = torch.cat(loc_flat,0).to(device)
    gt_loc = torch.cat(gt_loc_flat,0).to(device)
    cls = torch.cat(cls_flat,0).to(device)
    gt_cls = torch.cat(gt_cls_flat,0).to(device)
    iou_score = torch.cat(iou_score_flat,0).to(device)


    #计算所有anchor的iou值，计算正负例
    pos_mask = iou_score >= cfg.pos_thr #正例掩码
    pos_mask_float = pos_mask.float()
    pos_num = torch.sum(pos_mask_float) #正例个数

    #负样本挖掘
    no_cls = pos_mask.int()
    predictions = nn.Softmax(dim=1).forward(cls)
    neg_mask = (np.logical_and(np.logical_not(pos_mask),iou_score > - 0.5) > 0).to(device)#逻辑去反得到neg，并与iou>-0.5取交
    neg_mask_float = neg_mask.float()
    #从背景概率中从小到大取3倍的负例个数（ neg_values 有反向传播）
    neg_values = torch.where(neg_mask, predictions[:, 0], 1. - neg_mask_float) #True时为背景概率，False是为1.0
    neg_values_flat = torch.reshape(neg_values,(-1,))
    all_neg_sum = torch.sum(neg_mask_float).int()
    #负例数
    batch_size_list = [int(x) for x in cfg.batch.split(',')]
    neg_num = (negative_ratio*pos_num).int()+batch_size_list[0]
    neg_num = torch.min(neg_num,all_neg_sum)
    #取负概率最大的top-k个负例
    value,idx = torch.topk(-neg_values_flat,neg_num)
    max_neg_value = -value[-1]

    #最后的负例mask
    neg_mask =(np.logical_and(neg_mask,neg_values < max_neg_value) >0).to(device)
    neg_mask_float = neg_mask.float()
    neg_num = torch.sum(neg_mask_float)

    #计算loss
    #将gt_cls,no_cls 转成one_hot
    gt_one_hot = []
    no_one_hot = []
    for i in gt_cls:
        cnt = [0 for i in range(21)]
        cnt[i] = 1
        gt_one_hot.append(cnt)
    gt_one_hot = torch.Tensor(gt_one_hot).to(device)
    for i in no_cls:
        cnt = [0 for i in range(21)]
        cnt[i] = 1
        no_one_hot.append(cnt)
    no_one_hot = torch.Tensor(no_one_hot).to(device)
    #然后计算交叉熵损失 loss [1224,1] pos_loss和neg_loss都需要有梯度
    loss = -torch.sum(Variable(gt_one_hot) * F.log_softmax(cls, dim=1), dim=1)
    pos_loss = torch.div(torch.sum(loss * pos_mask_float), pos_num)
    loss = -torch.sum(Variable(no_one_hot) * F.log_softmax(cls, dim=1), dim=1)
    neg_loss = torch.div(torch.sum(loss * neg_mask_float), neg_num)

    #计算定位损失 loc_loss也需要有梯度
    loss = smooth_L1(loc-gt_loc,device)
    weight = (alpha * pos_mask_float).unsqueeze(1)
    loc_loss = torch.div(torch.sum(loss * weight),pos_num)

    loss = torch.add(loc_loss,torch.add(pos_loss,neg_loss))

    return loss,pos_loss,neg_loss,loc_loss,pos_num,neg_num


def compute_consistency_loss(cfg,predictions,ema_predictions,cls_logits,ema_clsg,loc_logits,ema_locg):
    loc_consistency_loss = compute_loc_consistency_loss(loc_logits,ema_locg)



def compute_loc_consistency_loss(detector_loc,ema_loc):
    #定位一致性损失  - l1,l1smooth,平方差损失
    #分类一致性损失 - kl，mse损失





def smooth_L1(x,device):
    """
    x^2 / 2 if abs(x) < 1
    abs(x) - 0.5 if abs(x) > 1
    """
    abs_x = torch.abs(x)
    min_x = Variable(torch.Tensor(np.minimum(abs_x.cpu().detach().numpy(), 1)).to(device))
    res = 0.5 * ((abs_x - 1) * min_x + abs_x)
    return res

def nn_distance(pc1,pc2)




if __name__ == '__main__':
    from operation.config import *

    cfg = Config(batch=4, is_training=True, clip_overlap=0.75)

    from dataset import *
    train_loader = torch.utils.data.DataLoader(THUMOSDetectionDataset(cfg, subset="train"),
                                               batch_size=1, shuffle=False, num_workers=2)

    for a, b in enumerate(train_loader):
        inputs = b[0]
        cls_label = b[1]
        loc_label = b[2]
        iou_label = b[3]
        clip_info = b[4]
        break

    from model.model import *

    model = Model(cfg)
    pre,cls_logit,loc_logit,end_point = model(inputs)
    # print(inputs.shape)
    # print(len(cls_label))
    # print(cls_label[0])
    # print(len(loc_label))
    # print(loc_label[0].shape)
    # print(len(cls_logit))
    # print(cls_logit[0].shape)

    print(clip_info)

    neg = compute_loss_no_ts(cfg,cls_logit,loc_logit,cls_label,loc_label,iou_label)
    # print(neg[0])
    # print(neg[1])

    # print(type(cls_logit))
    # print(type(loc_logit))
    # print(type(cls_label))












