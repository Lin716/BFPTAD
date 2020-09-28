import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as torch_init
import torch.nn.functional as F
# from torchsummary import summary

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)

class Model(nn.Module):
    def __init__(self,cfg):
        super(Model, self).__init__()
        self.feat_dim = cfg.feature_size
        self.channel = 1024
        self.FPN_channel = 256
        self.anchor_info = cfg.anchor_info
        self.classnum = cfg.classes_num

        #卷积 特征

        #t:32->32
        self.conv1 = nn.Sequential(
            nn.Conv1d(self.feat_dim,self.channel,kernel_size=1,stride=1,padding=0),
            nn.ReLU(inplace=True)
        )
        #t:32->16
        self.conv2 = nn.Sequential(
            nn.Conv1d(self.channel, self.channel, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        #t:16->8
        self.conv3 = nn.Sequential(
            nn.Conv1d(self.channel, self.channel, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        #t:8->4
        self.conv4 = nn.Sequential(
            nn.Conv1d(self.channel, self.channel, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        #t:4->2
        self.conv5 = nn.Sequential(
            nn.Conv1d(self.channel, self.channel, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        #P5
        self.P5 = nn.Sequential(
            nn.Conv1d(self.channel,self.FPN_channel,kernel_size=1,stride=1,padding=0),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.FPN_channel, self.FPN_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.P4 = fpn(self.channel, self.FPN_channel, 4)
        self.P3 = fpn(self.channel, self.FPN_channel, 8)
        self.P2 = fpn(self.channel, self.FPN_channel, 16)
        self.P1 = fpn(self.channel, self.FPN_channel, 32)

        self.N2 = reverse_fpn(self.FPN_channel)
        self.N3 = reverse_fpn(self.FPN_channel)
        self.N4 = reverse_fpn(self.FPN_channel)
        self.N5 = reverse_fpn(self.FPN_channel)

        #类别预测 (batch,tmp_num,t)
        self.tmp_num1 = 5 * self.classnum
        self.cls1 = nn.Conv1d(self.FPN_channel,self.tmp_num1,kernel_size=3,stride=1,padding=1)
        self.tmp_num2 = 3* self.classnum
        self.cls2 = nn.Conv1d(self.FPN_channel,self.tmp_num2,kernel_size=3,stride=1,padding=1)

        self.tmp_loc1 = 5*2
        self.loc1 = nn.Conv1d(self.FPN_channel,self.tmp_loc1,kernel_size=3,stride=1,padding=1)
        self.tmp_loc2 = 3*2
        self.loc2 = nn.Conv1d(self.FPN_channel, self.tmp_loc2, kernel_size=3, stride=1, padding=1)
        self.predict_fn = nn.Softmax(dim=-1)

        #权重初始化
        self.apply(weights_init)



    def forward(self, x):

        end_points = {}
        #卷积
        conv1 = self.conv1(x)
        end_points['conv1'] = conv1
        conv2 = self.conv2(conv1)
        end_points['conv2'] = conv2
        conv3 = self.conv3(conv2)
        end_points['conv3'] = conv3
        conv4 = self.conv4(conv3)
        end_points['conv4'] = conv4
        conv5 = self.conv5(conv4)
        end_points['conv5'] = conv5

        #金字塔
        p5 = self.P5(conv5)
        end_points['P5'] = p5
        p4 = self.P4(conv4,p5)
        end_points['P4'] = p4
        p3 = self.P3(conv3, p4)
        end_points['P3'] = p3
        p2 = self.P2(conv2, p3)
        end_points['P2'] = p2
        p1 = self.P1(conv1, p2)
        end_points['P1'] = p1

        #反向金字塔
        n1 = p1
        end_points['N1'] = n1
        n2 = self.N2(p2,n1)
        end_points['N2'] = n2
        n3 = self.N3(p3,n2)
        end_points['N3'] = n3
        n4 = self.N4(p4,n3)
        end_points['N4'] = n4
        n5 = self.N5(p5,n4)
        end_points['N5'] = n5

        #预测+定位
        predictions = []
        cls_logits = []
        loc_logits = []

        #每一层都进行预测 N1,N2,N3,N4,N5
        for i,layer in enumerate(self.anchor_info["feat_layers"]):

            inputs = end_points[layer]
            if self.anchor_info["normalizations"][i]:
                #对特征通道进行正则化
                net = F.normalize(inputs,dim=1)
            batch,_,t = list(inputs.shape)
            anchor_num = len(self.anchor_info["anchor_scales"][i])
            if i == 4:
                cls_pred = self.cls2(net)
                loc_pred = self.loc2(net)
            else:
                cls_pred = self.cls1(net)
                loc_pred = self.loc1(net)

            #分类预测
            cls_pred = torch.reshape(cls_pred,(batch,anchor_num,self.classnum,t))
            cls_pred = torch.transpose(cls_pred, 2, 3)
            cls_pred = torch.transpose(cls_pred, 1, 2)
            #定位
            loc_pred = torch.reshape(loc_pred,(batch,anchor_num,2,t))
            loc_pred = torch.transpose(loc_pred, 2, 3)
            loc_pred = torch.transpose(loc_pred, 1, 2)

            #预测类别的结果
            predictions.append(self.predict_fn(cls_pred))
            #先验框预测的类别概率,预测的位置信息
            cls_logits.append(cls_pred)
            loc_logits.append(loc_pred)

        return predictions,cls_logits,loc_logits,end_points


class fpn(nn.Module):
    def __init__(self,in_channels,out_channels,t):
        super(fpn, self).__init__()
        #横向连接
        self.latter_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )


        #相加后接一个3x1conv，消除上采样的混叠效应(aliasing effect)
        self.fconv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.apply(weights_init)
        # 反卷积
        self.deconv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=[1, t], stride=[1, 2],
                                         padding=[0, (t - 1) // 2])

    def forward(self,C,P):
        # point = {}
        C = self.latter_conv(C)
        # point['latter_conv'] = C
        P = torch.unsqueeze(P,2)
        # point['unsq'] = P
        P = self.deconv(P)
        # point['deconv'] = P
        P = torch.sum(P,dim=2)
        # point['sum'] = P
        add = self.fconv(C+P)
        # point['conv'] = add

        return add

class reverse_fpn(nn.Module):
    def __init__(self,channel):
        super(reverse_fpn, self).__init__()
        #下采样
        self.rfconv = nn.Sequential(
            nn.Conv1d(channel, channel, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.apply(weights_init)


    def forward(self,P,N):
        conv = self.rfconv(N)
        add = torch.add(P,conv)
        add = nn.ReLU(True)(add)
        return add


if __name__ == '__main__':
    from operation.config import *

    cfg = Config(batch=4, is_training=True, clip_overlap=0.75)

    model=Model(cfg)
    input=torch.Tensor(np.random.random((4,2048,32)))



    predict,cls,loc,end_point =model(input)
    print(len(predict))
    print(predict[0].shape)
    print(len(cls))
    print(cls[0].shape)
    print(len(loc))
    print(loc[0].shape)
    # print(end_point['conv1'].shape)
    # print(end_point['conv2'].shape)
    # print(end_point['conv3'].shape)
    # print(end_point['conv4'].shape)
    # print(end_point['conv5'].shape)
    # print(end_point['P5'].shape)
    # print(end_point['P4'].shape)
    # print(end_point['P3'].shape)
    # print(end_point['P2'].shape)
    # print(end_point['P1'].shape)
    # print(end_point['N2'].shape)
    # print(end_point['N3'].shape)
    # print(end_point['N4'].shape)
    # print(end_point['N5'].shape)

    # print(cls[4].shape)
    # print(loc[4].shape)
    # print(type(cls))
    # print(len(cls))



    # FPN = fpn(1024,256,8)
    # C = torch.randn(4,1024,8)
    #
    # P = torch.randn(4,256,4)
    #
    # add = FPN(C, P)
    #
    # # print(point['latter_conv'].shape)
    # # print(point['unsq'].shape)
    # # print(point['deconv'].shape)
    # # print(point['sum'].shape)
    # # print(point['conv'].shape)
    # print(type(add))

    # rpn = reverse_fpn(256)
    # N = torch.randn(4,256,16)
    # P = torch.randn(4,256,8)
    # add = rpn(P,N)
    # print(add.shape)







