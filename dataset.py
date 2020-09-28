# -*- coding:utf-8 -*-
import logging
import h5py
import numpy as np
import os
import torch.utils.data as data
import torch
from operation.config import *

class THUMOSDetectionDataset(data.Dataset):
    def __init__(self, cfg, subset="train"):
        self.is_training = cfg.is_training
        self.clip_len = cfg.clip_len
        self.anchor_info = cfg.anchor_info
        self.clip_overlap = cfg.clip_overlap
        #特征/视频文件：名字+帧数/clip的信息(name-起始帧)/clip的label信息（5层layer）
        if self.is_training:
            if subset == "train":
                self.feat_file = cfg.feat_train_file
                self.video_info = cfg.video_train_info_file
                self.clip_file = cfg.clip_gt_file
                self.label_file = cfg.label_file
            elif subset == "test":
                self.feat_file = cfg.feat_test_file
                self.video_info = cfg.video_test_info_file
                self.clip_file = cfg.clip_test_file
                self.label_file = cfg.test_label_file
            # 训练或测试下video的信息
            self.video = get_dict_from_pkl(self.video_info)
            # 训练或测试下clip的信息 videoname-起始帧
            self.clip = get_dict_from_pkl(self.clip_file)
            # 训练或测试下clip的label信息
            self.label = get_dict_from_pkl(self.label_file)
            # 训练或测试下的clip列表长度
            self.clip_list = list(self.label)
        elif not self.is_training and subset == "test":
            self.feat_file = cfg.feat_test_file
            self.video_info = cfg.video_test_info_file
            self.test_video_info = get_dict_from_pkl(self.video_info)
            self.all_test_clip = self._get_test_clip()
            self.clip_list = list(self.all_test_clip)


    #训练和测试的长度
    def __len__(self):
        return len(self.clip_list)


    def __getitem__(self, idx):
        clip_info = self.clip_list[idx]
        input_data = self._read_feat_data(clip_info) #32*2048维
        input_data = torch.Tensor(input_data.transpose())

        #特征尺度：[4(batch),2048,32]
        #标签:[5(时间尺度的个数),4(batch),时间尺度,5(1,2,1)]
        if self.is_training:

            cls_label,loc_label,iou_label = self._get_train_label(clip_info)


            return input_data,cls_label,loc_label,iou_label,clip_info #clip_info 为4 类型tuple
        else:


            return input_data,clip_info


    def _read_feat_data(self, clip_info):
        """
        获取视频特征中的一个clip的特征
        :param clip_info: clip信息
        :return:
        """
        if not os.path.exists(self.feat_file):
            logging.info("%s: file isn't exists!" % self.feat_file)
            return
        if self.feat_file.split('.')[-1] != 'h5':
            logging.info("%s: file type is error!" % self.feat_file)
            return

        video_name = clip_info.split('-')[0]
        # 根据开始帧，算出在特征中的起止snippet
        start = int(int(clip_info.split('-')[-1]) / 16)
        end = start + int(self.clip_len / 16)
        #读取文件
        with h5py.File(self.feat_file, 'r') as f:
            if end > len(f[video_name]):
                feat = np.zeros([end - start, 2048])
                feat[:len(f[video_name][start:]), :] = f[video_name][start:]
            else:
                feat = f[video_name][start:end]

        return feat


    def _get_train_label(self,clip_info):
        cls_label = []  # 一个clip中所有的标签
        loc_label = []
        iou_label = []

        #获取clip_info下的label
        label = self.label[clip_info]

        # 同一层的label放在一起
        for i in range(len(self.anchor_info["feat_map_dim"])):

            cls_label.append(label[i][0]) # 样本的cls label保存在一起 (5,5,1)
            loc_label.append(label[i][1]) # 样本的loc label保存在一起 (5,5,2)
            iou_label.append(label[i][2]) # 样本的iou score保存在一起 (5,5,1)

        return cls_label, loc_label, iou_label

    #获取测试的数据
    def _get_test_clip(self):
        all_test_clip = {}
        for video in self.test_video_info:
            video_length = self.test_video_info[video]
            if video_length > self.clip_len:
                for clip_start_idx in range(0, int(video_length - self.clip_len * self.clip_overlap),
                                            int(self.clip_len * (1 - self.clip_overlap))):
                    clip_name = video+"-"+str(clip_start_idx)
                    all_test_clip[clip_name] = clip_start_idx
            else:
                clip_name = video +"-"+str(0)
                all_test_clip[clip_name] = 0

        return sorted(all_test_clip)



if __name__ == '__main__':
    cfg = Config(batch=1, is_training=False, clip_overlap=0.5)
    data = THUMOSDetectionDataset(cfg, subset="test")


    # cls,loc,iou = data._get_train_label('video_test_0000004-0')
    # # feat = data._read_feat_data('video_validation_0000311-4992')
    # # print(feat)
    # print("cls:",cls)
    # print("loc:",type(loc[0]))
    # print("iou:",iou)

    train_loader = torch.utils.data.DataLoader(data,
                                               batch_size=1, shuffle=False, num_workers=2)

    for a,(b,c) in enumerate(train_loader):
        # print(type(a),type(b))


        print(c[0])
        # print(b[1])
        # print(b[2])
        # print(b[3])


        break

        # print(type(b[1][0]))
        # #cls_
        # print("特征维度：", b[0].shape)
        # print("cls：", b[1][0].shape)
        # print("loc：", b[2][0].shape)
        # print("iou：", b[3][0].shape)



    # TRAIN_DATASET = THUMOSDetectionDataset(cfg, subset="train")
    # print(TRAIN_DATASET.is_training)
    # data = TRAIN_DATASET._read_feat_data('video_validation_0000903-128')
    # print(type(data))
    # TRAIN_DATASET.__getitem__(1)
    # label= TRAIN_DATASET._get_train_label('video_validation_0000903-128')
    # print(len(label))
















