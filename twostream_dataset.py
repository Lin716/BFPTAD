import h5py
import numpy as np
import os
import torch.utils.data as data
import torch
from operation.sslConfig import *

class THUMOSLabeledTwoStreamDataset(data.Dataset):
    def __init__(self,cfg):
        #标记的部分  只加载 包含lable的 clip 以及 label  60%
        self.labeled_sample_list = cfg.labeled_sample_list

        self.feat_file = cfg.feat_train_file
        self.video_info = cfg.video_train_info_file
        self.clip_file = cfg.clip_gt_file
        self.label_file = cfg.label_file
        self.clip_len = cfg.clip_len
        self.anchor_info = cfg.anchor_info
        self.clip_overlap = cfg.clip_overlap
        #关于扰动的一些参数
        self.time_masking = cfg.time_masking
        self.time_maskpro = cfg.time_maskpro

        # 训练下video的信息
        self.video = get_dict_from_pkl(self.video_info)
        # 训练下clip的信息 videoname-起始帧
        self.clip = get_dict_from_pkl(self.clip_file)
        # 训练下clip的label信息
        self.label = get_dict_from_pkl(self.label_file)
        # 训练下的clip列表长度
        self.clip_list = list(self.label)

    def __len__(self):
        return len(self.clip_list)

    def __getitem__(self, idx):
        clip_info = self.clip_list[idx]
        input_data = self._read_feat_data(clip_info)  # 32*2048维

        '''此处可以对数据进行增强 暂且只使用对特征的增强'''
        if self.time_masking:
            # 对feature进行时间掩模屏蔽掉一些片段
           mask_feature = self._time_mask(input_data)
        input_data = torch.Tensor(input_data.transpose())
        mask_feature = torch.Tensor(mask_feature.transpose())


        cls_label, loc_label, iou_label = self._get_train_label(clip_info)

        ret_dict = {}
        ret_dict['clip_info'] = clip_info
        ret_dict['feature'] = input_data
        ret_dict['mask_feature'] = mask_feature
        ret_dict['cls_label'] = cls_label
        ret_dict['loc_label'] = loc_label
        ret_dict['iou_label'] = iou_label

        return ret_dict

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

            cls_label.append(label[i][0]) # 样本不同时间尺度的cls label保存在一起 (5,5,1)
            loc_label.append(label[i][1]) # 样本的loc label保存在一起 (5,5,2)
            iou_label.append(label[i][2]) # 样本的iou score保存在一起 (5,5,1)

        return cls_label, loc_label, iou_label


    # 时间掩模
    def _time_mask(self, feature):

        clip_len = self.clip_len // 16
        new_feature = np.zeros([clip_len, 2048])
        num = int(self.time_maskpro * clip_len)
        choice = np.random.choice([i for i in range(clip_len)], num, replace=False)
        for i in range(clip_len):
            if i not in choice:
                new_feature[i] = feature[i]

        return new_feature



class THUMOSUnlabeledTwoStreamDataset(data.Dataset):
    def __init__(self,cfg):
        self.labeled_sample_list = cfg.labeled_sample_list

        self.feat_file = cfg.feat_train_file
        self.video_info = cfg.video_train_info_file
        self.clip_overlap = cfg.clip_overlap
        self.clip_len = cfg.clip_len
        # 关于扰动的一些参数
        self.time_masking = cfg.time_masking
        self.time_maskpro = cfg.time_maskpro

        # 所有的video
        self.video = get_dict_from_pkl(self.video_info)

        self.unlabel_clip = self._get_unlabel_clip()
        self.clip_list = list(self.unlabel_clip)

    def __len__(self):
        return len(self.clip_list)

    def __getitem__(self, idx):
        clip_info = self.clip_list[idx]
        input_data = self._read_feat_data(clip_info)

        '''此处可以做增强变换 对input_data'''
        if self.time_masking:
            # 对feature进行时间掩模屏蔽掉一些片段
           mask_feature = self._time_mask(input_data)

        input_data = torch.Tensor(input_data.transpose())
        mask_feature = torch.Tensor(mask_feature.transpose())

        ret_dict = {}
        ret_dict['clip_info'] = clip_info
        ret_dict['feature'] = input_data
        ret_dict['mask_feature'] = mask_feature


        return ret_dict


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



    def _get_unlabel_clip(self):

        #标记的video
        labeled_sample = get_list_from_txt(self.labeled_sample_list)
        #未标记的video 并获取clip（overlap使用与其他相同的 0.75）
        unlabel_video = [ name for name in list(self.video) if name not in labeled_sample]

        clip = {}
        for video in unlabel_video:
            video_length = self.video[video]
            if video_length > self.clip_len:
                for clip_start_idx in range(0, int(video_length - self.clip_len * self.clip_overlap),
                                            int(self.clip_len * (1 - self.clip_overlap))):
                    clip_name = video + "-" + str(clip_start_idx)
                    clip[clip_name] = clip_start_idx
            else:
                clip_name = video + "-" + str(0)
                clip[clip_name] = 0
        return clip

    # 时间掩模
    def _time_mask(self, feature):

        clip_len = self.clip_len // 16
        new_feature = np.zeros([clip_len, 2048])
        num = int(self.time_maskpro * clip_len)
        choice = np.random.choice([i for i in range(clip_len)], num, replace=False)
        for i in range(clip_len):
            if i not in choice:
                new_feature[i] = feature[i]

        return new_feature





if __name__ == '__main__':
    from operation.sslConfig import *

    cfg = sslConfig(batch='2,4', is_training=True, clip_overlap=0.75)

    label_data = THUMOSLabeledTwoStreamDataset(cfg)
    print(len(label_data.clip_list))
    label_dataloader = torch.utils.data.DataLoader(label_data,
                                               batch_size=2, shuffle=False, num_workers=2)
    for a,b in enumerate(label_dataloader):
        print(b['clip_info'])
        print(type(b['feature']))
        print(type(b['mask_feature']))
        print(torch.equal(b['feature'][0],b['mask_feature'][0]))
        print(b['cls_label'])
        print(b['loc_label'])
        print(b['iou_label'])

        break

    unlabel_data = THUMOSUnlabeledTwoStreamDataset(cfg)
    print(len(unlabel_data.clip_list))
    unlabel_loader = torch.utils.data.DataLoader(unlabel_data,
                                               batch_size=4, shuffle=False, num_workers=2)
    for a,b in enumerate(unlabel_loader):
        print(b['clip_info'])
        print(b['feature'])
        print(b['mask_feature'])
        print(torch.equal(b['feature'],b['mask_feature']))
        break