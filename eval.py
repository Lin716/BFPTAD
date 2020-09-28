# -*- coding:utf-8 -*-
from model.model import *
import numpy as np
from tqdm import tqdm
from operation.config import *
import os
import h5py
from dataset import *
from loss_function import *
import torch.utils.data as data


# def read_feat_data(cfg, path ,clip_info):
#     if not os.path.exists(path):
#         logging.info("%s: file isn't exists!" % path)
#         return
#     if path.split('.')[-1] != 'h5':
#         logging.info("%s: file type is error!" % path)
#         return
#
#     video_name = clip_info.split('-')[0]
#     #根据开始帧，算出在特征中的起止snippet
#     start = int(int(clip_info.split('-')[-1]) / 16)
#     end = start + int(cfg.clip_len / 16)
#     with h5py.File(path, 'r') as f:
#         if end > len(f[video_name]):
#             feat = np.zeros([end-start, 2048])
#             feat[:len(f[video_name][start:]), :] = f[video_name][start:]
#         else:
#             feat = f[video_name][start:end]
#     feat = torch.Tensor(feat.transpose())
#     return feat


# def get_test_clip(cfg, test_video_info):
#     """获得所有测试clip"""
#     all_test_clip = {}  # 存储所有test video的clip
#     for video in test_video_info:
#         video_length = test_video_info[video]
#         # print(video, video_length)
#         if video_length > cfg.clip_len:
#             for clip_start_idx in range(0, int(video_length - cfg.clip_len * cfg.clip_overlap),
#                                         int(cfg.clip_len * (1 - cfg.clip_overlap))):
#                 clip_name = video + "-" + str(clip_start_idx)
#                 all_test_clip[clip_name] = clip_start_idx
#         else:
#             clip_name = video + "-" + str(0)
#             all_test_clip[clip_name] = 0
#     return all_test_clip


def calculate_anchor_coordinate_one_layer(feat_map_dim, anchor_scales, anchor_step, anchor_width, offset=0.5):

    c = np.mgrid[0:feat_map_dim]
    # 可以得到在原图上，相对原图比例大小的每个锚点中心坐标c
    c = (c.astype(np.float32) + offset) * anchor_step

    # 将锚点中心坐标扩大维度
    c = np.expand_dims(c, axis=-1)

    # 该特征图上每个点对应的anchor的数量
    anchors_num = len(anchor_scales)
    assert anchors_num > 0

    l = np.zeros((anchors_num,), dtype=np.float32)
    for i, scale in enumerate(anchor_scales):
        l[i] = scale * anchor_width

    return c, l


def get_prior_box_coor(cfg):
    """获得anchor的原始坐标"""
    prior_box_list = []
    for idx in range(len(cfg.anchor_info["feat_layers"])):  # 每个特征层
        time_dim = cfg.anchor_info["feat_map_dim"][idx]
        scales_num = len(cfg.anchor_info["anchor_scales"][idx])  # 每个点上的anchor数
        prior_box_vec = np.zeros([time_dim, scales_num, 2], dtype=np.float32)  # 存储先验框的中心和宽度
        # 获得每个特征层的每个anchor的中心坐标和宽度
        center, width = calculate_anchor_coordinate_one_layer(cfg.anchor_info["feat_map_dim"][idx],
                                                              cfg.anchor_info["anchor_scales"][idx],
                                                              cfg.anchor_info["anchor_scaling"][idx],
                                                              cfg.anchor_info["anchor_width"][idx],
                                                              cfg.anchor_info["anchor_offset"])
        # 计算每个anchor的起始结束坐标
        center = np.array(center)
        width = np.array(width)
        prior_box_vec[:, :, 0] = center  # 先验框中心
        prior_box_vec[:, :, 1] = width  # 先验框宽度
        prior_box_vec = np.expand_dims(prior_box_vec, axis=0)  # 对应batch维度
        prior_box_list.append(prior_box_vec)
    return prior_box_list


def decode(prior_box, loc_pred):
    """
    解码，将得到的位置预测值(中心和宽度)转为起始结束坐标
    :param prior_box: 先验框中心，宽度信息
    :param loc_pred:  位置预测值
    :return:
    """
    if len(prior_box) != len(loc_pred):
        logging.info("the layers' number has error!!")
        return

    loc_list = []  # 存储每层解码后的loc信息(起始结束坐标)
    for i in range(len(prior_box)):  # 每一层
        shape = prior_box[i].shape
        tmp = np.zeros(shape, dtype=np.float32)
        tmp[:, :, :, 0] = loc_pred[i][:, :, :, 0] * prior_box[i][:, :, :, 1] + prior_box[i][:, :, :, 0]  # 包含batch维度
        tmp[:, :, :, 1] = prior_box[i][:, :, :, 1] * np.exp(loc_pred[i][:, :, :, 1])
        loc_list.append(tmp)
    return loc_list


def drop_bound(prior_box, cls_pred, loc_pred):
    """
    将超出边界的anchor(即不做预测的anchor)抛弃
    :param prior_box: 先验框中心，宽度信息
    :param loc_pred:  位置预测值
    :return:
    """
    prior_box_concat = []
    for i in range(len(prior_box)):  # 每一层
        prior_box_concat.append(np.reshape(prior_box[i], [-1, 2]))
    prior_box1 = np.concatenate(prior_box_concat, axis=0)  # (all_anchor_num, 2)
    mask = np.bitwise_and(np.greater(prior_box1[:,0], cfg.clip_len), np.less(prior_box1[:,1], 0))  # 原始anchor超出边界的
    cls_pred[mask] = 0.
    loc_pred[mask] = 0.
    return cls_pred, loc_pred


def nms(cls, loc, topk=10, threhold=0.5):
    """
    NMS处理
    :param cls: 预测的分类分数
    :param loc: 预测的位置信息
    :param topk: 最多保留topk个检测结果
    :param threhold: 预测框之间的IOU筛选阈值
    :return:
    """
    start = loc[:, 0]
    end = loc[:, 1]

    box_length = end-start  # 每一个预测框的长度
    order = cls.argsort()[::-1]  # order是按照cls分数降序排序

    # 去除超出边界的检测结果
    order_list = []
    for i in order:
        # 起始结束大于0且分类分数大于0
        if (start[i] >= 0) and (end[i] > 0) and (start[i] < end[i]) and (cls[i] > 0):
            order_list.append(i)
    order = np.array(order_list)
    # print("order:", order)

    keep = []  # 保存最终保留的预测框
    while order.size > 0:
        i = order[0]
        # print("I:", i)
        # print("loc:", start[i], end[i])

        keep.append(i)
        if topk is not None:
            if len(keep) >= topk:
                return keep

        # 计算当前概率最大的框和其他框的相交框的坐标, 用到python的broadcast机制
        start1 = np.maximum(start[i], start[order[1:]])
        end1 = np.minimum(end[i], end[order[1:]])
        # 计算相交框的部分，不相交是负数，用0代替
        inter = np.maximum(0., end1 - start1)
        # 计算IOU
        iou = inter/ (box_length[i] + box_length[order[1:]] - inter)
        # 找到iou不高于阈值的框索引
        inds = np.where(iou <= threhold)[0]

        # 更新order序列,保留iou小于阈值的框
        order = order[inds + 1]  # 由于得到的框索引比原order序列小1([1:])，所以需要加1
    return keep


def get_one_cat_score_and_loc(one_video_res, cls_idx):
    """获得一个视频的某一类的所有分类分数和坐标信息"""
    cls_score_list = []  # 保存分类分数
    loc = []  # 保存坐标
    for i in range(len(one_video_res[0])):
        cls_score_list.append(one_video_res[0][i][:, cls_idx])
        loc.append(one_video_res[1][i])
    cls_score = np.concatenate(cls_score_list, axis=0).reshape(-1)
    loc = np.vstack(loc)
    return cls_score, loc


def get_max_prob_cat_in_one_video(video_res, num=2):
    """
    获得一个视频中每个anchor的最大分类分数及类别，抛弃背景类
    :param video_res: 一个视频的所有clip的检测结果
    :param num: 每个anchor取最大的num个后续用于nms
    :return:
    """
    one_video_res_dict = {}  # 创建一个20类的字典，每个类别保存对应的结果
    for i in range(1, cfg.classes_num):
        one_video_res_dict[i] = [[],[]]  # 分别保存分类分数和位置信息

    for clip_idx in range(len(video_res[0])):  # 一个视频有多个clip
        for i in range(len(video_res[0][clip_idx])):  # 一个clip有多个anchor
            anchor_cls = video_res[0][clip_idx][i, :]  # 第i个anchor的检测结果,(21,)
            anchor_loc = video_res[1][clip_idx][i, :]  # 第i个anchor的位置检测结果,(2,)

            # 获取最大的num个分类分数
            res_sort = np.argsort(anchor_cls)
            max_idx = res_sort[-num:]
            for i in range(len(max_idx)):
                if max_idx[i] == 0:  # 抛弃背景
                    continue
                else:
                    max_score = anchor_cls[max_idx[i]]  # 分类分数
                    one_video_res_dict[max_idx[i]][0].append(max_score)
                    one_video_res_dict[max_idx[i]][1].append(anchor_loc)

    for key in one_video_res_dict:
        if len(one_video_res_dict[key][0]) != 0:
            one_video_res_dict[key][0] = np.array(one_video_res_dict[key][0])
            one_video_res_dict[key][1] = np.concatenate(one_video_res_dict[key][1], axis=0).reshape(-1,2)
            # print("one_video_res_dict[key][0]:", one_video_res_dict[key][0].shape)
            # print("one_video_res_dict[key][1]:", one_video_res_dict[key][1].shape)
    return one_video_res_dict


def post_process(cfg,video_res, video, f):
    """
    对每个视频的检测结果进行后处理
    :param video_res: 每个视频的检测结果
    :param video: 视频名字
    :return:
    """
    # 整个视频一次NMS,每个视频所有clip的anchor聚集起来，然后每个anchor去掉背景,然后20类分别做nms
    if cfg.post_process == 1:
        for i_cls in range(1, cfg.classes_num):  # 每个类别都要做一次NMS, 不包含背景类
            i_cls_score, i_loc = get_one_cat_score_and_loc(video_res, i_cls)
            # 筛选, 分类分数小于阈值的改为0
            i_cls_score = np.expand_dims(i_cls_score, axis=0)  # shape(N,) -> (1,N)
            cls_neg_mask = np.less(i_cls_score, cfg.thr)
            i_cls_score[cls_neg_mask] = 0.
            i_cls_score = np.reshape(i_cls_score, -1)  # shape(1,N) -> (N,)

            i_keep = nms(i_cls_score, i_loc, topk=None, threhold=cfg.nms_thr)
            # if len(i_keep):
            #     print("i_keep:", i_keep)
            for j in i_keep:
                if cfg.dataset == "THUMOS14":
                    j_cls = cfg.cat_index_in_UCF[i_cls][1]  # 转为101类中的序号
                elif cfg.dataset == "activity":
                    j_cls = i_cls
                j_cls_score = i_cls_score[j]  # 每个视频中的第i类的第j个NMS结果的分类分数
                j_loc_start = i_loc[j, 0]  # 起始
                j_loc_end = i_loc[j, 1]  # 结束
                # 视频名, 起始, 结束, 类别, 分类分数
                f.write(str(video) + " " + str(j_loc_start / cfg.fps) + " " + str(j_loc_end / cfg.fps) + " " + str(
                    j_cls) + " " + str(j_cls_score) + "\n")
                if cfg.dataset == "THUMOS14":
                    # 类别是CliffDiving时, 同时保存一个Diving结果
                    if j_cls == 22:
                        f.write(str(video) + " " + str(j_loc_start / cfg.fps) + " " + str(
                            j_loc_end / cfg.fps) + " " + str(26) + " " + str(j_cls_score) + "\n")
    # 每个视频两次NMS,第一次对每个clip进行一次,然后将每个视频得到的检测结果(多个clip nms后的结果)再进行一次nms
    elif cfg.post_process == 2:
        clip_nms = {}  # 保存每个clip nms后的结果
        for i_cls in range(1, cfg.classes_num):  # 每个类别，20类
            clip_nms[i_cls] = [[], []]
        pbar.set_description("NMS in %s" % video)
        for clip in range(len(video_res[0])):  # 每个clip
            clip_cls = video_res[0][clip]  # (all_anchor_num, 21)
            clip_loc = video_res[1][clip]  # (all_anchor_num, 2)

            # 将超出整个视频长度的动作改为视频长度
            video_len = test_video_info[video]  # 视频长度
            mask = np.greater(clip_loc, video_len)
            clip_loc[mask] = video_len

            for i_cls in range(1, cfg.classes_num):  # 每个类别，20类
                i_cls_score = clip_cls[:, i_cls]  # (all_anchor_num,)
                #  # 筛选, 分类分数小于阈值的改为0
                i_cls_score = np.expand_dims(i_cls_score, axis=0)  # shape(N,) -> (1,N)
                cls_neg_mask = np.less(i_cls_score, cfg.thr)
                i_cls_score[cls_neg_mask] = 0.
                i_cls_score = np.reshape(i_cls_score, -1)  # shape(1,N) -> (N,)

                # 每个clip每个类别最多top-k个结果
                i_keep = nms(i_cls_score, clip_loc, topk=10, threhold=cfg.nms_thr)
                for j in i_keep:
                    j_cls_score = i_cls_score[j]  # 每个clip中的第i类的第j个NMS结果的分类分数
                    j_loc = clip_loc[j]  # 坐标
                    clip_nms[i_cls][0].append(j_cls_score)
                    clip_nms[i_cls][1].append(j_loc)
        # 对每个视频进行第2次nms
        for i_cls in range(1, cfg.classes_num):
            if len(clip_nms[i_cls][0]) == 0:
                continue
            i_cls_score = np.hstack(clip_nms[i_cls][0])
            i_cls_loc = np.vstack(clip_nms[i_cls][1])
            i_keep = nms(i_cls_score, i_cls_loc, topk=None, threhold=cfg.nms_thr)
            for j in i_keep:
                j_cls = cfg.cat_index_in_UCF[i_cls][1]  # 转为101类中的序号
                j_cls_score = i_cls_score[j]  # 每个clip中的第i类的第j个NMS结果的分类分数
                j_loc_start = i_cls_loc[j, 0]  # 起始
                j_loc_end = i_cls_loc[j, 1]  # 结束
                # 视频名, 起始, 结束, 类别, 分类分数
                f.write(str(video) + " " + str(j_loc_start / cfg.fps) + " " + str(j_loc_end / cfg.fps) + " " + str(
                    j_cls) + " " + str(j_cls_score) + "\n")
                if cfg.dataset == "THUMOS14":
                    # 类别是CliffDiving时, 同时保存一个Diving结果
                    if j_cls == 22:
                        f.write(str(video) + " " + str(j_loc_start / cfg.fps) + " " + str(
                            j_loc_end / cfg.fps) + " " + str(26) + " " + str(j_cls_score) + "\n")
    # 整个视频一次NMS,从每个视频的每个clip中的每个anchor找最大分类结果,每个视频的所有结果聚集起来，然后20类分别做nms
    elif cfg.post_process == 3:
        one_video_res = get_max_prob_cat_in_one_video(video_res, num=cfg.using_num)
        for i_cls in range(1, cfg.classes_num):  # 每个类别都要做一次NMS
            if len(one_video_res[i_cls][0]) == 0:
                continue
            # 筛选, 分类分数小于阈值的改为0
            one_video_res[i_cls][0] = np.expand_dims(one_video_res[i_cls][0], axis=0)  # shape(N,) -> (1,N)
            cls_neg_mask = np.less(one_video_res[i_cls][0], cfg.thr)
            one_video_res[i_cls][0][cls_neg_mask] = 0.
            one_video_res[i_cls][0] = np.reshape(one_video_res[i_cls][0], -1)  # shape(1,N) -> (N,)

            i_keep = nms(one_video_res[i_cls][0], one_video_res[i_cls][1], topk=None, threhold=cfg.nms_thr)
            # print("i_keep:", i_keep)
            for j in i_keep:
                j_cls = cfg.cat_index_in_UCF[i_cls][1]  # 转为101类中的序号
                j_cls_score = one_video_res[i_cls][0][j]
                j_loc_start = one_video_res[i_cls][1][j][0]
                j_loc_end = one_video_res[i_cls][1][j][1]
                # 视频名, 起始, 结束, 类别, 分类分数
                f.write(str(video) + " " + str(j_loc_start / cfg.fps) + " " + str(j_loc_end / cfg.fps) + " " + str(
                    j_cls) + " " + str(j_cls_score) + "\n")
                if cfg.dataset == "THUMOS14":
                    # 类别是CliffDiving时, 同时保存一个Diving结果
                    if j_cls == 22:
                        f.write(str(video) + " " + str(j_loc_start / cfg.fps) + " " + str(
                            j_loc_end / cfg.fps) + " " + str(26) + " " + str(j_cls_score) + "\n")


def eval(cfg,data,dataloader,model,ckpt):
    """
       eval
       :param cfg: 参数类的对象
       :param ckpt: 模型
       :param all_test_clip: 所有test视频的clip
       :param test_feat_file: 测试集特征
       :param test_video_info: 测试集视频帧数信息
       :param pkl_path: 检测结果保存路径(未进行后处理,保存为pkl文件)
       :param post_process_path: 后处理结果文件，保存为txt
       :return:
    """

    if not os.path.exists(cfg.ckpt_path):
        logging.info("The ckpt path is not exists!")
    if not os.path.exists(cfg.detection_res):
        os.makedirs(cfg.detection_res)


    # test视频clip 片段之间的overlap为0.5
    # test_video_info = get_dict_from_pkl(cfg.video_test_info_file)
    # all_test_clip = get_test_clip(cfg, test_video_info)
    # print("test clip num: %s\n" % len(all_test_clip))

    print("test clip num: %s\n" % len(data.clip_list))

    # 训练的次数及损失值
    ckpt_split = ckpt.split('.')[-2]
    epoch = ckpt_split.split('_')[-1]

    # 保存后处理结果文件
    res_file = "%s/res_fps_%s_epoch_%s.txt" % (cfg.detection_res, cfg.fps, epoch)
    f = open(res_file, 'w')

    #加载模型
    checkpoint = torch.load(ckpt, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])

    prior_box_coor = get_prior_box_coor(cfg) # 获得先验框的坐标,用于解码


    one_video_score_res = []  # 一个视频的分数检测结果保存在一个list中
    one_video_loc_res = []  # 一个视频的坐标检测结果保存在一个list中
    video_name = None


    for n_iter,(input_data,clip) in enumerate(dataloader):
        clip_info = clip[0]
        split = clip_info.split('-')
        clip_video_name = split[0]  # clip所在视频名
        clip_start = int(split[1])  # clip的起始帧

        if video_name is None:
            video_name = clip_video_name
        if video_name != clip_video_name:  # 改变视频时保存上一个视频的检测结果
            post_process(cfg, [one_video_score_res, one_video_loc_res], video_name, f)  # 后处理
            one_video_score_res = []  # 清空
            one_video_loc_res = []  # 清空
            video_name = clip_video_name


        with torch.no_grad():
            cls, cls_logits, loc, endpoint = model(input_data)


        # 将预测的部分转成numpy 后进行处理
        for i in range(len(cls)):
            cls[i] = cls[i].detach().numpy()
            loc[i] = loc[i].detach().numpy()

        #根据预测进行解码以及分析
        loc = decode(prior_box_coor, loc)  # 解码，得到动作的中心和宽度

        cls_concat = []
        loc_concat = []
        for j in range(len(cls)):
            cls_concat.append(np.reshape(cls[j], [-1, cfg.classes_num]))
            loc_concat.append(np.reshape(loc[j], [-1, 2]))
        cls_score = np.concatenate(cls_concat, axis=0)  # (all_anchor_num, classes_num)
        loc_pred = np.concatenate(loc_concat, axis=0)  # (all_anchor_num, 2)

        if not cfg.beyond_bound_train: # 如果超出边界的anchor不训练， 则将这些anchor抛弃
            cls_score, loc_pred = drop_bound(prior_box_coor, cls_score, loc_pred)

        # 将中心和宽度转为起始和结束坐标
        loc = np.zeros(loc_pred.shape)
        loc[:, 0] = loc_pred[:, 0] - loc_pred[:, 1] / 2
        loc[:, 1] = loc_pred[:, 0] + loc_pred[:, 1] / 2

        if cfg.clip_beyond_boundary_res:  # 超出边界的进行修剪
            loc_start_mask = np.less(loc, 0)
            loc[loc_start_mask] = 0
            loc_end_mask = np.greater(loc, cfg.clip_len)
            loc[loc_end_mask] = cfg.clip_len

            loc += clip_start # 最后将所有检测结果加上clip的起始坐标，得到在视频中的位置
        else:  # 超出边界的全部抛弃
            loc_mask = np.bitwise_and(np.greater(loc[:, 0], 0), np.less(loc[:, 1], cfg.clip_len))  # 没有超出边界的
            loc_neg_mask = np.bitwise_not(loc_mask)  # 超出边界的抛弃
            loc[loc_neg_mask] = 0.
            cls_score[loc_neg_mask] = 0.
            loc[loc_mask] += clip_start  # 没有超出边界的坐标加上clip的起始坐标，得到在视频中的位置

        # 将分类分数和起始结束结果保存  （这个是一个clip的）
        one_video_score_res.append(cls_score)
        one_video_loc_res.append(loc)

        if (n_iter+1) == len(data.clip_list):  # 保存最后一个视频的检测结果
            post_process(cfg, [one_video_score_res, one_video_loc_res], video_name, f)  # 后处理


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    cfg = Config(batch=1, is_training=False,clip_overlap=0.5)
    model = Model(cfg)

    # 文件夹下所有model测试
    for parent, dirnames, filenames in os.walk(cfg.ckpt_path):
        for i in range(len(filenames)):
            # print("file name:", filenames[i])
            # filename_split = filenames[i].split(".")

            # 每个ckeckpoint有3种文件，.meta,.index, .data-00000-of-00001，去除后缀可以得到保存的checkpoint文件名
            # if filename_split[-1] != 'meta':
            #     continue
            # 如果文件名包含loss值
            ckpt = cfg.ckpt_path + '/' + filenames[i]
            print("ckpt:", ckpt)
            eval(cfg, model,ckpt)

    # 单个文件测试
    # ckpt = "%s/model_epoch_70_loss_0.005.pkl" % cfg.ckpt_path
    # eval(cfg, ckpt)

