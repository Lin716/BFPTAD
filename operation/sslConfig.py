# -*- coding:utf-8 -*-
import os
import logging
import pickle

#配置的参数 预训练的参数，包含了训练+测试结果的部分
class sslConfig(object):
    """
    define a class to store parameters
    """
    def __init__(self, batch="2,4", is_training=True, clip_overlap=0.75):
        self.epochs = 100
        self.batch = batch
        self.is_training = is_training
        self.dataset = "THUMOS14"  # THUMOS14 or activity
        self.pos_thr = 0.5  # anchor为正例的阈值
        self.neg_thr = 0.5  # anchor为负例的阈值
        self.learning_rate = 0.0001  # 学习率

        #每个clip的长度512帧（THUMMOS：32，Anet：512，因为I3D特征）
        self.clip_len = 512
        self.fps = 25  # 视频提取为帧时的采样率
        self.model_name = "BFPTAD"  # 模型名字


        if self.dataset == "THUMOS14":
            self.classes_num = 21  # 类别数(包含背景)
        elif self.dataset == "activity":
            self.classes_num = 101  # 类别数(包含背景)
        else:
            logging.info("The dataset not use!")
            quit()

        """  Anchor  info  锚的信息"""
        self.anchor_info = {
            "feat_layers": ['N1', 'N2', 'N3', 'N4', 'N5'],  # 特征层名字
            "feat_map_dim": [32, 16, 8, 4, 2],  # 每个特征层的时间维度
            # "default_anchor_width": [1, 1, 1, 1, 1],  # 每个特征层的anchor默认宽度比例
            "anchor_scaling": [16, 32, 64, 128, 256],  # 特征图锚点框放大到原始图的缩放比例，与每一层的时间维度相乘得到clip的长度
            "anchor_width": [16, 48, 112, 240, 496],  # anchor的默认大小
            "anchor_scales": [  # 每个特征层上anchor的比率
                [0.5, 0.75, 1, 1.5, 2.0],  # N1
                [0.5, 0.75, 1, 1.5, 2.0],  # N2
                [0.5, 0.75, 1, 1.5, 2.0],  # N3
                [0.5, 0.75, 1, 1.5, 2.0],  # N4
                [0.5, 0.75, 1]],  # N5
            "anchor_offset": 0.5,
            "normalizations": [True, True, True, True, True],
            }

        self.clip_overlap = clip_overlap  # 同一视频clip之间的重叠率, Train: 0.75, Test: 0.5
        self.gt_clip_overlap = 0.75  # gt和clip重叠率大于该阈值, 则认为gt属于该clip

        #测试的部分（NMS+后处理）
        self.nms_thr = 0.3  # nms阈值
        self.thr = 0.1  # test中, 分类分数小于该阈值的都变为0
        self.post_process = 1  # 1 or 2 or 3 对应不同的后处理方式
        self.using_num = 3  # 第3种后处理方式取每个anchor的最大using_Num个分类分数，用于nms
        self.beyond_bound_train = True  # 超出边界的anchor是否参与训练
        self.clip_beyond_boundary_res = False  # 检测结果超出边界的是抛弃还是修剪, True将超出边界的修剪为边界值, False抛弃超出边界的
        self.focal_loss = False  # 使用 focal loss:True, 使用hard negative mining: False

        """  Path  文件的路径"""
        self.annotations = "annotation/" + self.dataset  # ground truth文件
        #特征文件
        #特征维数
        self.feat_train_file = r'C:/dataset/BFPTAD/%s_I3d_train_feat.h5' % self.dataset  # 训练集特征文件
        self.feat_test_file = r'C:/dataset/BFPTAD/%s_I3d_test_feat.h5' % self.dataset
        self.feature_size = 2048


        self.ckpt_path = "Checkpoints"
        self.res_path = "Res"
        self.log_path = "Logs"

        #结果文件、checkpoint文件，res文件，log文件的路径
        self.detection_res = "%s/%s" % (self.res_path, self.dataset) + '/ssl'   # 后处理结果文件，保存为txt
        self.ckpt_path = self.ckpt_path + '/' + self.dataset +'/ssl'
        self.res_path = self.res_path + '/' + self.dataset +'/ssl'
        self.log_path = self.log_path + '/' + self.dataset +'/ssl'

        self.preprocess_dir = "preprocess_info/%s/" % self.dataset  # 保存各种预处理文件的路径

        #文件名：视频帧
        if self.dataset == "THUMOS14":
            self.video_train_info_file = self.preprocess_dir + "train_video.pkl"
            self.video_test_info_file = self.preprocess_dir + "test_video.pkl"

        elif self.dataset == "activity":
            self.video_train_info_file = self.preprocess_dir + "train_video.pkl"
            self.video_test_info_file = self.preprocess_dir + "validation_video.pkl"


        #关于label的文件：groundtruth文件，clip的gt(包含训练和验证的)
        self.gt_file = self.preprocess_dir + "gt.pkl"
        self.gt_test_file = self.preprocess_dir + "gt_test.pkl"


        #暂且直接利用60%的label样本直接训练 train
        self.clip_gt_file = self.preprocess_dir + "clip_60.pkl"
        self.label_file = self.preprocess_dir + "train_label_60.pkl"

        #测试 test
        self.clip_test_file = self.preprocess_dir + "clip_test.pkl"
        self.test_label_file = self.preprocess_dir+"test_label.pkl"


        #半监督
        self.labeled_sample_list = 'semiIndex.txt'
        self.detector_checkpoint = 'Checkpoints/'+self.dataset+'/model_best.pkl'  #加载的模型
        self.ema_decay = 0.999
        self.time_masking = True
        self.time_maskpro = 0.3
        #一致性损失的权重
        self.consistency_weight = 10.0



        if self.dataset == "THUMOS14":
            self.cat_index = {
                "Background": 0,
                "BaseballPitch": 1,
                "BasketballDunk": 2,
                "Billiards": 3,
                "CleanAndJerk": 4,
                "CliffDiving": 5,
                "CricketBowling": 6,
                "CricketShot": 7,
                "Diving": 8,
                "FrisbeeCatch": 9,
                "GolfSwing": 10,
                "HammerThrow": 11,
                "HighJump": 12,
                "JavelinThrow": 13,
                "LongJump": 14,
                "PoleVault": 15,
                "Shotput": 16,
                "SoccerPenalty": 17,
                "TennisSwing": 18,
                "ThrowDiscus": 19,
                "VolleyballSpiking": 20
            }
        elif self.dataset == "activity":
            self.cat_index = {
                "Background": 0,
                "Archery": 1,
                "Ballet": 2,
                "Bathing dog": 3,
                "Belly dance": 4,
                "Breakdancing": 5,
                "Brushing hair": 6,
                "Brushing teeth": 7,
                "Bungee jumping": 8,
                "Cheerleading": 9,
                "Chopping wood": 10,
                "Cleaning shoes": 11,
                "Cleaning windows": 12,
                "Clean and jerk": 13,
                "Cricket": 14,
                "Cumbia": 15,
                "Discus throw": 16,
                "Dodgeball": 17,
                "Doing karate": 18,
                "Doing kickboxing": 19,
                "Doing motocross": 20,
                "Doing nails": 21,
                "Doing step aerobics": 22,
                "Drinking beer": 23,
                "Drinking coffee": 24,
                "Fixing bicycle": 25,
                "Getting a haircut": 26,
                "Getting a piercing": 27,
                "Getting a tattoo": 28,
                "Grooming horse": 29,
                "Hammer throw": 30,
                "Hand washing clothes": 31,
                "High jump": 32,
                "Hopscotch": 33,
                "Horseback riding": 34,
                "Ironing clothes": 35,
                "Javelin throw": 36,
                "Kayaking": 37,
                "Layup drill in basketball": 38,
                "Long jump": 39,
                "Making a sandwich": 40,
                "Mixing drinks": 41,
                "Mowing the lawn": 42,
                "Paintball": 43,
                "Painting": 44,
                "Ping-pong": 45,
                "Plataform diving": 46,
                "Playing accordion": 47,
                "Playing badminton": 48,
                "Playing bagpipes": 49,
                "Playing field hockey": 50,
                "Playing flauta": 51,
                "Playing guitarra": 52,
                "Playing harmonica": 53,
                "Playing kickball": 54,
                "Playing lacrosse": 55,
                "Playing piano": 56,
                "Playing polo": 57,
                "Playing racquetball": 58,
                "Playing saxophone": 59,
                "Playing squash": 60,
                "Playing violin": 61,
                "Playing volleyball": 62,
                "Playing water polo": 63,
                "Pole vault": 64,
                "Polishing forniture": 65,
                "Polishing shoes": 66,
                "Preparing pasta": 67,
                "Preparing salad": 68,
                "Putting on makeup": 69,
                "Removing curlers": 70,
                "Rock climbing": 71,
                "Sailing": 72,
                "Shaving legs": 73,
                "Shaving": 74,
                "Shot put": 75,
                "Shoveling snow": 76,
                "Skateboarding": 77,
                "Smoking a cigarette": 78,
                "Smoking hookah": 79,
                "Snatch": 80,
                "Spinning": 81,
                "Springboard diving": 82,
                "Starting a campfire": 83,
                "Tai chi": 84,
                "Tango": 85,
                "Tennis serve with ball bouncing": 86,
                "Triple jump": 87,
                "Tumbling": 88,
                "Using parallel bars": 89,
                "Using the balance beam": 90,
                "Using the pommel horse": 91,
                "Using uneven bars": 92,
                "Vacuuming floor": 93,
                "Walking the dog": 94,
                "Washing dishes": 95,
                "Washing face": 96,
                "Washing hands": 97,
                "Windsurfing": 98,
                "Wrapping presents": 99,
                "Zumba": 100
            }
        self.cat_index_in_UCF = {  # 20类在UCF101类中的序号，真实类别序号
            0:("Background",0),
            1:("BaseballPitch",7),
            2:("BasketballDunk",9),
            3:("Billiards",12),
            4:("CleanAndJerk",21),
            5:("CliffDiving",22),
            6:("CricketBowling",23),
            7:("CricketShot",24),
            8:("Diving",26),
            9:("FrisbeeCatch",31),
            10:("GolfSwing",33),
            11:("HammerThrow",36),
            12:("HighJump",40),
            13:("JavelinThrow",45),
            14:("LongJump",51),
            15:("PoleVault",68),
            16:("Shotput",79),
            17:("SoccerPenalty",85),
            18:("TennisSwing",92),
            19:("ThrowDiscus",93),
            20:("VolleyballSpiking",97)
            }


def save_dict_to_pkl(input_dict, pkl_file):
    """将字典保存到pkl文件中"""
    assert isinstance(input_dict, dict)
    assert isinstance(pkl_file, str)
    if pkl_file.split('.')[-1] != 'pkl':
        logging.info("the %s' type isn't pkl file!" % pkl_file)
        return
    f = open(pkl_file, 'wb')
    pickle.dump(input_dict, f)
    f.close()


def get_dict_from_pkl(pkl_file):
    """从pkl文件中读取字典数据"""
    assert isinstance(pkl_file, str)
    if not os.path.exists(pkl_file):
        logging.info("the %s' isn't exists!" % pkl_file)
        return {}
    if pkl_file.split('.')[-1] != 'pkl':
        logging.info("the %s' type isn't pkl file!" % pkl_file)
        return {}

    f = open(pkl_file, 'rb')
    dict_data = pickle.load(f)
    return dict_data


def get_list_from_txt(file):
    idx = []
    with open(file,'r') as f:
        lines = f.readlines()
        for line in lines:
            idx.append(line. strip('\n'))
    return idx



if __name__=='__main__':
    cfg = sslConfig(batch="2,4", is_training=True, clip_overlap=0.75)
    print(cfg.label_file)
    # train_label = get_dict_from_pkl("../preprocess_info/THUMOS14/train_label_60.pkl")
    # train_label_list = list(train_label)
    # print(train_label_list)