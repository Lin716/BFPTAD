from twostream_dataset import *
import torch.optim as optim
import torch.utils.data as data
from eval import *
import utils.ramps as ramps
import torch

# --------------------------------Global Setting------------------------------------#
cfg = sslConfig(batch='2,4', is_training = True,clip_overlap = 0.75)

if not os.path.exists(cfg.log_path):
    os.makedirs(cfg.log_path)
if not os.path.exists(cfg.ckpt_path):
    os.makedirs(cfg.ckpt_path)

# 记录log
LOG_FOUT = open(os.path.join(cfg.log_path, 'log_train.txt'), 'a')


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


batch_size_list = [int(x) for x in cfg.batch.split(',')]
BATCH_SIZE = batch_size_list[0] + batch_size_list[1]  # 标记2，未标记4

# ---------------------------------------------------------------------------------#

#构建模型
def create_model(ema = False):
    model = Model(cfg)
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


#更新ema模型的变量
def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct 使用真实的平均值直到指数平均更加正确
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


# #获取当前epoch下一致性损失的权重是多少 -暂且不改变一致性损失的权重
# def get_current_consistency_weight(epoch):
#     # Consistency ramp-up
#     return FLAGS.consistency_weight * ramps.sigmoid_rampup(epoch, FLAGS.consistency_rampup)


def train_one_epoch(detector,ema_detector,label_dataloader,unlabel_dataloader,optimizer,global_step,device):
    detector.train()
    ema_detector.train()
    #获取当前的一致性损失的权重
    # consistency_weight = get_current_consistency_weight(EPOCH_CNT)
    log_string('Current consistency weight: %f' % cfg.consistency_weight)

    unlabel_dataloader_iterator = iter(unlabel_dataloader)

    for batch_idx,(batch_data_label) in enumerate(label_dataloader):
        try:
            #无标签的部分
            batch_data_unlabel = next(unlabel_dataloader_iterator)
        except StopIteration:
            unlabel_dataloader_iterator = iter(unlabel_dataloader)
            batch_data_unlabel = next(unlabel_dataloader)

        #对input_data进行拼接
        for key in batch_data_unlabel:
            if key == 'clip_info':
                for clip in batch_data_unlabel[key]:
                    batch_data_label[key].append(clip)
            else:
                batch_data_label[key] = torch.cat((batch_data_label[key],batch_data_unlabel[key]))

        for key in batch_data_label:
            if isinstance(batch_data_label[key], torch.Tensor):
                batch_data_label[key] = batch_data_label[key].to(device)


        #教师模型无扰动 ， 学生模型输入mask的特征

        student_input = batch_data_label['mask_feature']
        teacher_input = batch_data_label['feature']

        optimizer.zero_grad()

        predictions,cls_logits,loc_logits,endpoint = detector(student_input)
        ema_pre,ema_clsg,ema_locg,ema_endpoint = ema_detector(teacher_input)

        #计算loss -监督和一致性
        #监督
        supervised_cls = []
        supervised_loc = []

        for i,layer in enumerate(cfg.anchor_info["feat_layers"]):
            supervised_cls.append(cls_logits[i][:batch_size_list[0]])
            supervised_loc.append(loc_logits[i][:batch_size_list[0]])

        detector_loss = compute_detector_loss(cfg,supervised_cls,supervised_loc,batch_data_label['cls_label'], batch_data_label['loc_label'],batch_data_label['iou_label'],device)
        print('detection loss',detector_loss[0])
        #一致性损失
        consistency_loss = compute_consistency_loss(cfg,predictions,ema_pre,cls_logits,ema_clsg,loc_logits,ema_locg) #参数都是在cuda上的


        # loss = detector_loss+consistency_loss

        detector_loss[0].backward()
        optimizer.step()

        global_step += 1

        #更新ema的变量
        # update_ema_variables(detector,ema_detector,cfg.ema_decay,global_step)

        #输出统计的loss等

    return global_step





def train():

    #加载数据
    label_dataset = THUMOSLabeledTwoStreamDataset(cfg)

    unlabel_dataset = THUMOSUnlabeledTwoStreamDataset(cfg)

    label_dataloader = torch.utils.data.DataLoader(label_dataset,batch_size=batch_size_list[0], shuffle=False, num_workers=2)

    unlabel_dataloader = torch.utils.data.DataLoader(unlabel_dataset,batch_size=batch_size_list[1], shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #构建网络
    detector = create_model().to(device) #学生模型
    ema_detector = create_model(ema=True).to(device) #教师模型


    # 优化器
    optimizer = optim.Adam(detector.parameters(), lr=cfg.learning_rate) #只对学生模型进行优化
    #加载预先训练的模型
    if os.path.exists(cfg.detector_checkpoint):
        checkpoint = torch.load(cfg.detector_checkpoint,map_location=torch.device('cpu'))
        pretrained_dict = checkpoint['state_dict']
        detector.load_state_dict(pretrained_dict)
        ema_detector.load_state_dict(pretrained_dict)
        pretrained_epoch = checkpoint['epoch']
        print('Loaded pretrained checkpoint %s (epoch %d)' %(cfg.detector_checkpoint,pretrained_epoch))
    else:
        print("Don't exist the pretrained checkpoint")


    #进行训练
    global EPOCH_CNT
    global_step =0

    for epoch in range(cfg.epochs):
        EPOCH_CNT = epoch
        log_string('\n********EPOCH  %03d, STEP %d *********' % (epoch+1, global_step))
        global_step = train_one_epoch(detector,ema_detector,label_dataloader,unlabel_dataloader,optimizer,global_step,device)



if __name__ == '__main__':
    train()