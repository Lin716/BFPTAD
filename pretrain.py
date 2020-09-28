import time
import torch.optim as optim
import torch.utils.data as data
from eval import *
from utils.utils import *
from evaluation.eval_detection import  ANETdetection
import pandas as pd
def train(cfg):
    '''
    训练
    :param cfg:
    :return:
    '''
    #构建存储文件夹
    if not os.path.exists(cfg.ckpt_path):
        os.makedirs(cfg.ckpt_path)


    #读取训练数据
    train_loader = torch.utils.data.DataLoader(THUMOSDetectionDataset(cfg, subset="train"),
                                               batch_size=cfg.batch, shuffle=True, num_workers=4)

    #验证数据
    # val_dataloader = torch.utils.data.DataLoader(THUMOSDetectionDataset(cfg, subset="test"),
    #                                            batch_size=cfg.batch, shuffle=False, num_workers=4)

    # 测试数据
    cfg.is_training = False
    cfg.clip_overlap = 0.5
    test_data = THUMOSDetectionDataset(cfg, subset="test")
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    train_label = get_dict_from_pkl(cfg.label_file)
    train_iter_num = int(len(list(train_label)) / cfg.batch)
    print("train_iter_num",train_iter_num)


    # 构建模型/优化器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    #测试map结果存放的地方
    result_map = []

    for epoch in range(3):
        train_one_epoch(train_loader,model,optimizer,epoch,train_iter_num)
        if epoch >= 0 and (epoch+1) % 1 == 0:
            evaluate_one_epoch(cfg,test_data,test_loader,model,epoch,result_map)
    print(result_map)
    new_df = pd.DataFrame(result_map,columns=['tiou','0.1','0.2','0.3','0.4','0.5','0.6','0.7'])
    new_df.to_csv('./result.csv',index=False)


def train_one_epoch(dataloder,model,optimizer,epoch,train_iter_num):
    start = time.time()
    model.train()
    epoch_loss = 0
    epoch_pos = 0
    epoch_neg = 0
    epoch_loc = 0


    for n_iter,(input_data,cls_label,loc_label,iou_label,clip_info) in enumerate(dataloder):

        predictions,cls_logits,loc_logits,endpoint = model(input_data)

        loss = compute_loss_no_ts(cfg,cls_logits,loc_logits,cls_label,loc_label,iou_label)
        #反向传播
        optimizer.zero_grad()
        loss[0].backward()
        optimizer.step()

        #输出迭代的loss以及整个epoch的loss
        if((n_iter+1) % 50 == 0):
            print("iter: %d, loss:%.5f, pos:%.5f, neg:%.5f, loc:%.5f" % (
                n_iter + 1, loss[0], loss[1], loss[2],loss[3]))

        epoch_loss += loss[0].cpu().detach().numpy()
        epoch_pos += loss[1].cpu().detach().numpy()
        epoch_neg += loss[2].cpu().detach().numpy()
        epoch_loc += loss[3].cpu().detach().numpy()


    duration = time.time()-start
    print("Training epoch: %d, loss:%.5f, pos:%.5f, neg:%.5f, loc:%.5f, time:%.3f" %(
        epoch+1,epoch_loss / (train_iter_num+1), epoch_pos / (train_iter_num+1),epoch_neg / (train_iter_num+1),epoch_loc / (train_iter_num+1), duration))

    #保存模型
    if epoch >= 0 and (epoch+1) % 1 == 0:
        state = {'epoch': epoch + 1, 'state_dict': model.state_dict()}
        torch.save(state, cfg.ckpt_path + "/model_epoch_%d.pkl" % (epoch + 1))



def evaluate_one_epoch(cfg,data,dataloader,model,epoch,result_map):

    start = time.time()

    #加载模型
    ckpt = "%s/model_epoch_%d.pkl" % (cfg.ckpt_path,epoch+1)

    #整个数据集得到得到检测的结果 result
    eval(cfg,data,dataloader,model,ckpt)

    repath = "%s/res_fps_25_epoch_%d.txt" % (cfg.detection_res,epoch+1)
    resultTojson(cfg,repath)


    #计算map ,如果检测得到的map要大的话就保存一个best model 一般0.5下大的话，其他结果也比较好
    tious = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]
    anet_detection = ANETdetection(ground_truth_filename='./evaluation/thumos_gt.json',
                                   prediction_filename=cfg.detection_res+"/json/res_fps_25_epoch_%d.json" % (epoch+1),subset='test',tiou_thresholds=tious)
    mAPs,average_mAP = anet_detection.evaluate()
    for (tiou,mAP) in zip(tious,mAPs):
        print("mAP at tIOU {} is {}".format(tiou,mAP))

    item = [ckpt]+mAPs.tolist()
    result_map.append(item)


    duration = time.time() - start
    print("total time:",duration)


''' 原始的部分 利用测试的进行验证'''

    # epoch_loss = 0
    # epoch_pos = 0
    # epoch_neg = 0
    # epoch_loc = 0
    #
    # for n_iter,(input_data,cls_label,loc_label,iou_label,clip_info) in enumerate(dataloader):
    #
    #     # input_data = input_data.cuda()
    #     # cls_label = cls_label.cuda()
    #     # loc_label = loc_label.cuda()
    #     # iou_label  = iou_label.cuda()
    #
    #     with torch.no_grad():
    #
    #     predictions, cls_logits, loc_logits, endpoint = model(input_data)
    #     loss = compute_loss_no_ts(cfg, cls_logits, loc_logits, cls_label, loc_label, iou_label)
    #
    #     epoch_loss += loss[0].cpu().detach().numpy()
    #     epoch_pos += loss[1].cpu().detach().numpy()
    #     epoch_neg += loss[2].cpu().detach().numpy()
    #     epoch_loc += loss[3].cpu().detach().numpy()
    # duration = time.time() - start
    # print(" Test epoch: %d, loss:%.5f, pos:%.5f, neg:%.5f, loc:%.5f, time:%.3f" % (
    #     epoch + 1, epoch_loss / test_iter_num, epoch_pos / test_iter_num, epoch_neg / test_iter_num,
    #     epoch_loc / test_iter_num, duration))
    # global best_loss
    # if epoch_loss < best_loss:
    #     best_loss = epoch_loss
    #     torch.save(state, cfg.ckpt_path + "/model_best.pkl")



if __name__ == '__main__':
    cfg = Config(batch=4, is_training=True, clip_overlap=0.75)
    train(cfg)
    # # test(cfg)
    # test_data = THUMOSDetectionDataset(cfg, subset="test")
    # test_loader = torch.utils.data.DataLoader(test_data,
    #                                           batch_size=1, shuffle=False)
    # model = Model(cfg)
    # # ckpt = "%s/model_epoch_75.pkl" % cfg.ckpt_path
    # evaluate_one_epoch(cfg,test_data,test_loader,model,74)