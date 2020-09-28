
import json


#将检测的txt转成json共计算map
cat_label_in_UCF101 = {
            7:"BaseballPitch",
            9:"BasketballDunk",
            12:"Billiards",
            21:"CleanAndJerk",
            22:"CliffDiving",
            23:"CricketBowling",
            24:"CricketShot",
            26:"Diving",
            31:"FrisbeeCatch",
            33:"GolfSwing",
            36:"HammerThrow",
            40:"HighJump",
            45:"JavelinThrow",
            51:"LongJump",
            68:"PoleVault",
            79:"Shotput",
            85:"SoccerPenalty",
            92:"TennisSwing",
            93:"ThrowDiscus",
            97:"VolleyballSpiking"
}

def resultTojson(cfg, path):
    dect_dict = {}
    video_name = None
    video_seg = []
    path_split = path.split('.')[-2]

    epoch = int(path_split.split('_')[-1])

    with open(path, 'r') as f:
        for line in f:

            info = line.split(" ")
            cur_video_name = info[0]
            if video_name is None:
                video_name = cur_video_name
            if video_name != cur_video_name:
                dect_dict[video_name] = video_seg
                video_seg = []
                video_name = cur_video_name
            dict = {}
            dict["label"] = cat_label_in_UCF101[int(info[3])]
            dict["score"] = float(info[4])
            start = float(info[1])
            end = float(info[2])
            dict["segment"] = [start, end]
            video_seg.append(dict)
            

    out_dict = {"version":"THUMOS14","results":dect_dict, "external_data": {}}
    with open('%s/json/res_fps_25_epoch_%d.json' % (cfg.detection_res , epoch), "w") as out:
        json.dump(out_dict, out)





if __name__ == '__main__':
    path = 'Res/THUMOS14/res_fps_25_epoch_2.txt'
    resultTojson(path)
