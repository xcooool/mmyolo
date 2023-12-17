# Copyright (c) OpenMMLab. All rights reserved.
"""Perform MMYOLO inference on a video as:

```shell
wget -P checkpoint https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth # noqa: E501, E261.

python demo/video_demo.py \
    demo/video_demo.mp4 \
    configs/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py \
    checkpoint/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth \
    --out demo_result.mp4
```
"""
import argparse
import math
import os
import cv2
import mmcv
from mmcv.transforms import Compose
from mmdet.apis import inference_detector, init_detector
from mmengine.utils import track_iter_progress
from mmyolo.utils.labelme_utils import LabelmeFormat
from mmyolo.registry import VISUALIZERS
from tqdm import tqdm
import json
from mmdet.apis.inference import  DetDataSample, SampleList
from mmengine.structures import InstanceData
import torch
from BoxGridProcess import  BoxGridProcess
import math

def parse_args():
    parser = argparse.ArgumentParser(description='MMYOLO video demo')
    parser.add_argument('video', help='Video file')
    parser.add_argument('config', help='Config file')
    parser.add_argument(
    '--out-dir', default='../output_debug', help='Path to output file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Bbox score threshold')
    parser.add_argument('--out', type=str, help='Output video file')
    parser.add_argument('--show', action='store_true', help='Show video')
    parser.add_argument('--sim', type=float, default=0.5, help='cal the similarity of box compared with last frame')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=1,
        help='The interval of show (s), 0 is block')
    parser.add_argument(
        '--class-name',
        nargs='+',
        type=str,
        help='Only Save those classes if set')
    args = parser.parse_args()
    return args

def eye2sreen(x, y):
    # 将unity中的眼动数据 和 屏幕中的数据对齐
    return float(x) + 1784 / 2, 720 / 2 - float(y)

def point_in_circle(x1, y1, x2, y2, x, y):
    # 计算中点坐标
    mx = (x1 + x2) / 2
    my = (y1 + y2) / 2

    # 计算半径
    r = 280

    # 计算待判断点与圆心的距离
    d = math.sqrt((x - mx) ** 2 + (y - my) ** 2)

    # 判断点是否在圆内
    if d <= r:
        return True
    else:
        return False

from skimage.metrics import structural_similarity as ssim

def calculate_similarity(matrix1, matrix2):
    # 计算SSIM，显式设置win_size和data_range
    # 返回值的范围是[-1,1]
    similarity_index = ssim(matrix1, matrix2, multichannel=True, win_size=3, data_range=255)

    similarity_index = (similarity_index + 1) / 2

    return similarity_index

def is_midpoint_inside_rectangle(x1, y1, x2, y2, rect_x1, rect_y1, rect_x2, rect_y2):
    # 计算两点直线的中点
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2

    # 判断中点是否在矩形内部
    return rect_x1 <= mid_x <= rect_x2 and rect_y1 <= mid_y <= rect_y2
def cal_close(cur_rgb_box, last_rgb_box):
    x11 = cur_rgb_box[0]
    y11 = cur_rgb_box[1]
    x12 = cur_rgb_box[2]
    y12 = cur_rgb_box[3]
    x21 = last_rgb_box[0]
    y21 = last_rgb_box[1]
    x22 = last_rgb_box[2]
    y22 = last_rgb_box[3]
    mid_x1 = (x11 + x12) / 2
    mid_y1 = (y11 + y12) / 2
    mid_x2 = (x21 + x22) / 2
    mid_y2 = (y21 + y22) / 2
    distance = math.sqrt((mid_x1 - mid_x2)**2 + (mid_y1 - mid_y2)**2)
    return max(0, min(1, distance / math.sqrt((x21 - x22)**2 + (y21 - y22)**2))), is_midpoint_inside_rectangle(
        x11,y11,x12,y12,
        x21,y21,x22,y22
    )


def normalize_distance(distance, min_distance, max_distance):
    normalized_distance = (distance - min_distance) / (max_distance - min_distance)
    return normalized_distance


def integrate_parameters(rgb_similarity, normalized_distance, alpha):
    # 使用权重alpha整合两个参数
    integrated_value = alpha * rgb_similarity + (1 - alpha) * (1 - normalized_distance)

    # 将整合后的值映射到[0, 1]的范围
    integrated_normalized = max(0, min(1, integrated_value))

    return integrated_normalized
def main():
    args = parse_args()
    assert args.out or args.show, \
        ('Please specify at least one operation (save/show the '
         'video) with the argument "--out" or "--show"')
    cur_dir = os.getcwd()
    print(cur_dir)
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    # build test pipeline
    model.cfg.test_dataloader.dataset.pipeline[
        0].type = 'mmdet.LoadImageFromNDArray'
    test_pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_detector
    visualizer.dataset_meta = model.dataset_meta



     # get model class name
    dataset_classes = model.dataset_meta.get('classes')

    # ready for labelme format if it is needed
    to_label_format = LabelmeFormat(classes=dataset_classes)

    if args.class_name is not None:
        for class_name in args.class_name:
            if class_name in dataset_classes:
                continue
            show_data_classes(dataset_classes)
            raise RuntimeError(
                'Expected args.class_name to be one of the list, '
                f'but got "{class_name}"')


    video_reader = mmcv.VideoReader(args.video)
    video_writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            args.out, fourcc, 50,
            (video_reader.width, video_reader.height))
    # 读取眼动数据
    eye_data = []
    eye_circles = []
    with open('data_1_有驾照.txt', 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                data = {}
                elements = line.split(',')
                data['frame'] = elements[0].strip()
                data['x'] = elements[1].strip()
                data['y'] = elements[2].strip()
                x = float(data['x']) + 1784 / 2
                y = 720 /2 - float(data['y'])
                eye_circles.append([x, y])
                eye_data.append(data)

    straight = 0 # 直行过程中正视前方的帧数
    straight_interval = [0, 120] # 0-120帧：直行的时间起始

    traffic_light = 0 # 有红绿灯时看红绿灯的帧数
    traffic_light_interval = [1080, 1200] # 13~15s：看到红绿灯的帧数

    # 储存的上一帧的所有物体和其rgb_box， 每次都和上一帧比较， 所以会快很多
    this_frame = {}
    label2count = {}
    all_frames = {}
    for idx in tqdm(range(len(video_reader))):
        # if idx == 100:
        #     break
        last_frame = this_frame
        this_frame = {}
        # --------------------------------------读取帧-----------------------------------
        frame = video_reader[idx]
        # 不现场推理了 ，直接从文件夹读取json文件
        json_file = f'../output_driving/{idx}_frame.json'
        # Load the JSON data
        with open(json_file, 'r') as f:
            json_data = json.load(f)

        shapes = json_data['shapes']
        if len(shapes) == 0:
            continue
        image_height = json_data['imageHeight']
        image_width = json_data['imageWidth']

        # Create a DetDataSample object
        data_sample = DetDataSample()
        data_sample.img_height = image_height
        data_sample.img_width = image_width

        # --------------------------------------解析帧-----------------------------------
        labels = []
        bboxes = []
        scores = []
        count = [] # 当前实例第几个

        # 判断直行过程，由于直行过程的满足率和目标检测没有关系，所以在for shape循环外层
        if idx >= straight_interval[0] and idx <= straight_interval[1]:
            x, y = eye2sreen(eye_data[idx]['x'], eye_data[idx]['y'])
            if 600 <= x <= 1200 and 260 <= y <= 460:
                straight += 1

        # 逐个分析这一帧中的instance
        box_processor = BoxGridProcess(frame)
        for shape in shapes:
            label = shape['label']
            points = shape['points']
            x_min = points[0][0]
            y_min = points[0][1]
            x_max = points[1][0]
            y_max = points[1][1]

            bbox = [x_min, y_min, x_max, y_max]

            # ----------------------------2023.12.16 START----------------------------
            # 和上一帧中的物体计算外形的相似度，超过某一个阈值就判定是一个物体，最外层用一个hashmap储存对应的实例名称和其外形
            # box_rgb_means: [10 * 10 * 3]
            box_rgb_means = box_processor.process_box(x_min, y_min, x_max, y_max)
            # 和上一帧中的所有物体比较
            exit = False # 如果上一帧存在和类似的Box
            max_sim = 0
            max_instance_label = ""

            for instance_label, instance_rgb_box in last_frame.items():
                # 外形RGB的相似程度
                rgb_similarity = calculate_similarity(box_rgb_means[4], instance_rgb_box[4])
                # 位移  越小越好
                distance, in_box = cal_close(box_rgb_means, instance_rgb_box)
                # 再加个alpha，用于权衡两个参数的比重 ， 但是inbox在与上一帧比较的过程中起到决定性作用
                similarity = integrate_parameters(rgb_similarity, distance,  0.3) if in_box == False else 1
                a = instance_label.split("_") # a[0]:label a[1]:count这个label的个数
                if similarity > args.sim and similarity >= max_sim and a[0] == label:
                    max_sim = similarity
                    max_instance_label = instance_label
                    exit = True
            if exit == True: # 说明上一帧存在当前实例
                this_frame[max_instance_label] = box_rgb_means
                all_frames[max_instance_label] = box_rgb_means # 时时更新all_frames中的实例
                a = max_instance_label.split("_")
                count.append(int(a[1]))
                scores.append(max_sim)
            else: # 说明上一帧不存在这一个实例
                # 从all_frames匹配外形
                all_max_sim = 0
                all_max_instance_label = ""
                flag = False
                for instance_label, instance_rgb_box in all_frames.items():
                    # 外形RGB的相似程度
                    rgb_similarity = calculate_similarity(box_rgb_means[4], instance_rgb_box[4])
                    distance, in_box = cal_close(box_rgb_means, instance_rgb_box)
                    # inbox不起到决定性作用，因为是和之前所有的实例相比，所以没有时间连续性，要说有，也可能是中途几帧没有识别出来
                    similarity = integrate_parameters(rgb_similarity, distance,  0.3) if in_box == True else 0
                    a = instance_label.split("_")  # a[0]:label a[1]:count这个label的个数
                    if similarity > args.sim  and similarity >= max_sim and a[0] == label:
                        all_max_sim = similarity
                        all_max_instance_label = instance_label
                        flag = True
                if flag: # 说明过去的帧中存在改实例
                    this_frame[all_max_instance_label] = box_rgb_means
                    all_frames[all_max_instance_label] = box_rgb_means
                    a = all_max_instance_label.split("_")
                    count.append(int(a[1]))
                    scores.append(all_max_sim)
                else: # 过去的帧中也没有当前实例，所以要重新添加
                    if label in label2count:
                        update_count = label2count[label] + 1
                        label2count[label] = update_count
                    else:
                        label2count[label] = 0
                    all_frames[f"{label}_{label2count[label]}"] = box_rgb_means
                    this_frame[f"{label}_{label2count[label]}"] = box_rgb_means
                    count.append(label2count[label])
                    scores.append(1)
            # ----------------------2023.12.16 END------------------------------

            bboxes.append(bbox)
            labels.append(dataset_classes.index(label))
            # scores.append(1)

            # # 如果眼睛注视点在矩形里面
            # if x_min <= float(eye_data[idx]['x']) + 1784 / 2 <= x_max and y_min <= 720 / 2 - float(eye_data[idx]['y']) <= y_max:
            #     bboxes.append(bbox)
            #     labels.append(dataset_classes.index(label))
            #     scores.append(1)
            # # 看到红绿灯，计算满足率
            # if idx >= traffic_light_interval[0] and idx <= traffic_light_interval[1]:
            #     x, y = eye2sreen(eye_data[idx]['x'], eye_data[idx]['y'])
            #     # print(f"labgel:{label}, x={x},y={y}, x1={x_min}, y1={y_min}, x2={x_max}, y2={y_max}")
            #     if label == "traffic light" and point_in_circle(x_min,y_min,x_max,y_max, x, y):
            #         traffic_light += 1
            #         print("----in the circle----")


        # --------------------------------可视化---------------------------------
        count_tensor = torch.tensor(count)
        labels_tensor = torch.tensor(labels)
        bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32)
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        # 比较特殊，因为按道理label,bbox,score的长度都是相等的，但是eye只有一个，所以用0补全
        if len(bboxes) == 1:
            eye_tensor = torch.tensor([eye_circles[idx]], dtype=torch.float32)
        else:
            eye_list = [eye_circles[idx]]
            for _ in range(len(bboxes) - 1):
                eye_list.append([0,0])
            eye_tensor = torch.tensor(eye_list, dtype=torch.float32)
        # Set the tensors in pred_instances
        pred_instances = InstanceData()
        pred_instances['bboxes'] = bboxes_tensor
        pred_instances['labels'] = labels_tensor
        pred_instances['count'] = count_tensor
        pred_instances['scores'] =  scores_tensor   # Replace with the appropriate metainfo tensor
        pred_instances['eyecircle'] = eye_tensor
        # Create a SampleList to store the pred_instances
        # sample_list = [pred_instances]

        # Set the SampleList in the DetDataSample object
        data_sample.pred_instances = pred_instances
        # -----------------------------------------

        # -----------------------------------------
        visualizer.add_datasample(
            name='video',
            image=frame,
            data_sample=data_sample,
            draw_gt=False,
            show=False,
            pred_score_thr=0.0)
        frame = visualizer.get_image()

        if args.show:
            cv2.namedWindow('video', 0)
            mmcv.imshow(frame, 'video', args.wait_time)
        if args.out:
            video_writer.write(frame)

    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    # print(f"直行的4s之内, 有{straight / 30 :.2f} 秒正视前方.")
    # print(f"红绿灯期间，有{traffic_light / 30 :.2f} 秒注视着红绿灯.")

if __name__ == '__main__':
    main()
