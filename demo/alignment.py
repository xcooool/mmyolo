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
    for idx in tqdm(range(len(video_reader))):
        # --------------------------------------读取帧-----------------------------------
        frame = video_reader[idx]
        # 不现场推理了 ，直接从文件夹读取json文件
        json_file = f'../output_driving/{idx}_frame.json'
        # Load the JSON data
        with open(json_file, 'r') as f:
            json_data = json.load(f)

        # Extract necessary information from the JSON data
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

        # 判断直行过程，由于直行过程的满足率和目标检测没有关系，所以在for shape循环外层
        if idx >= straight_interval[0] and idx <= straight_interval[1]:
            x, y = eye2sreen(eye_data[idx]['x'], eye_data[idx]['y'])
            if 600 <= x <= 1200 and 260 <= y <= 460:
                straight += 1

        for shape in shapes:
            label = shape['label']
            points = shape['points']
            x_min = points[0][0]
            y_min = points[0][1]
            x_max = points[1][0]
            y_max = points[1][1]

            bbox = [x_min, y_min, x_max, y_max]
            # 和上一帧中的物体计算外形的相似度，超过某一个阈值就判定是一个物体，最外层用一个hashmap储存对应的实例名称和其外形

            # 如果眼睛注视点在矩形里面
            if x_min <= float(eye_data[idx]['x']) + 1784 / 2 <= x_max and y_min <= 720 / 2 - float(eye_data[idx]['y']) <= y_max:
                bboxes.append(bbox)
                labels.append(dataset_classes.index(label))
                scores.append(1)
            # 看到红绿灯，计算满足率
            if idx >= traffic_light_interval[0] and idx <= traffic_light_interval[1]:
                x, y = eye2sreen(eye_data[idx]['x'], eye_data[idx]['y'])
                # print(f"labgel:{label}, x={x},y={y}, x1={x_min}, y1={y_min}, x2={x_max}, y2={y_max}")
                if label == "traffic light" and point_in_circle(x_min,y_min,x_max,y_max, x, y):
                    traffic_light += 1
                    print("----in the circle----")


        # --------------------------------可视化---------------------------------
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
        # break

    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    print(f"直行的4s之内, 有{straight / 30 :.2f} 秒正视前方.")
    print(f"红绿灯期间，有{traffic_light / 30 :.2f} 秒注视着红绿灯.")

if __name__ == '__main__':
    main()
