import argparse
import copy
import json
import os
import sys

try:
    import apex
except:
    print("No APEX!")
import numpy as np
import torch
import yaml
from det3d import torchie
from det3d.datasets import build_dataloader, build_dataset
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis import (
    batch_processor,
    build_optimizer,
    get_root_logger,
    init_dist,
    set_random_seed,
    train_detector,
)
from det3d.torchie.trainer import get_dist_info, load_checkpoint
from det3d.torchie.trainer.utils import all_gather, synchronize
from torch.nn.parallel import DistributedDataParallel
import pickle 
import time 

import onnxruntime


FEATURE_NUM = 10
MAX_PILLARS = 30000
MAX_PIONT_IN_PILLARS = 20
BEV_W = 512
BEV_H = 512
OUTPUT_W = 128
OUTPUT_H = 128

X_MIN = -51.2
X_MAX = 51.2
Y_MIN = -51.2
Y_MAX = 51.2
Z_MIN = -5.0
Z_MAX = 3.0
X_STEP = 0.2
Y_STEP = 0.2

TASK_NUM = 6
SCORE_THREAHOLD = 0.1
OUT_SIZE_FACTOR = 4.0

NMS_PRE_MAX_SIZE = 1000
NMS_POST_MAX_SIZE = 83
NMS_THREAHOLD = 0.2


def save_pred(pred, root):
    with open(os.path.join(root, "prediction.pkl"), "wb") as f:
        pickle.dump(pred, f)


def convert_box(info):
    boxes =  info["gt_boxes"].astype(np.float32)
    names = info["gt_names"]
    assert len(boxes) == len(names)
    detection = {}
    detection['box3d_lidar'] = boxes
    # dummy value 
    detection['label_preds'] = np.zeros(len(boxes)) 
    detection['scores'] = np.ones(len(boxes))
    return detection


##############################
#        pre-process
##############################
def preprocess(points):
    """
    Preprocess lidar points into avilable model inputs.
    
    :params
        points: np.ndarray, [num_points, dim==5]
    :return
        feature: np.ndarray, [dim==10, num_pillars, num_points_per_pillar]
        indices: np.ndarray, [num_pillars, 2]
    """
    feature = np.zeros((FEATURE_NUM, MAX_PILLARS, MAX_PIONT_IN_PILLARS), dtype=np.float32)
    indices = np.zeros((MAX_PILLARS, 2), dtype=np.int64)
    
    # delete points out of range
    range_mask = \
        (points[:, 0]>X_MIN)*(points[:, 0]<X_MAX) * \
        (points[:, 1]>Y_MIN)*(points[:, 1]<Y_MAX) * \
        (points[:, 2]>Z_MIN)*(points[:, 2]<Z_MAX)
    points = points[range_mask]
    
    # get pillars by BEV idx
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    xIdx = ((x-X_MIN)/X_STEP).astype(np.int64)
    yIdx = ((y-Y_MIN)/Y_STEP).astype(np.int64)
    pillarIdx = yIdx*BEV_W + xIdx
    argsort_pillarIdx = pillarIdx.argsort()
    points = points[argsort_pillarIdx]
    pillarIdx = pillarIdx[argsort_pillarIdx]
    _, pointIdx = np.unique(pillarIdx, return_index=True)
    pillars = np.array(np.split(points, pointIdx[1:]), dtype=object)
    
    # sort pillars by number of points
    pointNum = np.array([pillar.shape[0] for pillar in pillars])
    argsort_pointNum = (-pointNum).argsort()
    pillars = pillars[argsort_pointNum][:MAX_PILLARS]
    pointNum = np.minimum(pointNum[argsort_pointNum][:MAX_PILLARS], MAX_PIONT_IN_PILLARS)
    pointCount = np.zeros((MAX_PILLARS), dtype=np.int64)
    pointCount[:pointNum.shape[0]] = pointNum
    
    # compute model inputs
    for i, pillar in enumerate(pillars):
        pillar_points = pillar[:MAX_PIONT_IN_PILLARS]
        num_points = pointCount[i]
        
        x, y = pillar_points[:, 0], pillar_points[:, 1]
        xIdx = ((x-X_MIN)/X_STEP).astype(np.int64)
        yIdx = ((y-Y_MIN)/Y_STEP).astype(np.int64)
        pillarIdx = yIdx*BEV_W + xIdx
        indices[i, 1] = pillarIdx[0]
        
        feature[:5, i, :num_points] = pillar_points.transpose((1,0))
        feature[8, i, :num_points] = x - (xIdx*X_STEP + X_MIN + X_STEP/2)
        feature[9, i, :num_points] = y - (yIdx*Y_STEP + Y_MIN + Y_STEP/2)
        
    valid_mask = np.arange(MAX_PIONT_IN_PILLARS)[None] < pointCount[:, None]
    centers = (feature[:3] * valid_mask).sum(-1, keepdims=True) / (valid_mask.sum(-1, keepdims=True)+1e-5)
    feature[5:8] = feature[:3] - centers
    feature *= valid_mask

    return feature[None], indices[None]


##############################
#        post-process
##############################
def _rotate_around_center(x, y, cx, cy, theta):
    cos = np.cos(theta)
    sin = np.sin(theta)
    new_x = cx + (x-cx)*cos - (y-cy)*sin
    new_y = cy + (x-cx)*sin + (y-cy)*cos
    return new_x, new_y

def _get_bev_box(box):
    x, y, w, l, theta = \
        box[:, 0], box[:, 1], box[:, 3], box[:, 4], box[:, -1]
    x1, x2 = x - w/2, x + w/2
    y1, y2 = y - l/2, y + l/2
    corners = np.array(
        [[x1, y1], [x1, y2],
         [x2, y1], [x2, y2]]
    )
    
    bev_boxes = np.zeros((corners.shape[-1], 4, 2))
    for i, cor in enumerate(corners):
        bev_boxes[:, i, 0], bev_boxes[:, i, 1] = \
            _rotate_around_center(cor[0], cor[1], x, y, -theta)
    
    return bev_boxes


def aligned_IoU_BEV(box_q, box_seq):
    max_q, min_q = box_q.max(-2), box_q.min(-2)
    max_seq, min_seq = box_seq.max(-2), box_seq.min(-2)
    
    sArea_q = (max_q[0] - min_q[0]) * (max_q[1] - min_q[1])
    sArea_seq = (max_seq[:, 0] - min_seq[:, 0]) * (max_seq[:, 1] - min_seq[:, 1])
    
    sInter_w = np.minimum(max_q[0], max_seq[:, 0]) - np.maximum(min_q[0], min_seq[:, 0])
    sInter_h = np.minimum(max_q[1], max_seq[:, 1]) - np.maximum(min_q[1], min_seq[:, 1])
    
    sInter = np.maximum(sInter_w, 0.0) * np.maximum(sInter_h, 0.0)
    sUnion = sArea_q + sArea_seq - sInter
    
    IoU = sInter / sUnion
    
    return IoU

def aligned_NMS_BEV(box_feature, box_score):
    select_idx = []
    
    bev_boxes = _get_bev_box(box_feature)
    order = (-box_score).argsort()
    order = order[:NMS_PRE_MAX_SIZE]
    
    while len(order) > 0:
        i = order[0]
        select_idx.append(i)
        if len(order) == 1:
            break
        iou = aligned_IoU_BEV(bev_boxes[i], bev_boxes[order[1:]])
        idx = (iou < NMS_THREAHOLD).nonzero()[0]
        if len(idx) == 0:
            break
        order = order[idx+1]
    select_idx = select_idx[:NMS_POST_MAX_SIZE]
    
    return select_idx


def postprocess(ort_outputs):
    """
    refer to det3d/models/bbox_heads/center_head.py
    """
    regName = ["594", "618", "642", "666", "690", "714"]
    heightName = ["598", "622", "646", "670", "694", "718"]
    rotName = ["606", "630", "654", "678", "702", "726"]
    velName = ["610", "634", "658", "682", "706", "730"]
    dimName = ["736", "740", "744", "748", "752", "756"]
    scoreName = ["737", "741", "745", "749", "753", "757"]
    clsName = ["738", "742", "746", "750", "754", "758"]
    clsOffsetPerTask = [0, 1, 3, 5, 6, 8]
    
    final_outputs = {
        "box3d_lidar": [],
        "scores": [],
        "label_preds": []
    }
    
    for taskIdx in range(TASK_NUM):
        reg = ort_outputs[regName[taskIdx]][0]
        height = ort_outputs[heightName[taskIdx]][0]
        rot = ort_outputs[rotName[taskIdx]][0]
        vel = ort_outputs[velName[taskIdx]][0]
        dim = ort_outputs[dimName[taskIdx]][0]
        score = ort_outputs[scoreName[taskIdx]][0]
        cls = ort_outputs[clsName[taskIdx]][0]

        xIdx = np.arange(OUTPUT_W)[None]
        yIdx = np.arange(OUTPUT_H)[:, None]
        x = (xIdx + reg[0])*OUT_SIZE_FACTOR*X_STEP + X_MIN
        y = (yIdx + reg[1])*OUT_SIZE_FACTOR*Y_STEP + Y_MIN
        
        score_mask = (score > SCORE_THREAHOLD)
        range_mask = \
            (x>X_MIN)*(x<X_MAX)* \
            (y>Y_MIN)*(y<Y_MAX)* \
            (height[0]>Z_MIN)*(height[0]<Z_MAX)
        valid_mask = score_mask * range_mask
        
        theta = np.arctan2(rot[0], rot[1])[None]
        box_feature = np.concatenate(
            [x[None], y[None], height, dim, vel, theta], axis=0
        ).transpose((1, 2, 0))
        box_feature = box_feature[valid_mask]
        box_score = score[valid_mask]
        box_cls = cls[valid_mask] + clsOffsetPerTask[taskIdx]
        
        select_idx = aligned_NMS_BEV(
            copy.deepcopy(box_feature),
            copy.deepcopy(box_score)
        )
        
        final_outputs["box3d_lidar"].append(box_feature[select_idx])
        final_outputs["scores"].append(box_score[select_idx])
        final_outputs["label_preds"].append(box_cls[select_idx])
        
    for k, v in final_outputs.items():
        final_outputs[k] = torch.from_numpy(np.concatenate(v, axis=0))

    return final_outputs


##############################
#            MAIN
##############################
def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work_dir", required=True, help="the dir to save logs and models")
    parser.add_argument("--checkpoint", help="the dir to checkpoint which the model read from")
    parser.add_argument("--testset", action="store_true")
    parser.add_argument("--save_data", action="store_true")
    args = parser.parse_args()
    return args


def main():

    # torch.manual_seed(0)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(0)

    args = parse_args()

    cfg = Config.fromfile(args.config)

    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
        os.makedirs(args.work_dir, exist_ok=True)

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)

    if args.testset:
        print("Use Test Set")
        dataset = build_dataset(cfg.data.test)
    else:
        print("Use Val Set")
        dataset = build_dataset(cfg.data.val)

    ort_session = onnxruntime.InferenceSession(
        args.checkpoint,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )

    logger.info(f"work dir: {args.work_dir}")
    prog_bar = torchie.ProgressBar(len(dataset))

    detections = {}

    time_start = time.time() 

    for i, data in enumerate(dataset):
        token = data['metadata']['token']
        
        if args.save_data:
            onnx_data_dir = os.path.join(args.work_dir, "data")
            os.makedirs(onnx_data_dir, exist_ok=True)
            
            if not args.testset:
                info = dataset._nusc_infos[i]
                gt_annos = convert_box(info)
            else:
                gt_annos=None
            
            np.savez(
                os.path.join(onnx_data_dir, f"{token}.npz"),
                points=data["points"].astype(np.float32),
                gt_annos=gt_annos
            )
        
        points=data["points"].astype(np.float32)
        feature, indices = preprocess(points)
        ort_inputs = {"input.1": feature, "indices_input": indices}
        
        output_names = [output.name for output in ort_session.get_outputs()]
        ort_outputs = ort_session.run(output_names, ort_inputs)
        ort_outputs = dict(zip(output_names, ort_outputs))
        
        final_outputs = postprocess(ort_outputs)
        final_outputs["metadata"] = {
            "token": token,
            "num_point_features": 5
        }
        
        detections.update({token: final_outputs})
        
        prog_bar.update()

    time_end = time.time()

    print("\n Total time per frame: ", (time_end -  time_start) / len(dataset))

    all_predictions = all_gather(detections)
    predictions = {}
    for p in all_predictions:
        predictions.update(p)

    save_pred(predictions, args.work_dir)

    result_dict, _ = dataset.evaluation(copy.deepcopy(predictions), output_dir=args.work_dir, testset=args.testset)

    if result_dict is not None:
        for k, v in result_dict["results"].items():
            print(f"Evaluation {k}: {v}")

if __name__ == "__main__":
    main()
