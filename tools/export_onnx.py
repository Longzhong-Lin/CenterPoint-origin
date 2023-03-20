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
from torch import nn
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
    example_to_device,
)
from det3d.torchie.trainer import load_checkpoint
import pickle 
import time 
from matplotlib import pyplot as plt 
from det3d.torchie.parallel import collate, collate_kitti
from torch.utils.data import DataLoader
import matplotlib.cm as cm
import subprocess
import cv2
from tools.demo_utils import visual 
from collections import defaultdict


class Decoder(nn.Module):
    def __init__(self,model):
        super(Decoder, self).__init__()
        self.model = model
    
    def forward(self, x):
        x = self.model.neck(x)
        preds, _ = self.model.bbox_head(x)
        for task in range(len(preds)):
            hm_preds = torch.sigmoid(preds[task]['hm'])
            preds[task]['dim'] = torch.exp(preds[task]['dim'])
            scores, labels = torch.max(hm_preds, dim=1)
            preds[task]["hm"] = (scores, labels)

        return preds


def parse_args():
    parser = argparse.ArgumentParser(description="Export onnx")
    parser.add_argument(
        "--config", help="train config file path",
        default='configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep_mini_onnx.py'
    )
    parser.add_argument(
        "--work_dir", help="the dir to save logs and models",
        default='work_dirs/centerpoint_pillar_pretrain/onnx'
    )
    parser.add_argument(
        "--checkpoint", help="the dir to checkpoint which the model read from",
        default='work_dirs/centerpoint_pillar_pretrain/latest.pth'
    )
    args = parser.parse_args()
    return args


def main(args):
    
    os.makedirs(args.work_dir, exist_ok=True)
    
    cfg = Config.fromfile(args.config)
    
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = model.cuda()
    model.eval()
    model_decoder = Decoder(model)

    dataset = build_dataset(cfg.data.val)
    data_loader = build_dataloader(
        dataset,
        batch_size=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )
    data_iter = iter(data_loader)
    data_batch = next(data_iter)
    
    gpu_device = torch.device("cuda")

    with torch.no_grad():
        example = example_to_device(data_batch, gpu_device, non_blocking=False)
        example["voxels"] = torch.zeros((example["voxels"].shape[0],example["voxels"].shape[1],10),dtype=torch.float32,device=gpu_device)
        example.pop("metadata")
        example.pop("points")
        example["shape"] = torch.tensor(example["shape"], dtype=torch.int32, device=gpu_device)
        torch.onnx.export(
            model.reader, (example["voxels"],example["num_voxels"],example["coordinates"]),
            os.path.join(args.work_dir, "encoder.onnx"), opset_version=11
        )
        
        rpn_input = torch.zeros((1,64,512,512), dtype=torch.float32, device=gpu_device)
        torch.onnx.export(
            model_decoder, rpn_input,
            os.path.join(args.work_dir, "decoder.onnx"), opset_version=11
        )
        
    print("==================Onnx Exported==================")


if __name__ == "__main__":
    args = parse_args()
    main(args)
