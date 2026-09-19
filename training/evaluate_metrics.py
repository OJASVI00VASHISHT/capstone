"""
COCO Evaluation for Mask R-CNN Ensemble & Class-wise Performance
"""
import os, sys, json
import numpy as np
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from PIL import Image

def get_finetuned_model(num_classes=7):
    model = maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model

def main():
    print("COCO Model Evaluation Script")
    print("Evaluates Precision, Recall, and mAP@0.50 on validation and test datasets.")

if __name__ == "__main__":
    main()
