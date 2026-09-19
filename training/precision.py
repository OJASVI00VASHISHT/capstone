import os
import sys
import json
import time
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from torchmetrics.detection import MeanAveragePrecision

# ---------------------------------------------------------------------------
# 1. Configuration & Label Mapping
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_IMG_DIR = "val_anu"
VAL_JSON_PATH = "instances_default.json"

MODELS = {
    "Previous Model": "mask_rcnn_model.pth",
    "July Model": "mask_rcnn_model_july.pth",
    "2000-Image Model": "mask_rcnn_model_2000.pth"
}

# Model 6-class schema:
MODEL_CLASSES = {
    1: "full_house",
    2: "wall",
    3: "road",
    4: "tree",
    5: "car",
    6: "shop"
}

# ---------------------------------------------------------------------------
# 2. Model Builder
# ---------------------------------------------------------------------------
def get_model(model_path, num_classes=7):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask, 256, num_classes
    )

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print(f"Loaded weights from {model_path}")
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model.to(DEVICE)
    model.eval()
    return model

# ---------------------------------------------------------------------------
# 3. Ground Truth Loader with Automatic Class Remapping
# ---------------------------------------------------------------------------
def load_val_ground_truth(val_json_path, val_img_dir):
    with open(val_json_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    raw_cat_map = {cat["id"]: cat["name"] for cat in val_data.get("categories", [])}
    name_to_model_id = {v: k for k, v in MODEL_CLASSES.items()}
    val_to_model_cat = {}
    for raw_id, name in raw_cat_map.items():
        if name in name_to_model_id:
            val_to_model_cat[raw_id] = name_to_model_id[name]

    ann_by_img = {}
    for ann in val_data.get("annotations", []):
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    targets = []
    image_list = []

    for img_info in val_data.get("images", []):
        img_filename = img_info["file_name"]
        img_path = os.path.join(val_img_dir, img_filename)
        
        if not os.path.exists(img_path):
            continue

        height = img_info["height"]
        width = img_info["width"]
        anns = ann_by_img.get(img_info["id"], [])

        boxes = []
        labels = []
        masks = []

        for ann in anns:
            raw_cat_id = ann["category_id"]
            if raw_cat_id not in val_to_model_cat:
                continue

            mapped_label = val_to_model_cat[raw_cat_id]
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue

            boxes.append([x, y, x + w, y + h])
            labels.append(mapped_label)

            mask = np.zeros((height, width), dtype=np.uint8)
            for seg in ann.get("segmentation", []):
                pts = np.array(seg).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [pts], 1)
            masks.append(mask)

        if len(boxes) == 0:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks": torch.zeros((0, height, width), dtype=torch.uint8)
            }
        else:
            target = {
                "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
                "masks": torch.as_tensor(np.array(masks), dtype=torch.uint8)
            }

        targets.append(target)
        image_list.append((img_path, (height, width)))

    return image_list, targets

# ---------------------------------------------------------------------------
# 4. Evaluation Runner
# ---------------------------------------------------------------------------
def compute_mean_iou(preds, targets):
    total_iou = 0.0
    matched_count = 0

    for pred, target in zip(preds, targets):
        p_boxes = pred["boxes"].cpu()
        t_boxes = target["boxes"].cpu()
        if len(p_boxes) == 0 or len(t_boxes) == 0:
            continue

        ious = torchvision.ops.box_iou(p_boxes, t_boxes)
        max_iou_per_gt = ious.max(dim=0).values
        total_iou += max_iou_per_gt.sum().item()
        matched_count += len(t_boxes)

    return total_iou / matched_count if matched_count > 0 else 0.0

def evaluate_model(model, image_list, targets, score_thresh=0.05):
    map_metric_bbox = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    map_metric_segm = MeanAveragePrecision(iou_type="segm", class_metrics=True)

    formatted_preds = []
    formatted_targets = []

    print(f"Running inference across {len(image_list)} images...")
    t0 = time.time()

    with torch.no_grad():
        for i, (img_path, (orig_h, orig_w)) in enumerate(image_list):
            img_bgr = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_tensor = torch.as_tensor(img_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

            pred = model(img_tensor)[0]

            keep = pred["scores"] >= score_thresh
            boxes = pred["boxes"][keep].cpu()
            scores = pred["scores"][keep].cpu()
            labels = pred["labels"][keep].cpu()
            masks = (pred["masks"][keep] > 0.5).squeeze(1).to(torch.uint8).cpu()

            formatted_preds.append({
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
                "masks": masks
            })

            target = targets[i]
            formatted_targets.append({
                "boxes": target["boxes"].cpu(),
                "labels": target["labels"].cpu(),
                "masks": target["masks"].cpu()
            })

    elapsed = time.time() - t0
    print(f"Inference completed in {elapsed:.2f}s ({elapsed/len(image_list)*1000:.1f} ms/image)")

    map_metric_bbox.update(formatted_preds, formatted_targets)
    res_bbox = map_metric_bbox.compute()

    map_metric_segm.update(formatted_preds, formatted_targets)
    res_segm = map_metric_segm.compute()

    mean_iou = compute_mean_iou(formatted_preds, formatted_targets)

    return {
        "mean_iou": mean_iou,
        "bbox": {
            "mAP": res_bbox["map"].item(),
            "AP50": res_bbox["map_50"].item(),
            "AP75": res_bbox["map_75"].item(),
            "AR1": res_bbox["mar_1"].item(),
            "AR10": res_bbox["mar_10"].item(),
            "AR100": res_bbox["mar_100"].item(),
            "per_class_AP": {MODEL_CLASSES.get(i+1, f"Class {i+1}"): (res_bbox["map_per_class"][i].item() if i < len(res_bbox["map_per_class"]) else 0.0) for i in range(6)}
        },
        "segm": {
            "mAP": res_segm["map"].item(),
            "AP50": res_segm["map_50"].item(),
            "AP75": res_segm["map_75"].item(),
            "AR1": res_segm["mar_1"].item(),
            "AR10": res_segm["mar_10"].item(),
            "AR100": res_segm["mar_100"].item(),
            "per_class_AP": {MODEL_CLASSES.get(i+1, f"Class {i+1}"): (res_segm["map_per_class"][i].item() if i < len(res_segm["map_per_class"]) else 0.0) for i in range(6)}
        }
    }

# ---------------------------------------------------------------------------
# 5. Main Comparison Execution
# ---------------------------------------------------------------------------
def main():
    print("=" * 95)
    print("PRECISION & RECALL BENCHMARK COMPARISON ON VALIDATION SET (val_anu)")
    print("=" * 95)
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    image_list, targets = load_val_ground_truth(VAL_JSON_PATH, VAL_IMG_DIR)
    print(f"Loaded {len(image_list)} validation samples from {VAL_IMG_DIR}\n")

    all_results = {}

    for name, path in MODELS.items():
        if not os.path.exists(path):
            print(f"Skipping {name} (file {path} not found)")
            continue
        print("-" * 95)
        print(f"Evaluating: {name} ({path})")
        print("-" * 95)
        model = get_model(path)
        all_results[name] = evaluate_model(model, image_list, targets)
        del model
        torch.cuda.empty_cache()
        print()

    # 3. Print Results Comparison Table
    print("\n" + "=" * 95)
    print("COMPARATIVE EVALUATION RESULTS SUMMARY")
    print("=" * 95)

    headers = list(all_results.keys())
    header_str = f"{'Metric':<22} | " + " | ".join([f"{h:<20}" for h in headers])
    print(f"\n{header_str}")
    print("-" * 95)

    # IoU
    iou_str = f"{'Mean Box IoU':<22} | " + " | ".join([f"{all_results[h]['mean_iou']*100:<19.2f}%" for h in headers])
    print(iou_str)
    print("-" * 95)

    # BBox
    print("--- BOUNDING BOX METRICS ---")
    for metric in ["mAP", "AP50", "AP75", "AR1", "AR10", "AR100"]:
        row = f"{('BBox ' + metric):<22} | " + " | ".join([f"{all_results[h]['bbox'][metric]*100:<19.2f}%" for h in headers])
        print(row)

    print("-" * 95)
    print("--- SEGMENTATION MASK METRICS ---")
    for metric in ["mAP", "AP50", "AP75", "AR1", "AR10", "AR100"]:
        row = f"{('Mask ' + metric):<22} | " + " | ".join([f"{all_results[h]['segm'][metric]*100:<19.2f}%" for h in headers])
        print(row)

    print("\n" + "-" * 95)
    print("--- PER-CLASS BBOX AP@50:95 ---")
    for cls_name in MODEL_CLASSES.values():
        row = f"{cls_name:<22} | " + " | ".join([f"{all_results[h]['bbox']['per_class_AP'].get(cls_name, 0.0)*100:<19.2f}%" for h in headers])
        print(row)

    # Save to JSON
    output_file = "precision_comparison.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)
    print(f"\nFull results saved to {output_file}")

if __name__ == "__main__":
    main()
