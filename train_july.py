import json
import os
import sys
import time
import numpy as np
import cv2
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn

# ---------------------------------------------------------------------------
# 1. Dataset Definition
# ---------------------------------------------------------------------------
class COCODataset(Dataset):
    def __init__(self, json_path, image_dir):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.image_dir = image_dir
        self.images = self.data["images"]
        self.annotations = self.data["annotations"]

        # map image_id -> annotations
        self.ann_map = {}
        for ann in self.annotations:
            self.ann_map.setdefault(ann["image_id"], []).append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.image_dir, img_info["file_name"])

        # Read image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Image not found: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]

        anns = self.ann_map.get(img_info["id"], [])

        boxes = []
        labels = []
        masks = []

        for ann in anns:
            # Skip invalid segmentation
            if "segmentation" not in ann or len(ann["segmentation"]) == 0:
                continue

            # BOX -> convert [x,y,w,h] -> [x1,y1,x2,y2]
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])

            # Labels
            labels.append(ann["category_id"])

            # MASK from polygon
            mask = np.zeros((height, width), dtype=np.uint8)
            for seg in ann["segmentation"]:
                pts = np.array(seg).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [pts], 1)

            masks.append(mask)

        # Handle empty cases
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, height, width), dtype=torch.uint8)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks
        }

        # Convert image to tensor [C, H, W] normalized to [0, 1]
        img = torch.as_tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

        return img, target

# ---------------------------------------------------------------------------
# 2. Model Architecture
# ---------------------------------------------------------------------------
def get_model(num_classes):
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")

    # box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )

    # mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask, 256, num_classes
    )

    return model

def collate_fn(batch):
    return tuple(zip(*batch))

# ---------------------------------------------------------------------------
# 3. Training Execution
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Starting Mask R-CNN Training on July Dataset")
    print("=" * 60)

    # Previous paths (kept as comments):
    # train_dataset = COCODataset("my_merged_coco_dataset/merged_dataset.json", "my_merged_coco_dataset/images")
    
    # New July Dataset paths:
    json_path = "july/instances_default.json"
    image_dir = "july-images"
    output_model_path = "mask_rcnn_model_july.pth"

    print(f"Annotation file: {json_path}")
    print(f"Images directory: {image_dir}")
    print(f"Target model output: {output_model_path}")

    train_dataset = COCODataset(json_path, image_dir)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 7 classes: 0 = __background__, 1..6 = foreground classes
    model = get_model(num_classes=7)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    num_epochs = 25
    total_batches = len(train_loader)
    print(f"Total Epochs: {num_epochs} | Batches per epoch: {total_batches}")
    print("-" * 60)

    start_training_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0.0

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == total_batches:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{total_batches}] - Current Batch Loss: {loss.item():.4f}")

        lr_scheduler.step()
        epoch_duration = time.time() - epoch_start_time
        avg_epoch_loss = total_loss / total_batches
        print(f"--> Epoch {epoch+1} Completed in {epoch_duration:.1f}s | Total Loss = {total_loss:.4f} | Avg Batch Loss = {avg_epoch_loss:.4f}")

        # Save checkpoint after each epoch
        # Previous model save path:
        # torch.save(model.state_dict(), "mask_rcnn_model.pth")
        torch.save(model.state_dict(), output_model_path)
        print(f"    [Checkpoint saved to {output_model_path}]")
        print("-" * 60)

    total_time = time.time() - start_training_time
    print(f"Training finished in {total_time/60:.2f} minutes.")
    print(f"Final model saved to {output_model_path}")

if __name__ == "__main__":
    main()
