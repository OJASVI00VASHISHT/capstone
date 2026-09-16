"""
Building Height Estimation from Monocular Street-View Imagery
============================================================
Hierarchical Reference-Object Calibrated Pipeline (Prioritizing Ground Anchors)
"""

import os
import sys
import glob
import argparse
import math
import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from PIL import Image
from transformers import pipeline as hf_pipeline
from tqdm import tqdm

DEFAULT_FOV_DEG = 95.0
DEFAULT_HOUSE_CONF = 0.40
DEFAULT_REF_CONF = 0.40
DEFAULT_IOU_THRESH = 0.50

REF_HEIGHTS = {
    "person": 1.70,
    "bicycle": 1.00,
    "car_sedan": 1.50,
    "car_suv": 1.70,
    "motorcycle": 1.10,
    "bus": 3.20,
    "truck": 3.50,
}

COCO_REF_CLASSES = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    6: "bus",
    8: "truck",
}

MIN_REF_HEIGHT_PX = 25
MIN_REF_WIDTH_PX = 12
FALLBACK_Z_MIN_M = 5.0
FALLBACK_Z_MAX_M = 15.0
FALLBACK_HEIGHT_BOOST = 1.20


def get_finetuned_model(num_classes=7):
    model = maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask, 256, num_classes
    )
    return model


def get_coco_model():
    return maskrcnn_resnet50_fpn(weights="DEFAULT")


def compute_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter + 1e-8)


def ensemble_nms(candidates, iou_thresh=0.50):
    if not candidates:
        return []
    s = sorted(candidates, key=lambda c: c["score"], reverse=True)
    kept = []
    while s:
        best = s[0]
        kept.append(best)
        s = [c for c in s[1:] if compute_iou(best["box"], c["box"]) <= iou_thresh]
    return sorted(kept, key=lambda c: c["box"][0])


def ref_nms(detections, iou_thresh=0.50):
    if not detections:
        return []
    by_class = {}
    for d in detections:
        by_class.setdefault(d["ref_class"], []).append(d)
    kept = []
    for _, dets in by_class.items():
        s = sorted(dets, key=lambda c: c["score"], reverse=True)
        cls_kept = []
        while s:
            best = s[0]
            cls_kept.append(best)
            s = [c for c in s[1:] if compute_iou(best["box"], c["box"]) <= iou_thresh]
        kept.extend(cls_kept)
    return kept


def robust_roofline_and_base(d_norm, x1, y1, x2, y2, img_h):
    box_w, box_h = x2 - x1, y2 - y1
    if box_w < 20 or box_h < 20:
        return y1, y2, False, False

    cx1 = x1 + int(0.25 * box_w)
    cx2 = x1 + int(0.75 * box_w)
    mid_y1, mid_y2 = y1 + int(0.30 * box_h), y1 + int(0.60 * box_h)
    facade_sample = d_norm[mid_y1:mid_y2, cx1:cx2]
    facade_d = float(np.median(facade_sample)) if facade_sample.size > 0 else float(d_norm[y1:y2, x1:x2].mean())
    sky_thresh = max(18.0, facade_d * 0.55)

    new_y1 = y1
    y1_strip = d_norm[max(0, y1):min(y1 + 10, y2), cx1:cx2]
    if y1_strip.size > 0 and float(y1_strip.mean()) < sky_thresh:
        for scan_y in range(y1, min(y2, y1 + int(0.5 * box_h))):
            if float(d_norm[scan_y, cx1:cx2].mean()) >= sky_thresh:
                new_y1 = scan_y
                break
    else:
        max_up = min(y1, int(box_h * 0.60))
        for scan_y in range(y1 - 1, max(0, y1 - max_up), -1):
            row_d = float(d_norm[scan_y, cx1:cx2].mean())
            if row_d < sky_thresh or (facade_d - row_d > 35.0):
                break
            new_y1 = scan_y

    new_y2 = y2
    fg_thresh = facade_d * 1.7
    for scan_y in range(y2, y2 + min(img_h - y2, int(box_h * 0.20))):
        row_d = float(d_norm[scan_y, cx1:cx2].mean())
        if row_d > fg_thresh or abs(row_d - facade_d) > 30.0:
            break
        new_y2 = scan_y

    return new_y1, new_y2, (new_y1 <= 15), (new_y1 < y1) or (new_y2 > y2)


def get_ref_height(ref_class, box_w, box_h):
    if ref_class == "car":
        aspect = box_w / max(box_h, 1)
        if aspect > 2.2:
            return REF_HEIGHTS["car_sedan"], "car(sedan)"
        return REF_HEIGHTS["car_suv"], "car(SUV)"
    return REF_HEIGHTS.get(ref_class, 1.50), ref_class


def get_depth_value(d_norm, x1, y1, x2, y2):
    crop = d_norm[y1:y2, x1:x2]
    valid = crop[crop > 15]
    if valid.size > 0:
        return float(np.percentile(valid, 75))
    return float(np.median(crop)) if crop.size > 0 else 50.0


def compute_azimuth_angle_rad(cx, img_w, fov_deg):
    half_w = img_w / 2.0
    half_fov = math.radians(fov_deg / 2.0)
    norm_x = (cx - half_w) / half_w
    return norm_x * half_fov


def fallback_height(h_pixel, d_val, img_w, fov_deg):
    f_pixel = (img_w / 2.0) / np.tan(np.radians(fov_deg / 2.0))
    d_hat = d_val / 255.0
    inv_z = (1.0 / FALLBACK_Z_MAX_M) + d_hat * ((1.0 / FALLBACK_Z_MIN_M) - (1.0 / FALLBACK_Z_MAX_M))
    perp_dist_m = float(np.clip(1.0 / max(inv_z, 1e-4), FALLBACK_Z_MIN_M, FALLBACK_Z_MAX_M))
    return (h_pixel * perp_dist_m) / f_pixel * FALLBACK_HEIGHT_BOOST, perp_dist_m


class BuildingHeightEstimator:
    def __init__(self, m1_path, m2_path, fov_deg=DEFAULT_FOV_DEG, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.fov_deg = fov_deg
        print(f"[Estimator] Loading models on {self.device} (FOV = {self.fov_deg}°)...")

        self.m1 = get_finetuned_model(7)
        self.m1.load_state_dict(torch.load(m1_path, map_location=self.device))
        self.m1.to(self.device).eval()

        self.m2 = get_finetuned_model(7)
        self.m2.load_state_dict(torch.load(m2_path, map_location=self.device))
        self.m2.to(self.device).eval()

        self.m_coco = get_coco_model().to(self.device).eval()

        dev_id = 0 if self.device.type == "cuda" else -1
        self.depth_pipe = hf_pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=dev_id
        )
        print("[Estimator] All models ready.")

    def estimate(self, image_input):
        if isinstance(image_input, str):
            img_bgr = cv2.imread(image_input)
            if img_bgr is None:
                raise ValueError(f"Unable to read image at {image_input}")
        else:
            img_bgr = image_input

        h_img, w_img = img_bgr.shape[:2]
        scale = 1024 / max(h_img, w_img)
        nw, nh = int(w_img * scale), int(h_img * scale)
        img_r = cv2.resize(img_bgr, (nw, nh))
        img_rgb = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)

        dr = self.depth_pipe(Image.fromarray(img_rgb))
        d_raw = np.array(dr["depth"])
        d_norm = ((d_raw - d_raw.min()) / (d_raw.max() - d_raw.min() + 1e-8) * 255).astype(np.uint8)

        f_pixel = (nw / 2.0) / np.tan(np.radians(self.fov_deg / 2.0))
        img_t = torch.as_tensor(img_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0

        with torch.no_grad():
            p1 = self.m1([img_t.to(self.device)])[0]
            p2 = self.m2([img_t.to(self.device)])[0]

        house_candidates = []
        for preds, mname in [(p1, "Unified-2000"), (p2, "July-400")]:
            boxes = preds["boxes"].cpu().numpy()
            labels = preds["labels"].cpu().numpy()
            scores = preds["scores"].cpu().numpy()
            for i in range(len(scores)):
                if labels[i] == 1 and scores[i] >= DEFAULT_HOUSE_CONF:
                    house_candidates.append({"box": boxes[i], "score": float(scores[i]), "model": mname})
        houses_detected = ensemble_nms(house_candidates, DEFAULT_IOU_THRESH)

        with torch.no_grad():
            p_coco = self.m_coco([img_t.to(self.device)])[0]

        ref_candidates = []
        coco_boxes = p_coco["boxes"].cpu().numpy()
        coco_labels = p_coco["labels"].cpu().numpy()
        coco_scores = p_coco["scores"].cpu().numpy()
        for i in range(len(coco_scores)):
            lbl = int(coco_labels[i])
            if lbl in COCO_REF_CLASSES and coco_scores[i] >= DEFAULT_REF_CONF:
                ref_candidates.append({
                    "box": coco_boxes[i],
                    "score": float(coco_scores[i]),
                    "ref_class": COCO_REF_CLASSES[lbl],
                    "coco_label": lbl,
                })
        ref_objects = ref_nms(ref_candidates, DEFAULT_IOU_THRESH)

        prepped_houses = []
        hid = 1
        for cand in houses_detected:
            x1, y1, x2, y2 = [int(v) for v in cand["box"]]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(nw, x2), min(nh, y2)
            raw_h, raw_w = y2 - y1, x2 - x1
            if raw_h < 25 or raw_w < 25:
                continue

            ext_y1, ext_y2, is_top_clipped, _ = robust_roofline_and_base(d_norm, x1, y1, x2, y2, nh)
            extended_h = ext_y2 - ext_y1
            house_depth = get_depth_value(d_norm, x1, ext_y1, x2, ext_y2)
            cx = (x1 + x2) / 2.0
            theta_rad = compute_azimuth_angle_rad(cx, nw, self.fov_deg)

            prepped_houses.append({
                "hid": hid,
                "cand": cand,
                "box": (x1, y1, x2, y2),
                "ext_box": (x1, ext_y1, x2, ext_y2),
                "h_px": extended_h,
                "depth": house_depth,
                "cx": cx,
                "theta_rad": theta_rad,
                "is_top_clipped": is_top_clipped,
                "direct_ref": None,
                "calibrated_h_m": None,
                "distance_m": None,
                "method": None,
                "ref_box_used": None,
                "anchor_parent_hid": None
            })
            hid += 1

        # 1. HIGH-PREFERENCE VEHICLE GROUND ANCHOR SEARCH
        # Vehicles (cars, buses, trucks) sitting in the lower ground frontage of the building
        class_priority = {"car": 10, "bus": 9, "truck": 9, "motorcycle": 3, "person": 2, "bicycle": 1}
        for h in prepped_houses:
            hx1, hy1, hx2, hy2 = h["ext_box"]
            best_ref = None
            best_score = -1.0

            for ref in ref_objects:
                rx1, ry1, rx2, ry2 = [int(v) for v in ref["box"]]
                ref_cx = (rx1 + rx2) / 2.0
                ref_cy = (ry1 + ry2) / 2.0
                
                # Check horizontal overlap with building boundary
                is_horiz_aligned = (hx1 - 80) <= ref_cx <= (hx2 + 80)
                if not is_horiz_aligned:
                    continue

                ref_w, ref_h = rx2 - rx1, ry2 - ry1
                if ref_h < MIN_REF_HEIGHT_PX or ref_w < MIN_REF_WIDTH_PX:
                    continue

                ref_depth = get_depth_value(d_norm, rx1, ry1, rx2, ry2)
                if ref_depth < 8:
                    continue

                ref_real_h, ref_label = get_ref_height(ref["ref_class"], ref_w, ref_h)
                raw_ratio = ref_depth / h["depth"] if h["depth"] > 5 else 1.0
                depth_ratio = float(np.clip(raw_ratio, 0.6, 1.8))

                # Weight vehicles heavily
                cls_wt = class_priority.get(ref["ref_class"], 1) / 10.0
                conf_wt = ref["score"]
                
                # Proximity to ground line
                dist_to_center = abs(ref_cx - h["cx"]) / max((hx2 - hx1), 1)
                prox_wt = max(0, 1.0 - dist_to_center)

                total_s = cls_wt * 0.5 + conf_wt * 0.3 + prox_wt * 0.2
                if total_s > best_score:
                    best_score = total_s
                    best_ref = {
                        "ref": ref,
                        "ref_real_h": ref_real_h,
                        "ref_label": ref_label,
                        "depth_ratio": depth_ratio,
                        "ref_h_px": ref_h,
                        "ref_box": (rx1, ry1, rx2, ry2),
                        "score": total_s
                    }

            if best_ref and best_ref["score"] >= 0.30:
                raw_h_m = (h["h_px"] / best_ref["ref_h_px"]) * best_ref["ref_real_h"] * best_ref["depth_ratio"]
                h["calibrated_h_m"] = raw_h_m
                h["method"] = f"ref:{best_ref['ref_label']}"
                h["direct_ref"] = best_ref
                h["ref_box_used"] = best_ref["ref_box"]
                
                ref_dist = (best_ref["ref_real_h"] * f_pixel) / best_ref["ref_h_px"]
                h["distance_m"] = ref_dist / best_ref["depth_ratio"] if best_ref["depth_ratio"] > 0.1 else ref_dist

        # 2. PASS 2: Inter-Building Reference Transfer
        anchor_houses = [h for h in prepped_houses if h["calibrated_h_m"] is not None and "car" in h["method"]]
        if not anchor_houses:
            anchor_houses = [h for h in prepped_houses if h["calibrated_h_m"] is not None]

        for h in prepped_houses:
            if h["calibrated_h_m"] is None and anchor_houses:
                closest_anchor = min(anchor_houses, key=lambda a: abs(h["cx"] - a["cx"]))
                x_separation = abs(h["cx"] - closest_anchor["cx"])
                
                if x_separation < (nw * 0.70):
                    depth_ratio = float(np.clip(closest_anchor["depth"] / max(h["depth"], 1), 0.6, 1.8))
                    transferred_h = closest_anchor["calibrated_h_m"] * (h["h_px"] / closest_anchor["h_px"]) * depth_ratio
                    
                    h["calibrated_h_m"] = transferred_h
                    h["distance_m"] = closest_anchor["distance_m"] / max(depth_ratio, 0.1)
                    h["method"] = f"ref:House#{closest_anchor['hid']}"
                    h["anchor_parent_hid"] = closest_anchor["hid"]
                    h["ref_box_used"] = closest_anchor["box"]

            if h["calibrated_h_m"] is None:
                real_h_m, house_dist = fallback_height(h["h_px"], h["depth"], nw, self.fov_deg)
                h["calibrated_h_m"] = real_h_m
                h["distance_m"] = house_dist
                h["method"] = "fallback"

        annotated = img_r.copy()
        buildings = []

        for ref in ref_objects:
            rx1, ry1, rx2, ry2 = [int(v) for v in ref["box"]]
            rx1, ry1, rx2, ry2 = max(0, rx1), max(0, ry1), min(nw, rx2), min(nh, ry2)
            ref_w, ref_h = rx2 - rx1, ry2 - ry1
            ref_real_h, ref_label = get_ref_height(ref["ref_class"], ref_w, ref_h)
            cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (0, 200, 255), 1)
            cv2.putText(
                annotated,
                f"{ref_label} {ref['score']*100:.0f}% ({ref_real_h:.1f}m)",
                (rx1 + 2, max(12, ry1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.30,
                (0, 200, 255),
                1
            )

        for h in prepped_houses:
            x1, y1, x2, y2 = h["box"]
            x1, ext_y1, x2, ext_y2 = h["ext_box"]
            real_h_m = float(h["calibrated_h_m"])
            house_dist = float(h["distance_m"])
            method = h["method"]
            floors = max(1, int(round(real_h_m / 3.2)))

            color = (0, 255, 0) if h["cand"]["model"] == "Unified-2000" else (255, 180, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
            cv2.rectangle(annotated, (x1, ext_y1), (x2, ext_y2), color, 3)

            if h["ref_box_used"]:
                rx1r, ry1r, rx2r, ry2r = h["ref_box_used"]
                cv2.line(
                    annotated,
                    ((x1 + x2) // 2, (ext_y1 + ext_y2) // 2),
                    ((rx1r + rx2r) // 2, (ry1r + ry2r) // 2),
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA
                )
                if "House" not in method:
                    cv2.rectangle(annotated, (rx1r, ry1r), (rx2r, ry2r), (0, 255, 255), 2)

            if ext_y1 < y1:
                cv2.arrowedLine(annotated, (x1 + (x2-x1)//2, y1), (x1 + (x2-x1)//2, ext_y1), (0, 0, 255), 2, tipLength=0.3)

            clip_str = " *[Clipped]" if h["is_top_clipped"] else ""
            meth_short = method.replace("fallback", "FB")
            label = f"H#{h['hid']}: {real_h_m:.1f}m ~{floors}fl{clip_str} | {meth_short} | {h['cand']['score']*100:.0f}%"
            (tw, th_t), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            ty = max(16, ext_y1 - 6)
            cv2.rectangle(annotated, (x1, ty - th_t - 4), (x1 + tw + 6, ty + 4), color, -1)
            cv2.putText(annotated, label, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1)

            buildings.append({
                "house_id": h["hid"],
                "score": h["cand"]["score"],
                "model": h["cand"]["model"],
                "method": method,
                "height_m": round(real_h_m, 2),
                "estimated_floors": floors,
                "distance_m": round(house_dist, 2),
                "is_top_clipped": h["is_top_clipped"],
                "azimuth_deg": round(math.degrees(h["theta_rad"]), 1),
                "bbox": [x1, ext_y1, x2, ext_y2],
            })

        return annotated, buildings
