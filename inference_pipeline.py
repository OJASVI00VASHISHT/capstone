"""
Pipeline v5: Reference-Object Calibrated Height Estimation
============================================================
Architecture:
  - Fine-tuned Model 1 (Unified Fixed 2000) + Model 2 (July + 400) → house detection
  - COCO Pretrained Mask R-CNN → reference object detection (person, car, motorcycle, bicycle, bus, truck)
  - Depth Anything V2 → relative depth (used for depth RATIOS, not absolute distance)

Height Calculation:
  Primary:   H_house = (h_house_px / h_ref_px) × H_ref × (d_ref / d_house)
  Fallback:  H_house = (h_house_px × D) / f_pixel × 1.20   [D from inverse disparity, 5-13m range]
"""

import os, sys, glob
sys.stdout.reconfigure(encoding='utf-8')

import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from PIL import Image
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls
from transformers import pipeline as hf_pipeline
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================
CONFIDENCE_THRESHOLD = 0.40       # For house detection
REF_CONFIDENCE_THRESHOLD = 0.50   # For reference objects (COCO model)
IOU_THRESHOLD = 0.50
HORIZONTAL_FOV_DEG = 95.0

# Fallback distance range (when no reference object available)
Z_MIN_M = 5.0
Z_MAX_M = 13.0
FALLBACK_HEIGHT_BOOST = 1.20  # +20% on fallback estimates

# Reference object known heights (meters)
REF_HEIGHTS = {
    "person": 1.70,
    "bicycle": 1.00,
    "car_sedan": 1.50,
    "car_suv": 1.70,
    "motorcycle": 1.10,
    "bus": 3.20,
    "truck": 3.50,
}

# COCO class IDs we care about for reference
COCO_REF_CLASSES = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    6: "bus",
    8: "truck",
}

# Max horizontal pixel distance to consider a reference object "nearby" a house
MAX_REF_HORIZONTAL_DIST = 350
# ============================================================


def get_finetuned_model(num_classes=7):
    """Load fine-tuned Mask R-CNN architecture (no weights)."""
    model = maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model


def get_coco_model():
    """Load pretrained COCO Mask R-CNN for reference object detection."""
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")
    return model


def compute_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter + 1e-8)


def ensemble_nms(candidates, iou_thresh=0.50):
    """NMS across ensemble candidates, keeping highest confidence."""
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
    """NMS for reference objects, per class."""
    if not detections:
        return []
    by_class = {}
    for d in detections:
        by_class.setdefault(d["ref_class"], []).append(d)
    
    kept = []
    for cls_name, dets in by_class.items():
        s = sorted(dets, key=lambda c: c["score"], reverse=True)
        cls_kept = []
        while s:
            best = s[0]
            cls_kept.append(best)
            s = [c for c in s[1:] if compute_iou(best["box"], c["box"]) <= iou_thresh]
        kept.extend(cls_kept)
    return kept


def robust_roofline_and_base(d_norm, x1, y1, x2, y2, img_h):
    """Robust roofline and base refinement using depth-map consistency."""
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w < 20 or box_h < 20:
        return y1, y2, False, False

    cx1 = x1 + int(0.25 * box_w)
    cx2 = x1 + int(0.75 * box_w)

    mid_y1 = y1 + int(0.30 * box_h)
    mid_y2 = y1 + int(0.60 * box_h)
    facade_sample = d_norm[mid_y1:mid_y2, cx1:cx2]
    if facade_sample.size == 0:
        facade_d = float(d_norm[y1:y2, x1:x2].mean())
    else:
        facade_d = float(np.median(facade_sample))

    sky_thresh = max(18.0, facade_d * 0.55)

    new_y1 = y1
    y1_strip = d_norm[max(0, y1):min(y1 + 10, y2), cx1:cx2]
    if y1_strip.size > 0 and float(y1_strip.mean()) < sky_thresh:
        for scan_y in range(y1, min(y2, y1 + int(0.5 * box_h))):
            row_d = float(d_norm[scan_y, cx1:cx2].mean())
            if row_d >= sky_thresh:
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
    max_down = min(img_h - y2, int(box_h * 0.20))
    for scan_y in range(y2, y2 + max_down):
        row_d = float(d_norm[scan_y, cx1:cx2].mean())
        if row_d > fg_thresh or abs(row_d - facade_d) > 30.0:
            break
        new_y2 = scan_y

    is_top_clipped = (new_y1 <= 15)
    was_extended = (new_y1 < y1) or (new_y2 > y2)
    return new_y1, new_y2, is_top_clipped, was_extended


def get_ref_height(ref_class, box_w, box_h):
    """Get known real-world height for a reference object.
    For cars, distinguish sedan vs SUV by aspect ratio."""
    if ref_class == "car":
        aspect = box_w / max(box_h, 1)
        if aspect > 2.5:
            return REF_HEIGHTS["car_sedan"], "car(sedan)"
        else:
            return REF_HEIGHTS["car_suv"], "car(SUV)"
    elif ref_class in REF_HEIGHTS:
        return REF_HEIGHTS[ref_class], ref_class
    else:
        return 1.50, ref_class  # default fallback


def get_depth_value(d_norm, x1, y1, x2, y2):
    """Get representative depth value for a bounding box region."""
    crop = d_norm[y1:y2, x1:x2]
    valid = crop[crop > 15]  # exclude sky
    if valid.size > 0:
        return float(np.percentile(valid, 75))
    return float(np.median(crop)) if crop.size > 0 else 50.0


def find_best_reference(house_box, house_depth, ref_objects, d_norm):
    """Find the best reference object for a house.
    Priority: nearby objects with high confidence and reliable class.
    Returns (ref_obj, ref_real_height, ref_label, ref_depth, depth_ratio) or None."""
    hx1, hy1, hx2, hy2 = house_box
    house_cx = (hx1 + hx2) / 2.0
    house_cy = (hy1 + hy2) / 2.0

    # Priority weights: person > car > bus > truck > motorcycle > bicycle
    class_priority = {"person": 5, "car": 4, "bus": 3, "truck": 2, "motorcycle": 1, "bicycle": 1}

    scored_refs = []
    for ref in ref_objects:
        rx1, ry1, rx2, ry2 = [int(v) for v in ref["box"]]
        ref_cx = (rx1 + rx2) / 2.0
        ref_cy = (ry1 + ry2) / 2.0
        
        # Horizontal distance
        x_dist = abs(house_cx - ref_cx)
        if x_dist > MAX_REF_HORIZONTAL_DIST:
            continue  # Too far away horizontally
        
        ref_w = rx2 - rx1
        ref_h = ry2 - ry1
        if ref_h < 40 or ref_w < 15:
            continue  # Fix 3: minimum 40px height for stable ratios
        
        # Fix 2: Motorcycles need higher confidence (>= 70%) since they're small/unreliable
        if ref["ref_class"] == "motorcycle" and ref["score"] < 0.70:
            continue
        
        ref_depth = get_depth_value(d_norm, rx1, ry1, rx2, ry2)
        if ref_depth < 10:
            continue  # Sky-like depth, unreliable
        
        ref_real_h, ref_label = get_ref_height(ref["ref_class"], ref_w, ref_h)
        
        # Depth ratio: d_ref / d_house (inverse disparity: higher d = closer)
        if house_depth > 5:
            raw_depth_ratio = ref_depth / house_depth
        else:
            raw_depth_ratio = 1.0
        
        # Fix 1: Clamp depth ratio to [0.5, 2.0] — prevents extreme over/under-estimates
        depth_ratio = float(np.clip(raw_depth_ratio, 0.5, 2.0))
        
        # Scoring: prefer nearby, high-confidence, high-priority class
        proximity_score = max(0, 1.0 - x_dist / MAX_REF_HORIZONTAL_DIST)
        cls_score = class_priority.get(ref["ref_class"], 1) / 5.0
        conf_score = ref["score"]
        
        # Prefer references at similar depth (ratio near 1.0)
        ratio_penalty = 1.0 - 0.4 * abs(depth_ratio - 1.0)  # max 0.4 penalty at ratio 0.5 or 2.0
        
        total_score = proximity_score * 0.3 + cls_score * 0.3 + conf_score * 0.2 + ratio_penalty * 0.2
        
        scored_refs.append({
            "ref": ref,
            "ref_real_h": ref_real_h,
            "ref_label": ref_label,
            "ref_depth": ref_depth,
            "depth_ratio": depth_ratio,
            "x_dist": x_dist,
            "total_score": total_score,
        })
    
    if not scored_refs:
        return None
    
    # Return the best-scoring reference
    best = max(scored_refs, key=lambda r: r["total_score"])
    return best


def fallback_height(h_pixel, d_val, img_w):
    """Fallback: inverse disparity depth + 20% boost."""
    f_pixel = (img_w / 2.0) / np.tan(np.radians(HORIZONTAL_FOV_DEG / 2.0))
    d_hat = d_val / 255.0
    inv_z = (1.0 / Z_MAX_M) + d_hat * ((1.0 / Z_MIN_M) - (1.0 / Z_MAX_M))
    perp_dist_m = float(np.clip(1.0 / max(inv_z, 1e-4), Z_MIN_M, Z_MAX_M))
    
    raw_height = (h_pixel * perp_dist_m) / f_pixel
    boosted_height = raw_height * FALLBACK_HEIGHT_BOOST
    return boosted_height, perp_dist_m


def set_cell_bg(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill}"/>'))


def main():
    print("=" * 80)
    print("PIPELINE v5: REFERENCE-OBJECT CALIBRATED HEIGHT ESTIMATION")
    print("=" * 80)

    work_dir = r"c:\Users\Ojasvi\PycharmProjects\cap"
    manual_dir = os.path.join(work_dir, "manual testing")
    out_dir = os.path.join(work_dir, "manual_testing_ensemble_outputs")
    out_docx = os.path.join(work_dir, "Manual_Testing_Ensemble_Height_Estimate.docx")
    out_docx2 = os.path.join(work_dir, "height estimate.docx")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"FOV: {HORIZONTAL_FOV_DEG}°  |  Fallback: Inv. Disparity ({Z_MIN_M}-{Z_MAX_M}m) +{int((FALLBACK_HEIGHT_BOOST-1)*100)}%")
    print(f"Reference objects: person, car, motorcycle, bicycle, bus, truck (COCO)")

    # ---- Load 3 models ----
    print("\n[1/4] Loading Fine-tuned Model 1: Unified Fixed 2000...")
    m1 = get_finetuned_model(7)
    m1.load_state_dict(torch.load(os.path.join(work_dir, "mask_rcnn_model_2000_fixed.pth"), map_location=device))
    m1.to(device); m1.eval()

    print("[2/4] Loading Fine-tuned Model 2: July + 400...")
    m2 = get_finetuned_model(7)
    m2.load_state_dict(torch.load(os.path.join(work_dir, "mask_rcnn_model_july_400.pth"), map_location=device))
    m2.to(device); m2.eval()

    print("[3/4] Loading COCO Pretrained Mask R-CNN for reference objects...")
    m_coco = get_coco_model()
    m_coco.to(device); m_coco.eval()

    print("[4/4] Loading Depth Anything V2...")
    dev_id = 0 if device.type == "cuda" else -1
    depth_pipe = hf_pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=dev_id)

    image_paths = sorted(glob.glob(os.path.join(manual_dir, "*.png")) +
                         glob.glob(os.path.join(manual_dir, "*.jpg")) +
                         glob.glob(os.path.join(manual_dir, "*.jpeg")))
    total_images = len(image_paths)
    print(f"\nProcessing {total_images} images...\n")

    all_results = []
    total_houses = 0
    ref_used_count = 0
    fallback_count = 0

    for img_path in tqdm(image_paths, desc="Pipeline v5"):
        img_name = os.path.basename(img_path)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        h_img, w_img = img_bgr.shape[:2]
        target = 1024
        scale = target / max(h_img, w_img)
        nw, nh = int(w_img * scale), int(h_img * scale)
        img_r = cv2.resize(img_bgr, (nw, nh))
        img_rgb = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # 1. Depth map
        dr = depth_pipe(img_pil)
        d_raw = np.array(dr["depth"])
        d_norm = ((d_raw - d_raw.min()) / (d_raw.max() - d_raw.min() + 1e-8) * 255).astype(np.uint8)

        f_pixel = (nw / 2.0) / np.tan(np.radians(HORIZONTAL_FOV_DEG / 2.0))
        img_t = torch.as_tensor(img_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0

        # 2. Detect houses with fine-tuned ensemble
        with torch.no_grad():
            p1 = m1([img_t.to(device)])[0]
            p2 = m2([img_t.to(device)])[0]

        house_candidates = []
        for preds, mname in [(p1, "Unified Fixed"), (p2, "July + 400")]:
            boxes = preds["boxes"].cpu().numpy()
            labels = preds["labels"].cpu().numpy()
            scores = preds["scores"].cpu().numpy()
            for i in range(len(scores)):
                if labels[i] == 1 and scores[i] >= CONFIDENCE_THRESHOLD:
                    house_candidates.append({"box": boxes[i], "score": float(scores[i]), "model": mname})

        houses_detected = ensemble_nms(house_candidates, IOU_THRESHOLD)

        # 3. Detect reference objects with COCO model
        with torch.no_grad():
            p_coco = m_coco([img_t.to(device)])[0]

        ref_candidates = []
        coco_boxes = p_coco["boxes"].cpu().numpy()
        coco_labels = p_coco["labels"].cpu().numpy()
        coco_scores = p_coco["scores"].cpu().numpy()
        
        for i in range(len(coco_scores)):
            lbl = int(coco_labels[i])
            if lbl in COCO_REF_CLASSES and coco_scores[i] >= REF_CONFIDENCE_THRESHOLD:
                ref_candidates.append({
                    "box": coco_boxes[i],
                    "score": float(coco_scores[i]),
                    "ref_class": COCO_REF_CLASSES[lbl],
                    "coco_label": lbl,
                })

        ref_objects = ref_nms(ref_candidates, IOU_THRESHOLD)

        # 4. Annotate and calculate heights
        annotated = img_r.copy()
        houses = []
        hid = 1

        # Draw reference objects first (lighter)
        for ref in ref_objects:
            rx1, ry1, rx2, ry2 = [int(v) for v in ref["box"]]
            rx1, ry1 = max(0, rx1), max(0, ry1)
            rx2, ry2 = min(nw, rx2), min(nh, ry2)
            ref_w = rx2 - rx1
            ref_h = ry2 - ry1
            ref_real_h, ref_label = get_ref_height(ref["ref_class"], ref_w, ref_h)
            
            # Dashed-style rectangle for reference objects (yellow)
            cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (0, 200, 255), 1)
            ref_text = f"{ref_label} {ref['score']*100:.0f}% h={ref_real_h:.1f}m"
            ry_text = max(12, ry1 - 4)
            cv2.putText(annotated, ref_text, (rx1 + 2, ry_text), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 200, 255), 1)

        for cand in houses_detected:
            x1, y1, x2, y2 = [int(v) for v in cand["box"]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(nw, x2), min(nh, y2)
            raw_h = y2 - y1
            raw_w = x2 - x1
            if raw_h < 25 or raw_w < 25:
                continue

            # Roofline extension
            ext_y1, ext_y2, is_top_clipped, was_extended = robust_roofline_and_base(d_norm, x1, y1, x2, y2, nh)
            extended_h = ext_y2 - ext_y1

            # House depth value
            house_depth = get_depth_value(d_norm, x1, ext_y1, x2, ext_y2)

            # Try reference-based height first
            ref_match = find_best_reference((x1, ext_y1, x2, ext_y2), house_depth, ref_objects, d_norm)

            if ref_match is not None:
                # Reference-calibrated height
                ref = ref_match["ref"]
                rx1r, ry1r, rx2r, ry2r = [int(v) for v in ref["box"]]
                ref_h_px = ry2r - ry1r
                ref_real_h = ref_match["ref_real_h"]
                ref_depth = ref_match["ref_depth"]
                depth_ratio = ref_match["depth_ratio"]
                ref_label = ref_match["ref_label"]
                
                # Core formula: H = (h_house_px / h_ref_px) × H_ref × (d_ref / d_house)
                real_h_m = (extended_h / ref_h_px) * ref_real_h * depth_ratio
                
                method = f"ref:{ref_label}"
                ref_used_count += 1
                
                # Distance estimate (back-calculated from reference)
                ref_dist = (ref_real_h * f_pixel) / ref_h_px
                house_dist = ref_dist / depth_ratio if depth_ratio > 0.1 else ref_dist
                
                # Draw reference link line
                ref_cx = (rx1r + rx2r) // 2
                ref_cy = (ry1r + ry2r) // 2
                house_cx_int = (x1 + x2) // 2
                house_cy_int = (ext_y1 + ext_y2) // 2
                cv2.line(annotated, (house_cx_int, house_cy_int), (ref_cx, ref_cy), (0, 255, 255), 1, cv2.LINE_AA)
                # Highlight the reference used
                cv2.rectangle(annotated, (rx1r, ry1r), (rx2r, ry2r), (0, 255, 255), 2)
            else:
                # Fallback: depth-based with boost
                real_h_m, house_dist = fallback_height(extended_h, house_depth, nw)
                method = "fallback(depth)"
                depth_ratio = 0.0
                ref_label = "none"
                ref_real_h = 0.0
                fallback_count += 1

            floors = max(1, int(round(real_h_m / 3.0)))

            # Visual annotations
            color = (0, 255, 0) if cand["model"] == "Unified Fixed" else (255, 180, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
            cv2.rectangle(annotated, (x1, ext_y1), (x2, ext_y2), color, 3)

            if ext_y1 < y1:
                cv2.arrowedLine(annotated, (x1 + raw_w // 2, y1), (x1 + raw_w // 2, ext_y1), (0, 0, 255), 2, tipLength=0.3)
            if ext_y2 > y2:
                cv2.arrowedLine(annotated, (x1 + raw_w // 2, y2), (x1 + raw_w // 2, ext_y2), (0, 0, 255), 2, tipLength=0.3)

            clip_str = " *[Clipped]" if is_top_clipped else ""
            method_short = method.replace("fallback(depth)", "FB")
            label = f"H#{hid}: {real_h_m:.1f}m ~{floors}fl{clip_str} | {method_short} | {cand['score']*100:.0f}%"
            (tw, th_t), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            ty = max(16, ext_y1 - 6)
            cv2.rectangle(annotated, (x1, ty - th_t - 4), (x1 + tw + 6, ty + 4), color, -1)
            cv2.putText(annotated, label, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1)

            houses.append({
                "house_id": hid,
                "score": cand["score"],
                "model": cand["model"],
                "orig_h_px": int(raw_h / scale),
                "ext_h_px": int(extended_h / scale),
                "is_top_clipped": is_top_clipped,
                "method": method,
                "ref_label": ref_label,
                "ref_real_h": ref_real_h,
                "depth_ratio": depth_ratio,
                "house_dist_m": house_dist,
                "height_m": real_h_m,
                "floors": floors,
            })
            total_houses += 1
            hid += 1

        cv2.imwrite(os.path.join(out_dir, f"ensemble_{img_name}"), annotated)
        all_results.append({"image_name": img_name, "houses": houses, "ref_count": len(ref_objects)})

    # ============================================================
    # Print summary
    # ============================================================
    print(f"\n{'='*80}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Total images:          {total_images}")
    print(f"Total houses detected: {total_houses}")
    print(f"  Reference-calibrated: {ref_used_count} ({ref_used_count*100/max(1,total_houses):.0f}%)")
    print(f"  Fallback (depth):     {fallback_count} ({fallback_count*100/max(1,total_houses):.0f}%)")

    all_h = [h["height_m"] for r in all_results for h in r["houses"]]
    if all_h:
        print(f"\nHeight distribution:")
        print(f"  Min: {min(all_h):.1f}m  |  Max: {max(all_h):.1f}m  |  Mean: {np.mean(all_h):.1f}m  |  Median: {np.median(all_h):.1f}m")

    # Per-image details
    print(f"\n{'='*100}")
    print(f"{'Image':>40} | {'House':>6} | {'Method':>20} | {'Height':>8} | {'Floors':>6} | {'Dist':>6} | {'Conf':>5}")
    print(f"{'-'*100}")
    for res in all_results:
        for h in res["houses"]:
            clip = "*" if h["is_top_clipped"] else " "
            print(f"{res['image_name']:>40} | H#{h['house_id']:>3} | {h['method']:>20} | {h['height_m']:>6.1f}m{clip} | {h['floors']:>4}fl | {h['house_dist_m']:>5.1f}m | {h['score']*100:>4.0f}%")

    # ============================================================
    # Build Word report
    # ============================================================
    print(f"\n{'='*80}")
    print("Building Word report...")
    doc = docx.Document()
    for sec in doc.sections:
        sec.top_margin = sec.bottom_margin = sec.left_margin = sec.right_margin = Inches(0.75)

    tp = doc.add_paragraph()
    tr = tp.add_run("Pipeline v5: Reference-Object Calibrated Height Estimation")
    tr.font.size = Pt(20); tr.font.bold = True; tr.font.color.rgb = RGBColor(15, 23, 42)
    tp.paragraph_format.space_after = Pt(2)

    sp = doc.add_paragraph()
    sr = sp.add_run(
        f"COCO Reference Objects (person, car, motorcycle, bicycle, bus, truck) + "
        f"Fine-tuned Ensemble (2 models) | FOV {HORIZONTAL_FOV_DEG:.0f}° | "
        f"Fallback: Inv. Disparity ({Z_MIN_M}-{Z_MAX_M}m) +{int((FALLBACK_HEIGHT_BOOST-1)*100)}%"
    )
    sr.font.size = Pt(10); sr.font.color.rgb = RGBColor(71, 85, 105)
    sp.paragraph_format.space_after = Pt(14)

    # Summary table
    doc.add_heading("1. Summary Statistics", level=1)
    all_d = [h["house_dist_m"] for r in all_results for h in r["houses"]]
    clipped_count = sum(1 for r in all_results for h in r["houses"] if h["is_top_clipped"])

    st = doc.add_table(rows=8, cols=2)
    st.alignment = WD_TABLE_ALIGNMENT.CENTER
    stats = [
        ("Total Images Processed", f"{total_images}"),
        ("Total Houses Detected (Conf >= 40%)", f"{total_houses}"),
        ("Height Method: Reference-Calibrated", f"{ref_used_count} / {total_houses} ({ref_used_count*100//max(1,total_houses)}%)"),
        ("Height Method: Fallback (Depth-Based)", f"{fallback_count} / {total_houses}"),
        ("Houses Clipped by Frame", f"{clipped_count} / {total_houses}"),
        ("Mean Building Height", f"{np.mean(all_h):.2f} meters" if all_h else "N/A"),
        ("Median Building Height", f"{np.median(all_h):.2f} meters" if all_h else "N/A"),
        ("Mean Distance from Camera", f"{np.mean(all_d):.2f} meters" if all_d else "N/A"),
    ]
    for i, (k, v) in enumerate(stats):
        st.cell(i, 0).paragraphs[0].add_run(k).font.bold = True
        st.cell(i, 1).paragraphs[0].add_run(v)
        set_cell_bg(st.cell(i, 0), "F1F5F9" if i % 2 == 0 else "FFFFFF")
        set_cell_bg(st.cell(i, 1), "F8FAFC" if i % 2 == 0 else "FFFFFF")

    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    # Methodology
    doc.add_heading("2. Height Estimation Methodology", level=1)
    doc.add_paragraph(
        "A. Reference-Object Calibration (Primary Method):\n"
        "   - COCO pretrained Mask R-CNN detects reference objects (person ≈1.7m, car ≈1.5-1.7m, motorcycle ≈1.1m, etc.)\n"
        "   - For each house, the nearest reference object is paired.\n"
        "   - Height formula: H_house = (h_house_px / h_ref_px) × H_ref × (d_ref / d_house)\n"
        "   - The depth ratio (d_ref / d_house) corrects for objects at different distances.\n\n"
        "B. Fallback — Depth-Based Estimation:\n"
        "   - When no reference object is available, inverse disparity depth mapping is used.\n"
        f"   - Distance range: {Z_MIN_M}m - {Z_MAX_M}m with a {int((FALLBACK_HEIGHT_BOOST-1)*100)}% height boost.\n\n"
        "C. Roofline Extension:\n"
        "   - Bounding boxes are extended upward/downward using depth-map continuity.\n"
        "   - Top-clipped buildings (y <= 15px) are marked with '*'."
    )

    # Main data table
    doc.add_heading("3. Detailed Results", level=1)
    tbl = doc.add_table(rows=total_houses + 1, cols=8)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER

    headers = ["Screenshot", "House", "Model (Conf)", "Method",
               "Reference Used", "Height (m)", "Framing", "Est. Floors"]
    cw = [Inches(1.4), Inches(0.45), Inches(0.95), Inches(1.1),
          Inches(1.0), Inches(0.8), Inches(0.9), Inches(0.55)]

    for ci, hd in enumerate(headers):
        c = tbl.cell(0, ci)
        c.width = cw[ci]
        r = c.paragraphs[0].add_run(hd)
        r.font.bold = True; r.font.size = Pt(8); r.font.color.rgb = RGBColor(255, 255, 255)
        set_cell_bg(c, "0F172A")
        c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    rp = 1
    for res in all_results:
        for h in res["houses"]:
            clip_text = "Top Clipped" if h["is_top_clipped"] else "Full View"
            h_text = f"{h['height_m']:.2f}m*" if h["is_top_clipped"] else f"{h['height_m']:.2f}m"
            ref_text = h["ref_label"] if h["ref_label"] != "none" else "—"
            if h["depth_ratio"] > 0:
                ref_text += f"\n(ratio: {h['depth_ratio']:.2f})"
            vals = [
                res["image_name"],
                f"#{h['house_id']}",
                f"{h['model']}\n({h['score']*100:.0f}%)",
                h["method"],
                ref_text,
                h_text,
                clip_text,
                f"{h['floors']} Story"
            ]
            bg = "F8FAFC" if rp % 2 == 1 else "FFFFFF"
            for ci, v in enumerate(vals):
                c = tbl.cell(rp, ci)
                c.width = cw[ci]
                r = c.paragraphs[0].add_run(v)
                r.font.size = Pt(7.5)
                if ci == 0: r.font.bold = True
                if ci == 5:
                    r.font.bold = True
                    r.font.color.rgb = RGBColor(2, 132, 199)
                if ci == 6:
                    if h["is_top_clipped"]:
                        r.font.color.rgb = RGBColor(217, 119, 6)
                    else:
                        r.font.color.rgb = RGBColor(22, 163, 74)
                if ci == 3:
                    if "ref:" in h["method"]:
                        r.font.color.rgb = RGBColor(22, 163, 74)
                    else:
                        r.font.color.rgb = RGBColor(217, 119, 6)
                set_cell_bg(c, bg)
                c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            rp += 1

    doc.save(out_docx)
    print(f"Saved: '{out_docx}'")
    try:
        doc.save(out_docx2)
        print(f"Saved: '{out_docx2}'")
    except Exception:
        pass
    print("=" * 80)
    print("DONE")


if __name__ == "__main__":
    main()
