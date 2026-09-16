"""
Compliance Engine: Evaluates estimated building heights against municipal zoning regulations
and renders visual compliance indicators directly on images.
"""
import cv2
import numpy as np
import base64
from zoning_db import get_zoning_for_location

def evaluate_and_annotate_compliance(annotated_bgr, buildings_data, lat, lon):
    zone_info = get_zoning_for_location(lat, lon)
    max_h = float(zone_info["max_allowed_height_m"])
    max_fl = int(zone_info["max_allowed_floors"])
    
    annotated = annotated_bgr.copy()
    compliance_results = []
    
    for b in buildings_data:
        h = float(b["height_m"])
        fl = int(b.get("estimated_floors", max(1, round(h / 3.0))))
        diff = round(h - max_h, 2)
        
        if h > max_h:
            status = "ILLEGAL"
            status_text = f"VIOLATION (+{diff:.1f}m)"
            badge_color = (0, 0, 220)
            text_color = (255, 255, 255)
        elif h >= (max_h * 0.95):
            status = "WARNING"
            status_text = f"NEAR LIMIT ({diff:.1f}m)"
            badge_color = (0, 165, 255)
            text_color = (0, 0, 0)
        else:
            status = "LEGAL"
            status_text = f"COMPLIANT ({abs(diff):.1f}m below)"
            badge_color = (34, 180, 34)
            text_color = (255, 255, 255)
            
        b_dict = {
            "house_id": b["house_id"],
            "height_m": h,
            "estimated_floors": fl,
            "max_allowed_height_m": max_h,
            "max_allowed_floors": max_fl,
            "height_difference_m": diff,
            "status": status,
            "status_text": status_text,
            "method": b.get("method", "ref"),
            "confidence": round(float(b.get("score", 1.0)) * 100, 1),
            "distance_m": b.get("distance_m", 0.0),
            "is_top_clipped": b.get("is_top_clipped", False)
        }
        
        if "bbox" in b and len(b["bbox"]) == 4:
            x1, y1, x2, y2 = b["bbox"]
            tag_label = f"[{status}] {h:.1f}m / Max {max_h:.1f}m"
            (tw, th), _ = cv2.getTextSize(tag_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            
            tag_y = max(th + 6, y1 - 22)
            cv2.rectangle(annotated, (x1, tag_y - th - 4), (x1 + tw + 8, tag_y + 4), badge_color, -1)
            cv2.putText(annotated, tag_label, (x1 + 4, tag_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), badge_color, 2)
            
        compliance_results.append(b_dict)
        
    banner_h = 32
    banner = np.zeros((banner_h, annotated.shape[1], 3), dtype=np.uint8)
    banner[:] = (15, 23, 42)
    
    zone_title = f"Zone: {zone_info['zone_name']} | Permissible Height Limit: {max_h:.1f}m ({max_fl} fl) | GPS: {lat:.4f}, {lon:.4f}"
    cv2.putText(banner, zone_title, (12, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    
    final_img = np.vstack([banner, annotated])
    _, buffer = cv2.imencode('.jpg', final_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    img_b64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "annotated_image_b64": f"data:image/jpeg;base64,{img_b64}",
        "zone_info": zone_info,
        "buildings": compliance_results,
        "has_violation": any(b["status"] == "ILLEGAL" for b in compliance_results),
        "total_buildings": len(compliance_results)
    }
