"""
Compliance Engine: Evaluates estimated building heights against municipal zoning regulations,
computes individual building GPS coordinates from camera heading & distance,
and automatically identifies the primary target building.
"""
import math
import cv2
import numpy as np
import base64
from zoning_db import get_zoning_for_location

METERS_PER_DEG_LAT = 111139.0

def compute_building_gps(cam_lat, cam_lon, distance_m, azimuth_deg, cam_heading_deg=0.0):
    total_angle_rad = math.radians(cam_heading_deg + azimuth_deg)
    d_north_m = distance_m * math.cos(total_angle_rad)
    d_east_m = distance_m * math.sin(total_angle_rad)
    
    delta_lat = d_north_m / METERS_PER_DEG_LAT
    meters_per_deg_lon = METERS_PER_DEG_LAT * math.cos(math.radians(cam_lat))
    delta_lon = d_east_m / max(meters_per_deg_lon, 1.0)
    
    house_lat = round(cam_lat + delta_lat, 6)
    house_lon = round(cam_lon + delta_lon, 6)
    
    return house_lat, house_lon, round(d_north_m, 1), round(d_east_m, 1)

def evaluate_and_annotate_compliance(annotated_bgr, buildings_data, cam_lat, cam_lon):
    zone_info = get_zoning_for_location(cam_lat, cam_lon)
    max_h = float(zone_info["max_allowed_height_m"])
    max_fl = int(zone_info["max_allowed_floors"])
    
    annotated = annotated_bgr.copy()
    compliance_results = []
    
    target_hid = None
    if buildings_data:
        right_candidates = [b for b in buildings_data if b.get("azimuth_deg", 0) > 0]
        if right_candidates:
            target_house = max(right_candidates, key=lambda b: b.get("azimuth_deg", 0))
            target_hid = target_house["house_id"]
        else:
            target_house = min(buildings_data, key=lambda b: b.get("distance_m", 999))
            target_hid = target_house["house_id"]
            
    for b in buildings_data:
        h = float(b["height_m"])
        fl = int(b.get("estimated_floors", max(1, round(h / 3.0))))
        diff = round(h - max_h, 2)
        is_target = (b["house_id"] == target_hid)
        
        b_dist = float(b.get("distance_m", 15.0))
        b_azimuth = float(b.get("azimuth_deg", 0.0))
        b_lat, b_lon, d_n, d_e = compute_building_gps(cam_lat, cam_lon, b_dist, b_azimuth, cam_heading_deg=0.0)
        
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
            "is_primary_target": is_target,
            "height_m": h,
            "estimated_floors": fl,
            "max_allowed_height_m": max_h,
            "max_allowed_floors": max_fl,
            "height_difference_m": diff,
            "status": status,
            "status_text": status_text,
            "method": b.get("method", "ref"),
            "confidence": round(float(b.get("score", 1.0)) * 100, 1),
            "distance_m": b_dist,
            "azimuth_deg": b_azimuth,
            "building_gps": {
                "latitude": b_lat,
                "longitude": b_lon,
                "offset_north_m": d_n,
                "offset_east_m": d_e
            },
            "is_top_clipped": b.get("is_top_clipped", False)
        }
        
        if "bbox" in b and len(b["bbox"]) == 4:
            x1, y1, x2, y2 = b["bbox"]
            target_prefix = "[TARGET] " if is_target else ""
            tag_label = f"{target_prefix}[{status}] {h:.1f}m / Max {max_h:.1f}m"
            (tw, th), _ = cv2.getTextSize(tag_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            
            tag_y = max(th + 6, y1 - 22)
            cv2.rectangle(annotated, (x1, tag_y - th - 4), (x1 + tw + 8, tag_y + 4), badge_color, -1)
            cv2.putText(annotated, tag_label, (x1 + 4, tag_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)
            
            if is_target:
                cv2.rectangle(annotated, (x1-3, y1-3), (x2+3, y2+3), (0, 215, 255), 3)
            else:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), badge_color, 2)
            
        compliance_results.append(b_dict)
        
    banner_h = 34
    banner = np.zeros((banner_h, annotated.shape[1], 3), dtype=np.uint8)
    banner[:] = (15, 23, 42)
    
    zone_title = f"Zone: {zone_info['zone_name']} | Permissible Limit: {max_h:.1f}m ({max_fl} fl) | Auto-Target: House #{target_hid}"
    cv2.putText(banner, zone_title, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)
    
    final_img = np.vstack([banner, annotated])
    _, buffer = cv2.imencode('.jpg', final_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    img_b64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "annotated_image_b64": f"data:image/jpeg;base64,{img_b64}",
        "zone_info": zone_info,
        "target_house_id": target_hid,
        "buildings": compliance_results,
        "has_violation": any(b["status"] == "ILLEGAL" for b in compliance_results),
        "total_buildings": len(compliance_results)
    }
