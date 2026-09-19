"""
Multi-strategy GPS and Metadata Extractor from Image Filenames and Dataset Catalogs
"""
import re
import os
import csv
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_APP_DIR)
CSV_CATALOG_PATH = os.path.join(_REPO_ROOT, "Connected Building Landscape.csv")


def _get_exif_gps(image_path):
    try:
        image = Image.open(image_path)
        exif = image._getexif()
        if not exif:
            return None
        
        gps_info = {}
        for tag, value in exif.items():
            decoded = TAGS.get(tag, tag)
            if decoded == "GPSInfo":
                for t in value:
                    sub_decoded = GPSTAGS.get(t, t)
                    gps_info[sub_decoded] = value[t]
        
        if not gps_info:
            return None
            
        def _to_degrees(val):
            d = float(val[0])
            m = float(val[1])
            s = float(val[2])
            return d + (m / 60.0) + (s / 3600.0)
            
        lat = _to_degrees(gps_info['GPSLatitude'])
        if gps_info.get('GPSLatitudeRef', 'N') != 'N':
            lat = -lat
            
        lon = _to_degrees(gps_info['GPSLongitude'])
        if gps_info.get('GPSLongitudeRef', 'E') != 'E':
            lon = -lon
            
        return lat, lon, "EXIF Metadata"
    except Exception:
        return None

def _lookup_csv_catalog(filename):
    if not os.path.exists(CSV_CATALOG_PATH):
        return None
        
    base = os.path.splitext(os.path.basename(filename))[0]
    clean_key = base.replace("aug_", "").replace("ens_", "").replace("val_", "").strip()
    
    try:
        with open(CSV_CATALOG_PATH, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("Name", "").strip()
                if name and (name == clean_key or name in clean_key):
                    lat = float(row.get("Latitude", 0))
                    lon = float(row.get("Longitude", 0))
                    if lat != 0 and lon != 0:
                        return lat, lon, f"Catalog Match ({name})"
    except Exception:
        pass
    return None

def extract_gps_from_filename_or_image(image_path_or_name):
    filename = os.path.basename(image_path_or_name)
    
    match = re.search(r'(?:lat[_-]?)?(-?\d{1,3}\.\d{3,8})[_\-, ]+(?:lon[_-]?)?(-?\d{1,3}\.\d{3,8})', filename, re.IGNORECASE)
    if match:
        lat, lon = float(match.group(1)), float(match.group(2))
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return lat, lon, "Filename Coordinates (Pattern A)"

    match = re.search(r'(\d{1,3}\.\d{3,8})\s*([NS])[_\-, ]+(\d{1,3}\.\d{3,8})\s*([EW])', filename, re.IGNORECASE)
    if match:
        lat = float(match.group(1)) * (1 if match.group(2).upper() == 'N' else -1)
        lon = float(match.group(3)) * (1 if match.group(4).upper() == 'E' else -1)
        return lat, lon, "Filename Coordinates (Pattern B)"

    cat_match = _lookup_csv_catalog(filename)
    if cat_match:
        return cat_match

    if os.path.exists(image_path_or_name):
        exif_match = _get_exif_gps(image_path_or_name)
        if exif_match:
            return exif_match

    return 28.6139, 77.2090, "Default Regional Anchor"
