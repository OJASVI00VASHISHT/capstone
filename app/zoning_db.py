"""
Municipal Zoning and Permissible Building Height Database
"""
import sqlite3
import os
import math

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_APP_DIR)
DB_PATH = os.path.join(_REPO_ROOT, "zoning_regulations.db")


DEFAULT_ZONES = [
    ("Residential Zone R-1 (Low Density)", 28.6139, 77.2090, 5000, 9.0, 3, "Municipal Corporation - Zone A", "Max G+2 floors allowed (9.0 meters)"),
    ("Residential Zone R-2 (Medium Density)", 28.5355, 77.3910, 8000, 12.0, 4, "Municipal Corporation - Zone B", "Max G+3 floors allowed (12.0 meters)"),
    ("Commercial Zone C-1 (Mixed Use)", 28.4595, 77.0266, 6000, 15.0, 5, "Urban Development Authority", "Max G+4 commercial floors allowed (15.0 meters)"),
    ("High-Density Corridor C-2", 28.7041, 77.1025, 7000, 24.0, 8, "Metropolitan Planning Board", "High-rise corridor up to 24.0 meters"),
    ("Osaka Urban Sample Area", 34.7385, 135.5641, 10000, 10.0, 3, "Osaka City Planning", "Standard residential height limit (10.0 meters)"),
    ("General Default Zone", 0.0, 0.0, 100000000, 10.0, 3, "Standard Municipal By-laws", "Default residential permissible height (10.0 meters)"),
]

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS zones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            zone_name TEXT NOT NULL,
            center_lat REAL NOT NULL,
            center_lon REAL NOT NULL,
            radius_meters REAL NOT NULL,
            max_allowed_height_m REAL NOT NULL,
            max_allowed_floors INTEGER NOT NULL,
            jurisdiction TEXT,
            description TEXT
        )
    """)
    cursor.execute("SELECT COUNT(*) FROM zones")
    if cursor.fetchone()[0] == 0:
        cursor.executemany("""
            INSERT INTO zones (zone_name, center_lat, center_lon, radius_meters, max_allowed_height_m, max_allowed_floors, jurisdiction, description)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, DEFAULT_ZONES)
        conn.commit()
    conn.close()

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def get_zoning_for_location(lat, lon):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM zones WHERE zone_name != 'General Default Zone'")
    zones = cursor.fetchall()
    
    best_zone = None
    min_dist = float('inf')
    
    for z in zones:
        dist = haversine_distance(lat, lon, z["center_lat"], z["center_lon"])
        if dist <= z["radius_meters"] and dist < min_dist:
            min_dist = dist
            best_zone = dict(z)
            best_zone["distance_to_center_m"] = round(dist, 1)
            
    if best_zone is None:
        cursor.execute("SELECT * FROM zones WHERE zone_name = 'General Default Zone'")
        default_z = cursor.fetchone()
        best_zone = dict(default_z) if default_z else {
            "zone_name": "Standard Municipal Zone",
            "max_allowed_height_m": 10.0,
            "max_allowed_floors": 3,
            "jurisdiction": "General Bylaws",
            "description": "Standard residential limit (10.0m)"
        }
        best_zone["distance_to_center_m"] = 0
        
    conn.close()
    return best_zone

def get_all_zones():
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM zones")
    zones = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return zones

if __name__ == "__main__":
    init_db()
    print("Zoning Database initialized.")
