"""
POI langs je route (simpel++): upload GPX + kies datum/tijd.
- Types: bakkerijen, koffiebars/cafés, supermarkten, fietswinkels
- Filters per type + sorteren (km-punt, naam, type, afstand)
- Buffer rond route (m), km-punt & afstand
- Kaart + tabel + CSV
"""

import io
import time
import math
from typing import Tuple, List, Dict

import streamlit as st
import pandas as pd
import requests
from requests.exceptions import RequestException
import gpxpy
import gpxpy.gpx
from shapely.geometry import LineString, Point
from shapely.ops import transform as shp_transform
from pyproj import Transformer
import folium
from streamlit_folium import st_folium
from datetime import datetime, time as dt_time

# ---------- Session state init ----------
for key, default in [
    ("gpx_bytes", None),
    ("route_coords", None),   # [(lon,lat), ...]
    ("last_df", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ------------------------ Helpers ------------------------
def get_transformers(ref_lat: float, ref_lon: float) -> Tuple[Transformer, Transformer]:
    zone = int((ref_lon + 180) / 6) + 1
    epsg = 32600 + zone if ref_lat >= 0 else 32700 + zone
    try:
        fwd = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        inv = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    except Exception:
        fwd = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        inv = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    return fwd, inv

def to_meters(geom, fwd: Transformer):
    return shp_transform(lambda x, y: fwd.transform(x, y), geom)

# ------------------------ GPX parsing + schoonmaak ------------------------
def _finite(x):
    return x is not None and math.isfinite(x)

def parse_gpx_to_linestring(gpx_bytes: bytes) -> LineString:
    gpx = gpxpy.parse(io.StringIO(gpx_bytes.decode("utf-8")))
    coords: List[Tuple[float, float]] = []

    def add_point(lon, lat):
        if _finite(lon) and _finite(lat):
            coords.append((float(lon), float(lat)))

    for track in gpx.tracks:
        for segment in track.segments:
            for p in segment.points:
                add_point(p.longitude, p.latitude)

    if not coords:
        for route in gpx.routes:
            for p in route.points:
                add_point(p.longitude, p.latitude)

    # dups & mini-jitter weg
    cleaned = []
    last = None
    for xy in coords:
        if last is None or (abs(xy[0]-last[0]) > 1e-7 or abs(xy[1]-last[1]) > 1e-7):
            cleaned.append(xy); last = xy

    if len(cleaned) < 2:
        raise ValueError("Geen (genoeg) bruikbare punten in GPX (track).")

    line = LineString(cleaned)
    if not line.is_valid or line.length == 0:
        raise ValueError("Ongeldige routegeometrie in GPX.")
    return line

# ------------------------ Overpass (cache + retries) ------------------------
@st.cache_data(ttl=600, show_spinner=False)
def overpass_pois_in_bbox(bbox: Tuple[float, float, float, float]) -> List[Dict]:
    minx, miny, maxx, maxy = bbox
    query = f"""
    [out:json][timeout:60];
    (
      /* ✅ bakkers */
      node["shop"="bakery"]({miny},{minx},{maxy},{maxx});
      way["shop"="bakery"]({miny},{minx},{maxy},{maxx});
      relation["shop"="bakery"]({miny},{minx},{maxy},{maxx});

      /* koffiebars/cafés */
      node["amenity"="cafe"]({miny},{minx},{maxy},{maxx});
      way["amenity"="cafe"]({miny},{minx},{maxy},{maxx});
      relation["amenity"="cafe"]({miny},{minx},{maxy},{maxx});

      /* supermarkten */
      node["shop"="supermarket"]({miny},{minx},{maxy},{maxx});
      way["shop"="supermarket"]({miny},{minx},{maxy},{maxx});
      relation["shop"="supermarket"]({miny},{minx},{maxy},{maxx});

      /* fietswinkels */
      node["shop"="bicycle"]({miny},{minx},{maxy},{maxx});
      way["shop"="bicycle"]({miny},{minx},{maxy},{maxx});
      relation["shop"="bicycle"]({miny},{minx},{maxy},{maxx});
    );
    out center tags;
    """
    url = "https://overpass-api.de/api/interpreter"
    last_err = None
    for attempt in range(3):
        try:
            r = requests.post(url, data={"data": query}, timeout=90)
            r.raise_for_status()
            return r.json().get("elements", [])
        except RequestException as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Overpass-fout na retries: {last_err}")

def element_to_point_and_type(elem: Dict) -> Tuple[Point, Dict, str]:
    tags = elem.get("tags", {}) or {}
    if elem.get("type") == "node":
        lon, lat = elem.get("lon"), elem.get("lat")
    else:
        center = elem.get("center") or {}
        lon, lat = center.get("lon"), center.get("lat")
    if not (_finite(lon) and _finite(lat)):
        raise ValueError("POI zonder geldige coördinaten")

    if tags.get("shop") == "bakery" or tags.get("amenity") == "bakery":
        poi_type = "bakery"
    elif tags.get("amenity") == "cafe":
        text = (tags.get("name", "") + " " + tags.get("cuisine", "") + " " + tags.get("brand", "")).lower()
        poi_type = "coffee_bar" if ("coffee" in text or "koffie" in text) else "cafe"
    elif tags.get("shop") == "supermarket":
        poi_type = "supermarket"
    elif tags.get("shop") == "bicycle":
        poi_type = "bicycle_shop"
    else:
        poi_type = "other"

    return Point(float(lon), float(lat)), tags, poi_type

# ------------------------ Analyse ------------------------
def analyse_pois_simple(route_ll: LineString, buffer_m: float) -> pd.DataFrame:
    mid = route_ll.interpolate(0.5, normalized=True)
    ref_lon, ref_lat = mid.x, mid.y
    fwd, _ = get_transformers(ref_lat, ref_lon)

    route_m = to_meters(route_ll, fwd)
    if route_m.length <= 0 or not math.isfinite(route_m.length):
        raise ValueError("Route heeft lengte 0 na projectie.")
    route_buffer_m = route_m.buffer(max(1.0, float(buffer_m)))

    minx, miny, maxx, maxy = route_ll.bounds
    elements = overpass_pois_in_bbox((minx, miny, maxx, maxy))

    rows = []
    for e in elements:
        try:
            pt_ll, tags, poi_type = element_to_point_and_type(e)
            pt_m = to_meters(pt_ll, fwd)
            if not route_buffer_m.contains(pt_m):
                continue
            d_m = route_m.distance(pt_m)
            proj = route_m.project(pt_m)
            if not (math.isfinite(d_m) and math.isfinite(proj)):
                continue
            rows.append({
                "km_punt": round(proj / 1000.0, 1),
                "type": poi_type,
                "name": tags.get("name"),
                "street": tags.get("addr:street"),
                "housenumber": tags.get("addr:housenumber"),
                "postcode": tags.get("addr:postcode"),
                "city": tags.get("addr:city"),
                "opening_hours": tags.get("opening_hours"),
                "distance_m": round(d_m, 0),
                "lat": pt_ll.y,
                "lon": pt_ll.x,
            })
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if not df.empty:
        order = ["km_punt", "type", "name", "street", "housenumber", "postcode", "city",
                 "opening_hours", "distance_m", "lat", "lon"]
        df = df[order].sort_values(["km_punt", "type", "distance_m"]).reset_index(drop=True)
    return df

# ------------------------ Kaart ------------------------
def make_map(route_coords: List[Tuple[float, float]], df: pd.DataFrame) -> folium.Map:
    # route_coords = [(lon,lat), ...]
    mid_idx = max(0, (len(route_coords) // 2) - 1)
    mid_lon, mid_lat = route_coords[mid_idx]
    m = folium.Map(location=(mid_lat, mid_lon), zoom_start=12)
    folium.PolyLine(locations=[(lat, lon) for lon, lat in route_coords],
                    weight=4, opacity=0.85).add_to(m)
    for _, r in df.iterrows():
        name = r.get("name") or r.get("type")
        oh_line = r.get("opening_hours") or ""
        popup = folium.Popup(
            f"<b>{name}</b> <i>({r['type']})</i><br/>"
            f"km {r['km_punt']} — d={int(r['distance_m'])} m<br/>"
            f"{(r.get('street') or '')} {(r.get('housenumber') or '')}<br/>"
            f"{(r.get('postcode') or '')} {(r.get('city') or '')}<br/>"
            f"{oh_line}",
            max_width=320,
        )
        folium.Marker(location=(r.lat, r.lon), popup=popup).add_to(m)
    return m

# ------------------------ UI ------------------------
st.set_page_config(page_title="POI langs je route", layout="wide")
st.title("🗺️ POI langs je route — bakkerijen • koffiebars • supermarkten • fietswinkels")

with st.sidebar:
    st.markdown("**Stap 1.** Upload je **GPX Track (.gpx)**")
    gpx_file = st.file_uploader("GPX bestand", type=["gpx"], key="gpx_upload")
    buffer_m = st.number_input("Buffer rond route (meter)", min_value=50, max_value=2000, value=300, step=25)

    st.markdown("**Filters — types**")
    f_bakery = st.checkbox("Bakkerijen", value=True)
    f_cafe   = st.checkbox("Koffiebars / cafés", value=True)
    f_super  = st.checkbox("Supermarkten", value=True)
    f_bike   = st.checkbox("Fietswinkels", value=True)

    st.markdown("**Sorteren**")
    sort_by = st.selectbox("Sorteer op", ["km_punt", "name", "type", "distance_m"], index=0)
    sort_asc = st.checkbox("Oplopend", value=True)

    run_btn = st.button("Zoek POI's")

# Bewaar upload direct (blijft staan bij rerun)
if gpx_file is not None:
    st.session_state["gpx_bytes"] = gpx_file.read()

# Actie: berekenen
if run_btn:
    if not st.session_state["gpx_bytes"]:
        st.warning("Upload eerst een GPX track bestand.")
        st.stop()
    try:
        route_ll = parse_gpx_to_linestring(st.session_state["gpx_bytes"])
        st.session_state["route_coords"] = list(route_ll.coords)  # bewaar lon/lat
    except Exception as e:
        st.error(f"Fout bij inlezen GPX: {e}")
        st.stop()

    with st.spinner("POI's ophalen en filteren…"):
        try:
            df = analyse_pois_simple(LineString(st.session_state["route_coords"]), buffer_m)
            st.session_state["last_df"] = df  # bewaar resultaten
        except Exception as e:
            st.error(f"Kon POI's niet ophalen: {e}")
            st.stop()

# Weergave: toon laatste resultaten als die bestaan
if st.session_state["last_df"] is not None and st.session_state["route_coords"] is not None:
    df = st.session_state["last_df"].copy()

    # Typefilters
    keep_types = []
    if f_bakery: keep_types.append("bakery")
    if f_cafe:   keep_types.extend(["cafe", "coffee_bar"])
    if f_super:  keep_types.append("supermarket")
    if f_bike:   keep_types.append("bicycle_shop")
    if keep_types:
        df = df[df["type"].isin(keep_types)]
    else:
        df = df.iloc[0:0]

    # Sorteren
    df = df.sort_values(sort_by, ascending=sort_asc, na_position="last").reset_index(drop=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Resultaten")
        if df.empty:
            st.info("Geen POI's voor de huidige filters/buffer.")
        else:
            show_cols = ["km_punt", "type", "name", "opening_hours",
                         "distance_m", "street", "housenumber", "postcode", "city", "lat", "lon"]
            st.dataframe(df[show_cols], width='stretch')
            csv = df[show_cols].to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv, file_name="pois_langs_route.csv", mime="text/csv")

    with col2:
        st.subheader("Kaart")
        m = make_map(st.session_state["route_coords"], df)
        st_folium(m, height=600, width=None)
else:
    st.info("Upload een GPX en klik ‘Zoek POI’s’. Resultaten blijven staan na herladen.")

st.caption("Data: © OpenStreetMap contributors via Overpass API. Openingstijden komen uit OSM's 'opening_hours' tag en kunnen onvolledig of foutief zijn.")
