"""Satellitendaten je Feld: Sentinel-2 (NDVI, NDRE) und Sentinel-1 (VV/VH), Schnitterkennung.

Zwei Wege:
  1. CDSE Sentinel Hub Statistical API (empfohlen): braucht einen kostenlosen OAuth-Client,
     Zugangsdaten in .streamlit/secrets.toml  ->  [cdse] client_id / client_secret
     (alternativ Umgebungsvariablen CDSE_CLIENT_ID / CDSE_CLIENT_SECRET).
  2. AWS Earth Search (ohne Konto): liest Sentinel-2-COGs direkt; braucht zusätzlich
     `pip install pystac-client rasterio`. Langsamer, nur Sentinel-2.

Ergebnisse werden je Feld als CSV in data/ gecacht und nur inkrementell ergänzt.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import time

import numpy as np
import pandas as pd
import requests

from .config import DATA_DIR, ROOT

FIELDS_FILE = DATA_DIR / "felder.geojson"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
STATS_URL = "https://sh.dataspace.copernicus.eu/api/v1/statistics"
S2_START = dt.date(2017, 3, 1)
MIN_CLEAR = 0.8           # Mindestanteil wolkenfreier Feldpixel
INNER_BUFFER_M = 10.0     # Randpixel (Wege, Hecken, Wald) ausschließen

EVAL_S2 = """//VERSION=3
function setup() {
  return {
    input: [{bands: ["B04", "B05", "B08", "B8A", "SCL", "dataMask"]}],
    output: [
      {id: "ndvi", bands: 1, sampleType: "FLOAT32"},
      {id: "ndre", bands: 1, sampleType: "FLOAT32"},
      {id: "dataMask", bands: 1}
    ]
  };
}
function evaluatePixel(s) {
  // SCL 4 = Vegetation, 5 = unbewachsen; Wolken, Schatten, Schnee, Wasser fallen heraus
  var ok = (s.SCL == 4 || s.SCL == 5) ? 1 : 0;
  var ndvi = (s.B08 - s.B04) / (s.B08 + s.B04 + 1e-6);
  var ndre = (s.B8A - s.B05) / (s.B8A + s.B05 + 1e-6);
  return {ndvi: [ndvi], ndre: [ndre], dataMask: [s.dataMask * ok]};
}"""

EVAL_S1 = """//VERSION=3
function setup() {
  return {
    input: [{bands: ["VV", "VH", "dataMask"]}],
    output: [
      {id: "vv", bands: 1, sampleType: "FLOAT32"},
      {id: "vh", bands: 1, sampleType: "FLOAT32"},
      {id: "dataMask", bands: 1}
    ]
  };
}
function evaluatePixel(s) {
  return {vv: [s.VV], vh: [s.VH], dataMask: [s.dataMask]};
}"""


# ---------------------------------------------------------------------------
# Felder
# ---------------------------------------------------------------------------
def load_fields() -> list[dict]:
    if not FIELDS_FILE.exists():
        return []
    fc = json.loads(FIELDS_FILE.read_text(encoding="utf-8"))
    out = []
    for i, f in enumerate(fc.get("features", [])):
        pr = f.get("properties") or {}
        ring = f["geometry"]["coordinates"][0]
        out.append({
            "id": pr.get("id", f"feld_{i + 1}"),
            "name": pr.get("name", f"Feld {i + 1}"),
            "flaeche_ha": pr.get("flaeche_ha") or round(polygon_area_ha(ring), 2),
            "geometry": f["geometry"],
            "lat": float(np.mean([p[1] for p in ring])),
            "lon": float(np.mean([p[0] for p in ring])),
        })
    return out


def _to_local(ring, lat0):
    k = math.cos(math.radians(lat0))
    return [(math.radians(x) * 6371000 * k, math.radians(y) * 6371000) for x, y, *_ in ring]


def polygon_area_ha(ring) -> float:
    lat0 = float(np.mean([p[1] for p in ring]))
    xy = _to_local(ring, lat0)
    a = 0.5 * abs(sum(xy[i][0] * xy[i + 1][1] - xy[i + 1][0] * xy[i][1] for i in range(len(xy) - 1)))
    return a / 1e4


def inner_buffer(geometry: dict, meters: float = INNER_BUFFER_M) -> dict:
    """Polygon um 'meters' nach innen verkleinern (lokale Projektion). Braucht shapely; ohne shapely unverändert."""
    try:
        from shapely.geometry import shape, mapping
        from shapely.affinity import scale
    except ImportError:
        return geometry
    ring = geometry["coordinates"][0]
    lat0 = float(np.mean([p[1] for p in ring]))
    k = math.cos(math.radians(lat0))
    mx, my = 6371000 * math.pi / 180 * k, 6371000 * math.pi / 180
    g = scale(shape(geometry), xfact=mx, yfact=my, origin=(0, 0))
    g = g.buffer(-meters)
    if g.is_empty or g.area < 2000:  # zu klein: Original verwenden
        return geometry
    g = scale(g, xfact=1 / mx, yfact=1 / my, origin=(0, 0))
    return mapping(g)


# ---------------------------------------------------------------------------
# CDSE Zugang
# ---------------------------------------------------------------------------
def cdse_credentials() -> tuple[str, str] | None:
    cid, sec = os.environ.get("CDSE_CLIENT_ID"), os.environ.get("CDSE_CLIENT_SECRET")
    if cid and sec:
        return cid, sec
    f = ROOT / ".streamlit" / "secrets.toml"
    if f.exists():
        import tomllib
        s = tomllib.loads(f.read_text(encoding="utf-8")).get("cdse", {})
        if s.get("client_id") and s.get("client_secret"):
            return s["client_id"], s["client_secret"]
    return None


class CDSE:
    def __init__(self, client_id: str, client_secret: str):
        self.cid, self.sec = client_id, client_secret
        self._token, self._exp = None, 0.0

    def token(self) -> str:
        if self._token and time.time() < self._exp - 60:
            return self._token
        r = requests.post(TOKEN_URL, data={"grant_type": "client_credentials",
                                           "client_id": self.cid, "client_secret": self.sec}, timeout=60)
        r.raise_for_status()
        js = r.json()
        self._token, self._exp = js["access_token"], time.time() + js.get("expires_in", 600)
        return self._token

    def statistics(self, body: dict) -> dict:
        for attempt in range(4):
            r = requests.post(STATS_URL, json=body, timeout=300,
                              headers={"Authorization": f"Bearer {self.token()}"})
            if r.status_code == 429:  # Ratenbegrenzung
                time.sleep(5 * (attempt + 1))
                continue
            if r.status_code >= 400:
                raise RuntimeError(f"CDSE {r.status_code}: {r.text[:400]}")
            return r.json()
        raise RuntimeError("CDSE: zu viele Anfragen, bitte später erneut versuchen")


def _res_deg(lat: float, meters: float) -> tuple[float, float]:
    return meters / (111320 * math.cos(math.radians(lat))), meters / 110540


def _stats_body(geometry, lat, start, end, data, evalscript, res_m):
    rx, ry = _res_deg(lat, res_m)
    return {
        "input": {
            "bounds": {"geometry": geometry,
                       "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"}},
            "data": [data],
        },
        "aggregation": {
            "timeRange": {"from": f"{start}T00:00:00Z", "to": f"{end}T23:59:59Z"},
            "aggregationInterval": {"of": "P1D"},
            "evalscript": evalscript,
            "resx": rx, "resy": ry,
        },
        "calculations": {"default": {"statistics": {"default": {"percentiles": {"k": [25, 50, 75]}}}}},
    }


def _parse_stats(js: dict, outputs: list[str]) -> pd.DataFrame:
    rows = []
    for item in js.get("data", []):
        if "error" in item:
            continue
        date = item["interval"]["from"][:10]
        r = {"date": date}
        for o in outputs:
            st_ = item["outputs"][o]["bands"]["B0"]["stats"]
            n, nod = st_.get("sampleCount", 0), st_.get("noDataCount", 0)
            r["clear"] = (n - nod) / n if n else 0.0
            if n - nod <= 0:
                r[o] = r[f"{o}_q25"] = r[f"{o}_q75"] = np.nan
                continue
            pc = st_.get("percentiles", {})
            r[o] = pc.get("50.0", st_.get("mean"))
            r[f"{o}_q25"], r[f"{o}_q75"] = pc.get("25.0"), pc.get("75.0")
            r[f"{o}_mean"] = st_.get("mean")
        rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def fetch_cdse(field: dict, kind: str, start: dt.date, end: dt.date, client: CDSE) -> pd.DataFrame:
    geom = inner_buffer(field["geometry"])
    parts = []
    y0 = start
    while y0 <= end:  # Jahresweise anfragen (Laufzeitgrenze der API)
        y1 = min(end, dt.date(y0.year, 12, 31))
        if kind == "s2":
            data = {"type": "sentinel-2-l2a", "dataFilter": {"maxCloudCoverage": 90}}
            js = client.statistics(_stats_body(geom, field["lat"], y0, y1, data, EVAL_S2, 10))
            parts.append(_parse_stats(js, ["ndvi", "ndre"]))
        else:
            data = {"type": "sentinel-1-grd",
                    "dataFilter": {"acquisitionMode": "IW", "polarization": "DV", "resolution": "HIGH",
                                   "orbitDirection": "ASCENDING"},
                    "processing": {"backCoeff": "GAMMA0_TERRAIN", "orthorectify": True,
                                   "demInstance": "COPERNICUS_30", "speckleFilter": {"type": "NONE"}}}
            js = client.statistics(_stats_body(geom, field["lat"], y0, y1, data, EVAL_S1, 20))
            df = _parse_stats(js, ["vv", "vh"])
            if not df.empty:  # linear -> dB (Mittelwert im Linearen bilden)
                for o in ("vv", "vh"):
                    df[f"{o}_db"] = 10 * np.log10(df[f"{o}_mean"].clip(lower=1e-5))
            parts.append(df)
        y0 = dt.date(y0.year + 1, 1, 1)
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts) if parts else pd.DataFrame()


# ---------------------------------------------------------------------------
# AWS Earth Search (ohne Konto, nur S2)
# ---------------------------------------------------------------------------
def fetch_earth_search(field: dict, start: dt.date, end: dt.date, progress=None) -> pd.DataFrame:
    try:
        import pystac_client
        import rasterio
        from rasterio.features import geometry_mask
        from rasterio.warp import transform_geom
        from rasterio.windows import from_bounds
    except ImportError as e:
        raise RuntimeError("Für Earth Search bitte installieren: pip install pystac-client rasterio") from e

    geom = inner_buffer(field["geometry"])
    cat = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")
    items = list(cat.search(collections=["sentinel-2-l2a"], intersects=geom,
                            datetime=f"{start}/{end}", query={"eo:cloud_cover": {"lt": 90}}).items())
    rows = []
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
    seen = set()
    for i, it in enumerate(sorted(items, key=lambda x: x.datetime)):
        day = it.datetime.date()
        if progress:
            progress(i / max(1, len(items)), f"{day} ({i + 1}/{len(items)})")
        if day in seen:
            continue
        try:
            arrs = {}
            with rasterio.open(it.assets["scl"].href) as src:
                g = transform_geom("EPSG:4326", src.crs, geom)
                xs = [p[0] for p in g["coordinates"][0]]
                ys = [p[1] for p in g["coordinates"][0]]
                bounds = (min(xs), min(ys), max(xs), max(ys))
            for key in ("red", "nir", "rededge1", "nir08", "scl"):
                a = it.assets[key]
                with rasterio.open(a.href) as src:
                    win = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
                    shape_ = (max(1, int(round((bounds[3] - bounds[1]) / 10))), max(1, int(round((bounds[2] - bounds[0]) / 10))))
                    data = src.read(1, window=win, out_shape=shape_, boundless=True, fill_value=0).astype(float)
                    if key != "scl":
                        rb = (a.extra_fields.get("raster:bands") or [{}])[0]
                        data = data * rb.get("scale", 1e-4) + rb.get("offset", 0.0)
                    arrs[key] = data
            tr = rasterio.transform.from_bounds(*bounds, arrs["scl"].shape[1], arrs["scl"].shape[0])
            inside = ~geometry_mask([g], out_shape=arrs["scl"].shape, transform=tr)
            ok = inside & np.isin(arrs["scl"], [4, 5])
            n = inside.sum()
            if n == 0:
                continue
            ndvi = (arrs["nir"] - arrs["red"]) / (arrs["nir"] + arrs["red"] + 1e-6)
            ndre = (arrs["nir08"] - arrs["rededge1"]) / (arrs["nir08"] + arrs["rededge1"] + 1e-6)
            r = {"date": pd.Timestamp(day), "clear": ok.sum() / n}
            for name, v in (("ndvi", ndvi), ("ndre", ndre)):
                vv = v[ok]
                r[name] = np.median(vv) if len(vv) else np.nan
                r[f"{name}_q25"] = np.percentile(vv, 25) if len(vv) else np.nan
                r[f"{name}_q75"] = np.percentile(vv, 75) if len(vv) else np.nan
            rows.append(r)
            seen.add(day)
        except Exception:  # einzelne defekte Szenen überspringen
            continue
    df = pd.DataFrame(rows)
    return df.set_index("date").sort_index() if not df.empty else df


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
def cache_path(field_id: str, kind: str):
    return DATA_DIR / f"sat_{kind}_{field_id}.csv"


def load_cached(field_id: str, kind: str) -> pd.DataFrame | None:
    p = cache_path(field_id, kind)
    if not p.exists():
        return None
    return pd.read_csv(p, parse_dates=["date"], index_col="date")


def update(field: dict, backend: str = "auto", progress=None) -> dict[str, pd.DataFrame]:
    """Lädt fehlende Satellitendaten nach und gibt {'s2': df, 's1': df} zurück."""
    end = dt.date.today()
    out = {}
    creds = cdse_credentials()
    use_cdse = backend == "cdse" or (backend == "auto" and creds is not None)
    if use_cdse and creds is None:
        raise RuntimeError("Keine CDSE-Zugangsdaten gefunden (.streamlit/secrets.toml, Abschnitt [cdse]).")
    client = CDSE(*creds) if use_cdse else None
    kinds = ("s2", "s1") if use_cdse else ("s2",)
    for kind in kinds:
        old = load_cached(field["id"], kind)
        start = S2_START if old is None or old.empty else (old.index.max().date() - dt.timedelta(days=10))
        if progress:
            progress(0.0, f"{field['name']}: {kind.upper()} ab {start}")
        new = fetch_cdse(field, kind, start, end, client) if use_cdse else fetch_earth_search(field, start, end, progress)
        df = new if old is None else pd.concat([old[old.index < pd.Timestamp(start)], new])
        if not df.empty:
            df = df[~df.index.duplicated(keep="last")].sort_index()
            df.to_csv(cache_path(field["id"], kind), index_label="date")
        out[kind] = df
    if "s1" not in out:
        out["s1"] = load_cached(field["id"], "s1")
    return out


# ---------------------------------------------------------------------------
# Schnitterkennung
# ---------------------------------------------------------------------------
def detect_cuts(s2: pd.DataFrame, s1: pd.DataFrame | None = None, min_drop: float = 0.15,
                min_before: float = 0.6, max_gap_days: int = 25, min_sep_days: int = 25) -> pd.DataFrame:
    """Schnitt = NDVI-Abfall um mindestens min_drop zwischen zwei klaren Aufnahmen
    (vorher ≥ min_before, Abstand ≤ max_gap_days). Geschätztes Datum: Mitte zwischen den Aufnahmen.
    Sentinel-1: Abfall der VH-Rückstreuung um ≥ 1 dB im selben Zeitraum gilt als Bestätigung."""
    if s2 is None or s2.empty:
        return pd.DataFrame(columns=["datum", "frueh", "spaet", "ndvi_vor", "ndvi_nach", "unsicherheit_tage", "s1_bestaetigt"])
    c = s2[(s2["clear"] >= MIN_CLEAR) & s2["ndvi"].notna()]
    c = c[(c.index.month >= 4) & (c.index.month <= 10)]
    rows = []
    dates, vals = list(c.index), list(c["ndvi"])
    for i in range(1, len(dates)):
        gap = (dates[i] - dates[i - 1]).days
        drop = vals[i - 1] - vals[i]
        if gap <= max_gap_days and drop >= min_drop and vals[i - 1] >= min_before:
            mid = dates[i - 1] + (dates[i] - dates[i - 1]) / 2
            conf = None
            if s1 is not None and not s1.empty and "vh_db" in s1:
                w = s1.loc[dates[i - 1] - pd.Timedelta(days=12): dates[i] + pd.Timedelta(days=12), "vh_db"].dropna()
                if len(w) >= 2:
                    before = w[w.index <= mid]
                    after = w[w.index > mid]
                    if len(before) and len(after):
                        conf = bool(before.iloc[-1] - after.iloc[0] >= 1.0)
            rows.append({"datum": mid.normalize(), "frueh": dates[i - 1], "spaet": dates[i],
                         "ndvi_vor": round(vals[i - 1], 3), "ndvi_nach": round(vals[i], 3),
                         "unsicherheit_tage": gap / 2, "s1_bestaetigt": conf})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Doppelte Treffer desselben Schnitts zusammenfassen
    keep = [0]
    for j in range(1, len(df)):
        if (df.loc[j, "datum"] - df.loc[keep[-1], "datum"]).days < min_sep_days:
            if df.loc[j, "ndvi_vor"] - df.loc[j, "ndvi_nach"] > df.loc[keep[-1], "ndvi_vor"] - df.loc[keep[-1], "ndvi_nach"]:
                keep[-1] = j
        else:
            keep.append(j)
    return df.loc[keep].reset_index(drop=True)


def synthetic_s2(model_trace_by_year: dict[int, np.ndarray], seed: int = 3) -> pd.DataFrame:
    """Kunst-Satellitenreihe aus Modell-LAI (nur Testmodus): NDVI = 0.9 - 0.75·exp(-0.6·LAI) + Rauschen,
    Aufnahmen alle 5 Tage, 55 % bewölkt."""
    rng = np.random.default_rng(seed)
    rows = []
    for y, tr in model_trace_by_year.items():
        for d in range(0, 365, 5):
            lai = tr[d, 7] if tr.shape[1] > 7 else 2.0
            ndvi = 0.9 - 0.75 * math.exp(-0.6 * lai) + rng.normal(0, 0.03)
            clear = 1.0 if rng.random() > 0.55 else rng.random() * 0.5
            day = dt.date(y, 1, 1) + dt.timedelta(days=d)
            rows.append({"date": pd.Timestamp(day), "clear": clear, "ndvi": ndvi,
                         "ndvi_q25": ndvi - 0.03, "ndvi_q75": ndvi + 0.03, "ndre": ndvi * 0.6})
    return pd.DataFrame(rows).set_index("date")
