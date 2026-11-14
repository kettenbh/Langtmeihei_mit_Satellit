"""Satellitendaten je Feld: Sentinel-2 (NDVI, NDRE, LAI) und Sentinel-1 (VV/VH), Schnitterkennung.

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
from .snap_lai import evalscript_s2, lai_numpy

FIELDS_FILE = DATA_DIR / "felder.geojson"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
STATS_URL = "https://sh.dataspace.copernicus.eu/api/v1/statistics"
S2_START = dt.date(2017, 3, 1)
MIN_CLEAR = 0.8           # Mindestanteil wolkenfreier Feldpixel
INNER_BUFFER_M = 10.0     # Randpixel (Wege, Hecken, Wald) ausschließen

EVAL_S2 = evalscript_s2()   # NDVI, NDRE, LAI (SNAP-Netz), siehe heu/snap_lai.py

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
            # sampleCount zählt alle Pixel der Bounding Box, auch außerhalb des Feldes. Der rohe Anteil
            # erreicht deshalb nie 1; normiere_clear() rechnet ihn auf die Feldpixel um.
            r["valid_px"], r["sample_px"] = n - nod, n
            r["clear"] = (n - nod) / n if n else 0.0
            if n - nod <= 0:
                r[o] = r[f"{o}_q25"] = r[f"{o}_q75"] = np.nan
                continue
            pc = st_.get("percentiles", {})
            r[o] = pc.get("50.0", st_.get("mean"))
            r[f"{o}_q25"], r[f"{o}_q75"] = pc.get("25.0"), pc.get("75.0")
            r[f"{o}_mean"] = st_.get("mean")
            r[f"{o}_std"] = st_.get("stDev")
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
            parts.append(_parse_stats(js, ["ndvi", "ndre", "lai"]))
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
            for key in ("green", "red", "rededge1", "rededge2", "rededge3", "nir", "nir08", "swir16", "swir22", "scl"):
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
            # LAI (SNAP-Netz). Winkel aus den STAC-Metadaten; fehlt der Beobachtungswinkel, gilt
            # näherungsweise nadirnah (5°) und senkrechter relativer Azimut (Fallback, ungenauer als CDSE).
            pr = it.properties
            sun_zen = 90.0 - float(pr.get("view:sun_elevation", 45.0))
            sun_az = float(pr.get("view:sun_azimuth", 160.0))
            view_zen = float(pr.get("view:incidence_angle", pr.get("s2:mean_incidence_zenith_angle", 5.0)))
            view_az = float(pr.get("view:azimuth", pr.get("s2:mean_incidence_azimuth_angle", sun_az - 90.0)))
            lai = lai_numpy({"B03": arrs["green"], "B04": arrs["red"], "B05": arrs["rededge1"],
                             "B06": arrs["rededge2"], "B07": arrs["rededge3"], "B8A": arrs["nir08"],
                             "B11": arrs["swir16"], "B12": arrs["swir22"]}, view_zen, sun_zen, sun_az, view_az)
            r = {"date": pd.Timestamp(day), "clear": ok.sum() / n}
            for name, v in (("ndvi", ndvi), ("ndre", ndre), ("lai", lai)):
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
def normiere_clear(df: pd.DataFrame, min_treffer: int = 3) -> pd.DataFrame:
    """Klar-Anteil bezogen auf die Feldpixel statt auf die Bounding Box.

    Die Statistical API zählt in sampleCount alle Pixel des umschließenden Rechtecks. Pixel außerhalb
    des Feldes sind immer noData, deshalb lag der rohe Anteil bei Wiese Ost höchstens bei 0,70.
    Bezug ist die größte Zahl gültiger Pixel der Reihe (= wolkenfreie Aufnahme). Normiert wird nur,
    wenn mindestens `min_treffer` Aufnahmen diesen Wert (±2 %) erreichen. Ältere Caches ohne Pixelzahlen
    werden über den größten Klar-Anteil normiert. Mehrfaches Anwenden ändert nichts.
    """
    if df is None or df.empty or "clear" not in df:
        return df
    df = df.copy()
    if "valid_px" in df and df["valid_px"].notna().any():
        v = df["valid_px"].astype(float)
        ref = v.max()
        if ref > 0 and (v >= 0.98 * ref).sum() >= min_treffer:
            df["clear"] = np.where(v.notna(), (v / ref).clip(upper=1.0), df["clear"])
        return df
    c = df["clear"].astype(float)
    ref = c.max()
    if 0 < ref < 0.98 and (c >= 0.98 * ref).sum() >= min_treffer:
        df["clear"] = (c / ref).clip(upper=1.0)
    return df


def cache_path(field_id: str, kind: str):
    return DATA_DIR / f"sat_{kind}_{field_id}.csv"


def load_cached(field_id: str, kind: str) -> pd.DataFrame | None:
    p = cache_path(field_id, kind)
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["date"], index_col="date")
    return normiere_clear(df) if kind == "s2" else df


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
        if kind == "s2" and old is not None and "lai" not in old.columns:
            old = None   # Cache aus der Zeit vor dem LAI-Evalscript: einmal komplett neu laden
        start = S2_START if old is None or old.empty else (old.index.max().date() - dt.timedelta(days=10))
        if progress:
            progress(0.0, f"{field['name']}: {kind.upper()} ab {start}")
        new = fetch_cdse(field, kind, start, end, client) if use_cdse else fetch_earth_search(field, start, end, progress)
        df = new if old is None else pd.concat([old[old.index < pd.Timestamp(start)], new])
        if not df.empty:
            df = df[~df.index.duplicated(keep="last")].sort_index()
            if kind == "s2" and use_cdse:
                df = normiere_clear(df)
            df.to_csv(cache_path(field["id"], kind), index_label="date")
        out[kind] = df
    if "s1" not in out:
        out["s1"] = load_cached(field["id"], "s1")
    return out


# ---------------------------------------------------------------------------
# Schnitterkennung
# ---------------------------------------------------------------------------
NUTZUNG_GRUENLAND = "Grünland"
NUTZUNG_ACKER = "Acker/Umbruch"


def nutzung_je_jahr(s2: pd.DataFrame, max_ndvi_offen: float = 0.3, min_offen: int = 2) -> dict[int, str]:
    """Nutzung je Jahr aus der NDVI-Reihe: Grünland ist von März bis Oktober nie großflächig offen.

    Ein Jahr gilt als "Acker/Umbruch", wenn mindestens `min_offen` klare Aufnahmen von März bis Oktober
    einen Feld-Median-NDVI unter `max_ndvi_offen` haben (offener Boden: Saatbett, Stoppel nach der
    Getreideernte, Umbruch). Eine einzelne Aufnahme reicht nicht, weil Dunst, den die SCL-Maske nicht
    erkennt, einzelne Tage auf NDVI 0,1 bis 0,2 drücken kann (z. B. 29.10.2024 auf allen drei Feldern).
    Gemähtes Grünland bleibt dagegen über 0,45, auch in Trockenphasen.
    """
    if s2 is None or s2.empty or "ndvi" not in s2:
        return {}
    c = s2[(s2["clear"] >= MIN_CLEAR) & s2["ndvi"].notna()]
    out = {}
    for y in sorted(set(s2.index.year)):
        d = c[(c.index.year == y) & (c.index.month >= 3) & (c.index.month <= 10)]
        offen = int((d["ndvi"] < max_ndvi_offen).sum())
        out[int(y)] = NUTZUNG_ACKER if offen >= min_offen else NUTZUNG_GRUENLAND
    return out


SCHNITT_SPALTEN = ["datum", "frueh", "spaet", "ndvi_vor", "ndvi_nach", "lai_vor", "lai_nach",
                   "unsicherheit_tage", "s1_bestaetigt", "kriterium", "nutzung"]


def detect_cuts(s2: pd.DataFrame, s1: pd.DataFrame | None = None, min_before: float = 0.6,
                ndvi_drop: float = 0.12, ndvi_drop_stark: float = 0.2, lai_min_vor: float = 1.5,
                lai_drop_rel: float = 0.35, lai_max_nach: float = 3.0, lai_ndvi_drop: float = 0.04,
                fenster_tage: int = 40, max_gap_days: int = 40, min_sep_days: int = 25,
                rebound_days: int = 12, min_ndvi_nach: float = 0.25, nur_gruenland: bool = True) -> pd.DataFrame:
    """Schnitte als Einbruch von einem Hoch auf ein Tief (klare Aufnahmen April bis Oktober).

    Kandidat ist jedes Tief: eine Aufnahme, nach der der NDVI wieder steigt, die nächste Aufnahme mehr
    als 15 Tage später kommt oder die letzte des Jahres.
    Das Hoch ist die Aufnahme mit dem größten LAI (ohne LAI: NDVI) in den fenster_tage davor. So wird
    auch ein Schnitt erkannt, der sich über mehrere Aufnahmen verteilt (Mähen, Heu liegt einige Tage,
    Abfahren), was paarweise Vergleiche bei dichter Aufnahmefolge übersehen. Ein Schnitt liegt vor, wenn
    der NDVI am Hoch ≥ min_before war und
      (a) der LAI um mindestens lai_drop_rel (relativ) fällt, am Hoch ≥ lai_min_vor, am Tief ≤ lai_max_nach,
          und der NDVI gleichzeitig um mindestens lai_ndvi_drop sinkt ("LAI"), oder
      (b) der NDVI um mindestens ndvi_drop_stark fällt ("NDVI stark"), oder
      (c) der NDVI um mindestens ndvi_drop fällt und der LAI dabei sinkt ("NDVI").
    Der LAI reagiert auf einen Schnitt deutlich stärker als der NDVI, der bei Grünland wegen der grünen
    Stoppel oft nur um 0,05 bis 0,15 fällt. Die NDVI-Bedingung in (a) fängt einzelne LAI-Ausreißer ab.

    Artefakte (Dunst, Schatten, den die SCL-Maske nicht erkennt):
      - Tief unter min_ndvi_nach: gemähtes Grünland bleibt grün, so tiefe Werte sind Umbruch oder Dunst;
      - die nächste klare Aufnahme innerhalb rebound_days erreicht wieder NDVI und 90 % des LAI vom Hoch.

    Zeitfenster: "frueh" ist die letzte Aufnahme, die noch nahe am Hoch liegt, "spaet" die erste, die
    schon nahe am Tief liegt (je 25 % der NDVI-Spanne). Datum = Mitte, Unsicherheit = halber Abstand;
    liegen mehr als max_gap_days dazwischen, wird der Kandidat verworfen. Treffer mit weniger als
    min_sep_days Abstand werden zusammengefasst (stärkerer Einbruch gewinnt).
    Sentinel-1: Abfall der VH-Rückstreuung um ≥ 1 dB im selben Zeitraum gilt als Bestätigung.
    In Jahren, die nutzung_je_jahr als Acker/Umbruch einstuft, ist ein NDVI-Abfall meist Ernte,
    Abreife oder Umbruch; diese Jahre werden mit nur_gruenland=True übersprungen.
    """
    leer = pd.DataFrame({c: pd.Series(dtype="datetime64[ns]" if c in ("datum", "frueh", "spaet") else
                                      ("object" if c in ("kriterium", "nutzung", "s1_bestaetigt") else "float"))
                         for c in SCHNITT_SPALTEN})
    if s2 is None or s2.empty or "ndvi" not in s2:
        return leer
    nutzung = nutzung_je_jahr(s2)
    c = s2[(s2["clear"] >= MIN_CLEAR) & s2["ndvi"].notna()]
    c = c[(c.index.month >= 4) & (c.index.month <= 10)]
    rows = []
    for jahr in sorted(set(c.index.year)):
        nu = nutzung.get(int(jahr), NUTZUNG_GRUENLAND)
        if nur_gruenland and nu != NUTZUNG_GRUENLAND:
            continue
        cj = c[c.index.year == jahr]
        dates = list(cj.index)
        nd = cj["ndvi"].to_numpy(dtype=float)
        la = cj["lai"].to_numpy(dtype=float) if "lai" in cj else np.full(len(cj), np.nan)
        for i in range(1, len(dates)):
            if i + 1 < len(dates) and nd[i + 1] < nd[i] and (dates[i + 1] - dates[i]).days <= 15:
                continue                                    # kein Tief: es geht gleich weiter abwärts
            fen = [j for j in range(i) if (dates[i] - dates[j]).days <= fenster_tage]
            if not fen:
                continue
            lai_ok = np.isfinite(la[i]) and all(np.isfinite(la[j]) for j in fen)
            h = max(fen, key=lambda j: la[j]) if lai_ok else max(fen, key=lambda j: nd[j])
            d_ndvi = max(nd[j] for j in fen if j >= h) - nd[i]
            nd_hoch = nd[h]
            if nd_hoch < min_before or d_ndvi <= 0:
                continue
            rel = (la[h] - la[i]) / la[h] if lai_ok and la[h] > 0 else np.nan
            krit = None
            if lai_ok and la[h] >= lai_min_vor and rel >= lai_drop_rel and la[i] <= lai_max_nach \
                    and d_ndvi >= lai_ndvi_drop:
                krit = "LAI"
            elif d_ndvi >= ndvi_drop_stark:
                krit = "NDVI stark"
            elif d_ndvi >= ndvi_drop and (not lai_ok or rel > 0):
                krit = "NDVI"
            if krit is None or nd[i] < min_ndvi_nach:
                continue
            if i + 1 < len(dates) and (dates[i + 1] - dates[i]).days <= rebound_days:
                nd_back = nd[i + 1] >= nd_hoch - 0.03
                la_back = (not lai_ok) or (np.isfinite(la[i + 1]) and la[i + 1] >= 0.9 * la[h])
                if nd_back and la_back:
                    continue
            # Zeitfenster eingrenzen
            hi_ref = max(nd[j] for j in range(h, i))
            span = hi_ref - nd[i]
            f = max(j for j in range(h, i) if nd[j] >= hi_ref - 0.25 * span)
            s_ = min(j for j in range(f + 1, i + 1) if nd[j] <= nd[i] + 0.25 * span)
            a, b = dates[f], dates[s_]
            gap = (b - a).days
            if gap > max_gap_days:
                continue
            mid = a + (b - a) / 2
            conf = None
            if s1 is not None and not s1.empty and "vh_db" in s1:
                w = s1.loc[a - pd.Timedelta(days=12): b + pd.Timedelta(days=12), "vh_db"].dropna()
                before, after = w[w.index <= mid], w[w.index > mid]
                if len(before) and len(after):
                    conf = bool(before.iloc[-1] - after.iloc[0] >= 1.0)
            rows.append({"datum": mid.normalize(), "frueh": a, "spaet": b,
                         "ndvi_vor": round(float(hi_ref), 3), "ndvi_nach": round(float(nd[i]), 3),
                         "lai_vor": round(float(la[h]), 2) if lai_ok else np.nan,
                         "lai_nach": round(float(la[i]), 2) if lai_ok else np.nan,
                         "unsicherheit_tage": gap / 2, "s1_bestaetigt": conf, "kriterium": krit, "nutzung": nu,
                         "_staerke": rel if np.isfinite(rel) else d_ndvi})
    df = pd.DataFrame(rows)
    if df.empty:
        return leer     # keine Schnitte erkannt: leere Tabelle mit allen Spalten
    df = df.sort_values("datum").reset_index(drop=True)
    keep = [0]
    for j in range(1, len(df)):
        if (df.loc[j, "datum"] - df.loc[keep[-1], "datum"]).days < min_sep_days:
            if df.loc[j, "_staerke"] > df.loc[keep[-1], "_staerke"]:
                keep[-1] = j
        else:
            keep.append(j)
    return df.loc[keep, SCHNITT_SPALTEN].reset_index(drop=True)


def synthetic_s2(model_trace_by_year: dict[int, np.ndarray], seed: int = 3) -> pd.DataFrame:
    """Kunst-Satellitenreihe aus Modell-LAI (nur Testmodus): NDVI = 0.9 - 0.75·exp(-0.6·LAI) + Rauschen,
    LAI = Modell-LAI + Rauschen (σ = max(0,5; 0,25·LAI)),
    Aufnahmen alle 5 Tage, 55 % bewölkt."""
    rng = np.random.default_rng(seed)
    rows = []
    for y, tr in model_trace_by_year.items():
        for d in range(0, 365, 5):
            lai = tr[d, 7] if tr.shape[1] > 7 else 2.0
            ndvi = 0.9 - 0.75 * math.exp(-0.6 * lai) + rng.normal(0, 0.03)
            lai_obs = max(0.0, lai + rng.normal(0, max(0.5, 0.25 * lai)))
            clear = 1.0 if rng.random() > 0.55 else rng.random() * 0.5
            day = dt.date(y, 1, 1) + dt.timedelta(days=d)
            rows.append({"date": pd.Timestamp(day), "clear": clear, "ndvi": ndvi,
                         "ndvi_q25": ndvi - 0.03, "ndvi_q75": ndvi + 0.03, "ndre": ndvi * 0.6,
                         "lai": lai_obs, "lai_q25": lai_obs - 0.3, "lai_q75": lai_obs + 0.3})
    return pd.DataFrame(rows).set_index("date")
