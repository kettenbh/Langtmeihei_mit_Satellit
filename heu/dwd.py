"""DWD Climate Data Center (CDC): tägliche Klimadaten (KL) der nächstgelegenen Station.

Quelle: https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/kl/
Frei, ohne Konto. Die Station wird aus der offiziellen Stationsliste nach Abstand gewählt
(keine fest eingetragene Stationsnummer). Die Verdunstung ET0 rechnet dieses Modul selbst
nach FAO-56 Penman-Monteith (Allen et al. 1998), weil der DWD sie in KL nicht mitliefert.

Ausgabe im selben Spaltenformat wie heu/weather.py (tmean, tmax, tmin, rain, rad, et0, wind)
plus rh (relative Feuchte, %). Cache: data/dwd_kl_<id>.csv
"""
from __future__ import annotations

import io
import math
import os
import re
import zipfile

import numpy as np
import pandas as pd
import requests

from .config import DATA_DIR, STANDORT

BASE = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/kl/"
STATIONS_URL = BASE + "historical/KL_Tageswerte_Beschreibung_Stationen.txt"


# ---------------------------------------------------------------------------
# Stationen
# ---------------------------------------------------------------------------
def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def station_list() -> pd.DataFrame:
    cache = DATA_DIR / "dwd_kl_stationen.txt"
    if cache.exists():
        txt = cache.read_text(encoding="latin-1")
    else:
        r = requests.get(STATIONS_URL, timeout=60)
        r.raise_for_status()
        txt = r.content.decode("latin-1")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_text(txt, encoding="latin-1")
    rows = []
    for line in txt.splitlines():
        parts = line.split()
        if len(parts) < 7 or not parts[0].isdigit():
            continue
        try:
            rows.append({
                "id": int(parts[0]),
                "von": pd.to_datetime(parts[1], format="%Y%m%d"),
                "bis": pd.to_datetime(parts[2], format="%Y%m%d"),
                "hoehe": float(parts[3]), "lat": float(parts[4]), "lon": float(parts[5]),
                "name": " ".join(p for p in parts[6:] if p not in ("Frei",))[:60],
            })
        except ValueError:
            continue
    return pd.DataFrame(rows)


def nearest_stations(lat: float = STANDORT.lat, lon: float = STANDORT.lon, start_year: int = 1991,
                     n: int = 5) -> pd.DataFrame:
    """Nächste Stationen, die seit start_year Daten haben und noch aktiv sind."""
    st = station_list()
    st["km"] = _haversine_km(lat, lon, st["lat"], st["lon"])
    ok = st[(st["von"].dt.year <= start_year) & (st["bis"] >= pd.Timestamp.today() - pd.Timedelta(days=60))]
    return ok.sort_values("km").head(n).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Abruf
# ---------------------------------------------------------------------------
def _read_zip(content: bytes) -> pd.DataFrame:
    z = zipfile.ZipFile(io.BytesIO(content))
    name = next(n for n in z.namelist() if n.startswith("produkt_klima_tag"))
    df = pd.read_csv(z.open(name), sep=";", na_values=[-999, "-999"], skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["MESS_DATUM"].astype(str), format="%Y%m%d")
    return df.set_index("date")


def _fetch_station(sid: int) -> pd.DataFrame:
    idx = requests.get(BASE + "historical/", timeout=60)
    idx.raise_for_status()
    m = re.findall(rf"tageswerte_KL_{sid:05d}_\d{{8}}_\d{{8}}_hist\.zip", idx.text)
    parts = []
    if m:
        r = requests.get(BASE + "historical/" + sorted(set(m))[-1], timeout=180)
        r.raise_for_status()
        parts.append(_read_zip(r.content))
    r = requests.get(BASE + f"recent/tageswerte_KL_{sid:05d}_akt.zip", timeout=120)
    if r.ok:
        parts.append(_read_zip(r.content))
    if not parts:
        raise RuntimeError(f"Keine KL-Dateien für DWD-Station {sid} gefunden.")
    raw = pd.concat(parts)
    return raw[~raw.index.duplicated(keep="last")].sort_index()


# ---------------------------------------------------------------------------
# ET0 nach FAO-56
# ---------------------------------------------------------------------------
def _es(t):
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))


def et0_fao56(dates: pd.DatetimeIndex, tmax, tmin, rh, wind10_ms, sun_h, lat: float, z: float):
    """Tägliche Gras-Referenzverdunstung [mm] und Globalstrahlung [MJ/m²].

    Strahlung aus Sonnenscheindauer (Ångström, a=0,25, b=0,50); fehlt sie, nach Hargreaves (kRs=0,16).
    Fehlt die Feuchte, wird ea aus Tmin geschätzt (FAO-56 Gl. 48)."""
    tmax, tmin = np.asarray(tmax, float), np.asarray(tmin, float)
    tmean = (tmax + tmin) / 2
    J = dates.dayofyear.values
    phi = math.radians(lat)
    dr = 1 + 0.033 * np.cos(2 * np.pi * J / 365)
    dec = 0.409 * np.sin(2 * np.pi * J / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(dec), -1, 1))
    Ra = 24 * 60 / np.pi * 0.0820 * dr * (ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws))
    N = 24 / np.pi * ws
    sun = np.asarray(sun_h, float)
    rs_ang = (0.25 + 0.5 * np.clip(sun / N, 0, 1)) * Ra
    rs_har = 0.16 * np.sqrt(np.maximum(tmax - tmin, 0)) * Ra
    Rs = np.where(np.isnan(sun), rs_har, rs_ang)
    Rso = (0.75 + 2e-5 * z) * Ra
    es = (_es(tmax) + _es(tmin)) / 2
    rh = np.asarray(rh, float)
    ea = np.where(np.isnan(rh), _es(tmin), es * rh / 100)
    Rns = 0.77 * Rs
    sigma = 4.903e-9
    Rnl = sigma * (((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2) * (0.34 - 0.14 * np.sqrt(np.maximum(ea, 0))) \
        * (1.35 * np.clip(Rs / np.maximum(Rso, 1e-6), 0.3, 1.0) - 0.35)
    Rn = Rns - Rnl
    u2 = np.nan_to_num(np.asarray(wind10_ms, float), nan=2.0) * 4.87 / math.log(67.8 * 10 - 5.42)
    P = 101.3 * ((293 - 0.0065 * z) / 293) ** 5.26
    gamma = 0.000665 * P
    delta = 4098 * _es(tmean) / (tmean + 237.3) ** 2
    et0 = (0.408 * delta * Rn + gamma * 900 / (tmean + 273) * u2 * (es - ea)) / (delta + gamma * (1 + 0.34 * u2))
    return np.maximum(et0, 0.0), Rs


def to_standard(raw: pd.DataFrame, lat: float, z: float) -> pd.DataFrame:
    g = lambda c: raw[c] if c in raw else pd.Series(np.nan, index=raw.index)
    df = pd.DataFrame({
        "tmean": g("TMK"), "tmax": g("TXK"), "tmin": g("TNK"), "rain": g("RSK"),
        "rh": g("UPM"), "wind_ms": g("FM"), "sun": g("SDK"),
    }, index=raw.index)
    et0, rs = et0_fao56(df.index, df["tmax"], df["tmin"], df["rh"], df["wind_ms"], df["sun"], lat, z)
    df["et0"], df["rad"] = et0, rs
    df["wind"] = df["wind_ms"] * 3.6
    return df[["tmean", "tmax", "tmin", "rain", "rad", "et0", "wind", "rh"]]


# ---------------------------------------------------------------------------
# Öffentliche Funktion
# ---------------------------------------------------------------------------
def load_dwd(start_year: int = 1961, refresh: bool = False) -> tuple[pd.DataFrame, str]:
    """Tageswerte der nächstgelegenen geeigneten DWD-Station ab start_year.

    Lücken der Hauptstation werden mit der zweitnächsten Station gefüllt (ohne Höhenkorrektur, MVP).
    HEU_SYNTHETIC=1 liefert Kunstwetter."""
    if os.environ.get("HEU_SYNTHETIC") == "1":
        from .weather import synthetic_weather
        d = synthetic_weather(start_year, pd.Timestamp.today().date() - pd.Timedelta(days=2), seed=7)
        d["rh"] = 75.0
        return d, "SYNTHETISCH (Testmodus, DWD-Ersatz)"
    stations = nearest_stations(start_year=start_year)
    if stations.empty:
        stations = nearest_stations(start_year=1991)
    if stations.empty:
        raise RuntimeError("Keine aktive DWD-Station in der Stationsliste gefunden.")
    frames, used = [], []
    for _, s in stations.head(2).iterrows():
        cache = DATA_DIR / f"dwd_kl_{int(s['id']):05d}.csv"
        if cache.exists() and not refresh and \
                (pd.Timestamp.today() - pd.Timestamp(os.path.getmtime(cache), unit="s")).days < 1:
            df = pd.read_csv(cache, parse_dates=["date"], index_col="date")
        else:
            df = to_standard(_fetch_station(int(s["id"])), float(s["lat"]), float(s["hoehe"]))
            df.rename_axis("date").to_csv(cache)
        frames.append(df[df.index.year >= start_year])
        used.append(f"{s['name']} (Nr. {int(s['id'])}, {s['km']:.0f} km, {s['hoehe']:.0f} m)")
    main = frames[0]
    if len(frames) > 1:
        main = main.combine_first(frames[1].reindex(main.index.union(frames[1].index)))
    main = main.sort_index()
    main["rain"] = main["rain"].fillna(0.0)
    main = main.interpolate(limit=5)
    return main, "DWD-Station " + used[0] + (f"; Lücken aus {used[1]}" if len(used) > 1 else "")


def vollstaendige_jahre(df: pd.DataFrame, col: str = "tmean", max_fehl: float = 0.05) -> list[int]:
    j = df[col].groupby(df.index.year).agg(lambda s: s.isna().mean())
    n = df[col].groupby(df.index.year).size()
    return [int(y) for y in j.index if j[y] <= max_fehl and n[y] >= 360]
