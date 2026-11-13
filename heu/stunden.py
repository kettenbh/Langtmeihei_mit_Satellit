"""Stündliche Wetterdaten für Hitzestress (Modul 1) und Klimatrend (THI-Stunden).

- Historie: Open-Meteo Archiv (ERA5), stündlich, 1991 bis heute, Cache data/wetter_stunde_cache.csv.gz
- Vorhersage: Open-Meteo Ensemble-API, ECMWF-ENS (51 Member, 15 Tage) und ICON-EPS des DWD (40 Member, ca. 7 Tage)
- Rückblick-Demo: "Vorhersage" aus dem tatsächlichen Wetter nach einem Stichtag plus Störungen, die mit der
  Vorlaufzeit wachsen. So lässt sich im Winter zeigen, wie die Warnung im Sommer ausgesehen hätte.

Spalten: t [°C], rh [%], ws [m/s, 10 m], rad [W/m², Globalstrahlung]
"""
from __future__ import annotations

import datetime as dt
import os

import numpy as np
import pandas as pd
import requests

from .config import DATA_DIR, STANDORT

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
HOURLY = {"temperature_2m": "t", "relative_humidity_2m": "rh", "wind_speed_10m": "ws", "shortwave_radiation": "rad"}
CACHE = DATA_DIR / "wetter_stunde_cache.csv.gz"
ENSEMBLE_MODELLE = {"ecmwf_ifs025": "ECMWF-ENS", "icon_seamless": "ICON-EPS (DWD)"}


def _synthetic_hourly(start_year: int, end: dt.date, seed: int = 1) -> pd.DataFrame:
    from .weather import synthetic_weather
    d = synthetic_weather(start_year, end, seed)
    rng = np.random.default_rng(seed + 11)
    idx = pd.date_range(d.index[0], d.index[-1] + pd.Timedelta(hours=23), freq="h")
    day = idx.normalize()
    tmin = d["tmin"].reindex(day).to_numpy()
    tmax = d["tmax"].reindex(day).to_numpy()
    h = idx.hour.to_numpy()
    t = (tmin + tmax) / 2 + (tmax - tmin) / 2 * np.sin(2 * np.pi * (h - 9) / 24)
    rh = np.clip(88 - 4.5 * (t - tmin) + rng.normal(0, 6, len(idx)), 25, 100)
    sun = np.clip(np.sin(np.pi * (h - 5) / 15), 0, None)
    rad = sun * d["rad"].reindex(day).to_numpy() * 1e6 / (3600 * 9.5)
    ws = np.clip(2.5 + rng.normal(0, 1.2, len(idx)), 0, None)
    return pd.DataFrame({"t": t, "rh": rh, "ws": ws, "rad": rad}, index=idx).rename_axis("time")


def _fetch_archive(start: dt.date, end: dt.date) -> pd.DataFrame:
    params = {"latitude": STANDORT.lat, "longitude": STANDORT.lon, "start_date": start.isoformat(),
              "end_date": end.isoformat(), "hourly": ",".join(HOURLY), "wind_speed_unit": "ms",
              "timezone": "Europe/Berlin"}
    r = requests.get(ARCHIVE_URL, params=params, timeout=180)
    r.raise_for_status()
    js = r.json()
    df = pd.DataFrame(js["hourly"]).rename(columns=HOURLY)
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")[list(HOURLY.values())]


def load_hourly(start_year: int = 1991, refresh: bool = False) -> tuple[pd.DataFrame, str]:
    today = dt.date.today()
    if os.environ.get("HEU_SYNTHETIC") == "1":
        return _synthetic_hourly(start_year, today - dt.timedelta(days=5)), "SYNTHETISCH (Testmodus)"
    df = None
    if CACHE.exists() and not refresh:
        df = pd.read_csv(CACHE, parse_dates=["time"], index_col="time")
        if df.index.min().year > start_year:
            df = None
    if df is None:
        parts = []
        for y0 in range(start_year, today.year + 1, 5):   # in 5-Jahres-Blöcken abrufen
            y1 = min(y0 + 4, today.year)
            parts.append(_fetch_archive(dt.date(y0, 1, 1), min(dt.date(y1, 12, 31), today)))
        df = pd.concat(parts)
    else:
        last = df.dropna(subset=["t"]).index.max().date()
        if (today - last).days > 2:
            new = _fetch_archive(last - dt.timedelta(days=7), today)
            df = pd.concat([df[df.index < pd.Timestamp(last - dt.timedelta(days=7))], new])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE, compression="gzip")
    df = df.loc[: df.dropna(subset=["t", "rh"]).index.max()]
    return df.interpolate(limit=6), f"Open-Meteo Archiv (ERA5) stündlich, {STANDORT.lat:.3f}°N {STANDORT.lon:.3f}°E"


# ---------------------------------------------------------------------------
# Ensemble-Vorhersage
# ---------------------------------------------------------------------------
def _fetch_ensemble(model: str, days: int) -> pd.DataFrame:
    params = {"latitude": STANDORT.lat, "longitude": STANDORT.lon, "hourly": ",".join(HOURLY),
              "models": model, "forecast_days": days, "wind_speed_unit": "ms", "timezone": "Europe/Berlin"}
    r = requests.get(ENSEMBLE_URL, params=params, timeout=120)
    r.raise_for_status()
    h = r.json()["hourly"]
    time = pd.to_datetime(h["time"])
    out = []
    members = sorted({k.split("_member")[1] for k in h if "_member" in k})
    for m in ["00"] + members:
        cols = {}
        for api, short in HOURLY.items():
            key = api if m == "00" else f"{api}_member{m}"
            if key in h:
                cols[short] = h[key]
        if len(cols) == len(HOURLY):
            d = pd.DataFrame(cols, index=time)
            d["member"] = f"{ENSEMBLE_MODELLE[model]} {m}"
            out.append(d)
    return pd.concat(out).rename_axis("time")


def load_ensemble(days: int = 15) -> tuple[pd.DataFrame, str]:
    """Langes Format: Index time, Spalten t, rh, ws, rad, member."""
    if os.environ.get("HEU_SYNTHETIC") == "1":
        hist, _ = load_hourly(2020)
        start = pd.Timestamp(dt.date.today())
        # Kunstvorhersage: Sommerwetter des Vorjahres ab gleichem Kalendertag verschoben
        return pseudo_ensemble(hist, hist.index.max() - pd.Timedelta(days=days + 1), days, start=start), \
            "SYNTHETISCH (Testmodus)"
    parts, used = [], []
    for model, label in ENSEMBLE_MODELLE.items():
        try:
            parts.append(_fetch_ensemble(model, days))
            used.append(label)
        except Exception:  # ein Modell darf fehlen
            continue
    if not parts:
        raise RuntimeError("Keine Ensemble-Vorhersage abrufbar (Open-Meteo Ensemble-API).")
    return pd.concat(parts), "Open-Meteo Ensemble: " + ", ".join(used)


def pseudo_ensemble(hourly: pd.DataFrame, stichtag: pd.Timestamp, days: int = 15, n: int = 30,
                    seed: int = 0, start: pd.Timestamp | None = None) -> pd.DataFrame:
    """Rückblick-Demo: tatsächliches Wetter nach dem Stichtag + Störungen, die mit der Vorlaufzeit wachsen
    (Temperatur-Spread ca. 0,6 °C am Tag 1 bis ca. 3,5 °C am Tag 15, typisch für Mitteleuropa im Sommer)."""
    rng = np.random.default_rng(seed)
    s = pd.Timestamp(stichtag).normalize() + pd.Timedelta(days=1)
    truth = hourly.loc[s: s + pd.Timedelta(days=days) - pd.Timedelta(hours=1)]
    lead = (truth.index - s) / pd.Timedelta(days=1)
    sd_t = 0.6 + 0.2 * lead.to_numpy()
    out = []
    shift = (pd.Timestamp(start).normalize() - s) if start is not None else pd.Timedelta(0)
    for m in range(n):
        # glatte Störung: Zufallsweg über die Tage, dazu kleine Stundenschwankung
        walk = np.cumsum(rng.normal(0, 1, days + 1))[np.floor(lead).astype(int)] / np.sqrt(np.floor(lead) + 1)
        dT = walk * sd_t
        d = truth.copy()
        d["t"] = d["t"] + dT
        d["rh"] = np.clip(d["rh"] - 2.5 * dT + rng.normal(0, 3, len(d)), 15, 100)
        d["ws"] = np.clip(d["ws"] * rng.lognormal(0, 0.15), 0, None)
        d.index = d.index + shift
        d["member"] = f"Demo {m:02d}"
        out.append(d)
    return pd.concat(out).rename_axis("time")
