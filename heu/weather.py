"""Wetterdaten: Abruf, Cache und Aufbereitung zu 365-Tage-Jahresarrays.

Quelle im MVP: Open-Meteo Historical Weather API (ERA5/ERA5-Land-Reanalyse, ca. 9-25 km).
Kostenlos, ohne Schlüssel. Später durch DWD HYRAS/Stationsdaten ersetzbar
(gleiches Spaltenformat in den Cache schreiben).
"""
from __future__ import annotations

import datetime as dt
import os

import numpy as np
import pandas as pd
import requests

from .config import DATA_DIR, STANDORT

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY_VARS = {
    "temperature_2m_mean": "tmean",
    "temperature_2m_max": "tmax",
    "temperature_2m_min": "tmin",
    "precipitation_sum": "rain",
    "shortwave_radiation_sum": "rad",          # MJ/m²/Tag
    "et0_fao_evapotranspiration": "et0",       # mm/Tag
    "wind_speed_10m_max": "wind",              # km/h
}
COLS = list(DAILY_VARS.values())
CACHE = DATA_DIR / "wetter_cache.csv"


def _fetch_open_meteo(start: dt.date, end: dt.date, lat: float, lon: float) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": ",".join(DAILY_VARS),
        "timezone": "Europe/Berlin",
    }
    r = requests.get(ARCHIVE_URL, params=params, timeout=120)
    r.raise_for_status()
    js = r.json()
    if "daily" not in js:
        raise RuntimeError(f"Unerwartete Antwort von Open-Meteo: {str(js)[:300]}")
    df = pd.DataFrame(js["daily"]).rename(columns=DAILY_VARS)
    df["date"] = pd.to_datetime(df.pop("time"))
    return df.set_index("date")[COLS]


def synthetic_weather(start_year: int, end: dt.date, seed: int = 1) -> pd.DataFrame:
    """Plausibles Kunstwetter (nur für Tests/Demo ohne Internet)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(f"{start_year}-01-01", end, freq="D")
    doy = idx.dayofyear.values
    year_anom = {y: rng.normal(0, 0.8) for y in idx.year.unique()}
    ya = np.array([year_anom[y] for y in idx.year])
    tmean = 8.3 + 9.5 * np.sin(2 * np.pi * (doy - 110) / 365) + ya + rng.normal(0, 3, len(idx))
    tmean = pd.Series(tmean).rolling(3, min_periods=1).mean().values
    wet = rng.random(len(idx)) < (0.38 + 0.12 * np.sin(2 * np.pi * (doy - 100) / 365))
    rain = np.where(wet, rng.gamma(0.9, 6.5, len(idx)), 0.0)
    rad = np.clip(11 + 9 * np.sin(2 * np.pi * (doy - 80) / 365) - 4 * wet + rng.normal(0, 2, len(idx)), 1, 32)
    et0 = np.clip(0.1 * rad + 0.08 * np.maximum(tmean, 0) - 0.4 * wet, 0.1, 7)
    return pd.DataFrame(
        {"tmean": tmean, "tmax": tmean + 5, "tmin": tmean - 5, "rain": rain,
         "rad": rad, "et0": et0, "wind": 12 + rng.normal(0, 4, len(idx))},
        index=idx,
    ).rename_axis("date")


def load_weather(start_year: int, refresh: bool = False) -> tuple[pd.DataFrame, str]:
    """Liefert Tageswetter ab 1.1.start_year bis zum letzten verfügbaren Tag.

    Rückgabe: (DataFrame, Quellenbeschreibung). Mit der Umgebungsvariable
    HEU_SYNTHETIC=1 wird Kunstwetter erzeugt (Offline-Test).
    """
    today = dt.date.today()
    if os.environ.get("HEU_SYNTHETIC") == "1":
        return synthetic_weather(start_year, today - dt.timedelta(days=5)), "SYNTHETISCH (Testmodus)"

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = None
    if CACHE.exists():
        df = pd.read_csv(CACHE, parse_dates=["date"], index_col="date")
        if df.index.min().year > start_year:
            df = None
    if df is None or refresh:
        df = _fetch_open_meteo(dt.date(start_year, 1, 1), today, STANDORT.lat, STANDORT.lon)
    else:
        last = df.dropna(subset=["tmean"]).index.max().date()
        if (today - last).days > 2:
            new = _fetch_open_meteo(last - dt.timedelta(days=7), today, STANDORT.lat, STANDORT.lon)
            df = pd.concat([df[df.index < pd.Timestamp(last - dt.timedelta(days=7))], new])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.to_csv(CACHE)
    # Archiv hat ca. 5 Tage Verzug: offene Tage am Ende abschneiden
    valid = df.dropna(subset=["tmean", "rad", "et0"])
    df = df.loc[: valid.index.max()]
    df["rain"] = df["rain"].fillna(0.0)
    df = df.interpolate(limit=3).ffill()
    return df, f"Open-Meteo Archiv (ERA5), {STANDORT.lat:.3f}°N {STANDORT.lon:.3f}°E"


def year_arrays(df: pd.DataFrame, year: int) -> dict[str, np.ndarray] | None:
    """Ein Kalenderjahr als 365-Tage-Arrays (29. Feb entfällt). None, wenn unvollständig."""
    d = df[df.index.year == year]
    d = d[~((d.index.month == 2) & (d.index.day == 29))]
    if len(d) < 365:
        return None
    return {c: d[c].to_numpy(dtype=float) for c in COLS}


def partial_year(df: pd.DataFrame, year: int) -> tuple[dict[str, np.ndarray], int]:
    """Laufendes Jahr: Arrays bis zum letzten Datentag (Länge n)."""
    d = df[df.index.year == year]
    d = d[~((d.index.month == 2) & (d.index.day == 29))]
    return {c: d[c].to_numpy(dtype=float) for c in COLS}, len(d)


def doy365(date: dt.date) -> int:
    """0-basierter Tagesindex im 365-Tage-Jahr."""
    d0 = dt.date(2001, 1, 1)  # Nicht-Schaltjahr
    return (dt.date(2001, date.month, date.day) - d0).days


def date_from_doy(year: int, i: int) -> dt.date:
    d = dt.date(2001, 1, 1) + dt.timedelta(days=int(i))
    return dt.date(year, d.month, d.day)
