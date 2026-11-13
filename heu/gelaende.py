"""Feuchtstellen auf den Weiden aus dem Geländemodell (Lebensraum der Zwergschlammschnecke).

Eingabe: DGM1-Kachel(n) der Bayerischen Vermessungsverwaltung (Open Data, GeoTIFF, EPSG:25832, 1 m),
manuell heruntergeladen von geodaten.bayern.de (OpenData -> Digitales Geländemodell 1 m) und in
data/dgm/ abgelegt. Lesen benötigt `rasterio` (optional, `pip install rasterio`).

Topografischer Feuchteindex TWI = ln(a / tan β) (Beven & Kirkby 1979), a = spezifische Einzugsfläche
(D8-Abflussakkumulation), β = Hangneigung. Für Tempo wird auf 5 m gemittelt.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .config import DATA_DIR

DGM_DIR = DATA_DIR / "dgm"


def wgs84_to_utm32(lat: float, lon: float) -> tuple[float, float]:
    """WGS84 -> ETRS89/UTM Zone 32N (EPSG:25832), Krüger-Reihe (Genauigkeit << 1 m)."""
    a, f = 6378137.0, 1 / 298.257222101
    k0, lon0, fe = 0.9996, math.radians(9.0), 500000.0
    n = f / (2 - f)
    A = a / (1 + n) * (1 + n**2 / 4 + n**4 / 64)
    al = [n / 2 - 2 * n**2 / 3 + 5 * n**3 / 16, 13 * n**2 / 48 - 3 * n**3 / 5, 61 * n**3 / 240]
    phi, lam = math.radians(lat), math.radians(lon) - lon0
    e = math.sqrt(f * (2 - f))
    t = math.sinh(math.atanh(math.sin(phi)) - e * math.atanh(e * math.sin(phi)))
    xi = math.atan2(t, math.cos(lam))
    eta = math.atanh(math.sin(lam) / math.sqrt(1 + t * t))
    E = fe + k0 * A * (eta + sum(al[j] * math.cos(2 * (j + 1) * xi) * math.sinh(2 * (j + 1) * eta) for j in range(3)))
    N = k0 * A * (xi + sum(al[j] * math.sin(2 * (j + 1) * xi) * math.cosh(2 * (j + 1) * eta) for j in range(3)))
    return E, N


def kacheln() -> list[Path]:
    return sorted(DGM_DIR.glob("*.tif")) if DGM_DIR.exists() else []


def lade_dgm(aufloesung_m: float = 5.0):
    """Mosaik aller Kacheln in data/dgm/, gemittelt auf aufloesung_m. Rückgabe (z, x0, y0, res)."""
    import rasterio
    from rasterio.merge import merge

    files = kacheln()
    if not files:
        raise FileNotFoundError("Keine DGM-Kacheln in data/dgm/ gefunden.")
    srcs = [rasterio.open(f) for f in files]
    arr, trans = merge(srcs)
    z = arr[0].astype(float)
    nod = srcs[0].nodata
    if nod is not None:
        z[z == nod] = np.nan
    res0 = trans.a
    k = max(1, int(round(aufloesung_m / res0)))
    h, w = (z.shape[0] // k) * k, (z.shape[1] // k) * k
    z = np.nanmean(z[:h, :w].reshape(h // k, k, w // k, k), axis=(1, 3))
    return z, trans.c, trans.f, res0 * k


def twi(z: np.ndarray, res: float) -> np.ndarray:
    """D8-Abflussakkumulation und TWI. z: Höhen (Zeilen von Nord nach Süd)."""
    z = np.where(np.isnan(z), np.nanmax(z) + 100, z)
    ny, nx = z.shape
    nb = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    dist = np.array([math.hypot(dy, dx) * res for dy, dx in nb])
    zp = np.pad(z, 1, mode="edge")
    drops = np.stack([(z - zp[1 + dy: 1 + dy + ny, 1 + dx: 1 + dx + nx]) / dist[i] for i, (dy, dx) in enumerate(nb)])
    best = np.argmax(drops, axis=0)
    slope = np.maximum(drops.max(axis=0), 0.001)          # tan β, mindestens 0,1 %
    acc = np.ones_like(z)                                  # Zellen, inkl. eigener
    order = np.argsort(z, axis=None)[::-1]                 # von oben nach unten
    by, bx = np.divmod(order, nx)
    for yy, xx in zip(by, bx):
        if drops[best[yy, xx], yy, xx] <= 0:
            continue
        dy, dx = nb[best[yy, xx]]
        ty, tx = yy + dy, xx + dx
        if 0 <= ty < ny and 0 <= tx < nx:
            acc[ty, tx] += acc[yy, xx]
    a_spez = acc * res * res / res
    return np.log(a_spez / slope)


def feuchte_auswertung(felder: list[dict], aufloesung_m: float = 5.0, schwelle: float | None = None):
    """TWI-Raster und je Feld der Anteil feuchter Fläche (TWI über Schwelle; Standard: 90. Perzentil des Ausschnitts)."""
    from shapely import contains_xy
    from shapely.geometry import Polygon

    z, x0, y0, res = lade_dgm(aufloesung_m)
    t = twi(z, res)
    ny, nx = t.shape
    xs = x0 + (np.arange(nx) + 0.5) * res
    ys = y0 - (np.arange(ny) + 0.5) * res
    if schwelle is None:
        schwelle = float(np.nanpercentile(t, 90))
    X, Y = np.meshgrid(xs, ys)
    ergebnis, umrisse = [], []
    for f in felder:
        geom = f["geometry"]
        ring = geom["coordinates"][0] if geom["type"] == "Polygon" else geom["coordinates"][0][0]
        utm = [wgs84_to_utm32(lat, lon) for lon, lat in ring]
        poly = Polygon(utm)
        inside = contains_xy(poly, X, Y)
        vals = t[inside]
        umrisse.append((f["name"], [p[0] for p in utm], [p[1] for p in utm]))
        ergebnis.append({"Feld": f["name"], "Zellen": int(inside.sum()),
                         "Anteil feucht [%]": float(np.mean(vals > schwelle) * 100) if len(vals) else np.nan,
                         "TWI Median": float(np.median(vals)) if len(vals) else np.nan})
    return {"twi": t, "xs": xs, "ys": ys, "schwelle": schwelle, "felder": ergebnis, "umrisse": umrisse}
