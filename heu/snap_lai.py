"""Blattflächenindex (LAI) aus Sentinel-2 mit dem neuronalen Netz des ESA-SNAP-Biophysikprozessors.

Herkunft der Gewichte (heu/snap_lai_weights.json): ESA SNAP S2 Toolbox, Biophysical Processor
(Weiss & Baret, S2ToolBox Level-2 ATBD). Die Zahlen sind aus dem Sentinel-Hub-Custom-Script
sentinel-2/lai/script.js übernommen (github.com/sentinel-hub/custom-scripts, Lizenz CC BY-SA 4.0).
Bei Weitergabe der App diese Herkunft nennen; Änderungen an der Gewichtsdatei unter derselben Lizenz.

Netz: 11 Eingänge (B03, B04, B05, B06, B07, B8A, B11, B12 als L2A-Reflexion 0..1, cos Beobachtungs-
zenit, cos Sonnenzenit, cos relativer Azimut), 5 versteckte tanh-Neuronen, linearer Ausgang.
Eingänge werden auf [-1, 1] normiert, der Ausgang auf LAI zurückskaliert.

Wie im Custom Script ohne die SNAP-Eingangsprüfung (Konvexhülle der Trainingsdaten). Ersatzweise
wird der LAI auf [0, LAI_MAX] begrenzt. Für dichtes Grünland ist das Netz nicht trainiert: es
unterschätzt hohe LAI eher. Deshalb wird der Operator in Stufe A gegen das Modell kalibriert
(Plan 3.1, heu/beobachtung.py).

Zwei Wege mit denselben Gewichten:
  - evalscript_s2(): JavaScript für die CDSE Statistical API (rechnet je Pixel beim Anbieter),
  - lai_numpy(): NumPy-Version für den Earth-Search-Fallback und für Tests.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

WEIGHTS_FILE = Path(__file__).with_name("snap_lai_weights.json")
LAI_MAX = 8.0
BANDS = ("B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12")


@lru_cache(maxsize=1)
def weights() -> dict:
    return json.loads(WEIGHTS_FILE.read_text(encoding="utf-8"))


def _norm(x, lo, hi):
    return 2.0 * (x - lo) / (hi - lo) - 1.0


def lai_numpy(bands: dict, view_zenith_deg, sun_zenith_deg, sun_azimuth_deg, view_azimuth_deg,
              clip: bool = True) -> np.ndarray:
    """LAI je Pixel. bands: {"B03": Array, ...} als L2A-Reflexion (0..1). Winkel in Grad (Skalar oder Array)."""
    W = weights()
    nm = W["normierung"]
    rad = np.pi / 180.0
    x = [_norm(np.asarray(bands[b], dtype=float), *nm[b.lower()]) for b in BANDS]
    x.append(_norm(np.cos(np.asarray(view_zenith_deg, dtype=float) * rad), *nm["viewZen"]))
    x.append(_norm(np.cos(np.asarray(sun_zenith_deg, dtype=float) * rad), *nm["sunZen"]))
    x.append(np.cos((np.asarray(sun_azimuth_deg, dtype=float) - np.asarray(view_azimuth_deg, dtype=float)) * rad))
    x = np.broadcast_arrays(*x)
    X = np.stack(x, axis=-1)                                   # (..., 11)
    h = np.tanh(X @ np.array(W["versteckt"]["w"]).T + np.array(W["versteckt"]["bias"]))  # tansig = tanh
    out = h @ np.array(W["ausgang"]["w"]) + W["ausgang"]["bias"]
    lo, hi = W["denormierung"]
    lai = 0.5 * (out + 1.0) * (hi - lo) + lo
    return np.clip(lai, 0.0, LAI_MAX) if clip else lai


def _js_neuron(i: int) -> str:
    W = weights()
    names = ["b03", "b04", "b05", "b06", "b07", "b8a", "b11", "b12", "vz", "sz", "ra"]
    terms = " + ".join(f"({w!r})*{n}" for w, n in zip(W["versteckt"]["w"][i], names))
    return f"  var h{i} = Math.tanh({W['versteckt']['bias'][i]!r} + {terms});"


def evalscript_s2() -> str:
    """Evalscript für CDSE: NDVI, NDRE und LAI (SNAP-Netz), Wolkenmaske über SCL wie bisher."""
    W = weights()
    nm = W["normierung"]
    lo, hi = W["denormierung"]
    norm_lines = "\n".join(
        f"  var {b.lower()} = nrm(s.{b}, {nm[b.lower()][0]!r}, {nm[b.lower()][1]!r});" for b in BANDS)
    neurons = "\n".join(_js_neuron(i) for i in range(len(W["versteckt"]["bias"])))
    out = " + ".join(f"({w!r})*h{i}" for i, w in enumerate(W["ausgang"]["w"]))
    return f"""//VERSION=3
// LAI: ESA SNAP Biophysical Processor, Gewichte aus sentinel-hub/custom-scripts (CC BY-SA 4.0)
function setup() {{
  return {{
    input: [{{bands: ["B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "SCL", "dataMask",
                     "viewZenithMean", "viewAzimuthMean", "sunZenithAngles", "sunAzimuthAngles"]}}],
    output: [
      {{id: "ndvi", bands: 1, sampleType: "FLOAT32"}},
      {{id: "ndre", bands: 1, sampleType: "FLOAT32"}},
      {{id: "lai", bands: 1, sampleType: "FLOAT32"}},
      {{id: "dataMask", bands: 1}}
    ]
  }};
}}
function nrm(x, lo, hi) {{ return 2 * (x - lo) / (hi - lo) - 1; }}
function evaluatePixel(s) {{
  // SCL 4 = Vegetation, 5 = unbewachsen; Wolken, Schatten, Schnee, Wasser fallen heraus
  var ok = (s.SCL == 4 || s.SCL == 5) ? 1 : 0;
  var ndvi = (s.B08 - s.B04) / (s.B08 + s.B04 + 1e-6);
  var ndre = (s.B8A - s.B05) / (s.B8A + s.B05 + 1e-6);
  var d2r = Math.PI / 180;
{norm_lines}
  var vz = nrm(Math.cos(s.viewZenithMean * d2r), {nm['viewZen'][0]!r}, {nm['viewZen'][1]!r});
  var sz = nrm(Math.cos(s.sunZenithAngles * d2r), {nm['sunZen'][0]!r}, {nm['sunZen'][1]!r});
  var ra = Math.cos((s.sunAzimuthAngles - s.viewAzimuthMean) * d2r);
{neurons}
  var o = {W['ausgang']['bias']!r} + {out};
  var lai = 0.5 * (o + 1) * ({hi!r} - {lo!r}) + {lo!r};
  lai = Math.min({LAI_MAX!r}, Math.max(0, lai));
  return {{ndvi: [ndvi], ndre: [ndre], lai: [lai], dataMask: [s.dataMask * ok]}};
}}"""
