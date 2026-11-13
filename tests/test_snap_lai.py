"""SNAP-LAI-Netz: NumPy-Version, generiertes Evalscript und Plausibilität."""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from heu import snap_lai as L  # noqa: E402

GRAS = dict(B03=.07, B04=.04, B05=.10, B06=.28, B07=.36, B8A=.42, B11=.20, B12=.10)
BODEN = dict(B03=.10, B04=.13, B05=.15, B06=.17, B07=.18, B8A=.20, B11=.28, B12=.24)


def test_plausible_values():
    assert 2.0 < L.lai_numpy(GRAS, 5, 40, 160, 100) < 5.0
    assert L.lai_numpy(BODEN, 5, 40, 160, 100) < 0.8


def test_clip_range():
    rng = np.random.default_rng(0)
    b = {k: rng.uniform(0, 0.6, 500) for k in L.BANDS}
    lai = L.lai_numpy(b, 5, 40, 160, 100)
    assert lai.min() >= 0 and lai.max() <= L.LAI_MAX


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js nicht installiert")
def test_evalscript_matches_numpy(tmp_path):
    rng = np.random.default_rng(1)
    smp = []
    for _ in range(100):
        s = {b: float(rng.uniform(0.01, 0.45)) for b in L.BANDS}
        s.update(viewZenithMean=float(rng.uniform(0, 12)), viewAzimuthMean=float(rng.uniform(0, 360)),
                 sunZenithAngles=float(rng.uniform(25, 68)), sunAzimuthAngles=float(rng.uniform(100, 250)),
                 B08=s["B8A"], SCL=4, dataMask=1)
        smp.append(s)
    (tmp_path / "smp.json").write_text(json.dumps(smp))
    (tmp_path / "gen.js").write_text(L.evalscript_s2())
    (tmp_path / "run.js").write_text(
        "const fs=require('fs'),vm=require('vm');const d=process.argv[2];"
        "const S=JSON.parse(fs.readFileSync(d+'/smp.json'));const c={};vm.createContext(c);"
        "vm.runInContext(fs.readFileSync(d+'/gen.js','utf8'),c);"
        "console.log(JSON.stringify(S.map(s=>c.evaluatePixel(s).lai[0])));")
    js = np.array(json.loads(subprocess.check_output(["node", str(tmp_path / "run.js"), str(tmp_path)])))
    g = lambda k: np.array([s[k] for s in smp])  # noqa: E731
    py = L.lai_numpy({b: g(b) for b in L.BANDS}, g("viewZenithMean"), g("sunZenithAngles"),
                     g("sunAzimuthAngles"), g("viewAzimuthMean"))
    np.testing.assert_allclose(js, py, atol=1e-9)
