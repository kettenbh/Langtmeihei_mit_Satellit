"""Schneller Funktionstest mit Kunstwetter (ohne Internet): HEU_SYNTHETIC=1 python scripts/selftest.py"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HEU_SYNTHETIC", "1")
import numpy as np
from heu.config import ANNAHMEN as A
from heu.model import ModVegeParams
from heu.weather import load_weather, date_from_doy
from heu import forecast as F

t=time.time()
df, src = load_weather(A.trainings_start)
print(src, df.index.min().date(), df.index.max().date())
clim = F.training_years(df, A)
p, ach = F.calibrate_ni(clim, ModVegeParams(WHC=A.nutzbare_feldkapazitaet_mm), A)
print(f"NI={p.NI:.3f} Brutto-Mittel={ach:.1f} dt/ha  ({time.time()-t:.1f}s)")
hist = F.history_table(F.simulate_years(clim, p, A))
print(hist.describe().round(1).T[["mean","std","min","max"]])
cur = F.run_current_year(df, A.simulations_jahr, clim, p, A)
print("Ist 2026:", [(str(date_from_doy(2026,c.day)), round(c.netto_kg/100,1), round(c.omd,3), round(c.nel,2)) for c in cur.final.cuts])
for d in (90, 130, 160, 200, 250):
    pr = F.forecast(cur, clim, d, p, A)
    print(date_from_doy(2026,d), "Saison q10/50/90:", np.percentile(pr.saison,[10,50,90]).round(1), "S1:", np.nanpercentile(pr.schnitte[1],[10,50,90]).round(1))
t=time.time(); fan = F.fan(cur, clim, p, A, 59, 300, 7); print("fan", fan.shape, f"{time.time()-t:.1f}s")
print(F.cut_timing(cur, clim, 130, p, A).round(2).head(14))

# --- Neue Module (Hitze, Parasiten, Futterqualität, Klima) ------------------
import pandas as pd
from heu.config import HITZE, PARASITEN
from heu import hitze as HZ, parasiten as P, qualitaet as Q, klima as K, stunden as S, dwd as D
t = time.time()
print("Schnitte 2026 mit Qualität:", [(c.stadium, round(c.nel_heu, 2), round(c.xp_heu)) for c in cur.final.cuts])
print(Q.schnittkurve(cur, clim, 130, p, A, 30)[["tag", "netto_dt_q50", "nel_q50", "xp_q50", "energie_gj_q50"]].iloc[::6].round(2))
h, _ = S.load_hourly(1991)
print("Hitzestunden je Jahr:", HZ.jahreswerte(h, HITZE)["hitzestunden_68"].describe().round(0).to_dict())
ens = S.pseudo_ensemble(h, pd.Timestamp(f"{h.index.max().year - 1}-07-10"), 15)
print(HZ.ensemble_tage(ens, HITZE)[["thi_max_q50", "p_stufe1", "p_stufe2", "stufe_wahrscheinlich"]].head(5).round(2))
reihen = P.indexreihen(df, PARASITEN, None, ["Wiese Ost"])
print(P.risikotabelle(reihen, pd.Timestamp(f"{df.index.max().year}-08-15"), list(range(1991, 2026))).round(1))
dd, src = D.load_dwd(1961)
kz = K.kennzahlen(dd, D.vollstaendige_jahre(dd)[:-1], A)
print(src, "Kennzahl-Jahre:", len(kz), "Trend Vegetationsbeginn:", {k: round(float(v), 3) for k, v in K.trend(kz["vegetationsbeginn_gts200"]).items() if k in ("steigung_jahrzehnt", "p")})
print(f"neue Module OK ({time.time() - t:.1f}s)")
