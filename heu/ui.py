"""Gemeinsame, gecachte Datenlader für alle Streamlit-Seiten."""
from __future__ import annotations

import datetime as dt
import os

import numpy as np
import streamlit as st

from . import forecast as F
from .config import ANNAHMEN, Annahmen
from .model import ModVegeParams, State, run
from .weather import load_weather

SYNTHETIC = os.environ.get("HEU_SYNTHETIC") == "1"


def annahmen() -> Annahmen:
    """Die auf der Hauptseite eingestellten Annahmen (sonst Standard)."""
    return st.session_state.get("A", ANNAHMEN)


def token() -> int:
    return st.session_state.get("token", 0)


@st.cache_resource(show_spinner="Lade Tageswetter (ERA5) …")
def taeglich(start_year: int, tok: int):
    return load_weather(start_year, refresh=tok > 0)


@st.cache_resource(show_spinner="Kalibriere ModVege und simuliere die Jahre …")
def modell(a: Annahmen, tok: int, quelle: str = "era5"):
    if quelle == "dwd":
        df, src = dwd(1961, tok)
    else:
        df, src = taeglich(a.trainings_start, tok)
    jahre_all = {}
    from .weather import year_arrays
    for y in range(int(df.index.year.min()), a.trainings_ende + 1):
        w = year_arrays(df, y)
        if w is not None and not any(np.isnan(v).any() for v in w.values()):
            jahre_all[y] = w
    clim = {y: w for y, w in jahre_all.items() if y >= a.trainings_start}
    p, ach = F.calibrate_ni(clim, ModVegeParams(WHC=a.nutzbare_feldkapazitaet_mm), a)
    sims = F.simulate_years(jahre_all, p, a)
    traces = {}
    for y, w in jahre_all.items():
        tr = []
        run(w, p, a, trace=tr)
        traces[y] = np.array(tr)
    cur = F.run_current_year(df, a.simulations_jahr, clim, p, a) if quelle != "dwd" else None
    return dict(df=df, src=src, clim=clim, alle=jahre_all, p=p, achieved=ach, sims=sims, traces=traces, cur=cur)


@st.cache_resource(show_spinner="Lade stündliche Wetterdaten 1991 bis heute (beim ersten Mal 1–3 Minuten) …")
def stuendlich(tok: int):
    from .stunden import load_hourly
    return load_hourly(1991, refresh=False)


@st.cache_resource(show_spinner="Lade DWD-Klimadaten der nächsten Station …")
def dwd(start_year: int, tok: int):
    from .dwd import load_dwd
    return load_dwd(start_year, refresh=tok > 0)


@st.cache_data(ttl=3 * 3600, show_spinner="Lade Ensemble-Wettervorhersage …")
def ensemble(stunde: str):
    from .stunden import load_ensemble
    return load_ensemble(15)


def heute_stunde() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H")


def testmodus_hinweis():
    if SYNTHETIC:
        st.warning("Testmodus (HEU_SYNTHETIC=1): alle Daten sind Kunstwetter, keine echten Messungen.")
