"""Modul 3: Futterqualität – Energie (MJ NEL) und Rohprotein (XP) je Schnitttermin, Zielkonflikt Menge/Qualität."""
from __future__ import annotations

import datetime as dt
from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from heu import qualitaet as Q
from heu import ui
from heu.plots import BLAU, GRAU, GRUEN, ROT, ORANGE
from heu.weather import date_from_doy, doy365

st.set_page_config(page_title="Futterqualität", page_icon="🌱", layout="wide")
st.title("🌱 Futterqualität: Energie und Rohprotein je Schnitttermin")
ui.testmodus_hinweis()

A0 = ui.annahmen()
with st.sidebar:
    st.header("Qualitätsannahmen")
    klee = st.slider("Kleeanteil an der TM", 0.0, 0.5, A0.kleeanteil, 0.05,
                     help="Leguminosen haben deutlich mehr Rohprotein. Schätzung aus Bestandsbonitur.")
    xa = st.number_input("Rohasche [g/kg TM]", 60.0, 160.0, A0.rohasche_g_kg, 5.0,
                         help="Aus Futteranalysen; hohe Werte deuten auf Futterverschmutzung hin.")
    bel = st.radio("Trocknung", ["Bodenheu", "Belüftungstrocknung"], index=1 if A0.belueftung else 0) == "Belüftungstrocknung"
    ziel_nel = st.slider("Qualitätsziel Heu [MJ NEL/kg TM]", 4.8, 6.6, 5.6, 0.1)
    horizon = st.slider("Betrachtete Tage ab Stichtag", 14, 70, 45, 7)

A = replace(A0, kleeanteil=klee, rohasche_g_kg=xa, belueftung=bel)
try:
    M = ui.modell(A, ui.token())
except Exception as e:
    st.error(f"Wetterdaten konnten nicht geladen werden: {e}")
    st.stop()
cur, clim, p, sims = M["cur"], M["clim"], M["p"], M["sims"]
JAHR = A.simulations_jahr
if cur is None or cur.n < 70:
    st.warning("Für das Simulationsjahr liegen noch zu wenige Wetterdaten vor.")
    st.stop()

# --- Kalibrierung an eigenen Analysen ---------------------------------------
rows = []
for y, s in list(sims.items()) + [(JAHR, cur.final)]:
    for i, c in enumerate(s.cuts, start=1):
        rows.append({"jahr": y, "schnitt": i, "datum": pd.Timestamp(date_from_doy(y, c.day)),
                     "netto_dt": c.netto_kg / 100, "nel_heu": c.nel_heu, "xp_heu": c.xp_heu, "stadium": c.stadium})
modell_cuts = pd.DataFrame(rows)
analysen = Q.lade_analysen()
k_nel, k_xp, n_k = Q.korrektur_aus_analysen(analysen, modell_cuts)
use_k = False
if n_k:
    use_k = st.sidebar.checkbox(f"Mit {n_k} eigenen Futteranalysen korrigieren "
                                f"(NEL {k_nel:+.2f}, XP {k_xp:+.0f})", value=True)
KORR = (k_nel, k_xp) if use_k else (0.0, 0.0)

min_d = date_from_doy(JAHR, doy365(dt.date(2001, 4, 1)))
max_d = date_from_doy(JAHR, min(cur.n - 1, doy365(dt.date(2001, 10, 1))))
stichtag = st.sidebar.slider("Stichtag", min_d, max_d, min(max_d, dt.date(JAHR, 5, 10)), format="DD.MM.YYYY")
ST = doy365(stichtag)

kurve = Q.schnittkurve(cur, clim, ST, p, A, horizon=horizon, korrektur=KORR)
if kurve.empty:
    st.info("Kein erntbarer Aufwuchs im betrachteten Zeitraum.")
    st.stop()
kurve["datum"] = [date_from_doy(JAHR, t) for t in kurve["tag"]]

# --- Kennzahlen ---------------------------------------------------------------
i_opt = int(kurve["energie_gj_q50"].idxmax())
opt = kurve.loc[i_opt]
ok_q = kurve[kurve["nel_q50"] >= ziel_nel]
heute = kurve.iloc[0]
stad = Q.stadium(heute["st"], p.ST1, p.ST2, int(heute["schnitt_nr"]), int(heute["tage_seit"]))
st.markdown(f"**{int(heute['schnitt_nr'])}. Aufwuchs am {stichtag:%d.%m.%Y}:** {stad}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("NEL Heu bei Schnitt heute", f"{heute['nel_q50']:.2f} MJ", help="Median über alle Wetterszenarien, nach Trocknungsabschlag, je kg TM")
c2.metric("Rohprotein Heu bei Schnitt heute", f"{heute['xp_q50']:.0f} g/kg TM")
c3.metric("Maximaler Energieertrag am", f"{opt['datum']:%d.%m.}", f"{opt['energie_gj_q50']:.1f} GJ NEL/ha", delta_color="off")
c4.metric(f"Ziel {ziel_nel:.1f} MJ NEL hält bis",
          f"{ok_q['datum'].iloc[-1]:%d.%m.}" if len(ok_q) else "unterschritten")

# --- Zielkonflikt-Diagramm ---------------------------------------------------
x = list(kurve["datum"])
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                    specs=[[{}], [{"secondary_y": True}], [{}]],
                    subplot_titles=["Heu netto [dt TM/ha]", "Energie Heu [MJ NEL/kg TM] und Rohprotein [g/kg TM]",
                                    "Energieertrag [GJ NEL/ha]"])


def band(name, row, color, fill, sec=False):
    fig.add_scatter(x=x + x[::-1], y=list(kurve[f"{name}_q90"]) + list(kurve[f"{name}_q10"])[::-1], fill="toself",
                    fillcolor=fill, line=dict(width=0), showlegend=False, row=row, col=1)
    fig.add_scatter(x=x, y=kurve[f"{name}_q50"], line=dict(color=color, width=2), name={"netto_dt": "Heu netto", "nel": "NEL Heu", "energie_gj": "Energieertrag"}[name], row=row, col=1)


band("netto_dt", 1, BLAU, "rgba(31,111,209,0.15)")
band("nel", 2, ROT, "rgba(217,83,79,0.15)")
fig.add_scatter(x=x, y=kurve["xp_q50"], line=dict(color=GRUEN, dash="dot"), name="Rohprotein (rechte Achse)",
                row=2, col=1, secondary_y=True)
fig.update_yaxes(title_text="MJ NEL", row=2, col=1, secondary_y=False)
fig.update_yaxes(title_text="g XP", row=2, col=1, secondary_y=True, showgrid=False)
band("energie_gj", 3, ORANGE, "rgba(232,137,43,0.18)")
fig.add_hline(y=ziel_nel, line_dash="dash", line_color="black", row=2, col=1, annotation_text="Qualitätsziel")
fig.add_vline(x=opt["datum"], line_color=ORANGE, line_dash="dash")
fig.update_layout(height=720, margin=dict(t=40, b=10), legend=dict(orientation="h", y=-0.08))
st.plotly_chart(fig, width="stretch")
st.caption("Bänder: 10–90 % über die Wetterszenarien 1991–2025 nach dem Stichtag. Grün gepunktet: Rohprotein (Median, "
           "rechte Achse). Annahme: bis zum jeweiligen Tag wird nicht gemäht. Orange gestrichelt: Maximum des "
           "Energieertrags (Median). Trocknungsabschlag je Regentag im Trocknungsfenster ist enthalten.")

# --- Historische Verteilung je Schnitt ---------------------------------------
st.subheader("Qualität je Schnitt im Vergleich zu 1991–2025")
hist = modell_cuts[modell_cuts["jahr"] != JAHR]
cur_cuts = modell_cuts[modell_cuts["jahr"] == JAHR]
cols = st.columns(3)
for i, col in enumerate(cols, start=1):
    h = hist[hist["schnitt"] == i]
    if h.empty:
        continue
    f = make_subplots(rows=2, cols=1, vertical_spacing=0.25, subplot_titles=["NEL [MJ/kg TM]", "XP [g/kg TM]"])
    f.add_histogram(x=h["nel_heu"] + KORR[0], marker_color=GRAU, nbinsx=12, row=1, col=1, showlegend=False)
    f.add_histogram(x=h["xp_heu"] + KORR[1], marker_color=GRAU, nbinsx=12, row=2, col=1, showlegend=False)
    c = cur_cuts[cur_cuts["schnitt"] == i]
    if len(c):
        f.add_vline(x=float(c["nel_heu"].iloc[0] + KORR[0]), line_color=ROT, line_width=3, row=1, col=1)
        f.add_vline(x=float(c["xp_heu"].iloc[0] + KORR[1]), line_color=ROT, line_width=3, row=2, col=1)
    f.update_layout(height=360, margin=dict(t=40, b=10), title=f"{i}. Schnitt")
    col.plotly_chart(f, width="stretch")
st.caption(f"Grau: Modellschnitte 1991–2025 mit der Schnittregel der Hauptseite. Rot: Simulation {JAHR} mit Ist-Wetter.")

show = cur_cuts.copy()
if len(show):
    show["datum"] = show["datum"].dt.strftime("%d.%m.%Y")
    show["nel_heu"] = (show["nel_heu"] + KORR[0]).round(2)
    show["xp_heu"] = (show["xp_heu"] + KORR[1]).round(0)
    st.dataframe(show.drop(columns="jahr").rename(columns={"schnitt": "Schnitt", "datum": "Datum", "netto_dt": "Heu netto [dt TM/ha]",
                                                           "nel_heu": "NEL Heu", "xp_heu": "XP Heu", "stadium": "Stadium"}).round(1),
                 hide_index=True, width="stretch")

with st.expander("Methodik und eigene Futteranalysen"):
    st.markdown(f"""
**Energie.** ModVege liefert die Verdaulichkeit der organischen Masse (OMD). Daraus: DOMD = OMD·(1000 − XA),
ME = 0,016·DOMD (MAFF 1975), NEL = 0,6·[1 + 0,004·(q − 57)]·ME mit q = ME/18,4·100 (GfE 2001).

**Rohprotein.** Kritische N-Verdünnungskurve für Gräser Nc = 4,8·W^−0,32 % (Lemaire & Salette 1984, W = grüne Masse
in t TM/ha), multipliziert mit dem kalibrierten Nährstoffindex NI = {p.NI:.2f}; gemischt mit Klee ({klee:.0%}, {A.n_leguminosen_pct} % N)
und toter Masse ({A.n_totmasse_pct} % N); XP = 6,25·N.

**Gras → Heu.** Abschlag {Q.HEU_ABSCHLAG['belueftung' if bel else 'boden']} (MJ NEL bzw. g XP, Grundwert + je Regentag).
Diese Abschläge und Klee-/Totmasse-N-Gehalte sind **Annahmen**.

**Eigene Analysen.** Lege `data/futteranalysen.csv` an (Vorlage `data/futteranalysen_vorlage.csv`) mit den Spalten
`datum, schnitt, weide, nel, xp, rohfaser`. Die App ordnet jede Analyse dem nächstgelegenen Modellschnitt (±10 Tage) zu
und korrigiert die mittlere Abweichung.

**Grenzen.** Kein Pflanzenbestandsmodell (Gräser/Kräuter/Klee fest), keine Live-Daten der LfL-Reifeprüfung
(lassen sich nur manuell vergleichen), Sentinel-2-NDRE fließt noch nicht ein. Erwartete Genauigkeit ohne
Kalibrierung etwa ±0,3–0,5 MJ NEL (Schätzung).
""")
