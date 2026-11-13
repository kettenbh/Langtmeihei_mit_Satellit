"""Modul 4: Klimatrend für den Standort Kirchweidach."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from heu import klima as K
from heu import ui
from heu.config import HITZE
from heu.dwd import vollstaendige_jahre
from heu.plots import BLAU, GRAU, ROT, hist_fig, trend_fig

st.set_page_config(page_title="Klimatrend", page_icon="📈", layout="wide")
st.title("📈 Klimatrend am Standort")
ui.testmodus_hinweis()
A = ui.annahmen()

c1, c2 = st.columns([2, 1])
quelle = c1.radio("Datenquelle", ["DWD-Station (empfohlen, ab 1961)", "ERA5 über Open-Meteo (ab 1991)"], horizontal=True,
                  help="Für Trends ist eine Messstation besser geeignet; ERA5 kann durch Änderungen im Beobachtungssystem "
                       "künstliche Sprünge enthalten.")
mit_thi = c2.checkbox("THI-Hitzestunden einbeziehen", value=False, help="Lädt stündliche ERA5-Daten (beim ersten Mal 1–3 Min.)")

q = "dwd" if quelle.startswith("DWD") else "era5"
try:
    M = ui.modell(A, ui.token(), q)
except Exception as e:
    st.error(f"{'DWD' if q == 'dwd' else 'ERA5'}-Daten nicht verfügbar: {e}")
    if q == "dwd":
        st.info("Weiter mit ERA5.")
        q = "era5"
        M = ui.modell(A, ui.token(), q)
    else:
        st.stop()
df = M["df"]
jahre = [y for y in vollstaendige_jahre(df) if y < pd.Timestamp.today().year]
st.caption(f"Quelle: {M['src']} · vollständige Jahre: {jahre[0]}–{jahre[-1]} ({len(jahre)})")

kz = K.kennzahlen(df, jahre, A)
mk = K.modell_kennzahlen({y: s for y, s in M["sims"].items() if y in jahre}, M["traces"], M["alle"])
kz = kz.join(mk, how="left")
if mit_thi:
    from heu import hitze as HZ
    hourly, _ = ui.stuendlich(ui.token())
    kz = kz.join(HZ.jahreswerte(hourly, HITZE)[["hitzestunden_68"]], how="left")

letztes = jahre[-1]
spalten = [c for c in K.BESCHREIBUNG if c in kz]
trends = {c: K.trend(kz[c]) for c in spalten}

# --- Kacheln ------------------------------------------------------------------
st.subheader(f"Überblick: {letztes} im Vergleich und Trend pro Jahrzehnt")
haupt = ["vegetationsbeginn_gts200", "hitzetage", "laengste_trockenphase", "kwb_apr_sep",
         "schnitte_modell", "erster_schnitt_tag", "heuwetterfenster", "wasserstresstage"]
cols = st.columns(4)
for i, c in enumerate([h for h in haupt if h in kz]):
    name, einheit, _ = K.BESCHREIBUNG[c]
    v = kz.loc[letztes, c]
    hist = kz[c].drop(letztes).dropna()
    pz = float((hist < v).mean() * 100) if len(hist) and not pd.isna(v) else np.nan
    t = trends[c]
    if "steigung_jahrzehnt" in t:
        delta = f"{t['steigung_jahrzehnt']:+.1f} {einheit}/10 J." + ("" if t["gesichert"] else " (nicht gesichert)")
    else:
        delta = "kein Trend berechenbar"
    wert = f"{v:.0f}" if not pd.isna(v) else "–"
    if "Tag im Jahr" in einheit and not pd.isna(v):
        wert = f"{(pd.Timestamp(letztes, 1, 1) + pd.Timedelta(days=int(v) - 1)):%d.%m.}"
    with cols[i % 4]:
        st.metric(name, wert, f"{pz:.0f}. Perzentil" if not np.isnan(pz) else None, delta_color="off")
        st.caption(f"Trend: {delta}")
st.caption("Perzentil: Anteil der übrigen Jahre mit niedrigerem Wert. Trend: Theil-Sen-Steigung; „gesichert“ heißt "
           "Mann-Kendall-Test p < 0,05. Bei Terminen (Tag im Jahr) bedeutet ein negativer Trend „früher“.")

# --- Einzelkennzahl ------------------------------------------------------------
st.subheader("Kennzahl im Detail")
wahl = st.selectbox("Kennzahl", spalten, format_func=lambda c: K.BESCHREIBUNG[c][0])
name, einheit, _ = K.BESCHREIBUNG[wahl]
t = trends[wahl]
st.plotly_chart(trend_fig(kz[wahl], t, name, einheit, markiere=letztes), width="stretch")
if "p" in t:
    st.markdown(f"Trend: **{t['steigung_jahrzehnt']:+.2f} {einheit} pro Jahrzehnt** (90-%-Bereich "
                f"{t['lo']:+.2f} bis {t['hi']:+.2f}), Kendalls τ = {t['tau']:.2f}, p = {t['p']:.3f} → "
                + ("**statistisch gesichert**." if t["gesichert"] else "**nicht gesichert** (Zufall nicht auszuschließen)."))
s = kz[wahl].dropna()
mitte = s.index[len(s) // 2]
c1, c2 = st.columns(2)
f = go.Figure()
f.add_histogram(x=s[s.index < mitte], name=f"{s.index.min()}–{mitte - 1}", marker_color=GRAU, opacity=0.6, nbinsx=12)
f.add_histogram(x=s[s.index >= mitte], name=f"{mitte}–{s.index.max()}", marker_color=ROT, opacity=0.6, nbinsx=12)
f.update_layout(barmode="overlay", height=300, margin=dict(t=30, b=10), xaxis_title=f"{name} [{einheit}]",
                yaxis_title="Anzahl Jahre", legend=dict(orientation="h", y=-0.3), title="Verschiebung der Verteilung")
c1.plotly_chart(f, width="stretch")
if "schnitte_modell" in kz:
    sz = kz["schnitte_modell"].dropna().astype(int)
    roll = pd.get_dummies(sz).rolling(10, min_periods=5).mean() * 100
    f2 = go.Figure()
    for n in roll.columns:
        f2.add_scatter(x=roll.index, y=roll[n], stackgroup="one", name=f"{n} Schnitte")
    f2.update_layout(height=300, margin=dict(t=30, b=10), yaxis_title="% der Jahre (gleitend 10 J.)",
                     title="Mögliche Schnittzahl (ModVege-Schnittregel)", legend=dict(orientation="h", y=-0.2))
    c2.plotly_chart(f2, width="stretch")

# --- Kalender-Heatmap ---------------------------------------------------------
st.subheader("Wann im Jahr wird es trocken?")
d = df[df.index.year.isin(jahre)]
kwb = (d["rain"] - d["et0"]).groupby([d.index.year, d.index.isocalendar().week.values]).sum().unstack()
kwb = kwb.loc[:, [w for w in kwb.columns if 10 <= w <= 40]]
f3 = go.Figure(go.Heatmap(z=kwb.values, x=[f"KW {w}" for w in kwb.columns], y=kwb.index, colorscale="RdBu",
                          zmid=0, zmin=-40, zmax=40, colorbar=dict(title="mm/Woche")))
f3.update_layout(height=min(900, 18 * len(kwb) + 100), margin=dict(t=20, b=10), yaxis=dict(autorange="reversed"))
st.plotly_chart(f3, width="stretch")
st.caption("Klimatische Wasserbilanz (Niederschlag − Gras-Referenzverdunstung) je Kalenderwoche, März bis September. "
           "Rot = Wasserdefizit.")

with st.expander("Tabelle aller Kennzahlen"):
    st.dataframe(kz.round(1), width="stretch")
    st.download_button("Als CSV", kz.round(2).to_csv().encode("utf-8"), file_name="klimakennzahlen.csv")
with st.expander("Methodik"):
    st.markdown(K.__doc__)
    st.markdown("**Noch nicht enthalten:** HYRAS-Raster, DWD-Bodenfeuchte (AMBAV), DWD-Phänologie, Klimaprojektionen "
                "(LfU Bayern). Fehlwerte der Hauptstation werden mit der zweitnächsten Station ohne Höhenkorrektur gefüllt.")
