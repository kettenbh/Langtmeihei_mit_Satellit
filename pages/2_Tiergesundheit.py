"""Module 1 und 2: Hitzestress (THI) sowie Parasiten- und Fliegendruck."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from heu import hitze as HZ
from heu import parasiten as P
from heu import ui
from heu.config import HITZE, PARASITEN
from heu.plots import BLAU, GRAU, GRUEN, ROT, band_fig, hist_fig
from heu.satellite import load_fields

WT = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]

st.set_page_config(page_title="Tiergesundheit", page_icon="🐄", layout="wide")
st.title("🐄 Tiergesundheit: Hitzestress, Parasiten und Fliegen")
ui.testmodus_hinweis()
A = ui.annahmen()
tabH, tabP, tabF = st.tabs(["Hitzestress (THI)", "Parasiten & Fliegen", "Feuchtstellen (Leberegel)"])

# =============================================================================
# Hitzestress
# =============================================================================
with tabH:
    try:
        hourly, src_h = ui.stuendlich(ui.token())
    except Exception as e:
        st.error(f"Stündliche Wetterdaten nicht verfügbar: {e}")
        hourly = None
    if hourly is not None:
        c1, c2, c3 = st.columns([2, 2, 1])
        modus = c1.radio("Vorhersage", ["Live (Ensemble-Wettervorhersage)", "Rückblick-Demo (Sommertag wählen)"],
                         help="Im Rückblick wird das tatsächliche Wetter nach dem Stichtag mit Vorhersage-typischer "
                              "Unsicherheit verrauscht. So lässt sich im Winter zeigen, wie die Warnung im Sommer aussah.")
        ort = c2.radio("Ort", ["Stall (THI)", "Weide (THI mit Sonne und Wind)"], horizontal=True)
        spalte = "thi" if ort.startswith("Stall") else "thi_weide"
        kuehe = c3.number_input("Kühe", 1, 500, HITZE.kuehe)
        ens = None
        if modus.startswith("Live"):
            try:
                ens, src_e = ui.ensemble(ui.heute_stunde())
            except Exception as e:
                st.error(f"Ensemble-Vorhersage nicht abrufbar: {e}. Bitte Rückblick-Demo nutzen.")
        else:
            letzt = hourly.index.max().date()
            vorschlag = dt.date(letzt.year if letzt.month >= 9 else letzt.year - 1, 7, 15)
            tag = st.date_input("Stichtag (Vorhersage ab dem Folgetag)", vorschlag,
                                min_value=dt.date(1992, 1, 1), max_value=letzt - dt.timedelta(days=16))
            from heu.stunden import pseudo_ensemble
            ens = pseudo_ensemble(hourly, pd.Timestamp(tag), 15)
            src_e = "Rückblick-Demo: ERA5-Ist-Wetter + wachsende Störung (30 Member)"
        if ens is not None and len(ens):
            tage = HZ.ensemble_tage(ens, HITZE, spalte)
            st.caption(f"Vorhersage: {src_e} · Historie: {src_h}")
            # Warnkarten
            cols = st.columns(3)
            for col, (d, r) in zip(cols, tage.head(3).iterrows()):
                k = int(r["stufe_wahrscheinlich"])
                p_mittel = r["p_stufe3"]
                col.markdown(
                    f"<div style='border-left:8px solid {HZ.STUFEN_FARBE[k]};padding:8px 12px;background:rgba(127,127,127,0.08)'>"
                    f"<b>{WT[d.weekday()]} {d:%d.%m.}</b><br><span style='font-size:1.4em'>Stufe: {HZ.STUFEN[k]}</span><br>"
                    f"THI max {r['thi_max_q50']:.0f} ({r['thi_max_q10']:.0f}–{r['thi_max_q90']:.0f}) · "
                    f"P(≥ leicht) {r['p_stufe2'] * 100:.0f} % · P(≥ mittel) {p_mittel * 100:.0f} %<br>"
                    f"<small>{HZ.HINWEISE[k]}</small></div>", unsafe_allow_html=True)
            # Stündlicher Fächer
            q = HZ.ensemble_stunden(ens, spalte)
            x = list(q.index)
            f = go.Figure()
            f.add_scatter(x=x + x[::-1], y=list(q[90]) + list(q[10])[::-1], fill="toself", fillcolor="rgba(217,83,79,0.15)",
                          line=dict(width=0), name="10–90 %")
            f.add_scatter(x=x + x[::-1], y=list(q[75]) + list(q[25])[::-1], fill="toself", fillcolor="rgba(217,83,79,0.30)",
                          line=dict(width=0), name="25–75 %")
            f.add_scatter(x=x, y=q[50], line=dict(color=ROT, width=2), name="Median")
            for y, lab in [(HITZE.schwelle_beginn, "68 Beginn"), (HITZE.schwelle_leicht, "72 leicht"),
                           (HITZE.schwelle_mittel, "80 mittel")]:
                f.add_hline(y=y, line_dash="dot", line_color="black", annotation_text=lab)
            f.update_layout(height=380, margin=dict(t=30, b=10), yaxis_title="THI", legend=dict(orientation="h", y=1.1))
            st.subheader("THI stündlich, nächste 15 Tage")
            st.plotly_chart(f, width="stretch")
            # Wahrscheinlichkeiten je Tag
            f2 = go.Figure()
            for k, name in [(2, "≥ leicht (72)"), (3, "≥ mittel (80)"), (4, "schwer (90)")]:
                f2.add_bar(x=tage.index, y=tage[f"p_stufe{k}"] * 100, name=name, marker_color=HZ.STUFEN_FARBE[k])
            f2.add_scatter(x=tage.index, y=tage["p_stufe1"] * 100, name="≥ Beginn (68)", line=dict(color=HZ.STUFEN_FARBE[1]))
            f2.update_layout(barmode="group", height=300, margin=dict(t=20, b=10), yaxis_title="Wahrscheinlichkeit [%]",
                             yaxis_range=[0, 100], legend=dict(orientation="h", y=1.15))
            st.plotly_chart(f2, width="stretch")
            # Weidefenster und Milchverlust
            e2 = HZ.mit_thi(ens)
            tagstd = e2[(e2.index.hour >= 5) & (e2.index.hour <= 22)]
            wf = tagstd.assign(ok=tagstd["thi_weide"] < HITZE.schwelle_beginn).groupby(
                [tagstd.index.normalize(), "member"])["ok"].sum().groupby(level=0).median()
            tab = pd.DataFrame({
                "Tag": [f"{WT[d.weekday()]} {d:%d.%m.}" for d in tage.index],
                "THI max (Median)": tage["thi_max_q50"].round(0).values,
                "Hitzestunden ≥ 68": tage["stunden_68_q50"].round(0).values,
                "Weidestunden 5–22 Uhr mit THI < 68": wf.reindex(tage.index).round(0).values,
                f"Milchverlust Herde [kg/Tag]": (tage["milchverlust_q50"] * kuehe).round(0).values,
            })
            st.dataframe(tab, hide_index=True, width="stretch")
            st.caption(f"Milchverlust: {HITZE.milchverlust_kg_je_thi} kg je Kuh und Tag je THI-Punkt des Tagesmittels über 68 "
                       "(Literaturspanne 0,2–0,4; mit Milchleistungsdaten kalibrieren). Stufe steigt um eins nach 3 Tagen "
                       "in Folge mit ≥ 8 Hitzestunden ohne nächtliche Erholung (6 h unter THI 65).")

        # Saisonvergleich
        st.subheader("Hitzestunden der Saison im Vergleich zu 1991–2025")
        hist = HZ.historie(hourly, HITZE)
        jahr = int(hourly.index.max().year) if hourly.index.max().month >= 5 else int(hourly.index.max().year) - 1
        if modus.startswith("Rückblick"):
            jahr = tag.year
        akt = hist.loc[jahr].dropna() if jahr in hist.index else None
        st.plotly_chart(band_fig(hist, akt, jahr, f"Kumulierte Stunden mit THI ≥ 68 ab 1. Mai ({jahr} rot)", "Stunden",
                                 x0_tag=120, hist_jahre=list(range(1991, 2026))), width="stretch")
        jw = HZ.jahreswerte(hourly, HITZE)
        c1, c2 = st.columns(2)
        c1.plotly_chart(hist_fig(jw["hitzestunden_68"].drop(jahr, errors="ignore"),
                                 float(jw.loc[jahr, "hitzestunden_68"]) if jahr in jw.index else None, jahr,
                                 "Hitzestunden Mai–Sep je Jahr"), width="stretch")
        c2.plotly_chart(hist_fig(jw["tage_thi72"].drop(jahr, errors="ignore"),
                                 float(jw.loc[jahr, "tage_thi72"]) if jahr in jw.index else None, jahr,
                                 "Tage mit THI max ≥ 72 je Jahr"), width="stretch")

# =============================================================================
# Parasiten und Fliegen
# =============================================================================
with tabP:
    try:
        M = ui.modell(A, ui.token())
        df = M["df"]
    except Exception as e:
        st.error(f"Wetterdaten nicht verfügbar: {e}")
        df = None
    if df is not None:
        plan = P.lade_weideplan()
        weiden = [f["name"] for f in load_fields()] or ["Weide"]
        if plan is not None:
            weiden = sorted(set(weiden) | set(plan["weide"].astype(str).str.strip()))
        letzt = df.index.max().date()
        c1, c2 = st.columns([1, 2])
        stichtag = c1.date_input("Stichtag", letzt, min_value=dt.date(1992, 3, 1), max_value=letzt)
        c2.info("Weideplan: " + ("data/weideplan.csv" if plan is not None else
                                 f"Standard (15 Jungrinder je Weide {PARASITEN.weide_austrieb} bis {PARASITEN.weide_abtrieb}); "
                                 "eigenen Plan aus data/weideplan_vorlage.csv anlegen"))
        reihen = P.indexreihen(df, PARASITEN, plan, weiden)
        hj = list(range(A.trainings_start, A.trainings_ende + 1))
        rt = P.risikotabelle(reihen, pd.Timestamp(stichtag), hj)
        st.subheader(f"Risiko am {stichtag:%d.%m.%Y} im Vergleich zum gleichen Kalendertag 1991–2025")
        st.dataframe(rt.round(1), hide_index=True, width="stretch")
        st.caption("Perzentil: Anteil der Jahre 1991–2025, die am gleichen Kalendertag einen niedrigeren Wert hatten. "
                   "Unter 50 grün, 50–80 gelb, über 80 rot. Die Indizes sind relative Wetterrisiken, keine Befunde.")
        for t in P.terminhinweise(df, PARASITEN, stichtag.year,
                                  reihen.get(f"Magen-Darm-Würmer {weiden[0]}")):
            st.markdown(f"- {t}")
        wahl = st.selectbox("Saisonverlauf anzeigen", list(reihen))
        m = P.jahresmatrix(reihen[wahl])
        akt = m.loc[stichtag.year].iloc[: (pd.Timestamp(stichtag).dayofyear)] if stichtag.year in m.index else None
        st.plotly_chart(band_fig(m, akt, stichtag.year, f"{wahl}: kumulierter Index", "Index", hist_jahre=hj),
                        width="stretch")
        with st.expander("Ollerenshaw-Monatswerte"):
            o = P.ollerenshaw(df, PARASITEN)
            if stichtag.year in o.index.get_level_values(0):
                st.dataframe(o.loc[stichtag.year].round(2), width="stretch")
            st.caption("Mt = n·(R − P + 5), R und P in Zoll, n = Regentage ≥ 0,25 mm; Monatswerte gedeckelt bei 100, "
                       "Mai und Oktober halb. Werte über etwa 400–475 im Sommer gelten in der britischen Literatur als "
                       "Leberegeljahr; für Kirchweidach zählt hier vor allem der Vergleich mit den eigenen Jahren.")
        with st.expander("Methodik und Grenzen"):
            st.markdown(open(P.__file__, encoding="utf-8").read().split('"""')[1])
            st.markdown("**Noch nicht enthalten:** 15-Tage-Vorhersage des Risikos, Lungenwurm, Gnitzen (Blauzungenkrankheit), "
                        "Kalibrierung an Kotproben, Tankmilch-Antikörpertests oder Schlachtbefunden.")

# =============================================================================
# Feuchtstellen aus dem Geländemodell
# =============================================================================
with tabF:
    from heu import gelaende as G
    st.markdown("Nasse Senken und Quellbereiche sind Lebensraum der Zwergschlammschnecke, des Zwischenwirts des Leberegels. "
                "Der topografische Feuchteindex (TWI) aus dem Geländemodell zeigt, wo sich Wasser sammelt.")
    files = G.kacheln()
    try:
        import rasterio  # noqa: F401
        has_rio = True
    except ImportError:
        has_rio = False
    if not files or not has_rio:
        st.info("So geht's:\n\n1. `pip install rasterio`\n2. Auf geodaten.bayern.de → OpenData → *Digitales Geländemodell 1 m* "
                "die Kachel(n) mit Wiese Ost und West herunterladen (GeoTIFF, EPSG:25832).\n"
                "3. Dateien nach `data/dgm/` legen und diese Seite neu laden.\n\n"
                f"Status: rasterio {'✓' if has_rio else '✗'} · Kacheln gefunden: {len(files)}")
    else:
        res = st.select_slider("Rasterweite", [2.0, 5.0, 10.0], 5.0, format_func=lambda v: f"{v:g} m")
        try:
            out = G.feuchte_auswertung(load_fields(), res)
            f = go.Figure(go.Heatmap(z=out["twi"], x=out["xs"], y=out["ys"], colorscale="YlGnBu",
                                     zmin=np.nanpercentile(out["twi"], 5), zmax=np.nanpercentile(out["twi"], 99),
                                     colorbar=dict(title="TWI")))
            for name, xs, ys in out["umrisse"]:
                f.add_scatter(x=xs, y=ys, mode="lines", line=dict(color="red", width=2), name=name)
            f.update_layout(height=600, yaxis=dict(scaleanchor="x"), margin=dict(t=20, b=10))
            st.plotly_chart(f, width="stretch")
            st.dataframe(pd.DataFrame(out["felder"]).round(1), hide_index=True, width="stretch")
            st.caption(f"Feucht = TWI über {out['schwelle']:.1f} (90. Perzentil des Ausschnitts). "
                       "Vor Ort prüfen: Binsen, Trittstellen, Quellaustritte. Ausbau: Bestätigung über Sentinel-1-Rückstreuung.")
        except Exception as e:
            st.error(f"Auswertung fehlgeschlagen: {e}")
