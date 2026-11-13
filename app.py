"""Heu-Prognose MVP – Streamlit-App.

Start:  streamlit run app.py
Offline-Test mit Kunstwetter:  HEU_SYNTHETIC=1 streamlit run app.py
"""
from __future__ import annotations

import datetime as dt
from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from heu import forecast as F
from heu.config import ANNAHMEN, STANDORT, Annahmen
from heu.model import ModVegeParams, State, omd_to_nel, run
from heu.prices import PRICE_FILE, load_prices, price_forecast
from heu.weather import date_from_doy, doy365, load_weather, year_arrays
from heu import satellite as SAT
import os

st.set_page_config(page_title="Heu-Prognose Kirchweidach", page_icon="🌾", layout="wide")

GRAU = "#9aa0a6"
BLAU = "#1f6fd1"
GRUEN = "#2e9d5b"
ROT = "#d9534f"


# ---------------------------------------------------------------------------
# Rechenkern (gecacht)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Lade Wetterdaten 1991 bis heute …")
def get_weather(start_year: int, token: int):
    return load_weather(start_year, refresh=token > 0)


@st.cache_resource(show_spinner="Kalibriere Modell an den Trainingsjahren und simuliere 2026 …")
def build(a: Annahmen, token: int):
    df, src = get_weather(a.trainings_start, token)
    clim = F.training_years(df, a)
    p, achieved = F.calibrate_ni(clim, ModVegeParams(WHC=a.nutzbare_feldkapazitaet_mm), a)
    sims = F.simulate_years(clim, p, a)
    hist = F.history_table(sims)
    cur = F.run_current_year(df, a.simulations_jahr, clim, p, a)
    # Klimatologie von Bodenwasser und Temperatursumme
    traces = {}
    for y, w in clim.items():
        tr = []
        run(w, p, a, trace=tr)
        traces[y] = np.array(tr)
    cur_trace = []
    s = State(WR=p.WHC)
    for d in range(cur.n):
        run(cur.w, p, a, state=s, start=d, end=d + 1, trace=cur_trace)
    return dict(df=df, src=src, clim=clim, p=p, achieved=achieved, hist=hist, cur=cur,
                traces=traces, cur_trace=np.array(cur_trace))


@st.cache_resource(show_spinner="Berechne Prognosen für alle Stichtage …")
def get_fan(a: Annahmen, token: int, perfect: int):
    m = build(a, token)
    return F.fan(m["cur"], m["clim"], m["p"], a, doy365(dt.date(2001, 3, 1)), doy365(dt.date(2001, 10, 31)), 7, perfect)


# ---------------------------------------------------------------------------
# Seitenleiste
# ---------------------------------------------------------------------------
if "token" not in st.session_state:
    st.session_state.token = 0

st.sidebar.title("🌾 Heu-Prognose")
FELDER = SAT.load_fields()
feld_namen = [f["name"] for f in FELDER] + ["Generisches Feld (2 ha)"]
feld_wahl = st.sidebar.selectbox("Feld", feld_namen)
FELD = next((f for f in FELDER if f["name"] == feld_wahl), None)
st.sidebar.caption(STANDORT.name + (f" · {FELD['flaeche_ha']:.2f} ha" if FELD else " · generisch"))

with st.sidebar.expander("Annahmen (generisch)", expanded=False):
    flaeche = st.number_input("Feldgröße [ha]", 0.1, 100.0, float(FELD["flaeche_ha"]) if FELD else STANDORT.flaeche_ha, 0.1,
                              key=f"flaeche_{feld_wahl}")
    ziel = st.number_input("Mittlerer Brutto-Jahresertrag 1991–2025 [dt TM/ha]", 30.0, 140.0,
                           ANNAHMEN.ziel_jahresertrag_dt_tm_ha, 5.0,
                           help="Kalibrierziel des Modells. Generische Annahme für Bio-Dauergrünland in Südostbayern.")
    nfk = st.number_input("Nutzbare Feldkapazität [mm]", 40.0, 250.0, ANNAHMEN.nutzbare_feldkapazitaet_mm, 10.0)
    tsum = st.number_input("Temperatursumme Schnittreife 1. Aufwuchs [°Cd]", 500.0, 1100.0,
                           ANNAHMEN.tsumme_erster_schnitt, 25.0)
    trocken = st.radio("Trocknung", ["Bodenheu (3 Trockentage)", "Belüftungstrocknung (2 Trockentage)"], index=0)
    ballen = st.number_input("Ballengewicht Heu [kg FM]", 100.0, 600.0, ANNAHMEN.ballengewicht_kg, 10.0)
    sigma = st.slider("Modellfehler je Schnitt (σ, log)", 0.0, 0.3, ANNAHMEN.modellfehler_sigma, 0.01)

A = replace(ANNAHMEN, ziel_jahresertrag_dt_tm_ha=ziel, nutzbare_feldkapazitaet_mm=nfk,
            tsumme_erster_schnitt=tsum, trockentage_noetig=3 if trocken.startswith("Boden") else 2,
            ballengewicht_kg=ballen, modellfehler_sigma=sigma)

st.session_state["A"] = A   # für die Seiten Futterqualität, Tiergesundheit, Klima

if st.sidebar.button("Wetterdaten aktualisieren"):
    st.session_state.token += 1
    st.cache_resource.clear()

try:
    M = build(A, st.session_state.token)
except Exception as e:  # Netz- oder Datenfehler sichtbar machen
    st.error(f"Wetterdaten konnten nicht geladen werden: {e}")
    st.info("Internetverbindung prüfen. Offline-Test: `HEU_SYNTHETIC=1 streamlit run app.py`")
    st.stop()

cur, hist, p, clim = M["cur"], M["hist"], M["p"], M["clim"]
JAHR = A.simulations_jahr
if cur.n < 70:
    st.warning(f"Für {JAHR} liegen erst {cur.n} Tage Wetterdaten vor.")
    st.stop()

min_d = date_from_doy(JAHR, doy365(dt.date(2001, 3, 1)))
max_d = date_from_doy(JAHR, cur.n - 1)
default_d = min(max_d, dt.date(JAHR, 5, 1))
stichtag_date = st.sidebar.slider("Stichtag der Prognose", min_d, max_d, default_d, format="DD.MM.YYYY",
                                  help="Simuliert, was die App an diesem Tag gewusst hätte: Wetter bis zum Stichtag ist bekannt, danach Szenarien aus 1991–2025.")
STICHTAG = doy365(stichtag_date)
perfect = st.sidebar.radio("Wettervorhersage im Rückblick", ["Keine (nur Klimatologie)", "Perfekte 7-Tage-Vorhersage"],
                           help="'Perfekt' nimmt das tatsächliche Wetter der nächsten 7 Tage als Vorhersage an: Obergrenze dessen, was eine gute Vorhersage bringt.")
PERFECT = 7 if perfect.startswith("Perfekte") else 0

ziele = {"Saisonertrag": "saison", **{f"{i}. Schnitt": f"schnitt{i}" for i in range(1, 5)}}
ziel_label = st.sidebar.selectbox("Zielgröße", list(ziele))
ZIEL = ziele[ziel_label]
einheit = st.sidebar.selectbox("Einheit", ["dt TM/ha", "% vom Mittel 1991–2025", "Ballen (Feld)", "t Heu (Feld)"])
zeige_ist = st.sidebar.checkbox(f"Ist-Wert {JAHR} anzeigen (Auflösung)", value=False,
                                help="'Ist' ist hier die Modellsimulation mit dem tatsächlichen Wetter 2026, keine Messung am Betrieb.")
st.sidebar.caption(f"Wetter: {M['src']}  \nDaten bis {max_d:%d.%m.%Y}")


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------
hist_vals = hist[ZIEL].dropna().to_numpy() if ZIEL in hist else np.array([])
hist_mean = float(np.mean(hist_vals)) if len(hist_vals) else np.nan


def conv(x):
    x = np.asarray(x, dtype=float)
    if einheit == "dt TM/ha":
        return x
    if einheit.startswith("%"):
        return x / hist_mean * 100
    kg_tm_feld = x * 100 * flaeche
    if einheit.startswith("Ballen"):
        return kg_tm_feld / A.tm_gehalt_heu / A.ballengewicht_kg
    return kg_tm_feld / A.tm_gehalt_heu / 1000


EINHEIT_KURZ = {"dt TM/ha": "dt TM/ha", "% vom Mittel 1991–2025": "%", "Ballen (Feld)": "Ballen", "t Heu (Feld)": "t Heu"}
WOCHENTAG = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]


def fmt(x):
    return f"{x:,.0f}".replace(",", ".") if abs(x) >= 100 else f"{x:,.1f}".replace(".", ",")


def ist_value():
    if ZIEL == "saison":
        return sum(c.netto_kg for c in cur.final.cuts) / 100
    i = int(ZIEL[-1])
    return cur.final.cuts[i - 1].netto_kg / 100 if len(cur.final.cuts) >= i else np.nan


def d2date(i):
    return date_from_doy(JAHR, int(i))


prog = F.forecast(cur, clim, STICHTAG, p, A, PERFECT)
samples = prog.saison if ZIEL == "saison" else prog.schnitte[int(ZIEL[-1])]
share_valid = np.mean(~np.isnan(samples))
samples = samples[~np.isnan(samples)]

# ---------------------------------------------------------------------------
# Kopf
# ---------------------------------------------------------------------------
st.title(f"Heu-Prognose {JAHR} – Simulation zum Stichtag {stichtag_date:%d.%m.%Y}")
st.caption("MVP mit generischen Annahmen: ein Feld, "
           f"{flaeche:g} ha ({feld_wahl}), {STANDORT.name}. Training: Wetter {A.trainings_start}–{A.trainings_ende}. "
           "Alle Erträge sind Modellwerte, noch nicht mit Betriebsdaten kalibriert.")

tab1, tab2, tabS, tab3, tab4, tab5 = st.tabs(["Ertragsprognose", "Schnittzeitpunkt", "Satellit", "Wetter 2026", "Heupreise", "Methodik & Annahmen"])

# ---------------------------------------------------------------------------
# Tab 1: Ertrag
# ---------------------------------------------------------------------------
with tab1:
    if len(samples) == 0 or share_valid < 0.3:
        st.info(f"In weniger als 30 % der Szenarien gibt es einen {ziel_label}. Andere Zielgröße wählen.")
    else:
        q10, q25, q50, q75, q90 = np.percentile(samples, [10, 25, 50, 75, 90])
        rang = float(np.mean(hist_vals < q50) * 100) if len(hist_vals) else np.nan
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"Median {ziel_label}", f"{fmt(conv(q50))} {EINHEIT_KURZ[einheit]}")
        c2.metric("80-%-Bereich", f"{fmt(conv(q10))} – {fmt(conv(q90))}")
        c3.metric("Relativ zum Mittel 1991–2025", f"{q50 / hist_mean * 100:.0f} %",
                  help="Mittel der simulierten Trainingsjahre mit derselben Schnittregel")
        c4.metric("Rang in der Historie", f"{rang:.0f}. Perzentil",
                  help="Anteil der Jahre 1991–2025, die schlechter waren als der Prognose-Median")
        if share_valid < 0.95:
            st.caption(f"Hinweis: Nur in {share_valid * 100:.0f} % der Szenarien kommt es zu einem {ziel_label}.")

        # Histogramm-Overlay
        allv = np.concatenate([conv(hist_vals), conv(samples)])
        lo, hi = np.nanmin(allv), np.nanmax(allv)
        size = (hi - lo) / 30 if hi > lo else 1
        fig = go.Figure()
        fig.add_histogram(x=conv(hist_vals), name=f"Historie {A.trainings_start}–{A.trainings_ende} (Simulation)",
                          marker_color=GRAU, opacity=0.55, histnorm="probability density",
                          xbins=dict(start=lo, end=hi + size, size=size))
        fig.add_histogram(x=conv(samples), name=f"Prognose am {stichtag_date:%d.%m.}",
                          marker_color=BLAU, opacity=0.65, histnorm="probability density",
                          xbins=dict(start=lo, end=hi + size, size=size))
        fig.add_vline(x=float(conv(hist_mean)), line_dash="dot", line_color=GRAU, annotation_text="Mittel Historie")
        if zeige_ist and not np.isnan(ist_value()):
            fig.add_vline(x=float(conv(ist_value())), line_color=ROT, line_width=2, annotation_text=f"Ist {JAHR}")
        fig.update_layout(barmode="overlay", height=380, margin=dict(t=30, b=10),
                          xaxis_title=f"{ziel_label} [{einheit}]", yaxis_title="Dichte",
                          legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, width="stretch")

        # Fächerdiagramm
        fan = get_fan(A, st.session_state.token, PERFECT)
        cols = [f"{ZIEL}_q{q}" for q in (10, 25, 50, 75, 90)]
        fd = fan.dropna(subset=cols)
        x = [d2date(t) for t in fd["tag"]]
        f2 = go.Figure()
        f2.add_scatter(x=x + x[::-1], y=list(conv(fd[cols[4]])) + list(conv(fd[cols[0]]))[::-1], fill="toself",
                       fillcolor="rgba(31,111,209,0.15)", line=dict(width=0), name="80-%-Bereich")
        f2.add_scatter(x=x + x[::-1], y=list(conv(fd[cols[3]])) + list(conv(fd[cols[1]]))[::-1], fill="toself",
                       fillcolor="rgba(31,111,209,0.30)", line=dict(width=0), name="50-%-Bereich")
        f2.add_scatter(x=x, y=conv(fd[cols[2]]), line=dict(color=BLAU, width=2), name="Median")
        f2.add_hline(y=float(conv(hist_mean)), line_dash="dot", line_color=GRAU, annotation_text="Mittel Historie")
        if zeige_ist and not np.isnan(ist_value()):
            f2.add_hline(y=float(conv(ist_value())), line_color=ROT, annotation_text=f"Ist {JAHR}")
        f2.add_vline(x=stichtag_date, line_dash="dash", line_color="black")
        f2.update_layout(height=360, margin=dict(t=30, b=10), yaxis_title=f"{ziel_label} [{einheit}]",
                         xaxis_title="Stichtag der Prognose", legend=dict(orientation="h", y=1.12))
        st.subheader("So verengt sich die Prognose im Lauf der Saison")
        st.plotly_chart(f2, width="stretch")

        # Kleine Vielfache: Histogramme an festen Stichtagen
        with st.expander("Histogramme an festen Stichtagen", expanded=True):
            tage = [dt.date(JAHR, m, 1) for m in (3, 4, 5, 6, 7, 8, 9)]
            tage = [t for t in tage if doy365(t) < cur.n]
            f3 = make_subplots(rows=len(tage), cols=1, shared_xaxes=True, vertical_spacing=0.02,
                               row_titles=[f"{t:%d.%m.}" for t in tage])
            for r, t in enumerate(tage, start=1):
                pr = F.forecast(cur, clim, doy365(t), p, A, PERFECT)
                v = pr.saison if ZIEL == "saison" else pr.schnitte[int(ZIEL[-1])]
                v = v[~np.isnan(v)]
                f3.add_histogram(x=conv(hist_vals), marker_color=GRAU, opacity=0.5, histnorm="probability density",
                                 xbins=dict(start=lo, end=hi + size, size=size), showlegend=False, row=r, col=1)
                if len(v):
                    f3.add_histogram(x=conv(v), marker_color=BLAU, opacity=0.7, histnorm="probability density",
                                     xbins=dict(start=lo, end=hi + size, size=size), showlegend=False, row=r, col=1)
                if zeige_ist and not np.isnan(ist_value()):
                    f3.add_vline(x=float(conv(ist_value())), line_color=ROT, row=r, col=1)
                f3.update_yaxes(showticklabels=False, row=r, col=1)
            f3.update_layout(barmode="overlay", height=110 * len(tage) + 60, margin=dict(t=10, b=10))
            f3.update_xaxes(title_text=f"{ziel_label} [{einheit}]", row=len(tage), col=1)
            st.plotly_chart(f3, width="stretch")
            st.caption("Grau: simulierte Jahre 1991–2025. Blau: Prognose am jeweiligen Stichtag "
                       f"({len(prog.szenario_jahre)} Wetterszenarien × {A.ziehungen_je_szenario} Ziehungen Modellfehler).")

# ---------------------------------------------------------------------------
# Tab 2: Schnittzeitpunkt
# ---------------------------------------------------------------------------
with tab2:
    done = prog.erledigt
    st.subheader("Bisherige Schnitte (Simulation mit Ist-Wetter)")
    if done:
        st.dataframe(pd.DataFrame([{
            "Schnitt": i + 1, "Datum": f"{d2date(c.day):%d.%m.%Y}",
            "Heu netto [dt TM/ha]": round(c.netto_kg / 100, 1),
            "Feldverlust": f"{c.verlust * 100:.0f} %",
            "NEL [MJ/kg TM]": round(c.nel, 2),
            "Ballen (Feld)": round(c.netto_kg * flaeche / A.tm_gehalt_heu / A.ballengewicht_kg),
            "Hinweis": "ohne volles Heuwetter gemäht" if c.erzwungen else "",
        } for i, c in enumerate(done)]), hide_index=True, width="stretch")
    else:
        st.write("Noch kein Schnitt bis zum Stichtag.")

    nxt = len(done) + 1
    if nxt <= A.max_schnitte:
        tage = prog.schnitt_tage[nxt]
        tage = tage[~np.isnan(tage)]
        if len(tage):
            med = d2date(np.median(tage))
            p10, p90 = d2date(np.percentile(tage, 10)), d2date(np.percentile(tage, 90))
            st.markdown(f"**Nächster ({nxt}.) Schnitt**: Median **{med:%d.%m.}**, 80 % der Szenarien zwischen "
                        f"{p10:%d.%m.} und {p90:%d.%m.} ({len(tage)} von {len(prog.szenario_jahre)} Szenarien mit Schnitt).")

    ct = F.cut_timing(cur, clim, STICHTAG, p, A, horizon=14, perfect_days=PERFECT)
    if ct.empty:
        st.info("Außerhalb der Schnittsaison.")
    else:
        extra = A.feldverlust_je_regentag * 2
        ct["nel"] = ct["omd"].apply(omd_to_nel)
        ct["exp_verlust"] = A.feldverlust_basis + (1 - ct["p_heuwetter"]) * extra
        ct["energie"] = ct["ertrag_dt"] * (1 - ct["exp_verlust"]) * ct["nel"]
        ct["bewertung"] = (ct["energie"] / ct["energie"].max() * 100).round(0)
        ct["datum"] = [d2date(t) for t in ct["tag"]]
        reif = ct[ct["p_reif"] >= 0.5]
        if len(reif):
            best = reif.loc[reif["bewertung"].idxmax()]
            st.success(f"Empfehlung: Mähen am **{WOCHENTAG[best['datum'].weekday()]} {best['datum']:%d.%m.}** – Heuwetter-Chance {best['p_heuwetter'] * 100:.0f} %, "
                       f"erwartet {best['ertrag_dt']:.1f} dt TM/ha brutto, NEL {best['nel']:.2f} MJ/kg TM.")
        else:
            st.info("In den nächsten 14 Tagen ist der Bestand laut Modell noch nicht schnittreif.")
        f4 = make_subplots(specs=[[{"secondary_y": True}]])
        f4.add_bar(x=ct["datum"], y=ct["p_heuwetter"] * 100, name=f"Chance auf {A.trockentage_noetig} Trockentage [%]",
                   marker_color=[GRUEN if r >= 0.5 else GRAU for r in ct["p_reif"]], opacity=0.7)
        f4.add_scatter(x=ct["datum"], y=ct["ertrag_dt"], name="Erntbare Masse [dt TM/ha]",
                       line=dict(color=BLAU, width=2), secondary_y=True)
        f4.add_scatter(x=ct["datum"], y=ct["nel"] * 10, name="NEL × 10 [MJ/kg TM]",
                       line=dict(color=ROT, dash="dot"), secondary_y=True)
        f4.update_layout(height=380, margin=dict(t=30, b=10), legend=dict(orientation="h", y=1.15))
        f4.update_yaxes(title_text="Heuwetter-Chance [%]", range=[0, 100], secondary_y=False)
        f4.update_yaxes(title_text="dt TM/ha · NEL×10", secondary_y=True)
        st.plotly_chart(f4, width="stretch")
        st.caption("Grüne Balken: Bestand in mindestens der Hälfte der Szenarien schnittreif. "
                   + ("Heuwetter-Chance der ersten 7 Tage aus 'perfekter Vorhersage' (Ist-Wetter), danach Klimatologie."
                      if PERFECT else "Heuwetter-Chance aus Klimatologie 1991–2025 (im Live-Betrieb: Ensemble-Wettervorhersage)."))
        st.dataframe(ct[["datum", "ertrag_dt", "nel", "p_heuwetter", "p_reif", "bewertung"]].rename(columns={
            "datum": "Tag", "ertrag_dt": "Masse [dt TM/ha]", "nel": "NEL", "p_heuwetter": "P(Heuwetter)",
            "p_reif": "P(schnittreif)", "bewertung": "Bewertung"}).round(2), hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
# Tab Satellit
# ---------------------------------------------------------------------------
def model_cut_dates(year: int) -> list[dt.date]:
    if year == JAHR:
        return [d2date(c.day) for c in cur.final.cuts]
    if year not in hist.index:
        return []
    r = hist.loc[year]
    return [date_from_doy(year, int(r[f"tag{i}"])) for i in range(1, 6) if f"tag{i}" in r and not pd.isna(r[f"tag{i}"])]


with tabS:
    if FELD is None:
        st.info("Für Satellitendaten bitte in der Seitenleiste ein echtes Feld wählen (data/felder.geojson).")
    else:
        synthetic = os.environ.get("HEU_SYNTHETIC") == "1"
        creds = SAT.cdse_credentials()
        c1, c2 = st.columns([3, 2])
        if synthetic:
            c1.warning("Testmodus: Kunst-Satellitendaten aus dem Modell, keine echten Aufnahmen.")
            tr_all = {y: t for y, t in M["traces"].items() if y >= 2017}
            s2 = SAT.synthetic_s2(tr_all)
            s1 = None
        else:
            c1.markdown("**Zugang:** " + ("CDSE (Sentinel-2 + Sentinel-1) ✓" if creds else
                        "kein CDSE-Zugang hinterlegt – Abruf über AWS Earth Search (nur Sentinel-2, `pip install pystac-client rasterio`)"))
            backend = c2.radio("Quelle", ["auto", "cdse", "earth-search"], horizontal=True,
                               help="auto = CDSE, wenn Zugangsdaten vorhanden, sonst Earth Search")
            if c2.button(f"Satellitendaten für {FELD['name']} abrufen/aktualisieren"):
                bar = st.progress(0.0, "Starte …")
                try:
                    SAT.update(FELD, backend, progress=lambda f, t: bar.progress(min(1.0, f), t))
                    bar.progress(1.0, "Fertig")
                except Exception as e:
                    st.error(f"Abruf fehlgeschlagen: {e}")
            s2 = SAT.load_cached(FELD["id"], "s2")
            s1 = SAT.load_cached(FELD["id"], "s1")
        if s2 is None or s2.empty:
            st.info("Noch keine Satellitendaten im Cache. Der erste Abruf (2017 bis heute) dauert über CDSE etwa "
                    "1 bis 3 Minuten je Feld, über Earth Search deutlich länger.")
        else:
            cuts = SAT.detect_cuts(s2, s1)
            jahre = sorted(set(s2.index.year), reverse=True)
            jahr = st.selectbox("Jahr", jahre, index=jahre.index(JAHR) if JAHR in jahre else 0)
            d = s2[s2.index.year == jahr]
            klar = d[d["clear"] >= SAT.MIN_CLEAR]
            trueb = d[(d["clear"] < SAT.MIN_CLEAR) & (d["clear"] > 0.2)]
            f7 = make_subplots(specs=[[{"secondary_y": True}]])
            f7.add_scatter(x=klar.index, y=klar["ndvi"], mode="markers+lines", name="NDVI (klar)",
                           line=dict(color=GRUEN), marker=dict(size=7),
                           error_y=dict(type="data", symmetric=False, array=(klar["ndvi_q75"] - klar["ndvi"]),
                                        arrayminus=(klar["ndvi"] - klar["ndvi_q25"]), thickness=1))
            f7.add_scatter(x=trueb.index, y=trueb["ndvi"], mode="markers", name="NDVI (teilbewölkt, ignoriert)",
                           marker=dict(color=GRAU, size=5, symbol="x"))
            if "ndre" in klar:
                f7.add_scatter(x=klar.index, y=klar["ndre"], mode="lines", name="NDRE", line=dict(color=BLAU, dash="dot"))
            if s1 is not None and not s1.empty and "vh_db" in s1:
                d1 = s1[s1.index.year == jahr]
                f7.add_scatter(x=d1.index, y=d1["vh_db"], mode="lines", name="Sentinel-1 VH [dB]",
                               line=dict(color="#8e6bbf", width=1), secondary_y=True)
            for _, r in cuts[cuts["datum"].dt.year == jahr].iterrows():
                f7.add_vrect(x0=r["frueh"], x1=r["spaet"], fillcolor=GRUEN, opacity=0.12, line_width=0)
                f7.add_vline(x=r["datum"], line_color=GRUEN, line_dash="dash")
            for md in model_cut_dates(jahr):
                f7.add_vline(x=pd.Timestamp(md), line_color=GRAU, line_dash="dot")
            f7.update_yaxes(title_text="Index", range=[0, 1], secondary_y=False)
            f7.update_yaxes(title_text="VH [dB]", secondary_y=True)
            f7.update_layout(height=420, margin=dict(t=30, b=10), legend=dict(orientation="h", y=1.15))
            st.plotly_chart(f7, width="stretch")
            st.caption("Grün gestrichelt: aus dem Satelliten erkannte Schnitte (Band = Zeitraum zwischen den Aufnahmen). "
                       "Grau gepunktet: Schnitte laut Modell-Schnittregel.")

            # Vergleich über alle Jahre
            rows = []
            for y in sorted(set(s2.index.year)):
                det = [r for _, r in cuts.iterrows() if r["datum"].year == y]
                mod = model_cut_dates(y)
                dy = s2[(s2.index.year == y) & (s2.index.month.isin([4, 5, 6, 7, 8, 9, 10]))]
                diffs = []
                for r in det:
                    if mod:
                        diffs.append(min(abs((r["datum"].date() - m).days) for m in mod))
                rows.append({"Jahr": y, "klare Aufnahmen Apr–Okt": int((dy["clear"] >= SAT.MIN_CLEAR).sum()),
                             "Schnitte erkannt": len(det),
                             "Daten erkannt": ", ".join(f"{r['datum']:%d.%m.}" for r in det),
                             "Schnitte Modell": len(mod),
                             "Daten Modell": ", ".join(f"{m:%d.%m.}" for m in mod),
                             "Ø Abweichung [Tage]": round(float(np.mean(diffs)), 1) if diffs else None})
            st.subheader("Erkannte Schnitte gegen Modell-Schnittregel")
            st.dataframe(pd.DataFrame(rows).sort_values("Jahr", ascending=False), hide_index=True, width="stretch")
            st.caption("Diese Gegenüberstellung ist die Grundlage für Schritt 3: Schnittregel und Modellparameter "
                       "werden an den erkannten Terminen und am NDVI/LAI-Verlauf kalibriert.")
            st.download_button("Satelliten-Zeitreihe als CSV", s2.to_csv().encode("utf-8"),
                               file_name=f"sentinel2_{FELD['id']}.csv")

# ---------------------------------------------------------------------------
# Tab 3: Wetter
# ---------------------------------------------------------------------------
with tab3:
    tr = M["traces"]
    years = sorted(tr)
    n = min(cur.n, STICHTAG + 1)
    xs = [d2date(i) for i in range(365)]
    xcur = xs[:n]

    def band(fig, arr, row, name):
        q = np.percentile(arr, [10, 50, 90], axis=0)
        fig.add_scatter(x=xs + xs[::-1], y=list(q[2]) + list(q[0])[::-1], fill="toself", fillcolor="rgba(150,150,150,0.25)",
                        line=dict(width=0), name=f"{name}: 10–90 % 1991–2025", row=row, col=1, showlegend=False)
        fig.add_scatter(x=xs, y=q[1], line=dict(color=GRAU, dash="dot"), row=row, col=1, showlegend=False)

    f5 = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                       subplot_titles=["Temperatursumme ab 1.1. [°Cd]", "Niederschlag kumuliert [mm]", "Bodenwasser [% nFK]"])
    tsum_cl = np.array([np.cumsum(np.maximum(clim[y]["tmean"], 0)) for y in years])
    rain_cl = np.array([np.cumsum(clim[y]["rain"]) for y in years])
    wr_cl = np.array([tr[y][:, 4] * 100 for y in years])
    band(f5, tsum_cl, 1, "T-Summe")
    band(f5, rain_cl, 2, "Regen")
    band(f5, wr_cl, 3, "Bodenwasser")
    f5.add_scatter(x=xcur, y=np.cumsum(np.maximum(cur.w["tmean"][:n], 0)), line=dict(color=ROT, width=2), row=1, col=1, showlegend=False)
    f5.add_scatter(x=xcur, y=np.cumsum(cur.w["rain"][:n]), line=dict(color=BLAU, width=2), row=2, col=1, showlegend=False)
    f5.add_scatter(x=xcur, y=M["cur_trace"][:n, 4] * 100, line=dict(color=GRUEN, width=2), row=3, col=1, showlegend=False)
    f5.add_hline(y=A.tsumme_erster_schnitt, line_dash="dash", line_color="black", row=1, col=1,
                 annotation_text="Schnittreife 1. Aufwuchs")
    f5.update_layout(height=760, margin=dict(t=40, b=10))
    st.plotly_chart(f5, width="stretch")
    st.caption(f"Farbig: {JAHR} bis Stichtag. Grau: Median und 10–90-%-Band der Jahre 1991–2025.")

# ---------------------------------------------------------------------------
# Tab 4: Preise
# ---------------------------------------------------------------------------
with tab4:
    seg = st.radio("Segment", ["konventionell", "bio"], horizontal=True)
    prices = load_prices(seg)
    if prices is None or prices["preis"].dropna().shape[0] < 24:
        st.info(
            "Noch keine ausreichende Preisreihe hinterlegt. Für eine Prognose werden mindestens 24 Monatswerte benötigt.\n\n"
            f"So geht's: `data/heupreise_vorlage.csv` nach `{PRICE_FILE.name}` kopieren und mit Monatspreisen füllen "
            "(z. B. aus LfL-Marktberichten, Raufutter-Notierungen der Fachpresse, Notierungen der Landwirtschaftskammer "
            "Oberösterreich oder eigenen Abschlüssen). Bewusst keine erfundenen Beispielpreise."
        )
    else:
        until = pd.Timestamp(stichtag_date)
        pf = price_forecast(prices, 6, until=until)
        f6 = go.Figure()
        hist_p = prices[prices.index <= until]
        f6.add_scatter(x=hist_p.index, y=hist_p["preis"], name="Preis", line=dict(color="black"))
        if pf is not None:
            px = list(pf.index)
            f6.add_scatter(x=px + px[::-1], y=list(pf["q90"]) + list(pf["q10"])[::-1], fill="toself",
                           fillcolor="rgba(31,111,209,0.15)", line=dict(width=0), name="80-%-Bereich")
            f6.add_scatter(x=px, y=pf["q50"], line=dict(color=BLAU), name="Median")
        later = prices[prices.index > until]
        if zeige_ist and len(later):
            f6.add_scatter(x=later.index, y=later["preis"], name="Ist danach", line=dict(color=ROT, dash="dot"))
        f6.update_layout(height=380, yaxis_title="€/dt", margin=dict(t=30, b=10))
        st.plotly_chart(f6, width="stretch")
        if pf is not None:
            st.dataframe(pf.round(1), width="stretch")
        st.caption("Methode: empirische Quantile der historischen Preisänderung je Horizont (gleicher Kalendermonat), "
                   "angewandt auf den letzten Preis vor dem Stichtag. Orientierung, keine Punktprognose.")

# ---------------------------------------------------------------------------
# Tab 5: Methodik
# ---------------------------------------------------------------------------
with tab5:
    st.markdown(f"""
**Modell.** Graswachstumsmodell ModVege (Jouven et al. 2006): vier Biomasse-Kompartimente, Wachstum aus Strahlung,
Temperatur, Bodenwasser und Saisoneffekt, Alterung bestimmt die Verdaulichkeit (OMD → NEL).
Der Nährstoffindex wurde so kalibriert, dass der mittlere Brutto-Jahresertrag {A.trainings_start}–{A.trainings_ende}
**{M['achieved']:.1f} dt TM/ha** beträgt (Ziel {A.ziel_jahresertrag_dt_tm_ha:.0f}); NI = {p.NI:.3f}.

**Schnittregel.** 1. Schnitt ab Temperatursumme {A.tsumme_erster_schnitt:.0f} °Cd und ≥ {A.min_ertrag_kg_tm_ha / 100:.0f} dt TM/ha
erntbarer Masse; Folgeschnitte frühestens nach {A.min_tage_zwischen_schnitten} Tagen. Gemäht wird nur bei
{A.trockentage_noetig} Tagen < {A.regen_grenze_mm} mm Regen und ET0 ≥ {A.min_et0_mm} mm; nach {A.max_wartetage} Tagen Warten
wird mit Zusatzverlust gemäht. Feldverlust {A.feldverlust_basis * 100:.0f} % + {A.feldverlust_je_regentag * 100:.0f} % je Regentag.

**Prognose zum Stichtag.** Zustand am Stichtag aus dem Ist-Wetter {JAHR}; danach {len(clim)} Szenarien mit dem Wetter je eines
Jahres {A.trainings_start}–{A.trainings_ende}. Je Szenario {A.ziehungen_je_szenario} Ziehungen eines log-normalen Modellfehlers
(σ = {A.modellfehler_sigma:.2f}) je künftigem Schnitt. Bereits gemähte Schnitte gelten als bekannt. Daher wird die Verteilung zum Erntetermin hin schmaler.

**Historie.** Die graue Verteilung sind die mit demselben Modell simulierten Jahre {A.trainings_start}–{A.trainings_ende}.
„Ist {JAHR}“ ist die Simulation mit dem tatsächlichen Wetter, **keine Messung**.

**Grenzen dieses MVP.** Keine Satellitendaten (generisches Feld), keine echten Ertragsdaten zur Validierung,
Wetter aus ERA5-Reanalyse (~9–25 km) statt DWD-Raster, Heuwetter im Rückblick ohne archivierte Ensemble-Vorhersagen,
Preise nur aus selbst gepflegter CSV. Nächste Schritte: Flächengeometrien, Sentinel-2-Assimilation, DWD-Daten,
Betriebsdaten für Kalibrierung und Hindcast-Validierung.
""")
    st.subheader("Simulierte Historie")
    show = hist.copy()
    for i in range(1, 6):
        if f"tag{i}" in show:
            show[f"tag{i}"] = show[f"tag{i}"].apply(lambda t: "" if pd.isna(t) else f"{date_from_doy(2001, int(t)):%d.%m.}")
    st.dataframe(show.round(1), width="stretch")
