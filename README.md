# Heu-Prognose MVP (Kirchweidach)

Streamlit-App, die die Heusaison 2026 für ein generisches Feld (2 ha, Kirchweidach, Lkr. Altötting) **im Rückblick simuliert**:
Für jeden Stichtag zwischen 1. März und dem letzten Datentag rechnet sie, was eine Prognose an diesem Tag gewusst hätte.
Das Wetter bis zum Stichtag ist bekannt, danach kommen 35 Szenarien aus den Trainingsjahren 1991–2025.

## Start auf dem Mac

Voraussetzung ist Python 3.10 oder neuer (`python3 --version`; sonst `brew install python`).

```bash
cd heu-prognose-mvp
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Der Browser öffnet sich unter http://localhost:8501. Beim ersten Start lädt die App rund 35 Jahre Tageswetter von
Open-Meteo (ERA5-Reanalyse, kostenlos, ohne Schlüssel) und speichert sie in `data/wetter_cache.csv`. Danach kalibriert
sie das Modell, was etwa 10 bis 30 Sekunden dauert. Spätere Starts gehen schneller.

Offline-Test mit Kunstwetter: `HEU_SYNTHETIC=1 streamlit run app.py` (oder `python scripts/selftest.py`).

## Was die App zeigt

| Reiter | Inhalt |
| --- | --- |
| Ertragsprognose | Histogramm der Prognose gegen die simulierte Historie 1991–2025, Fächerdiagramm über alle Stichtage, Histogramme am 1. jedes Monats (die Verengung) |
| Schnittzeitpunkt | Bisherige Schnitte, Zeitraum des nächsten Schnitts, Bewertung der nächsten 14 Tage (Masse, NEL, Heuwetter-Chance, Schnittreife) |
| Satellit | Sentinel-2-NDVI/NDRE (und mit CDSE-Zugang Sentinel-1-VH) je Feld 2017–heute, automatisch erkannte Schnitte, Vergleich mit der Modell-Schnittregel |
| Wetter 2026 | Temperatursumme, Niederschlag und Bodenwasser gegen das Klimaband |
| Heupreise | Quantilprognose, sobald `data/heupreise.csv` gefüllt ist (Vorlage liegt bei) |
| Methodik & Annahmen | Modellbeschreibung, Kalibrierung, simulierte Historie |

In der Seitenleiste stellst du Stichtag, Zielgröße (Saison oder 1. bis 4. Schnitt), Einheit (dt TM/ha, % vom Mittel,
Ballen, t Heu) und die Annahmen ein. „Ist-Wert anzeigen“ blendet das simulierte Ergebnis 2026 als Auflösung ein.

## Neue Seiten (Module 1–4)

Links in der Seitenleiste gibt es jetzt vier Seiten. Die Annahmen (Ertragsniveau, Schnittregel, Trocknung) stellst du
auf der Hauptseite ein; die anderen Seiten übernehmen sie.

| Seite | Inhalt | Datenquellen |
| --- | --- | --- |
| **Futterqualität** | Energie (MJ NEL) und Rohprotein je möglichem Schnitttag mit 10–90-%-Band, Energieertrag (GJ NEL/ha) und dessen Maximum, Qualitätsziel, Vergleich je Schnitt mit 1991–2025, Korrektur über eigene Futteranalysen | ModVege, ERA5, `data/futteranalysen.csv` |
| **Tiergesundheit → Hitzestress** | THI-Warnung für 3 Tage, stündlicher THI-Fächer 15 Tage (Stall oder Weide), Wahrscheinlichkeit je Stufe, Weidestunden, Milchverlust der Herde, Hitzestunden der Saison gegen 1991–2025. „Rückblick-Demo“ zeigt einen vergangenen Sommertag, z. B. für das Gespräch im Winter | Open-Meteo Ensemble (ECMWF-ENS, ICON-EPS), ERA5 stündlich |
| **Tiergesundheit → Parasiten & Fliegen** | Leberegel (Ollerenshaw, Wb-GDD), Magen-Darm-Würmer je Weide (GLOWORM-FL vereinfacht), Hornfliege, Kopffliege als Perzentil gegen 1991–2025, Termine für Kotproben | ERA5 täglich, `data/weideplan.csv` |
| **Tiergesundheit → Feuchtstellen** | Topografischer Feuchteindex aus dem DGM1 mit Feldumrissen, Anteil feuchter Fläche je Wiese | DGM1-Kacheln in `data/dgm/` (manuell laden), `rasterio` |
| **Klimatrend** | Vegetationsbeginn, Vegetationsperiode, Hitze-/Sommertage, Tropennächte, Trockenphasen, Wasserbilanz, SPEI, Heuwetterfenster, mögliche Schnittzahl, Wasserstress; Trend je Jahrzehnt mit Signifikanz, Verteilung alt gegen neu, Trockenheits-Kalender | DWD-Station ab 1961 (oder ERA5), ModVege |

Beim ersten Öffnen lädt die App zusätzlich die **stündlichen** ERA5-Daten ab 1991 (in 5-Jahres-Blöcken, etwa 1–3 Minuten,
Cache `data/wetter_stunde_cache.csv.gz`) und die **DWD-Tageswerte** der nächsten Station (die Station wird aus der
DWD-Stationsliste nach Abstand gewählt; Cache `data/dwd_kl_<nr>.csv`, täglich aktualisiert).

Eigene Daten (optional, Vorlagen liegen in `data/`):
- `data/futteranalysen.csv` aus `futteranalysen_vorlage.csv`: Schnittdatum, NEL, XP, Rohfaser früherer Heuanalysen
- `data/weideplan.csv` aus `weideplan_vorlage.csv`: welche Gruppe wann auf welcher Weide war (die Vorlage ist ein Beispiel, keine Betriebsangabe)

## Satellitendaten (Schritt 1)

Die Felder stehen in `data/felder.geojson` (Wiese Ost 3,67 ha, Wiese West 3,21 ha). Weitere Felder kannst du als
Features mit `id`, `name` und `flaeche_ha` ergänzen. In der Seitenleiste wählst du das Feld, im Reiter „Satellit“ lädst du die Daten.

**Empfohlen: CDSE-Zugang.** Lege `.streamlit/secrets.toml` an:

```toml
[cdse]
client_id = "sh-…"
client_secret = "…"
```

Dann holt die App Sentinel-2 (NDVI, NDRE, SCL-Wolkenmaske, 10 m Innenpuffer) und Sentinel-1 (VV/VH, Gamma0
geländekorrigiert, aufsteigende Bahn) über die Sentinel Hub Statistical API, jahresweise ab 2017. Die Ergebnisse
landen in `data/sat_s2_<feld>.csv` bzw. `data/sat_s1_<feld>.csv`, spätere Abrufe ergänzen nur die neuen Tage.

**Ohne Konto:** `pip install pystac-client rasterio`, dann „earth-search“ wählen. Das holt nur Sentinel-2 und ist deutlich langsamer.

**Schnitterkennung:** NDVI-Abfall um mindestens 0,15 zwischen zwei klaren Aufnahmen (mindestens 80 % des Feldes wolkenfrei,
vorher NDVI ≥ 0,6, höchstens 25 Tage Abstand). Ein Rückgang der VH-Rückstreuung um mindestens 1 dB gilt als Bestätigung.
Bei langen Wolkenlücken werden Schnitte verpasst.

## Aufbau

```
app.py            Streamlit-Oberfläche
heu/config.py     Alle generischen Annahmen (Standort, Ertragsniveau, Schnittregel, Verluste)
heu/weather.py    Wetterabruf (Open-Meteo), Cache, 365-Tage-Jahre
heu/model.py      ModVege-Graswachstum, Bodenwasser, Schnittregel, Heuwetter, OMD→NEL
heu/forecast.py   Kalibrierung, Ist-Simulation 2026, Szenario-Prognose, Fächer, Schnittbewertung
heu/prices.py     Preisreihe aus CSV, empirische Quantilprognose
heu/satellite.py  Felder, CDSE- und Earth-Search-Abruf, Cache, Schnitterkennung
heu/qualitaet.py  Futterqualität: NEL aus OMD, Rohprotein über N-Verdünnung, Heu-Abschlag, Zielkonflikt-Kurve
heu/hitze.py      THI, Weide-THI, Tageskennzahlen, Warnstufen, Ensemble-Wahrscheinlichkeiten
heu/parasiten.py  Ollerenshaw, Wb-GDD, Wurm-Kohortenmodell, Fliegen-Phänologie, Perzentile
heu/gelaende.py   DGM1 lesen, Feuchteindex (TWI), Auswertung je Feld
heu/klima.py      Klimakennzahlen, SPEI, Heuwetterfenster, Trends (Mann-Kendall, Theil-Sen)
heu/dwd.py        DWD-Stationsdaten (CDC), ET0 nach FAO-56
heu/stunden.py    Stündliches ERA5, Ensemble-Vorhersage, Rückblick-Demo
heu/plots.py      Gemeinsame Diagramme (Band 1991–2025, Histogramm, Trend)
heu/ui.py         Gecachte Datenlader für alle Seiten
heu/ensemble.py   ModVege als Ensemble (NumPy, N Mitglieder mit eigenen Parametern/Wetter/Schnitttagen)
heu/zwilling.py   Zwillingsexperiment: synthetische Wahrheit, S2-LAI-Beobachtungen, Kennzahlen (CRPS)
pages/            Seiten Futterqualität, Tiergesundheit, Klimatrend
scripts/selftest.py  Funktionstest ohne Internet
scripts/benchmark_ensemble.py  Laufzeit Skalar- gegen Ensemble-Modell
scripts/twin_experiment.py     Zwillingsexperiment (derzeit Open-Loop-Messlatte ohne Assimilation)
tests/            Regressionstests (pip install -r requirements-dev.txt; python -m pytest -q tests/)
docs/PLAN_hybridmodell_assimilation.md  Plan Hybridmodell mit Satelliten-Datenassimilation
```

## Hybridmodell mit Satelliten-Assimilation (in Arbeit)

Dieses Repo ist die Weiterentwicklung der Demo-App `heu-prognose-mvp` um die Datenassimilation
(Plan: `docs/PLAN_hybridmodell_assimilation.md`).

- **Paket 1 (fertig):** `heu/ensemble.py` rechnet ModVege für viele Mitglieder gleichzeitig. Die Ergebnisse
  stimmen mit `heu/model.py` bis auf Rundungsfehler überein (`tests/test_ensemble.py`). 64 Mitglieder ×
  35 Wetterjahre laufen etwa 20-mal schneller als mit der Skalarversion. Das Zwillingsexperiment liefert die
  Open-Loop-Messlatte.
- Als Nächstes: LAI-Beobachtungen aus Sentinel-2 (Paket 2) und der EnKF (Paket 3).

## Wichtige Grenzen

- **„Training“ heißt hier:** Wetter 1991–2025 als Klimatologie und Szenarien, plus Kalibrierung des Ertragsniveaus auf
  einen angenommenen Mittelwert (75 dt TM/ha brutto, in der Seitenleiste änderbar). Echte Ertragsdaten gibt es noch nicht.
- Auch „Ist 2026“ ist eine Modellsimulation mit echtem Wetter und keine Messung.
- Die Heuwetter-Chance im Rückblick beruht auf Klimatologie oder optional auf einer „perfekten“ 7-Tage-Vorhersage.
  Archivierte Ensemble-Vorhersagen kommen später.
- Die Satellitendaten werden angezeigt und Schnitte erkannt, fließen aber noch nicht ins Modell ein. Kalibrierung und Assimilation kommen in Schritt 3.
- Noch kein DWD-Raster (ERA5 hat 9–25 km Auflösung); das kommt in Schritt 2.
- Es sind keine Heupreise hinterlegt. Bewusst gibt es keine erfundenen Beispielwerte.

### Grenzen der neuen Module

- **Parasiten:** Die Koeffizienten des Wurmmodells (Rose et al. 2015) und die Detailregeln des Ollerenshaw-Index
  (Deckelung, Halbierung Mai/Oktober, Regentag ab 0,25 mm) stammen aus Zusammenfassungen und sind noch mit den
  Originalarbeiten abzugleichen. Die Hornfliegen-Gradtage (Basis 12,5 °C, 150 Gradtage) sind aus Angaben zur
  Entwicklungsdauer abgeleitet, nicht aus einer Originaltabelle. Alle Parasitenwerte sind relative Wetterrisiken.
- **Futterqualität:** Kleeanteil, N-Gehalte von Klee und Totmasse und die Heu-Abschläge sind Annahmen, bis eigene
  Analysen vorliegen. Live-Werte der LfL-Reifeprüfung und Sentinel-2-NDRE fließen noch nicht ein.
- **Hitze:** Der Milchverlust ist eine Literatur-Faustzahl. Im Stall ist es oft wärmer als draußen; ein eigener Logger wäre besser.
- **Klima:** Noch kein HYRAS-Raster, keine DWD-Bodenfeuchte und Phänologie, keine Klimaprojektionen.
  Lücken der nächsten Station werden ohne Höhenkorrektur aus der zweitnächsten gefüllt.
- Alles wurde in der Cloud-Umgebung nur mit Kunstwetter getestet (Open-Meteo, DWD und CDSE sind dort gesperrt).
