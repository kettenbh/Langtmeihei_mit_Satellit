# Plan: Hybridmodell mit Satelliten-Datenassimilation

Stand: 13.11.2026 · Status: **Entwurf zur Freigabe, noch kein Code**

Ziel: Die Heu-Ertragsprognose soll nicht mehr nur aus dem Wettermodell (ModVege + Wasserbilanz) kommen.
Aktuelle Sentinel-Aufnahmen sollen den Modellzustand laufend korrigieren. Dadurch wird die Verteilung
der Prognose ehrlicher und enger, je näher der Schnitt rückt. Später soll derselbe Rahmen auch Weizen, Gerste
und Mais abdecken.

---

## 0. Kurzfassung

1. **Was es schon gibt:** Die Assimilation von Blattflächenindex (LAI) aus Sentinel-2 in Pflanzenwachstumsmodelle
   ist gut erforscht, vor allem für Ackerkulturen (WOFOST, SAFY) mit dem Ensemble-Kalman-Filter (EnKF). Für Grünland
   gibt es deutlich weniger. Am nächsten an unserer App sind
   - Zhang et al. 2022: **ModVege** + LAI-Assimilation (MODIS, 4DVar, Parameter-Update), Innere Mongolei,
   - Reinermann et al. 2025 (DLR): **Grünlandertrag in Süddeutschland** aus Sentinel-2 im Vergleich mit dem
     Prozessmodell LandscapeDNDC. Wichtigster Einflussfaktor war die Schnitthäufigkeit.
   - Jörges & Hank (LMU, EGU 2025/2026): Hybrid aus Prozessmodell (LPJmL) und Machine Learning plus
     Assimilation für Bayern. Bisher nur Konferenzbeiträge.
   Eine fertige Open-Source-Lösung „ModVege + Sentinel-2 + EnKF“ habe ich nicht gefunden. Die Bausteine
   (PCSE mit WOFOST und LINGRA, SNAP-LAI-Netz in Python, DA-Bibliotheken) gibt es aber.
2. **Empfehlung für die App:** ein dreistufiges Hybridmodell
   - **Stufe A, Kalibrierung (offline, je Feld):** Bayessche Schätzung weniger ModVege-Parameter (NI, RUEmax, WHC,
     ST1/ST2) aus der Sentinel-2-Historie 2017–2025, zusammen mit den erkannten Schnitten.
   - **Stufe B, sequentielle Assimilation (in der Saison):** Ensemble-Kalman-Filter mit erweitertem Zustand
     (Biomassen + 1–2 Parameter). Er wird bei jeder wolkenfreien Aufnahme aktualisiert. Beobachtet werden LAI aus
     Sentinel-2 (SNAP-Netz), optional NDVI/NDRE über einen Beobachtungsoperator, später Sentinel-1.
     Schnitte werden als **Ereignis** assimiliert (Datum aus Satellit oder Eingabe des Landwirts).
   - **Stufe C, Restfehler-Korrektur (später):** ein kleines statistisches Modell (Ridge oder LightGBM) auf den
     verbleibenden Fehlern, sobald echte Erträge vorliegen (Ballenzahlen, Rising-Plate-Messungen).
3. **Einbau in die App:** Die heutige Prognose besteht aus „1 Zustand am Stichtag × 35 Wetterjahre“. Daraus wird
   „N Analyse-Mitglieder am Stichtag × 35 Wetterjahre“. Die Darstellung als sich verengende Verteilung bleibt
   gleich. Sie bekommt einen Schalter „mit/ohne Satellitenkorrektur“. Neu sind die Module `heu/assimilation.py`
   und `heu/beobachtung.py`. `forecast.py` und `satellite.py` werden erweitert.
4. **Ackerkulturen:** Für Weizen, Gerste und Mais nicht selbst modellieren, sondern **WOFOST über PCSE**
   (EUPL-1.2, kommerziell nutzbar) hinter dieselbe Assimilations-Schnittstelle hängen. Die Phänologie wird mit
   DWD-Phänologiedaten kalibriert. LAI-Assimilation ist dort Standard.
5. **Aufwand:** etwa 15–22 Arbeitstage bis zum Grünland-Hybridmodell (Stufen A+B mit Validierung). Für Stufe C
   und Ackerkulturen kommen je 8–12 Tage dazu. Stufe C hängt von Ertragsdaten ab.

---

## 1. Ausgangslage in der App

| Baustein | Datei | Stand |
|---|---|---|
| Graswachstum | `heu/model.py` | ModVege, 4 Kompartimente (GV, GR, DV, DR) + Alter, Eimer-Wasserbilanz `WR`, Temperatursumme `ST`. Regelbasierte Schnittentscheidung. Modell-LAI = `SLA · pLAM · GV · 0,1` |
| Kalibrierung | `heu/forecast.py::calibrate_ni` | nur NI per Bisektion auf einen angenommenen Mittelertrag (75 dt TM/ha) |
| Prognose | `heu/forecast.py::forecast` | Zustand am Stichtag (deterministisch, echtes Wetter) × 35 Klimajahre ab Stichtag, dazu ein log-normaler „Strukturfehler“ je Schnitt |
| Satellit | `heu/satellite.py` | CDSE Statistical API: S2 NDVI/NDRE (SCL-Wolkenmaske, 10 m Innenpuffer), S1 VV/VH. NDVI-Abfall-Schnitterkennung, Earth-Search-Fallback. `synthetic_s2` erzeugt Testdaten aus Modell-LAI |
| Darstellung | `app.py`, `heu/plots.py` | Fächerdiagramm (Quantile über Stichtage), Histogramm, Band 1991–2025 |

**Die größte Schwäche heute:** Die Unsicherheit am Stichtag ist null. Alles, was bis zum Stichtag passiert ist,
kommt aus dem Modell, also falsche Schnitttermine, Trockenschäden, Düngung, Bestandeslücken oder ein falsches
NI. Der log-normale Zuschlag ist pauschal. Genau hier setzt die Assimilation an. Sie gibt dem Zustand am
Stichtag eine Verteilung und zieht sie mit jeder Aufnahme zu den Beobachtungen.

---

## 2. Literatur- und Software-Recherche

> Hinweis: Kennzahlen stammen aus Abstracts und Zusammenfassungen. Vor der Umsetzung sollten die Originale
> geprüft werden. Mit (E) markierte Angaben stammen aus meinem Vorwissen und sind nicht frisch nachgeprüft.

### 2.1 Grünland

| Arbeit | Modell / Daten | Methode | Ergebnis, Bedeutung für uns |
|---|---|---|---|
| Zhang, Huang et al. 2022, *Remote Sensing* 14:3194 — [Link](https://www.mdpi.com/2072-4292/14/13/3194) | **ModVege**, AgERA5-Wetter, MODIS-LAI (500 m) | Zuerst MCMC-Kalibrierung (DREAM über SPOTPY) von ST1, ST2, LLS, BMGV0, BMD0 gegen LAI, Biomasse und ST2-Termin. Danach **4DVar-Assimilation** von LAI mit Update von ST2 und Anfangsbiomasse | RMSE 242 → 214 kg/ha. **Bestätigt, dass ModVege mit LAI assimilierbar ist** und welche Parameter empfindlich sind. Die Weide ist extensiv und ungeschnitten, mit MODIS-Pixeln statt Feldern |
| Reinermann, Boos, Kaim … Kiese 2025, *Biogeosciences* 22:4969 — [Link](https://elib.dlr.de/216906/), dazu ESA LPS 2025 — [Link](https://elib.dlr.de/215349/) | Sentinel-2 (Bänder, EVI, Tasseled-Cap-Feuchte, Tage seit letztem Schnitt), LandscapeDNDC, LfL-Richtwerte; **Ammer-Einzugsgebiet, Oberbayern** | XGBoost auf Feldbiomasse 2019–2021, Vergleich mit Prozessmodell | Test-R² 0,68 (RMSE 0,19 t/ha je Aufwuchs). Jahreserträge 4–9 t/ha. **Schnitthäufigkeit ist der wichtigste Faktor (r = 0,81).** Das Prozessmodell überschätzt den 1. Schnitt. Sehr nah an unserer Region und Referenz für die Größenordnung |
| Reinermann et al. 2022, *Remote Sensing* 14:1647 — [Link](https://www2.mdpi.com/2072-4292/14/7/1647) | S2-EVI + S1 (Rückstreuung, Kohärenz, PolSAR) | regelbasierte Schnitterkennung, Deutschland | F1 0,65 nur mit S2. S1 brachte kaum Verbesserung. Unsere Schnitterkennung ist ähnlich aufgebaut |
| Lobert, Schwieder et al. 2022 (DGPF) — [Link](https://www.dgpf.de/src/tagung/jt2022/proceedings/proceedings/paper/11_DLT2022_Lobert_et_al.pdf) | S1-Kohärenz, S2/Landsat-NDVI | 1D-CNN | hohe Präzision, geringe Trefferquote. Thünen-Schnittkarten Deutschland 2022–2024 gibt es frei (OpenAgrar) und eignen sich zur Plausibilisierung |
| Montaldo, Gaspa, Corona 2023, *Int. J. Digital Earth* 16:3810 — [Link](https://doaj.org/article/e36973878bbe4d2599b9b2b992d92160) | Landoberflächen-Vegetationsmodell, NDVI aus S2 und Landsat 8, Gras-Baum-System | **EnKF mit dynamischer Parameterkalibrierung** (Erhaltungsatmung, getrennt für Gras und Baum) | beseitigt systematischen Biomassefehler bei falschem Parameter. **Belegt den Ansatz „EnKF mit erweitertem Zustand inklusive Parameter“ für Gras** |
| Marwaha 2021, Diss. UCC Cork — [Link](https://cora.ucc.ie/handle/10468/12405) | S2-Bänder, NDVI, NDRE, Regen; 179 irische Milchviehbetriebe | Random Forest/ANFIS gegen Brereton-Grasmodell | RF R² 0,55 (Zuwachs kg TM/ha/Tag). Das reine Prozessmodell schnitt sehr schlecht ab. **Argument für den Hybrid statt reines Modell** |
| Basu et al. 2024, *Front. Environ. Sci.* — [Link](https://www.frontiersin.org/journals/environmental-science/articles/10.3389/fenvs.2024.1331659/full) | S2-OPTRAM-Bodenfeuchte + Bodenwasserdefizit-Modell, irischer Milchviehbetrieb | Regression | Idee für später: die **Bodenfeuchte** aus S2 oder S1 als zweite Beobachtung für die Wasserbilanz `WR` |
| Ruelle et al. 2018, MoSt-GG (E) | Moorepark-St-Gilles-Grasmodell | — | Alternative Grasmodell-Struktur, für uns nicht nötig |

### 2.2 Ackerkulturen und DA-Methodik

| Arbeit | Modell / Daten | Methode | Ergebnis |
|---|---|---|---|
| Kopanja, Mimić … van Evert 2024, CEUR Vol-3930 — [Link](https://ceur-ws.org/Vol-3930/paper14.pdf) | **WOFOST 7.2 in PCSE 5.4**, S2-LAI (SNAP), Mais/Soja, Serbien | **EnKF, 50 Mitglieder**, gestört: TDWI, SPAN | Mais-RMSE 1493 → 633 kg/ha mit S2-LAI. Fast eine Vorlage für unseren Ackerkulturen-Pfad |
| Liu, Huang et al. 2023, *Front. Plant Sci.* 14:1201179 — [Link](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2023.1201179/full) | WOFOST über PCSE, S2-LAI aus Vegetationsindex-Regression, Mais | Sobol-Sensitivität → MCMC-Kalibrierung → EnKF | Ertrags-R² 0,56. Zeigt den Ablauf **Sensitivität → Kalibrierung → EnKF** |
| Tůma 2022, Diss. CZU Prag — [Link](https://ftz.czu.cz/dl/105850?lang=cs) | WOFOST, Winterweizen, S2-LAI und **S1-RVI → „Radar-LAI“ per Random Forest** | EnKF | RMSE 4,23 (offen) → 3,43 (S2) bzw. 3,66 (S1). **S1 hilft, aber weniger als S2.** S1 ist nützlich als Lückenfüller bei Wolken |
| Silvestro, Casa et al. 2021, *Remote Sensing* 13:2138 — [Link](https://www.mdpi.com/2072-4292/13/11/2138) | SAFY mit Wasserbilanz, S2-LAI (SNAP) | EnKF, ca. 200 Mitglieder, erst synthetisch (OSSE), dann real | Zeigt den Nutzen eines **Zwillingsexperiments vor echten Daten**. Das können wir in der Cloud ohne Netz machen |
| Ma et al. 2022, *Sci. Rep.* 12:5473 — [Link](https://pmc.ncbi.nlm.nih.gov/articles/PMC8971471) | SAFY, S2-LAI, Winterweizen | Parameteroptimierung (SP-UCI) | Ertrags-R² 0,49–0,61. SAFY ist eine schlanke Alternative zu WOFOST |
| Ling et al. 2019, *GMD* 12:3119 — [Link](https://gmd.copernicus.org/articles/12/3119/2019/) | CLM4CN + DART, LAI global | Vergleich KF, **EnKF, EAKF**, Partikelfilter | **EAKF und EnKF schlagen den Partikelfilter** bei gleicher Ensemblegröße. Begründung für EnKF als Standard |
| Arakida et al. 2017, *NPG* 24:553 — [Link](https://npg.copernicus.org/articles/24/553/2017/) | SEIB-DGVM, LAI | Partikelfilter, Parameter + Zustand | PF ist sinnvoll bei stark nichtlinearen, nicht-gaußschen Größen (z. B. Phänologie) |
| Jörges & Hank (LMU) EGU 2025 — [Link](https://meetingorganizer.copernicus.org/EGU25/EGU25-13318.html), EGU 2026 — [Folien](https://presentations.copernicus.org/EGU26/EGU26-4037_supplement.pdf) | LPJmL + RF/XGBoost/LSTM/ResNet, Sentinel-3/MODIS-FAPAR, Bayern/EU | Hybrid ML + Prozessmodell, EnKF als nächster Schritt | ML+LPJmL hat den niedrigsten Fehler. **Bestätigt den Hybrid-Ansatz für Bayern**, allerdings regional und nicht schlaggenau |
| Dhillon et al. 2025, *Front. Plant Sci.* 16:1630087 — [Link](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2025.1630087/full) | LUE-Modell + Random Forest, Weizen/Raps Bayern | Hybrid ohne DA | Landkreis-R² 0,87 für Weizen |
| de Wit & van Diepen 2007, *Agric. For. Meteorol.* 146:38 (E) | WOFOST, Bodenfeuchte | EnKF | Klassiker zum EnKF in WOFOST, Grundlage des PCSE-EnKF-Beispiels |
| Evensen 1994/2003; Burgers et al. 1998; Anderson 2001 (E) | — | EnKF, gestörte Beobachtungen, EAKF | methodische Grundlagen |

**Fazit Literatur:**
- LAI-Assimilation mit EnKF bringt in Ackerkulturen typischerweise 20–50 % weniger Ertragsfehler. Bei Grünland
  ist der Nutzen weniger gut belegt. Der Grund ist, dass Schnitte die Zeitreihe zerhacken und NDVI bei dichtem
  Bestand sättigt (ab LAI etwa 3).
- Im Grünland ist der **Schnitttermin der wichtigste Faktor** (Reinermann). Ihn richtig zu haben ist mindestens so wichtig
  wie die LAI-Korrektur. Unser Plan behandelt Schnitte deshalb als eigenes Assimilationsereignis.
- Für die Auflösung eines einzelnen Feldes eignen sich nur Sentinel-2 (10–20 m) und Sentinel-1 (10 m). MODIS und
  Sentinel-3 (300–500 m) sind für 3–4 ha zu grob.

### 2.3 Software und GitHub

| Projekt | Was | Lizenz | Nutzen für uns |
|---|---|---|---|
| **PCSE** (`ajwdewit/pcse`, PyPI `pcse` 6.0.13, Juni 2026) — [PyPI](https://pypi.org/project/pcse/), [Doku](https://pcse.readthedocs.io/en/stable/) | Python-Rahmen mit **WOFOST, LINTUL3, LINGRA** (Grünland), Agromanagement, Wetter-Reader | **EUPL-1.2** (schwaches Copyleft, kommerziell nutzbar. Änderungen *an PCSE selbst* müssen bei Weitergabe offengelegt werden, eigener App-Code nicht) | Ackerkulturen-Pfad; LINGRA als zweites Grünlandmodell zum Gegenprüfen |
| `ajwdewit/pcse_notebooks` (E) | Jupyter-Beispiele, darunter ein **EnKF mit WOFOST** | EUPL (prüfen) | Vorlage für Schnittstelle `get_variable`/`set_variable` |
| `ajwdewit/WOFOST_crop_parameters` (E) | Parametersätze für Weizen, Gerste, Mais usw. | EUPL (prüfen) | Startparameter Ackerkulturen |
| `jgomezdans/accra_wkshp` — [GitHub](https://github.com/jgomezdans/accra_wkshp) | NCEO-Workshop: S2/MODIS-LAI, WOFOST-Mais, **EnKF-Beispiel** | nicht angegeben | gut zum Lernen, nicht als Abhängigkeit |
| **satellitetools** (Nevalainen, FMI) — [Doku](https://satellitetools.readthedocs.io/en/latest/_modules/satellitetools/biophys/biophys.html) | Python-Port des **SNAP-S2-Biophysik-Netzes** (LAI, FAPAR, FCOVER aus B3–B12 + Winkeln) | nicht angegeben. Die SNAP-Gewichte stammen aus ESA s2tbx (GPL-3, prüfen) | Referenz. Wir rechnen das Netz besser direkt im Sentinel-Hub-Evalscript (s. u.) |
| Sentinel-Hub Custom Scripts „LAI“ (E) | Evalscript mit demselben SNAP-Netz | CC-BY-SA (prüfen) | **Direkter Weg zu LAI** über unsere bestehende Statistical-API-Abfrage |
| **DAPPER** (`nansencenter/DAPPER`) — [GitHub](https://github.com/nansencenter/DAPPER) | Forschungsbaukasten: EnKF-Varianten, LETKF, iEnKS, Partikelfilter, Twin-Experimente | Lizenzdatei vorhanden (prüfen), Python ≥ 3.12 | zum Methodenvergleich offline. Als App-Abhängigkeit zu schwer |
| LaVEnDAR (`ppharris/lavendar`) — [GitHub](https://github.com/ppharris/lavendar) | 4D-EnVar für JULES | nicht angegeben | nur als Ideenquelle |
| filterpy (E) | Kalman-, EnKF- und PF-Klassen | MIT | seit Jahren nicht mehr gepflegt. **Lieber eigenen EnKF schreiben** (etwa 60 Zeilen NumPy) |
| SPOTPY (E) | Kalibrierung: DREAM, SCE-UA, MCMC, LHS | MIT | Stufe A (Kalibrierung), wie bei Zhang et al. 2022 |
| TSMP-DA (`HPSCTerrSys/TSMP_DA`) — [GitHub](https://github.com/HPSCTerrSys/TSMP_DA) | PDAF-Kopplung für Großmodelle | — | zu groß, nicht relevant |

**Keine fertige Lösung zum Übernehmen.** Den EnKF schreiben wir selbst, weil ModVege sehr klein ist (8 Zustände,
schneller Tagesschritt). Externe Bibliotheken kommen nur für die Kalibrierung (SPOTPY) und für Ackerkulturen
(PCSE) dazu.

**Datenprodukt-Alternative:** Der Copernicus Land Monitoring Service bietet mit HR-VPP fertige 10-m-Reihen aus
Sentinel-2 (u. a. PPI, LAI, FAPAR, offene Lizenz, auch kommerziell) (E, Verfügbarkeit und Variablen prüfen).
Das wäre ein Plan B, falls das eigene LAI-Evalscript nicht überzeugt. Google Earth Engine bleibt wegen der
kommerziellen Nutzung außen vor.

---

## 3. Methodenwahl für Grünland

### 3.1 Welche Größe assimilieren?

| Kandidat | Vorteil | Nachteil | Entscheidung |
|---|---|---|---|
| **LAI aus SNAP-Netz (S2)** | physikalische Modellgröße, direkt vergleichbar mit `SLA·pLAM·GV` | Netz für Ackerkulturen trainiert, bei Grünland eher Unterschätzung bei hohem LAI. Fehler etwa ±0,7–1,0 m²/m² | **primär** |
| NDVI über Operator `NDVI = a − b·exp(−k·LAI)` | robust, schon vorhanden | sättigt ab LAI ≈ 3, gerade vor dem 1. Schnitt kaum informativ | Ersatz, und zum Kalibrieren der Operatoren |
| NDRE / CIred-edge | sättigt später | rauschiger | als zusätzliche Beobachtung testen |
| S1 VH, VH/VV, RVI | wolkenunabhängig, alle 6–12 Tage | schwacher, feuchteabhängiger Zusammenhang mit Biomasse (Tůma: r ≈ 0,5–0,7 zu LAI) | Phase 2, nur per gelerntem Operator („Radar-LAI“) und mit großem Beobachtungsfehler |
| **Schnittereignis** (S2-NDVI-Abfall, S1-Bestätigung, Eingabe des Landwirts) | betrifft die größte Fehlerquelle | diskret, passt nicht in einen Kalman-Ansatz | **eigene Ereignis-Behandlung** (3.4) |
| Bodenfeuchte (S1 oder OPTRAM) | korrigiert die Wasserbilanz `WR` | aufwendig, unsicher | später optional |

Die Wolkenlage im Alpenvorland ist ein echtes Problem. In April und Mai sind oft nur 2–5 brauchbare
S2-Aufnahmen pro Monat zu erwarten. Deshalb bleibt die Mindestschwelle `MIN_CLEAR = 0,8`, aber der
Beobachtungsfehler wird zusätzlich vom Klar-Anteil abhängig gemacht.

### 3.2 Filter: EnKF mit erweitertem Zustand (Standard), Partikelfilter als Gegenprobe

**Zustandsvektor je Ensemble-Mitglied** (transformiert, damit nichts negativ wird):

```
x = [ log GV, log GR, log DV, log DR, logit(WR/WHC),  log NI, log RUEmax ]
       └──────── Modellzustand ────────┘             └─ Parameter (persistent) ─┘
```

Die Alter (aGV …) und ST werden nicht aktualisiert, sondern nur mitgeführt. ST kommt aus dem gemessenen Wetter
und ist praktisch sicher.

**Vorgehen je Saison (Analysephase bis heute):**

1. **Start (1. Januar oder Vegetationsbeginn):** N = 64–128 Mitglieder. Die Parameter werden aus der Posterior von
   Stufe A gezogen, die Anfangsbiomassen gestört (±30 %, log-normal).
2. **Vorhersageschritt:** Jedes Mitglied läuft mit dem *echten* Wetter bis zur nächsten Aufnahme. Dazu kommen
   stochastische Störungen. Wetterfehler: Strahlung ±10 %, Niederschlag log-normal ±25 % (das ERA5- bzw. DWD-Pixel
   ist nicht das Feld). Kleiner Modellfehler auf den Wachstumsterm: Faktor log-normal σ ≈ 0,05 pro Tag, AR(1)-korreliert.
3. **Analyseschritt** bei einer klaren Aufnahme mit Beobachtung y und Fehler R:
   - Modellabbild `h(x) = SLA·pLAM·GV·0,1` (LAI), bzw. der NDVI-Operator.
   - **Stochastischer EnKF mit gestörten Beobachtungen** (Burgers 1998). Bei einer skalaren Beobachtung reduziert
     sich das auf eine Kovarianz-Regression. Das ist einfach und robust, und in einer Python-Datei gut testbar.
     Alternative: ETKF/EAKF ohne Störungen, deterministisch. Laut Ling et al. 2019 etwas besser, aber
     komplizierter. Optional später.
   - **Inflation** (multiplikativ 1,02–1,10 oder „relaxation to prior spread“), damit das Ensemble nicht kollabiert.
     Wichtig, weil die Parameter sonst nach wenigen Aufnahmen „einfrieren“.
   - **Plausibilitätsprüfung (Gating):** Innovation |y − h̄| > 3·√(HPHᵀ + R) → Aufnahme verwerfen und protokollieren
     (Wolkenrest, Schatten, frisch gemäht und nicht erkannt).
   - Parameter-Updates gedämpft (Faktor 0,3–0,5), damit eine einzelne Aufnahme NI nicht umwirft.
4. **Ergebnis am Stichtag:** N Analyse-Mitglieder (Zustand + Parameter). Diese ersetzen den heutigen
   deterministischen `s0` in `forecast()`.

**Beobachtungsfehler R:** Für LAI σ = max(0,5; 0,25·LAI) × (1 + 2·(1 − clear)), dazu die räumliche Streuung im
Feld (q25–q75 aus der Statistical API). Danach wird R aus den Innovationsstatistiken der Hindcast-Jahre
nachjustiert (Desroziers-Diagnose (E)).

**Warum nicht 4DVar wie bei Zhang?** 4DVar braucht ein adjungiertes oder gradientenfreies Optimierungsverfahren über das
ganze Zeitfenster und liefert keine Verteilung. Wir brauchen aber eine Verteilung für die Darstellung. Die
Fenster-Idee übernehmen wir in Stufe A (Kalibrierung über ganze Jahre).

**Partikelfilter (SIR) als Gegenprobe:** ModVege ist so billig, dass N = 500–1000 Partikel möglich sind. Die
PF-Ergebnisse dienen in der Validierung als nichtlineare Referenz. Weicht der EnKF stark ab, deutet das auf
Nichtgaußizität hin, z. B. rund um Schnitte. Für den Normalbetrieb ist ein PF bei so wenigen Beobachtungen
anfällig für Gewichtsentartung.

### 3.3 Stufe A: Kalibrierung je Feld (offline)

- **Daten:** S2-LAI- und NDVI-Reihen 2017–2025 je Feld (wolkenfrei), erkannte Schnitte (`detect_cuts`), wenn
  vorhanden die Schnittdaten des Landwirts, dazu DWD- bzw. ERA5-Wetter.
- **Parameter:** NI, RUEmax, WHC, ST1, ST2, LLS, optional SLA (Sensitivität vorab mit Morris oder Sobol prüfen,
  höchstens 5 freie Parameter).
- **Likelihood:** LAI-Residuen (mit R aus 3.2) + Strafterm für Schnitttermin-Abweichung + *schwacher* Prior auf den
  Jahresertrag (LfL-Richtwert bzw. Annahme 75 dt, σ = 20 %), solange es keine echten Erträge gibt.
- **Verfahren:** DREAM(ZS) aus SPOTPY oder einfacher ein **Ensemble Smoother mit Mehrfachdatenassimilation
  (ES-MDA)** mit demselben Code wie der EnKF. ES-MDA ist für uns der elegantere Weg: eine Methode, ein Code
  (E: Emerick & Reynolds 2013).
- **Ergebnis:** eine Posterior-Stichprobe der Parameter je Feld, gespeichert als `data/kalibrierung_<feld>.csv`.
  Sie dient als Prior für Stufe B im neuen Jahr.
- **Wichtig:** In der Kalibrierung laufen die Schnitte *erzwungen* an den beobachteten Tagen (Modus
  `manage=False` + Schnittliste), nicht nach der Schnittregel. Sonst „lernt“ das Modell den Fehler der Regel.

### 3.4 Schnitte als Ereignisse

Heute entscheidet `run()` per Regel, wann geschnitten wird. Im Analysezeitraum (Vergangenheit bis heute) ist das
falsch, weil der echte Schnitt beobachtbar ist.

- **Quelle 1 (beste):** Der Landwirt trägt Schnittdaten ein (Eingabefeld oder `data/schnitte.csv`).
- **Quelle 2:** `detect_cuts` (S2-NDVI-Abfall, S1-Bestätigung). Liefert ein Zeitfenster `[frueh, spaet]`.
- **Behandlung im Ensemble:** Jedes Mitglied zieht sein Schnittdatum gleichverteilt aus `[frueh, spaet]`. Das ist
  eine einfache Form der Ereignis-Unsicherheit. Die nächste klare Aufnahme nach dem Schnitt (niedriger LAI) zieht
  die Mitglieder dann zusammen.
- Liegt ein vom Modell geplanter Schnitt in der Vergangenheit, aber der Satellit sieht keinen (NDVI bleibt hoch),
  wird der Schnitt in der Analyse *nicht* ausgeführt. Der Bestand wächst weiter. Das ist ein häufiger
  Fall bei Regenwetter.
- Die Schnittregel gilt nur noch in der **Zukunft** (Prognose ab Stichtag).

### 3.5 Prognose ab Stichtag (Kombination mit den Wetterszenarien)

Heute: 1 Zustand × 35 Jahre × k Ziehungen (log-normaler Strukturfehler).
Neu: **N Analyse-Mitglieder × 35 Jahre**. Bei N = 64 sind das 2240 Läufe pro Stichtag. Mit dem reinen
Python-Tagesschritt dauert das geschätzt einige Sekunden. Für das Fächerdiagramm über viele Stichtage
(`fan()`) bieten sich zwei Lösungen an:
- (a) Ziehung: Pro Wetterjahr nur m = 8 zufällige Mitglieder verwenden (280 Läufe).
- (b) **Vektorisierung** von `grow_one_day` über das Ensemble mit NumPy-Arrays statt Dataclass. Das bringt den größten
  Gewinn (Faktor 20–50) und lohnt sich auch für die Kalibrierung. Empfehlung: Vor Stufe B einmal umbauen, mit
  Regressionstest gegen die heutige Skalarversion.

Der log-normale Strukturfehler (`modellfehler_sigma`) bleibt, wird aber **kleiner** und aus den Hindcasts
geschätzt. Ein Teil der Unsicherheit steckt jetzt explizit in der Ensemble-Streuung der Parameter und Zustände.

### 3.6 Stufe C: statistische Restfehler-Korrektur (später)

Erst sinnvoll mit ≥ 20–30 echten Ertragswerten (Schnitt × Feld × Jahr). Quellen: Ballenzahl × mittleres
Ballengewicht × TM-Gehalt, Rising-Plate-Messungen vor dem Schnitt, Wiegungen.

- Ziel: Residuum `log(Ertrag_beobachtet / Ertrag_Analyse)`.
- Merkmale: Aufwuchsnummer, Tage seit letztem Schnitt, Analyse-LAI, letzter NDRE, Innovationsmittel der Saison,
  Wasserstress-Index, Temperatursumme, Gülle-Termine.
- Modell: zuerst Ridge-Regression (wenig Daten), später LightGBM. Ausgabe als Korrekturfaktor mit
  Unsicherheit (Quantilregression oder konforme Intervalle).
- So bleibt das Prozessmodell der Kern („physics-first“), und ML korrigiert nur den systematischen Rest. Das
  entspricht dem Hybrid-Konzept von Jörges & Hank und Dhillon et al.

---

## 4. Validierung (bevor irgendetwas in der App „scharf“ geschaltet wird)

1. **Zwillingsexperiment (OSSE), ohne Internet, in der Cloud machbar:** Ein „wahres“ Jahr mit bekannten
   Parametern (z. B. NI = 0,55 statt 0,7, ein Schnitt 9 Tage später als die Regel) wird simuliert. Daraus
   entstehen synthetische S2-Beobachtungen per `synthetic_s2` (Wolkenmuster, Rauschen). Dann wird geprüft:
   - ob der EnKF NI zurückgewinnt,
   - ob die Ertragsprognose am Stichtag 15.5. näher an der Wahrheit liegt als ohne DA,
   - ob die Streuung zur tatsächlichen Abweichung passt.
   Teil von `scripts/selftest.py` → `scripts/twin_experiment.py`.
2. **Hindcast 2018–2025 mit echten S2-Daten** (auf Huberts Mac): für jedes Jahr Prognosen zu festen Stichtagen
   (1.4., 15.4., 1.5., 15.5., 1.6. …) mit und ohne DA. Mangels echter Erträge vorerst nur indirekt bewertbar:
   - Vorhersage der *nächsten* LAI-Aufnahme (Innovation, Ein-Schritt-Prognose),
   - Schnitttermin-Prognose gegen erkannte Schnitte,
   - Abgleich mit LfL-Richtwerten und Reinermann-Größenordnungen (4–9 t TM/ha/Jahr).
3. **Kennzahlen:** RMSE und Bias der Innovation, **CRPS** der Ertragsverteilung, **PIT-Histogramm** bzw.
   Rangdiagramm (Kalibrierung der Verteilung), Verhältnis Spread zu Skill (sollte ≈ 1 sein).
4. **Abbruchkriterium:** Verbessert die DA im Hindcast die Ein-Schritt-LAI-Prognose nicht um mindestens 15 % oder
   wird die Verteilung schlechter kalibriert, wird nur Stufe A (Kalibrierung) eingebaut und B zurückgestellt.

---

## 5. Einbau in die App

### 5.1 Neue und geänderte Module

| Datei | Änderung |
|---|---|
| `heu/model.py` | (1) **`EnsembleState`**: dieselben Größen wie `State`, aber als NumPy-Arrays der Länge N. (2) `grow_ensemble_day(...)` vektorisiert, mathematisch identisch zu `grow_one_day`. (3) `run()` bekommt optional `forced_cuts: list[int]` (Analysephase) und `params` als Array je Mitglied. Die Skalarversion bleibt für Tests und die Qualitätsseite erhalten |
| **`heu/beobachtung.py`** (neu) | Einheitliches Beobachtungsformat: `datum, groesse (lai/ndvi/ndre/vh), wert, sigma, clear, quelle`. Qualitätskontrolle (Klar-Anteil, Ausreißer gegenüber gleitendem Median, Unterdrückung von Aufnahmen 0–3 Tage nach erkannten Schnitten). Beobachtungsoperatoren `h_lai`, `h_ndvi(a, b, k)` und ihre Kalibrierung |
| **`heu/assimilation.py`** (neu) | `transform/untransform` (log/logit), `enkf_update(X, y, R, h, inflation, gating)`, optional `etkf_update`, `sir_update` (Partikelfilter), `esmda(...)` für Stufe A, `analyse_saison(wetter, beobachtungen, schnitte, prior) -> Analyse` mit Ensemble-Schnappschüssen je Tag, Innovationsprotokoll und Parameter-Trajektorien |
| `heu/satellite.py` | Neues Evalscript **`EVAL_S2_LAI`** (SNAP-Netz, braucht B03–B12 sowie `sunZenithAngles`, `viewZenithMean`, `sunAzimuthAngles`, `viewAzimuthMean`). Ausgaben `lai`, `fapar`, `fcover` (mean, q25, q75, stDev) über dieselbe Statistical-API-Abfrage, als zusätzliche Spalten im Cache. Earth-Search-Fallback: Netz in NumPy (Port nach satellitetools bzw. SNAP-Gewichten, Lizenz prüfen) |
| `heu/forecast.py` | `CurrentYear` bekommt optional `analyse: Analyse`. `forecast()` nimmt die Mitglieder am Stichtag statt `s0` (mit Ziehung aus 3.5). `calibrate_ni` bleibt als Fallback ohne Satellitendaten. `Prognose` bekommt das Feld `quelle = "modell" | "assimiliert"` |
| `heu/config.py` | `Annahmen`: `da_aktiv`, `da_mitglieder` (64), `da_inflation` (1,05), `da_beobachtung` (`"lai"`/`"ndvi"`), `da_param_daempfung` (0,4), `da_gating_sigma` (3,0), `da_wetterfehler` |
| `heu/kalibrierung.py` (neu, Stufe A) | ES-MDA über die Jahre 2017–2025, Ausgabe `data/kalibrierung_<feld>.csv`. Läuft nur bei Knopfdruck und dauert Minuten, das Ergebnis wird gecacht |
| `heu/ui.py` | gecachte Lader für Analyse und Kalibrierung (`st.cache_data` mit Hash über Wetter-, Satelliten- und Parameterstand) |
| `app.py`, Reiter „Satellit“ | Siehe 5.2 |
| `scripts/selftest.py`, `scripts/twin_experiment.py` | OSSE aus Abschnitt 4. Läuft ohne Netz |
| `data/schnitte_vorlage.csv` (neu) | `feld, datum, schnitt_nr, ballen, ballengewicht_kg, bemerkung`: Eingabe des Landwirts, zugleich Grundlage für Stufe C |

### 5.2 Darstellung

- **Prognose-Reiter:** Schalter „Satellitenkorrektur“ (an/aus). Das Fächerdiagramm zeigt beide Varianten
  übereinander (assimiliert gefüllt, reines Modell als gestrichelte Quantile). So sieht der Landwirt, wie
  sehr der Satellit die Prognose verschiebt und verengt.
- **Satellit-Reiter:** Zeitreihe Modell-LAI (Ensemble-Band 10–90 %) gegen S2-LAI-Punkte (mit Fehlerbalken). Verworfene
  Aufnahmen werden grau dargestellt, erkannte und eingegebene Schnitte als senkrechte Linien.
- **Neues Diagramm „Was der Satellit gelernt hat“:** Verlauf von NI und RUEmax über die Saison (Ensemble-Band),
  mit verständlicher Beschriftung, z. B. „Nährstoffversorgung: eher unter Annahme“.
- **Hinweis-Texte:** „Letzte klare Aufnahme vor X Tagen“. Bei mehr als 20 Tagen ohne klare Aufnahme wird gewarnt,
  dass die Korrektur veraltet ist.
- Die Histogramm- bzw. Verteilungsdarstellung relativ zu den historischen Jahren bleibt unverändert. Die
  Verengung zum Schnitt hin wird jetzt *aus Daten* begründet und nicht nur durch weniger verbleibende Wettertage.

### 5.3 Datenfluss (täglicher Lauf)

```
Wetter (DWD/ERA5) ─┐
S2-LAI/NDVI (CDSE) ┼─► beobachtung.py (QC, σ) ─► assimilation.analyse_saison ─► Ensemble am Stichtag
Schnitte (S2/S1/LW)┘                                    ▲                              │
Kalibrierung (Stufe A, Prior) ──────────────────────────┘                              ▼
                                       forecast(): N Mitglieder × 35 Wetterjahre ─► Verteilung, Fächer, Schnittzeitpunkt
                                                                                    (Stufe C: × Korrekturfaktor)
```

Rechenzeit-Ziel: Analyse einer Saison < 2 s, Prognose für einen Stichtag < 3 s, Fächerdiagramm < 30 s auf dem Mac
(nach der Vektorisierung).

---

## 6. Ausweitung auf Weizen, Gerste, Mais

### 6.1 Architektur: Modell-Adapter

Die Assimilation soll nichts von ModVege wissen. Jedes Modell stellt eine kleine Schnittstelle bereit:

```
class Kulturmodell(Protocol):
    def init_ensemble(self, n, prior) -> None
    def step(self, tag, wetter) -> None            # einen Tag für alle Mitglieder
    def get_state(self) -> np.ndarray             # transformierter Zustandsvektor (N × d)
    def set_state(self, X: np.ndarray) -> None
    def beobachte(self, groesse: str) -> np.ndarray   # h(x) je Mitglied
    def ereignis(self, art: str, tag: int) -> None    # Schnitt, Aussaat, Ernte, Düngung
    def ertrag(self) -> np.ndarray
```

- **Grünland:** ModVege-Adapter (eigener Code), optional LINGRA aus PCSE zum Gegenprüfen.
- **Weizen, Gerste, Mais:** **WOFOST (PCSE)**. Je Mitglied gibt es eine PCSE-Engine. PCSE bietet
  `get_variable`/`set_variable` (z. B. LAI, TWLV, TWST, TWSO, SM), womit der EnKF wie im PCSE-Beispiel funktioniert.
  N = 50 ist bei WOFOST unkritisch (Sekunden pro Saison).
- Leichte Alternative: **SAFY** (wenige Parameter, LUE-Modell, mit S2-LAI gut kalibrierbar), falls WOFOST für
  Bio-Bedingungen zu schwer zu kalibrieren ist.

### 6.2 Kulturspezifische Punkte

| Thema | Winterweizen / Wintergerste | Silo- oder Körnermais |
|---|---|---|
| Kalibrierung Phänologie | TSUM1/TSUM2 aus **DWD-Phänologie-Sofortmelder bzw. Jahresmelder** (Ährenschieben, Gelbreife), Open Data | Aussaat- und Aufgangsdatum vom Landwirt, TSUM aus DWD-Phänologie (Fahnenschieben) |
| Assimilierte Größen | S2-LAI bis etwa Ährenschieben (danach fällt das Grün-LAI durch Seneszenz, Operator anpassen), optional S1-RVI im Winter | S2-LAI ab 6–8-Blatt-Stadium, S1-VH während der Bodenbedeckung sehr aussagekräftig |
| Parameter im erweiterten Zustand | SPAN, TDWI, ein Ertragsniveau-Faktor (Bio, N-Limitierung) | SPAN, TDWI, Ertragsniveau |
| Bio-Besonderheit | WOFOST rechnet potenziell oder wasserlimitiert, aber nicht N-limitiert (WOFOST 8 kennt N, Kalibrierung aufwendig). Ein **Ertragslücken-Faktor** gegen Landkreiserträge und Bio-Referenzen ist zunächst einfacher | dito; zusätzlich Unkrautdruck → Bestandeslücken, die S2 sieht und die DA direkt korrigiert |
| Referenzerträge | Bayerisches Landesamt für Statistik (Landkreis Altötting), LfL-Deckungsbeiträge, Bio-Abschläge | dito; bei Silomais TM-Gehalt zur Ernte als Zusatzprognose |
| Darstellung | wie Heu: Ertragsverteilung relativ zu historischen Jahren, die sich zur Ernte verengt; zusätzlich Erntetermin-Prognose | dito, für Silomais zusätzlich ein Fenster für den optimalen TM-Gehalt (30–35 %) |
| Fruchtfolge | Feldgeometrien je Jahr mit Kultur aus iBALIS bzw. Mehrfachantrag (vom Landwirt) | dito |

### 6.3 Reihenfolge

1. Grünland fertig validieren (Abschnitte 3–5).
2. Schnittstelle `Kulturmodell` herausziehen, ModVege-Adapter darauf umstellen.
3. Ein Ackerschlag mit Winterweizen und WOFOST über PCSE (Parameter aus `WOFOST_crop_parameters`), Zwillingsexperiment,
   Hindcast 2018–2025 gegen Landkreiserträge.
4. Gerste (geringer Mehraufwand gegenüber Weizen), dann Mais.
5. Eigene App-Seite „Ackerkulturen“ mit gemeinsamer Darstellung.

---

## 7. Arbeitspakete und Aufwand

| # | Paket | Tage | abhängig von |
|---|---|---|---|
| 1 | Vektorisierung `EnsembleState`/`grow_ensemble_day` + Regressionstest gegen Skalarversion | 2–3 | — |
| 2 | `beobachtung.py` + LAI-Evalscript in `satellite.py` + Cache-Erweiterung | 2–3 | CDSE-Zugang (vorhanden) |
| 3 | `assimilation.py`: EnKF, Inflation, Gating, Schnittereignisse, `analyse_saison` | 3–4 | 1, 2 |
| 4 | Zwillingsexperiment (OSSE) + Selftest | 1–2 | 3 |
| 5 | Stufe A: ES-MDA-Kalibrierung je Feld, Sensitivitätsanalyse | 2–3 | 3 |
| 6 | `forecast.py`-Umbau (Mitglieder × Wetterjahre), Strukturfehler neu schätzen | 1–2 | 3 |
| 7 | UI: Schalter, Vergleichsfächer, LAI-Reiter, Parameterverlauf, Warnhinweise | 2 | 6 |
| 8 | Hindcast 2018–2025 auf dem Mac, Kennzahlen, Bericht, Go/No-Go | 2–3 | 5, 6 |
| | **Summe Grünland-Hybrid (A+B)** | **15–22** | |
| 9 | Stufe C: Restfehler-Modell | 3–5 | Ertragsdaten |
| 10 | Ackerkulturen: Adapter + WOFOST/PCSE + Weizen-Hindcast | 6–9 | 1–8, Felddaten |
| 11 | Gerste + Mais | 3–5 | 10 |

Realistisches Ziel: Pakete 1–8 bis Ende Februar 2027. So kann der Hybrid in der Saison 2027 vom ersten Aufwuchs an
parallel zum reinen Modell laufen. Auf die Prognose umgeschaltet wird erst nach dem Hindcast-Go.

---

## 8. Risiken

| Risiko | Wirkung | Gegenmaßnahme |
|---|---|---|
| Wenige klare S2-Aufnahmen im Frühjahr | DA greift genau vor dem 1. Schnitt kaum | S1 als Lückenfüller (Phase 2), Warnhinweis, Inflation |
| SNAP-LAI im dichten Grünland verzerrt | systematische Fehlkorrektur | Operator `LAI_s2 → LAI_modell` linear kalibrieren (a, b in Stufe A mitschätzen), NDRE prüfen, Rising-Plate-Messungen als Referenz |
| Feld-Mischpixel (Ränder, Hecken, Wald im Osten) | verzerrte Werte | bestehender 10-m-Innenpuffer, bei Bedarf 20 m |
| Ensemble-Kollaps, eingefrorene Parameter | Scheinsicherheit | Inflation, Dämpfung, Spread-Skill-Kontrolle im Hindcast |
| Keine echten Erträge | Stufe C unmöglich, Absolutniveau unsicher | Schnittprotokoll mit Ballenzahlen ab 2027, Rising-Plate-Messgerät (ca. 300–600 €) als günstige Biomassemessung (E) |
| Lizenz (SNAP-Gewichte GPL, Custom-Script CC-BY-SA, PCSE EUPL) | Weitergabe-Pflichten | Vor Paket 2 und 10 klären. Im Evalscript liegt nur die Gewichtsmatrix, App-Code bleibt eigen |
| Cloud blockiert CDSE/DWD | Entwicklung nur synthetisch | Pakete 1, 3, 4, 6, 7 laufen offline. 2, 5, 8 laufen auf dem Mac |

---

## 9. Offene Fragen an Hubert und den Landwirt

1. **Schnittdaten 2017–2025** (Datum je Wiese), auch ungefähr. Das sind die wertvollsten Daten für die Kalibrierung.
2. **Ballenzahlen bzw. Gewichte** je Schnitt und Wiese, soweit vorhanden (für Stufe C und das Absolutniveau).
3. Wäre ein **Rising-Plate-Meter** (Weidezollstock) denkbar, gemessen etwa 1× pro Woche von April bis Juni auf beiden Wiesen?
   Das wäre die beste Referenz, um das Satelliten-LAI zu prüfen.
4. Güllegaben (Termine, Mengen). Sie erklären Sprünge in NI.
5. Welche Ackerschläge (Kultur, Jahr, Geometrie) kommen für die Ausweitung in Frage? Wird Mais als Silo- oder
   Körnermais angebaut?
6. Freigabe der Reihenfolge: zuerst Grünland A+B mit Hindcast, Ackerkulturen erst danach?

---

## 10. Quellen

Grünland und Hybrid:
- Zhang Y., Huang J. et al. (2022): Grassland Aboveground Biomass Estimation through Assimilating Remote Sensing Data into a Grass Simulation Model. *Remote Sensing* 14(13):3194. https://www.mdpi.com/2072-4292/14/13/3194
- Reinermann S., Boos C., Kaim A. et al. (2025): Grassland yield estimations – potentials and limitations of remote sensing in comparison to process-based modeling and field measurements. *Biogeosciences* 22:4969–4992. https://elib.dlr.de/216906/
- Reinermann S. et al. (2025): Grassland yield estimation based on Sentinel-2 time series … southern Germany. ESA Living Planet Symposium. https://elib.dlr.de/215349/
- Reinermann S. et al. (2022): Detection of Grassland Mowing Events for Germany by Combining Sentinel-1 and Sentinel-2 Time Series. *Remote Sensing* 14(7):1647. https://www2.mdpi.com/2072-4292/14/7/1647
- Lobert F., Röder N., Gocht A., Schwieder M., Erasmi S. (2022): Mowing Detection from Combined Sentinel-1, Sentinel-2, and Landsat 8 Time Series … DGPF. https://www.dgpf.de/src/tagung/jt2022/proceedings/proceedings/paper/11_DLT2022_Lobert_et_al.pdf
- Thünen: Grassland mowing events across Germany 2022–2024 (Datensätze). https://www.openagrar.de/receive/openagrar_mods_00094089
- Montaldo N., Gaspa A., Corona R. (2023): Assimilation of NDVI data in a land surface – vegetation model … tree-grass ecosystem. *Int. J. Digital Earth* 16(1):3810–3837. https://doaj.org/article/e36973878bbe4d2599b9b2b992d92160
- Marwaha R. (2021): National farm scale estimates of grass yield from satellite remote sensing. Diss., UCC. https://cora.ucc.ie/handle/10468/12405
- Basu R. et al. (2024): Identifying favourable conditions for farm scale trafficability and grass growth … *Front. Environ. Sci.* 12:1331659. https://www.frontiersin.org/journals/environmental-science/articles/10.3389/fenvs.2024.1331659/full
- Jouven M., Carrère P., Baumont R. (2006): Model predicting dynamics of biomass, structure and digestibility of herbage in managed permanent pastures. *Grass Forage Sci.* 61:112–124.

Ackerkulturen und Methodik:
- Kopanja M. et al. (2024): Data assimilation of LAI improved crop growth modeling … CEUR Vol-3930. https://ceur-ws.org/Vol-3930/paper14.pdf
- Liu J., Huang J. et al. (2023): A method for estimating yield of maize inbred lines by assimilating WOFOST model with Sentinel-2 satellite data. *Front. Plant Sci.* 14:1201179. https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2023.1201179/full
- Tůma L. (2022): The Assimilation of Synthetic Aperture Radar Satellite Data into the Simulation of Crop Model. Diss., CZU Prag. https://ftz.czu.cz/dl/105850?lang=cs
- Silvestro P. C., Casa R. et al. (2021): Synergistic Use of Multispectral Data and Crop Growth Modelling … *Remote Sensing* 13(11):2138. https://www.mdpi.com/2072-4292/13/11/2138
- Ma C. et al. (2022): Wheat growth monitoring and yield estimation based on remote sensing data assimilation into the SAFY crop growth model. *Sci. Rep.* 12:5473. https://pmc.ncbi.nlm.nih.gov/articles/PMC8971471
- Ling X.-L. et al. (2019): Comparison of different sequential assimilation algorithms for satellite-derived leaf area index using DART. *GMD* 12:3119–3133. https://gmd.copernicus.org/articles/12/3119/2019/
- Arakida H. et al. (2017): Non-Gaussian data assimilation of satellite-based LAI observations with an individual-based DGVM. *NPG* 24:553–567. https://npg.copernicus.org/articles/24/553/2017/
- Jörges C., Hank T., Fader M. (2025): Chances and Challenges of Data Assimilation for Seasonal Yield Predictions Using Sentinel-3 … EGU25-13318. https://meetingorganizer.copernicus.org/EGU25/EGU25-13318.html
- Jörges C., Hank T. (2026): Regional Seasonal Crop Yield Forecasts Through Hybrid Modeling and Remote Sensing Data Assimilation. EGU26-4037. https://presentations.copernicus.org/EGU26/EGU26-4037_supplement.pdf
- Dhillon M. S. et al. (2025): Landscape structure, climate variability, and soil quality shape crop biomass patterns … Bavaria. *Front. Plant Sci.* 16:1630087. https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2025.1630087/full
- (E) de Wit A. J. W., van Diepen C. A. (2007): Crop model data assimilation with the Ensemble Kalman filter for improving regional crop yield forecasts. *Agric. For. Meteorol.* 146:38–56.
- (E) Evensen G. (2003): The Ensemble Kalman Filter: theoretical formulation and practical implementation. *Ocean Dynamics* 53:343–367.
- (E) Burgers G., van Leeuwen P. J., Evensen G. (1998): Analysis scheme in the ensemble Kalman filter. *Mon. Wea. Rev.* 126:1719–1724.
- (E) Emerick A. A., Reynolds A. C. (2013): Ensemble smoother with multiple data assimilation. *Computers & Geosciences* 55:3–15.

Software:
- PCSE: https://pypi.org/project/pcse/ · https://pcse.readthedocs.io/en/stable/ · (E) https://github.com/ajwdewit/pcse_notebooks · (E) https://github.com/ajwdewit/WOFOST_crop_parameters
- satellitetools (SNAP-Biophysik in Python): https://satellitetools.readthedocs.io/en/latest/_modules/satellitetools/biophys/biophys.html
- DAPPER: https://github.com/nansencenter/DAPPER
- NCEO-Workshop (WOFOST + EnKF): https://github.com/jgomezdans/accra_wkshp
- LaVEnDAR: https://github.com/ppharris/lavendar
- CDSE Evalscript-Doku: https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Evalscript.html
