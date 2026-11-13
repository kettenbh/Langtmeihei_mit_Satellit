# Umsetzungsplan: Seite „Schädlinge“ (Borkenkäfer · Buchsbaumzünsler · Trockenstress Wald · Sentinel-2 Waldstück)

Stand: 06.11.2026 · Zielgruppe: Claude Code in VS Code mit Zugriff auf `heu-prognose-mvp/`
Dieser Plan ist als Arbeitsanweisung geschrieben. Jeder Schritt in Abschnitt 10 enthält einen Prompt, den du Claude Code
direkt geben kannst. Alle Parameter, die mit **[VERIFIZIEREN]** markiert sind, stammen nicht aus einer geprüften
Originalquelle und müssen vor produktiver Nutzung abgeglichen werden.

---

## 0. Ziel in einem Satz

Eine neue Streamlit-Seite `pages/4_Schädlinge.py` mit zwei Tabs (Borkenkäfer, Buchsbaumzünsler), die für das laufende
Jahr einen **Jahreskalender mit farbigen Aktionstagen** zeigt, darunter **KDE-Flächendiagramme** der Ereigniszeitpunkte
(historische Jahre 1991–2025 gegen das aktuelle Jahr), im Borkenkäfer-Tab zusätzlich eine **Trockenstressprognose für
den Fichtenbestand** und eine **Sentinel-2-Auswertung eines Waldstücks** aus einer eigenen GeoJSON.

## 1. Was der Nutzer sieht (Zielbild)

### Tab „Borkenkäfer (Buchdrucker)“
1. Kopfzeile: Stichtag (Standard heute bzw. letzter Datentag), Exposition (besonnt/Bestandesrand vs. beschattet),
   Kennzahlen-Karten: aktuelle Generation, Entwicklungsstand der Brut in %, „nächstes Ereignis“ mit Datum (Median, 10–90 %).
2. **Jahreskalender** (12 Zeilen × 31 Spalten oder Wochenraster), jeder Tag eingefärbt nach der wichtigsten Aktion:
   Kontrollsaison beginnt · Bohrmehlkontrolle · Befallene Bäume aufarbeiten & abfahren (Frist) · Fangsysteme
   leeren/kontrollieren · Ruhe (keine Aktion) · Vergangenheit (blass). Hover: alle Aktionen des Tages mit
   Wahrscheinlichkeit und Begründung. Darunter eine Legende und eine Liste „Nächste 14 Tage“.
3. **KDE-Flächendiagramme** (x = Kalendertag, y = Dichte), je Ereignis ein Panel oder überlagert:
   - Schwärmbeginn (Überwinterer, Generation 1)
   - Ausflug Jungkäfer Generation 1 (= Schwärmen Generation 2)
   - Ausflug Jungkäfer Generation 2 (= Schwärmen Generation 3)
   - optional Geschwisterbruten
   Graue Fläche: Verteilung über 1991–2025. Farbige Fläche: Verteilung für das aktuelle Jahr (Ist-Wetter bis Stichtag,
   danach Ensemble + Klimaszenarien). Senkrechte Linie, wenn das Ereignis schon eingetreten ist. Beschriftung
   „tritt in x % der Jahre ein“ (Generation 2/3 wird nicht in jedem Jahr fertig).
4. **Trockenstress Wald**: relativ extrahierbares Bodenwasser (REW) des Fichtenbestands, Fächer 10–90 % für 6 Wochen,
   Band 1991–2025, Ampel „P(REW < 0,4) in 14 Tagen“.
5. **Sentinel-2 Waldstück**: Zeitreihe NDVI, NDMI (NDWI nach Gao), NDRE, DSWI gegen das Mittel 2017–Vorjahr,
   Anomalie-Ampel, Karte der Pixel-Anomalie der letzten wolkenfreien Szene (Rot-/Graustadium-Verdachtsflächen).

### Tab „Buchsbaumzünsler“
1. Kopfzeile: Stichtag, Kennzahlen: aktive Generation, Gradtagsumme, „Spritzfenster offen?“.
2. **Jahreskalender**: Kontrolle (Gespinste, Kotkrümel) · Pheromonfalle aufhängen/kontrollieren ·
   **Spritzen Bt** (nur wenn Junglarven aktiv UND Wetter passt) · Absammeln/Abspritzen mit Wasser ·
   Ruhe (Diapause).
3. **KDE-Flächendiagramme**: Wiederaufnahme Fraß der Überwinterer, Junglarven Generation 1, Junglarven Generation 2,
   Junglarven Generation 3 (nur in warmen Jahren), jeweils historisch vs. aktuell.
4. Tabelle „Spritzfenster der nächsten 10 Tage“ aus dem Ensemble: Tag, P(Tmax ≥ 15 °C), P(trocken 24 h), Gesamtbewertung.

## 2. Architektur und Dateien

```
heu/
  config.py              ÄNDERN  + BorkenkaeferAnnahmen, ZuenslerAnnahmen, WaldAnnahmen
  phaenologie.py         NEU     gemeinsame Bausteine: Tageslänge, Gradtage, Szenario-Wetter, Ereignisse, KDE
  borkenkaefer.py        NEU     PHENIPS (Baier et al. 2007), Ereignisse, Aktionsregeln
  zuensler.py            NEU     Gradtagmodell Cydalima perspectalis, Ereignisse, Aktionsregeln, Spritzfenster
  wald_wasser.py         NEU     Bodenwasserbilanz Fichtenbestand (REW), Prognose
  wald_satellit.py       NEU     Sentinel-2 Waldindizes über CDSE (nutzt CDSE-Klasse aus satellite.py)
  kalender.py            NEU     Datenmodell Aktionstag + Plotly-Jahreskalender
  plots.py               ÄNDERN  + kde_area_fig()
  stunden.py             ÄNDERN  + load_ensemble_daily() (Tageswerte inkl. Niederschlag/ET0 aus dem Ensemble)
  ui.py                  ÄNDERN  + gecachte Lader für Schädlingsläufe
pages/
  4_Schädlinge.py        NEU
data/
  wald.geojson           NEU (Hubert)  Waldstück(e), gleiches Format wie felder.geojson
  schaedlinge_beobachtungen_vorlage.csv  NEU  eigene Beobachtungen zur Kalibrierung
tests/
  test_phaenologie.py, test_borkenkaefer.py, test_zuensler.py, test_wald_wasser.py   NEU (pytest)
scripts/selftest.py      ÄNDERN  + Abschnitt Schädlinge
docs/PLAN_schaedlinge.md (diese Datei)
```

**Datenfluss** (beide Schädlinge gleich aufgebaut):

```
ui.taeglich()  ──► Tageswetter 1991..heute (tmean, tmax, tmin, rain, rad[MJ/m²], et0, wind)
                    │
                    ├─► Historie: model(year) für jedes Jahr 1991–2025 ──► Ereignistage je Jahr ──► KDE grau
                    │
                    └─► aktuelles Jahr: Ist bis Stichtag
                           + Ensemble (15 Tage, ui.ensemble_taeglich)
                           + Klimajahr y (Rest des Jahres), für y in 1991–2025
                         = 35 × (Anzahl Ensemble-Member, gekappt) Szenarien
                         ──► model(szenario) ──► Ereignistage ──► KDE farbig + Wahrscheinlichkeiten je Tag
                                                                    │
                                                                    └─► Aktionsregeln ──► Kalender
```

Das Muster ist `forecast._scenario_weather` (Ist bis Stichtag, danach Klimajahr) und `parasiten.jahresmatrix`
(Tagesreihe → Jahr × Tag). Wie dort wird mit 365-Tage-Jahren gearbeitet (29.2. entfällt, `weather.doy365`).

## 3. Gemeinsame Bausteine: `heu/phaenologie.py`

```python
def tageslaenge_h(doy: np.ndarray | int, lat_deg: float) -> np.ndarray
```
Astronomische Tageslänge nach FAO-56 (Allen et al. 1998, Gl. 25 und 34): δ = 0,409·sin(2π/365·J − 1,39),
ωs = arccos(−tan φ·tan δ), N = 24/π·ωs. **J ist der echte Kalendertag 1..366**, nicht der 0-basierte Index.
Kontrollwert für 48,09 °N: N(7. Aug.) ≈ 14,5 h, N(26. Aug.) ≈ 13,5 h (in diesem Plan nachgerechnet).
[VERIFIZIEREN] Ob PHENIPS die astronomische Tageslänge oder eine mit Dämmerung verwendet; barrks rechnet mit einer
eigenen Funktion. Bei Abweichung Parameter `tageslaenge_def: Literal["astronomisch", "buergerlich"]` ergänzen.

```python
def gradtage(t: np.ndarray, basis: float, obergrenze: float | None = None) -> np.ndarray
```
Tägliche Gradtage max(0, min(t, obergrenze) − basis). Mittelwertmethode (Arnold 1960), wie in `parasiten.hornfliege`.

```python
@dataclass
class Szenario:
    name: str            # z. B. "2003 | ECMWF 07"
    jahr_klima: int      # Klimajahr für den Rest
    member: str | None
    w: dict[str, np.ndarray]   # 365-Tage-Arrays wie weather.year_arrays

def szenarien(df: pd.DataFrame, jahr: int, stichtag: int, clim: dict[int, dict],
              ens_daily: pd.DataFrame | None, max_szenarien: int = 175, seed: int = 0) -> list[Szenario]
```
- Ist-Arrays mit `weather.partial_year(df, jahr)`; Länge n.
- Tage `stichtag+1 … stichtag+15`: Ensemble-Member (Tageswerte), zufällig einem Klimajahr zugeordnet
  (jedes Klimajahr bekommt `k = ceil(max_szenarien/35)` Member, Rest kappen). Ohne Ensemble: direkt Klimajahr.
- Danach: Klimajahr `y` ab gleichem Kalendertag.
- Lücke zwischen letztem Datentag n und Stichtag (ERA5 hat ~5 Tage Verzug): Stichtag = min(gewünschter Stichtag, n−1);
  im UI anzeigen „Ist-Wetter bis TT.MM.“.
- Rückblick-Modus (Stichtag in einem vergangenen Jahr): kein Ensemble, nur Klimajahre, das Jahr selbst wird aus `clim`
  ausgeschlossen (sonst Informationsleck). Dieses Muster erlaubt Hubert im Winter zu zeigen, wie die Prognose im Juni
  ausgesehen hätte.

```python
def ereignistag(serie: np.ndarray, schwelle: float, ab_tag: int = 0) -> float
```
Erster Tag ≥ schwelle (ab ab_tag), sonst `np.nan`.

```python
@dataclass
class EreignisVerteilung:
    name: str
    hist_tage: np.ndarray          # je Historienjahr (nan = nicht eingetreten)
    akt_tage: np.ndarray           # je Szenario des aktuellen Jahres (nan = nicht eingetreten)
    eingetreten: int | None        # Tag, falls im aktuellen Jahr schon vor dem Stichtag eingetreten
    p_hist: float                  # Anteil Historienjahre mit Ereignis
    p_akt: float                   # Anteil Szenarien mit Ereignis

def kde_kurve(tage: np.ndarray, x: np.ndarray, bw: float | str = "scott",
              min_bw_tage: float = 3.0, gewichte: np.ndarray | None = None) -> np.ndarray
```
- `scipy.stats.gaussian_kde` auf den nicht-nan Tagen; Bandbreite mindestens `min_bw_tage` (sonst Nadeln, wenn alle
  Szenarien kurz vor dem Ereignis fast identisch sind). Umsetzung: `kde.set_bandwidth(max(kde.factor, min_bw_tage/std))`.
- Weniger als 3 verschiedene Werte: schmale Normalverteilung (σ = min_bw_tage) um den Mittelwert.
- **Dichte mit `p` skalieren** (Fläche = Eintrittswahrscheinlichkeit), damit „Gen. 3 nur in 20 % der Jahre“ sichtbar
  kleiner ist als Gen. 1.
- Optional `gewichte`: jüngere Jahre stärker (z. B. linear 1991 → 0,5, 2025 → 1,5), Schalter im UI „Klimawandel
  gewichten“. Standard aus.

```python
def tageswahrscheinlichkeit(akt_tage: np.ndarray, n_szen: int, fenster: tuple[int, int] = (0, 0)) -> np.ndarray
```
Für jeden Tag d des Jahres: Anteil Szenarien, in denen das Ereignis in [d+fenster0, d+fenster1] liegt. Basis für die
Aktionsregeln.

```python
def phasen_wahrscheinlichkeit(szen_reihen: np.ndarray, von: float, bis: float) -> np.ndarray
```
`szen_reihen` = Matrix Szenario × Tag (z. B. Entwicklungsstand der Brut); Ergebnis je Tag P(von ≤ wert < bis).

## 4. Borkenkäfer: `heu/borkenkaefer.py` (PHENIPS)

### 4.1 Modell und geprüfte Parameter

Quelle der Zahlen: PHENIPS nach Baier, Pennerstorfer & Schopf (2007), *Forest Ecology and Management* 249:171–186,
in der Implementierung des R-Pakets **barrks** (CRAN, Datei `R/model-phenips.R`), dort abgelesen am 06.11.2026.
Entwicklungsfunktion nach Wermelinger & Seifert (1998).

| Parameter | Wert | Bedeutung |
| --- | --- | --- |
| `onset_start` | 01.04. | Beginn Gradtagsumme für den Schwärmbeginn |
| `onset_basis_c` | 8,3 °C | Basis der Gradtage **aus Tmax** |
| `onset_gt` | 140 GT | Schwelle der Tmax-Gradtagsumme |
| `t_flug_c` | 16,5 °C | Tmax muss darüber liegen (Schwärmtag) |
| `t_basis_c` = `t_low_c` | 8,3 °C | Entwicklungsnullpunkt |
| `t_opt_c` | 30,4 °C | Optimum, darüber Abzug |
| `t_up_c` | 38,9 °C | obere Grenze, darüber keine Entwicklung |
| `gt_generation` | 557 GT | Ei bis Jungkäferausflug (dev = 1) |
| `dev_geschwister` | 0,5 | Elterngeneration ab dev 0,5 → Geschwisterbrut möglich |
| `dev_weisse_stadien_max` | 0,6 | Brut mit dev < 0,6 (Ei/Larve/Puppe) stirbt zum 31.10. |
| `tageslaenge_diapause_h` | 14,5 h | nach dem 21.6.: erster Tag mit N < 14,5 h → keine neue Brut |
| `modell_ende` | 31.10. | danach keine Entwicklung |

Hinweis: In der Vorbesprechung stand „Schwärmen ab ~110 GT“; barrks verwendet für PHENIPS **140 GT (Tmax, Basis 8,3 °C,
ab 1.4.)**. Der Plan übernimmt 140 und macht den Wert einstellbar.

**Rindentemperatur** (Baier 2007, Gl. A.3–A.5, verbatim aus barrks):

```
bt_mean = −0,173 + 0,0008518·rad + 1,054·tmean
bt_max  =  1,656 + 0,002955·rad + 0,534·tmax + 0,01884·tmax²
bt_diff = (−310,667 + 9,603·bt_max) / 24
```
Beschattet: `bt_mean = tmean`, `bt_max = tmax`.

**Effektive Temperatur je Tag** (verbatim-Logik aus `phenips_calc_teff`):

```
teff = 0                         wenn bt_mean < t_low oder bt_mean > t_up
teff = bt_mean                   sonst
teff -= bt_diff                  wenn bt_max > t_opt und bt_diff > 0
teff = max(0, teff − t_basis)
dev  = kumulierte Summe(teff) / 557
```

**[VERIFIZIEREN] Einheit von `rad`.** barrks dokumentiert die Strahlung als Tagessumme in W·h/m². ERA5/Open-Meteo
liefert `shortwave_radiation_sum` in MJ/m² → Umrechnung `rad_wh = rad_mj · 277,78`. Plausibilitätscheck:
Sommertag 25 °C, 25 MJ/m² ≈ 6 945 Wh/m² → bt_max ≈ 1,7 + 20,5 + 13,4 + 11,8 = 47 °C. Das ist für besonnte Rinde
am Bestandesrand möglich, aber hoch. Vor der Freigabe mit einem Referenzlauf in barrks gegenprüfen (Test T-B3 unten).
Falls barrks Tagesmittel in W/m² erwartet, ändert sich der Faktor auf `rad_mj·1e6/86400`.

### 4.2 Generationen-Logik

```python
@dataclass
class Brut:
    name: str            # "Gen 1", "Gen 1 Geschwister", "Gen 2", ...
    start: int           # Tag, an dem die Elterngeneration fliegt/anlegt
    dev: np.ndarray      # 365 Werte, nan vor start, 0..>1
    ausflug: float       # Tag mit dev ≥ 1 (nan = nicht fertig)
    stirbt: bool         # dev < 0,6 am 31.10.

@dataclass
class PhenipsErgebnis:
    schwaermbeginn: float          # Tag (nan = kein Schwärmen)
    diapause_tag: int              # erster Tag nach 21.6. mit N < 14,5 h
    bruten: list[Brut]
    teff: np.ndarray
    bt_mean: np.ndarray
    bt_max: np.ndarray

def rindentemperatur(tmean, tmax, rad_mj, besonnt: bool, ba: BorkenkaeferAnnahmen) -> tuple[np.ndarray, np.ndarray]
def teff(bt_mean, bt_max, ba) -> np.ndarray
def schwaermbeginn(tmax: np.ndarray, ba) -> float
def flugtage(tmax: np.ndarray, ba) -> np.ndarray[bool]          # tmax > 16,5
def phenips(w: dict, ba: BorkenkaeferAnnahmen, lat: float, besonnt: bool = True) -> PhenipsErgebnis
```

Ablauf in `phenips()` (entspricht barrks):
1. `onset` = erster Tag ab 1.4. mit Σ max(0, tmax − 8,3) ≥ 140 **und** tmax > 16,5.
2. Gen 1 startet am `onset`; dev kumuliert teff/557 ab diesem Tag.
3. Geschwisterbrut: am ersten Flugtag, an dem dev(Eltern) > 0,5 und noch keine Diapause, startet „Gen n Geschwister“
   (eigene dev-Reihe). Geschwisterbruten erzeugen selbst keine weiteren Geschwister (barrks: nur halbe Generationen
   `gen_<n+0.5>`).
4. Gen n+1 startet am ersten Flugtag mit dev(Gen n) ≥ 1 und Tag < `diapause_tag`.
5. Ab `diapause_tag` werden keine neuen Bruten gestartet; laufende entwickeln sich bis 31.10. weiter.
6. Am 31.10.: Bruten mit dev < 0,6 als „stirbt“ markieren (überwintern nur Käfer und späte Jungkäfer).

### 4.3 Ereignisse (für KDE und Kalender)

| Ereignis | Definition |
| --- | --- |
| `Schwärmen Gen 1` | `schwaermbeginn` |
| `Ausflug Jungkäfer Gen 1` | `bruten["Gen 1"].ausflug` (= Schwärmbeginn Gen 2) |
| `Ausflug Jungkäfer Gen 2` | `bruten["Gen 2"].ausflug` |
| `Ausflug Geschwister Gen 1` | optional, Schalter im UI |
| `Brutstadium` je Tag | dev < 0,6 „weiße Stadien“, 0,6 ≤ dev < 1 „Jungkäfer unter der Rinde (Reifungsfraß)“ |

[VERIFIZIEREN] Die Grenze 0,6 stammt aus dem Mortalitätsparameter von barrks und wird hier als Übergang
„weiße Stadien → Jungkäfer“ interpretiert. Für die Aktionsregeln ist das ausreichend; für eine Stadienanzeige
die Stadienanteile in Wermelinger & Seifert (1998) nachschlagen.

### 4.4 Aktionsregeln (Bio-Betrieb, keine Spritzung)

```python
@dataclass(frozen=True)
class Aktion:
    key: str          # "fallen", "bohrmehl", "aufarbeiten", "fangsystem", "ruhe"
    label: str
    farbe: str
    prio: int         # höher gewinnt bei der Einfärbung

def aktionen_borkenkaefer(akt: list[PhenipsErgebnis], hist_p: dict, stichtag: int,
                          ba: BorkenkaeferAnnahmen, trockenstress_p: np.ndarray | None) -> pd.DataFrame
# Rückgabe: Index Tag 0..364, Spalten je Aktion: p (0..1), aktiv (bool), grund (str); plus "haupt" (key mit höchster prio)
```

| Aktion | Regel (Tag d aktiv, wenn …) | Farbe |
| --- | --- | --- |
| **Kontrollsaison beginnt** (Fichtenränder begehen, Vorjahres-Befallsstellen prüfen) | P(Schwärmbeginn in [d, d+10]) ≥ 0,2 und d < Median Schwärmbeginn; einmalig das erste solche Fenster. Nur falls `pheromonfallen=True`: stattdessen „Fallen aufstellen“ | Blau |
| **Bohrmehlkontrolle** (Stammfuß, Rindenschuppen, Spinnweben) | d liegt in einer Schwärmphase: P(eine Generation fliegt an d ± 3) ≥ 0,3, **dann alle `kontroll_intervall_tage` (Standard 7)** ein Kontrolltag; bei Trockenstress P(REW < 0,4) ≥ 0,5 Intervall 5 Tage | Gelb |
| **Befallene Bäume aufarbeiten & abfahren** | Frist: zwischen Befallsbeginn und P(dev ≥ `dev_frist` = 0,8) ≥ 0,2 der laufenden Brut. Farbe Orange, die letzten 5 Tage vor der Frist Rot | Orange/Rot |
| **Fangsysteme kontrollieren/leeren** | Brunnhofwald: **keine Fallen** (`pheromonfallen=False`), Aktion standardmäßig ausgeblendet; nur wenn im UI „Fangsysteme vorhanden“ angehakt: während Schwärmphasen alle 7 Tage | Violett |
| **Winterkontrolle / Schadholz nach Sturm** | Nov.–März: einmal monatlich (statisch), außerdem nach Tagen mit Böen > 70 km/h (Spalte `wind`) | Grau-blau |
| **Ruhe** | sonst | Weiß/hellgrau |

Fachliche Grundlage (vor Freigabe gegenlesen): Im Bio-Betrieb kein Insektizid; Pheromonfallen dienen dem
**Monitoring**, nicht der Bekämpfung, und gehören in ausreichenden Abstand zum gefährdeten Bestand. Befallene Fichten
vor dem Ausflug der Jungkäfer aufarbeiten und aus dem Wald abfahren (Abstand zum Nadelholz) oder entrinden.
[VERIFIZIEREN] Abstände und Fristen mit dem LWF-Merkblatt „Borkenkäfer“ und der Bio-Verbandsrichtlinie abgleichen;
die Zahlen (Mindestabstand, Kontrollintervall) sind im Plan als Annahmen in `BorkenkaeferAnnahmen` geführt.

### 4.5 Kalibrierung mit Monitoringdaten

`data/schaedlinge_beobachtungen.csv` (Vorlage wird angelegt):

```
datum,art,typ,wert,ort,bemerkung
2025-04-22,borkenkaefer,fallenfang,350,Falle Nord,Pheromonfalle Buchdrucker
2025-06-10,borkenkaefer,befall,1,Waldstück Süd,Bohrmehl an 2 Fichten
2025-05-05,zuensler,falter,12,Garten,Pheromonfalle
2025-04-18,zuensler,raupen_aktiv,1,Hecke,erste Fraßspuren
```

`kalibriere_onset(beob, hist_ergebnisse) -> dict` vergleicht beobachteten ersten Fangtag (Schwelle z. B. ≥ 100 Käfer je
Falle und Woche, einstellbar) mit dem modellierten Schwärmbeginn je Jahr; Ausgabe mittlere Abweichung, RMSE und ein
vorgeschlagener Wert für `onset_gt`. Keine automatische Übernahme; das UI zeigt den Vorschlag.

Externe Vergleichsdaten: Das LWF betreibt ein Borkenkäfer-Monitoring mit Fallenfängen und Entwicklungsständen für
Bayern. [VERIFIZIEREN] Ob die Fangzahlen als Datei herunterladbar sind (dann Loader `lade_lwf_monitoring()`), sonst
manuell in die Beobachtungs-CSV übertragen. Eine passende Station in der Nähe von Kirchweidach (Lkr. Altötting/Traunstein)
auswählen.

## 5. Trockenstress Wald: `heu/wald_wasser.py`

Ziel: tägliche relative extrahierbare Bodenwassermenge `REW = (W − W_welk)/(W_fk − W_welk)` für einen Fichtenbestand,
Stressschwelle **REW < 0,4** nach Granier et al. (1999, *Ecological Modelling* 116:269–283, Modell BILJOU).
[VERIFIZIEREN] Schwelle 0,4 und Interzeptionsansatz im Original prüfen.

Wiederverwendung: Das Eimer-Prinzip aus `model.grow_one_day` (Zeilen „Wasserbilanz (MVP: Eimer-Modell …)“), aber als
eigene, vektorisierte Funktion, weil der Wald keine Graswachstumsdynamik braucht.

```python
@dataclass(frozen=True)
class WaldAnnahmen:
    nfk_mm: float = 120.0              # nutzbare Feldkapazität Wurzelraum Fichte (flach wurzelnd) [ANNAHME]
    interzeption_anteil: float = 0.30  # Anteil Freilandniederschlag, der im Kronendach verdunstet [ANNAHME]
    interzeption_max_mm: float = 4.0   # Kronenspeicher je Tag [ANNAHME]
    kc_fichte: float = 0.9             # Bestandes-Koeffizient auf ET0 (Transpiration+Boden) [ANNAHME]
    anteil_fichte: float = 0.30        # Brunnhofwald: Mischwald, ca. 30 % Fichte (Hubert 06.11.2026)
    kc_laub_winter: float = 0.3        # Laubholzanteil ohne Blätter (Nov.–April) [ANNAHME], siehe Abschnitt 15.2
    rew_stress: float = 0.4            # Granier 1999
    rew_start: float = 1.0             # 1. Januar voll
    winter_ohne_et_bis_tmean: float = 5.0   # keine Transpiration bei tmean < 5 °C [ANNAHME]

def wasserbilanz(w: dict, wa: WaldAnnahmen, rew0: float | None = None) -> pd.DataFrame
# Spalten: interzeption, bestandesniederschlag, et, rew, stress (bool), stress_tage_kum
```

Tagesschritt:
```
I    = min(interzeption_max_mm, interzeption_anteil · rain)
P_b  = rain − I
ETa  = kc · et0 · f(REW),   f(REW) = 1 für REW ≥ 0,4, sonst REW/0,4   (Granier-Reduktion)
ETa  = 0, wenn tmean < 5 °C
W    = clip(W + P_b − ETa, 0, nfk)   (Überschuss versickert)
```

Prognose: `szenarien()` aus `phaenologie.py` (Ist bis Stichtag + Ensemble 15 Tage + Klimajahre), Ergebnis-Matrix
Szenario × Tag, daraus Quantile 10/25/50/75/90 und `P(REW < 0,4)` je Tag für die nächsten 42 Tage.
Historie: `jahresmatrix` aus `parasiten.py` für das graue Band (via `plots.band_fig`). Zusätzlich Kennzahl
„Stresstage seit 1.4.“ als Perzentil gegen 1991–2025 (`parasiten.perzentil`, `parasiten.ampel`).

Kopplung an den Borkenkäfer: Trockenstress schwächt die Abwehr der Fichte (Harzfluss). Im Plan nur als
**Prioritätsverstärker** (kürzeres Kontrollintervall, Hinweistext), nicht als Eingang in PHENIPS.

Wetterquelle: Für 1991–2025 ERA5 (`ui.taeglich`), Option DWD (`ui.dwd`) wie auf der Seite Klimatrend. ET0 ist in beiden
vorhanden (ERA5 `et0`, DWD selbst gerechnet in `dwd.et0_fao56`).

## 6. Sentinel-2 Waldstück: `heu/wald_satellit.py`

### 6.1 Eingabe
`data/wald.geojson`, Format wie `felder.geojson` (Properties `id`, `name`, optional `flaeche_ha`, `baumart`).
Polygon oder MultiPolygon. Laden mit einer Variante von `satellite.load_fields()`:

```python
def load_waldstuecke(path: Path = DATA_DIR / "wald.geojson") -> list[dict]
```
Erweiterung für MultiPolygon: `satellite.load_fields` liest nur `coordinates[0]`; für Wald `shapely.geometry.shape`
verwenden und Schwerpunkt/Fläche daraus ableiten. Upload-Variante im UI: `st.file_uploader("GeoJSON Waldstück")` →
validieren (Typ, Koordinaten in WGS84, Fläche 0,1–500 ha) → in `data/wald.geojson` speichern.
Innenpuffer: 10 m wie bei den Wiesen (`satellite.inner_buffer`), damit Wegränder herausfallen.

### 6.2 Indizes (Evalscript für die CDSE Statistical API)

| Index | Formel (Sentinel-2 L2A) | Zweck |
| --- | --- | --- |
| NDVI | (B08 − B04)/(B08 + B04) | Vitalität, Rot-/Graustadium |
| NDMI (NDWI nach Gao 1996) | (B8A − B11)/(B8A + B11) | Wassergehalt der Krone, frühester Trockenstress-/Befallshinweis |
| NDRE | (B8A − B05)/(B8A + B05) | Chlorophyll, Rotkante |
| DSWI | (B08 + B03)/(B11 + B04) | Disease Stress Water Index (Galvão et al. 2005), in Borkenkäferstudien verwendet |

Wolkenmaske: SCL ∈ {4 (Vegetation)} für Wald; 5 (unbewachsen) zusätzlich zulassen, damit Kahlflächen nach Räumung
sichtbar bleiben. Statistik: Perzentile 10/25/50/75/90 statt nur Median, weil Befall zuerst als **unteres Perzentil**
(einzelne Bäume/Nester) sichtbar wird, lange bevor sich der Median bewegt.

```python
EVAL_S2_WALD: str   # analog EVAL_S2 in satellite.py, Ausgänge ndvi, ndmi, ndre, dswi, dataMask
def fetch_wald_s2(stueck: dict, start: dt.date, end: dt.date, client: CDSE) -> pd.DataFrame
def update_wald(stueck: dict, progress=None) -> pd.DataFrame          # Cache data/sat_wald_<id>.csv, inkrementell
```
Wiederverwendung aus `satellite.py`: `CDSE`, `cdse_credentials`, `_stats_body` (Perzentile-Liste als Parameter ergänzen,
ohne bestehende Aufrufe zu ändern: neues Argument `percentiles=(25, 50, 75)` mit Standardwert), `_parse_stats`
(erweitern, damit es beliebige Perzentile übernimmt), `inner_buffer`. Earth-Search-Fallback ist für den Wald nicht nötig
(B11 ist bei Earth Search vorhanden, aber der Aufwand lohnt nicht, solange CDSE läuft).

### 6.3 Auswertung

```python
def anomalie(s2: pd.DataFrame, index: str = "ndmi", q: str = "q10", referenz_jahre: tuple[int, int] | None = None,
             fenster_tage: int = 15) -> pd.DataFrame
```
- Saisonale Referenz: für jeden Kalendertag Median und MAD (robuste Streuung) aus den Referenzjahren (Standard 2017 bis
  Vorjahr), geglättet über ±`fenster_tage`. Nur Szenen Mai–September (Fichte im Winter mit Schnee/Sonnenstand unzuverlässig).
- z-Wert je Szene: (Wert − Median)/(1,4826·MAD).
- Ampel: z < −1,5 an zwei aufeinanderfolgenden wolkenfreien Szenen → „Verdacht“ (Orange), z < −2,5 → „deutliche
  Veränderung“ (Rot).

```python
def pixelkarte(stueck: dict, datum: dt.date, client: CDSE, index: str = "ndmi") -> np.ndarray
```
Optional (Ausbaustufe): Sentinel Hub **Process API** (`https://sh.dataspace.copernicus.eu/api/v1/process`) liefert ein
10-m-Raster des Index für die Szene und für eine Referenz (Median Juli/August des Vorjahres, über `mosaicking: ORBIT`
und Evalscript-Median). Differenzraster als `plotly.express.imshow` über dem Polygonumriss; Pixel mit Differenz
< −0,1 (NDMI) rot markieren. [VERIFIZIEREN] Kontingent der CDSE-Processing-Units für Hubert-Konto prüfen; ein
Waldstück von 5 ha × wenige Szenen ist klein.

**Erwartungsmanagement (im UI als Hinweis):** Grünbefall (Käfer in der Rinde, Krone noch grün) ist mit 10-m-Pixeln
kaum erkennbar; Studien finden schwache Signale in NDMI/DSWI einige Wochen vor der Rotverfärbung. Rot- und Graustadium
(Nadelverfärbung, Nadelverlust) sind gut sichtbar, aber dann ist die Brut meist ausgeflogen. Satellit ist deshalb
Kontrolle **wo** nachzuschauen ist und Dokumentation, PHENIPS sagt **wann**.

Testmodus: `synthetic_wald_s2(seed)` erzeugt saisonale Kurven mit Rauschen, Wolkenlücken und in 2024 einem Befallsnest
(q10 fällt ab Juli um 0,15). Gleiches Muster wie `satellite.synthetic_s2`.

## 7. Buchsbaumzünsler: `heu/zuensler.py`

### 7.1 Geprüfte Fakten und offene Parameter

Geprüft (Abstract Nacambo et al. 2014, *J. Appl. Entomol.* 138:14–26):
- Überwinterung überwiegend als **L3** in Europa,
- Diapause-Induktion bei Tageslänge **etwa 13,5 h**,
- 1,5–2 Monate Kälte zur Beendigung der Diapause,
- in Mitteleuropa **mindestens zwei Generationen**.

Geprüft (Lemić et al. 2025, *J. Central Eur. Agric.* 26(1):150–161, Kroatien, Gradtage ab 1.1.):
- Generation 1 ≈ 560–640 GT, Generation 2 ≈ 530–730 GT,
- Larven der Überwinterer ab ≈ 53 GT aktiv, Bekämpfungsfenster „Ende April/Anfang Mai (53 GT)“ und
  „Ende Juni/Anfang Juli (478 GT)“,
- Ei ≈ 4–6 Tage, Larvenentwicklung 30–35 Tage (Dauern im Feld).
- Basistemperatur im Text nicht eindeutig; 10 °C wird als üblicher Wert genannt.

**Nicht geprüft** (Originaltext Nacambo 2014 war nicht abrufbar): untere Entwicklungsschwelle je Stadium und die
Gradtagsumme je Generation. Die Vorbesprechung nannte „Basis ~8–10 °C, ~450–550 GT“. → **[VERIFIZIEREN]**: Tabelle der
Entwicklungsparameter in Nacambo et al. 2014 (Tab. 2/3) besorgen (über Bibliothek/Fernleihe oder Autorenkopie) und die
Werte unten ersetzen.

### 7.2 Modell

Bewusst einfach und vollständig parametrisiert, damit die Zahlen später ohne Codeänderung getauscht werden können:

```python
@dataclass(frozen=True)
class ZuenslerAnnahmen:
    basis_c: float = 10.0                 # [VERIFIZIEREN] Nacambo 2014; Lemić 2025 nennt 10 °C als üblich
    gt_start: str = "01-01"               # Gradtage ab 1.1. (wie Lemić 2025)
    gt_reaktivierung: float = 53.0        # Überwinterer fressen wieder (Lemić 2025)
    gt_rest_ueberwinterer: float = 250.0  # L3 -> Falter der Überwinterungsgeneration [ABGELEITET aus Lemić: Falter 174–443 GT]
    gt_generation: float = 600.0          # Ei -> Ei-Ablage nächste Gen. [Lemić 560–640; VERIFIZIEREN Nacambo]
    anteil_ei: float = 0.09               # Stadienanteile an gt_generation, ABGELEITET aus Lemić Tab. 3
    anteil_larve: float = 0.58            # (Ei 4–6 d, Larve 30–35 d, Puppe ~15–20 d, Präovipos. ~3 d)
    anteil_puppe: float = 0.28
    anteil_praeovi: float = 0.05
    junglarven_bis_anteil_larve: float = 0.4   # L1–L3 ≈ erste 40 % der Larvenzeit [ANNAHME]
    tageslaenge_diapause_h: float = 13.5  # Nacambo 2014
    max_generationen: int = 3
    # Bekämpfung (Bio: Bacillus thuringiensis)
    bt_tmax_min_c: float = 15.0           # Fraßaktivität nötig [ANNAHME, Herstellerangaben prüfen]
    bt_trocken_stunden: int = 24          # nach Behandlung kein Regen [ANNAHME]
    bt_regen_grenze_mm: float = 1.0
```

```python
@dataclass
class ZuenslerErgebnis:
    gt_kum: np.ndarray
    reaktivierung: float              # Tag
    falter_ueberwinterer: float       # Tag (Beginn Eiablage)
    generationen: list[dict]          # je Gen: {"eiablage": tag, "junglarven_von": tag, "junglarven_bis": tag,
                                      #          "falter": tag, "diapause": bool}
    stadium: np.ndarray[str]          # je Tag dominantes Stadium der jüngsten aktiven Gen.

def zuensler(w: dict, za: ZuenslerAnnahmen, lat: float) -> ZuenslerErgebnis
```

Ablauf:
1. GT = Σ max(0, tmean − basis) ab 1.1.
2. `reaktivierung` = erster Tag GT ≥ 53 (Überwinterer werden aktiv → **Kontrolle**, ggf. **Bt**).
3. `falter_ueberwinterer` = GT ≥ 53 + 250 → Eiablage Gen 1 beginnt.
4. Für Gen k: Eiablage bei GT_k; Schlupf (Junglarven) bei GT_k + anteil_ei·gt_generation; Junglarvenphase bis
   GT_k + (anteil_ei + junglarven_bis_anteil_larve·anteil_larve)·gt_generation; Falter bei
   GT_k + (anteil_ei+anteil_larve+anteil_puppe)·gt_generation; nächste Eiablage bei GT_k + gt_generation.
5. Diapause: Larven, die nach dem ersten Tag mit N < 13,5 h (nach 21.6.; bei 48,09 °N ≈ 26.8.) schlüpfen, gehen als
   L2–L3 in Diapause → keine weitere Generation. Ereignis zählt trotzdem als „Junglarven Gen k“.
6. Verteilung statt Punktwert: Weil Eiablage sich über Wochen zieht, je Ereignis zusätzlich **Beginn, Mitte, Ende**
   (Gradtage ±15 % der Phase); die KDE nutzt die Mitte, der Kalender die ganze Spanne.

### 7.3 Ereignisse für die KDE
`Wiederaufnahme Fraß (Überwinterer)`, `Junglarven Gen 1`, `Junglarven Gen 2`, `Junglarven Gen 3`.

### 7.4 Aktionsregeln

| Aktion | Regel | Farbe |
| --- | --- | --- |
| **Kontrolle** (Inneres der Pflanze, Gespinste, grüne Kotkrümel) | ab P(Reaktivierung bis d) ≥ 0,2 wöchentlich bis Diapause; zusätzlich täglich-markiert in den Junglarvenphasen | Gelb |
| **Pheromonfalle aufhängen** | 7 Tage vor Median Falterflug Überwinterer; Falle kontrollieren wöchentlich während Flugphasen | Blau |
| **Spritzen Bt (abends)** | P(Junglarven aktiv an d) ≥ 0,4 **und** Wetter am Tag d: tmax ≥ 15 °C, Regen d und d+1 < 1 mm (bei Ensemble: P ≥ 0,7) | Grün (dunkel) |
| **Spritzen möglich, Wetter unsicher** | Junglarven-Bedingung erfüllt, Wetter-P 0,4–0,7 | Grün (hell) |
| **Absammeln / mit Wasser abspritzen** | Larven aktiv (alle Stadien), kein Spritzwetter | Orange |
| **Ruhe (Diapause/Winter)** | sonst | Hellgrau |

Wetterbedingung für die Tage mit Vorhersage aus dem Ensemble (Tageswerte je Member), für Tage danach aus den
Klimajahren (dann nur „Spritzfenster wahrscheinlich in Woche X“, keine Tagesfarbe). Vergangene Tage mit Ist-Wetter.
[VERIFIZIEREN] Zugelassene Bt-Präparate (B. thuringiensis subsp. kurstaki bzw. aizawai) für Buchsbaum und deren
Temperaturangaben beim BVL; im Bio-Betrieb Betriebsmittelliste (FiBL) prüfen.

## 8. Kalender und Plots

### 8.1 `heu/kalender.py`

```python
AKTIONEN_BK: dict[str, Aktion]
AKTIONEN_BZ: dict[str, Aktion]

def kalender_fig(jahr: int, aktionen: pd.DataFrame, katalog: dict[str, Aktion], stichtag: int,
                 layout: Literal["monate", "wochen"] = "monate", height: int = 420) -> go.Figure
```
- Layout „monate“: `go.Heatmap` mit y = Monat (Jan oben), x = Tag 1–31; ungültige Tage (31.4.) leer.
- Diskrete Farben: Kategorien auf Ganzzahlen abbilden (`prio`-Reihenfolge), `colorscale` als Stufen
  `[[i/n, farbe], [(i+1)/n, farbe]]`, `showscale=False`, eigene Legende über unsichtbare `go.Scatter`-Spuren
  (eine je Aktion, `mode="markers"`, `marker_symbol="square"`).
- Vergangene Tage: Farbe mit 40 % Deckkraft (zweite Heatmap-Ebene mit weiß halbtransparent oder `opacity` je Zelle über
  zwei getrennte Traces „vergangen“ und „zukünftig“).
- Heute: Rahmen über `add_shape(type="rect")`.
- `hovertemplate` mit `customdata`: Datum, alle Aktionen des Tages mit p in %, Begründung.
- Zellen quadratisch: `yaxis.scaleanchor="x"`, `xgap=2, ygap=2`.

```python
def naechste_aktionen(aktionen: pd.DataFrame, katalog, stichtag: int, tage: int = 14) -> pd.DataFrame
```
Tabelle für `st.dataframe`: Datum, Wochentag, Aktion, Wahrscheinlichkeit, Grund.

### 8.2 `plots.kde_area_fig`

```python
def kde_area_fig(verteilungen: list[EreignisVerteilung], jahr: int, stichtag: int,
                 farben: list[str] | None = None, height: int = 380, getrennt: bool = False) -> go.Figure
```
- x: Datum des Jahres (`plots.tag_zu_datum`), 1.3.–31.10. als Standardbereich.
- je Ereignis: grau `fill="tozeroy"`, `fillcolor="rgba(150,150,150,0.25)"` (Historie), farbig 0,45 Deckkraft
  (aktuelles Jahr), Linie Median Historie gestrichelt, Linie Median aktuell durchgezogen.
- Eingetreten: `add_vline` mit Beschriftung „eingetreten TT.MM.“; die farbige Fläche entfällt dann.
- Legende: „Gen 2 Ausflug · 2026: 74 % · Historie: 61 %“.
- `getrennt=True`: `make_subplots(rows=len(verteilungen), shared_xaxes=True)` als Ridgeline; Standard für Handy-Breite.
- Senkrechte Linie „Stichtag“ gepunktet.

Farben aus `plots.py` (BLAU, GRUEN, ORANGE, ROT) plus Violett `#7b4fd1`.

## 9. Seite `pages/4_Schädlinge.py`

Gerüst nach dem Muster von `pages/2_Tiergesundheit.py`:

```python
st.set_page_config(page_title="Schädlinge", page_icon="🪲", layout="wide")
st.title("🪲 Schädlinge: Borkenkäfer und Buchsbaumzünsler")
ui.testmodus_hinweis()
tabB, tabZ = st.tabs(["Borkenkäfer (Buchdrucker)", "Buchsbaumzünsler"])
```

Gemeinsame Steuerung oben (vor den Tabs): Jahr/Stichtag (`st.date_input`, Standard letzter Datentag, Rückblick
erlaubt ab 1992), Wetterquelle (ERA5/DWD), Schalter „Ensemble verwenden“ (nur wenn Stichtag ≥ heute − 3).

**Caching** in `heu/ui.py`:

```python
@st.cache_resource(show_spinner="Berechne Borkenkäfer-Phänologie 1991–2025 …")
def bk_historie(tok: int, besonnt: bool, ba_hash: str) -> dict[int, PhenipsErgebnis]

@st.cache_data(ttl=3 * 3600, show_spinner="Berechne Szenarien für das laufende Jahr …")
def bk_aktuell(tok: int, jahr: int, stichtag: int, besonnt: bool, mit_ensemble: bool, stunde: str) -> dict

@st.cache_data(ttl=3 * 3600, show_spinner="Lade Ensemble-Tageswerte …")
def ensemble_taeglich(stunde: str) -> tuple[pd.DataFrame, str]
```
(analog für Zünsler und Wald). Dataclass-Annahmen über `dataclasses.astuple` → Hash als Cache-Schlüssel.

**Tab Borkenkäfer**, Reihenfolge der Elemente: Kennzahlen (st.metric ×4) → Kalender → „Nächste 14 Tage“ →
KDE (Expander „Wie lese ich das?“) → Trockenstress (Fächer + Band + Ampel) → Sentinel-2 Waldstück (Auswahl/Upload,
Button „Daten laden“, Zeitreihe q10/q50 mit Referenzband, Anomalie-Tabelle, optional Karte) →
Expander „Methodik & Quellen“ (Parameter-Tabelle aus `BorkenkaeferAnnahmen` automatisch gerendert).

**Tab Buchsbaumzünsler**: Kennzahlen → Kalender → Spritzfenster-Tabelle 10 Tage → KDE → Beobachtungen
(`st.data_editor` auf Basis der CSV, Speichern-Button) → Methodik & Quellen.

### 9.1 Ergänzung `stunden.load_ensemble_daily`

```python
ENS_DAILY_HOURLY = {"temperature_2m": "t", "precipitation": "rain", "shortwave_radiation": "rad",
                    "et0_fao_evapotranspiration": "et0", "wind_speed_10m": "ws"}

def load_ensemble_daily(days: int = 15) -> tuple[pd.DataFrame, str]
# Langes Format: Index date, Spalten tmean, tmax, tmin, rain, rad [MJ/m²], et0 [mm], wind [km/h max], member
```
- Ruft die Ensemble-API wie `_fetch_ensemble` ab, aber mit eigener Variablenliste (die bestehende `HOURLY`-Konstante und
  den Stundencache **nicht** ändern, sonst bricht der Cache `wetter_stunde_cache.csv.gz`).
- Aggregation je Member und Tag (Europe/Berlin): tmean = mean(t), tmax/tmin, rain = sum, rad = sum(W/m²·3600)/1e6,
  et0 = sum, wind = max(ws)·3,6.
- [VERIFIZIEREN] Ob `et0_fao_evapotranspiration` in der Ensemble-API je Member geliefert wird; sonst ET0 je Tag nach
  Hargreaves aus tmax/tmin/Extraterrestrischer Strahlung rechnen (Funktion in `phaenologie.py`).
- Testmodus: `pseudo_ensemble` auf Stundenbasis + gleiche Aggregation.

## 10. Umsetzungsreihenfolge mit Prompts für Claude Code

Jeder Schritt endet mit grünem `pytest` und `HEU_SYNTHETIC=1 python scripts/selftest.py`. Nach jedem Schritt committen.

**Schritt 1 – Grundlagen (0,5–1 Tag)**
> Lies `docs/PLAN_schaedlinge.md` Abschnitte 2, 3 und 8.2. Lege `heu/phaenologie.py` mit `tageslaenge_h`, `gradtage`,
> `Szenario`, `szenarien`, `ereignistag`, `EreignisVerteilung`, `kde_kurve`, `tageswahrscheinlichkeit`,
> `phasen_wahrscheinlichkeit` an, ergänze in `heu/config.py` die drei Annahmen-Dataclasses aus den Abschnitten 4.1, 5
> und 7.2 (Werte exakt übernehmen, Kommentare mit Quelle bzw. VERIFIZIEREN). Ergänze `plots.kde_area_fig`.
> Schreibe `tests/test_phaenologie.py` (Tests T-P1 bis T-P5). Nutze die vorhandenen Hilfen `weather.doy365`,
> `weather.partial_year`, `forecast._scenario_weather` als Vorbild, ändere bestehende Funktionen nicht.

**Schritt 2 – Ensemble-Tageswerte (0,5 Tag)**
> Ergänze `heu/stunden.py` um `load_ensemble_daily` nach Abschnitt 9.1, ohne `HOURLY` und den Stundencache zu ändern,
> und `heu/ui.py` um `ensemble_taeglich`. Synthetischer Pfad über `pseudo_ensemble`. Test T-E1.

**Schritt 3 – PHENIPS (1–1,5 Tage)**
> Implementiere `heu/borkenkaefer.py` nach Abschnitt 4.1–4.3 (Rindentemperatur, teff, Schwärmbeginn, Generationen,
> Geschwisterbruten, Diapause, Mortalität 31.10.). Strahlung aus ERA5 in MJ/m² mit Faktor 277,78 in Wh/m² umrechnen,
> Faktor als Konstante `MJ_ZU_WH`. Tests T-B1 bis T-B6.

**Schritt 4 – Wasserbilanz Wald (0,5 Tag)**
> Implementiere `heu/wald_wasser.py` nach Abschnitt 5 (vektorisiert über Szenarien, wo möglich). Tests T-W1 bis T-W3.

**Schritt 5 – Buchsbaumzünsler (1 Tag)**
> Implementiere `heu/zuensler.py` nach Abschnitt 7.2–7.4 inklusive Spritzfenster-Bewertung aus Ensemble-Tageswerten.
> Tests T-Z1 bis T-Z5.

**Schritt 6 – Aktionsregeln und Kalender (1 Tag)**
> Implementiere `heu/kalender.py` (Abschnitt 8.1) und die Funktionen `aktionen_borkenkaefer` (4.4) und
> `aktionen_zuensler` (7.4). Tests T-K1 bis T-K3.

**Schritt 7 – Seite (1–1,5 Tage)**
> Lege `pages/4_Schädlinge.py` nach Abschnitt 9 an, Muster `pages/2_Tiergesundheit.py`. Erst ohne Satellit.
> Ergänze `scripts/selftest.py` um einen Abschnitt „Schädlinge“, der Historie, aktuelles Jahr, KDE-Kurven und
> Aktionstabellen einmal durchrechnet und Laufzeiten ausgibt. Ergänze README (Seitentabelle).

**Schritt 8 – Sentinel-2 Waldstück (1–1,5 Tage)**
> Implementiere `heu/wald_satellit.py` nach Abschnitt 6 (Statistical API, Anomalie; Process-API-Karte als eigener,
> optionaler Teil), GeoJSON-Upload im Tab, Synthetik `synthetic_wald_s2`. Bestehende Signaturen in `satellite.py`
> dürfen nur abwärtskompatibel erweitert werden (Standardwerte). Tests T-S1 bis T-S3.

**Schritt 9 – Echtlauf auf dem Mac und Plausibilisierung (0,5–1 Tag)**
> Starte die App ohne Testmodus, prüfe Schwärmbeginn 1991–2025 gegen LWF-Angaben/eigene Beobachtungen, prüfe
> Rindentemperatur-Maxima (sollten im Sommer besonnt 35–50 °C, beschattet ≈ Lufttemperatur sein), lade das Waldstück.

Gesamt: etwa 8–10 Arbeitstage (inkl. Abschnitt 15).

## 11. Tests (pytest, ohne Netz)

Testdaten: synthetische 365-Tage-Arrays, in `tests/conftest.py` als Fixtures (`konstant(t, rad, rain)`,
`sinusjahr(mittel=8.5, amplitude=10)`), plus `HEU_SYNTHETIC=1`.

| ID | Prüft |
| --- | --- |
| T-P1 | `tageslaenge_h`: 48,09 °N, 21.6. ≈ 16,0 h (±0,1), 7.8. ≈ 14,5 h (±0,1), 26.8. ≈ 13,5 h (±0,1), 21.12. ≈ 8,3 h (±0,15) |
| T-P2 | `gradtage`: Basis 10, t=[5,10,15] → [0,0,5] |
| T-P3 | `szenarien`: Länge 365 je Szenario; Tage ≤ Stichtag identisch mit Ist; im Rückblick-Modus fehlt das Jahr selbst |
| T-P4 | `kde_kurve`: Integral über x ≈ p (±2 %), konstanter Input erzeugt keine NaN |
| T-P5 | `tageswahrscheinlichkeit`: Ereignis in allen Szenarien an Tag 100 → p[100]=1, sonst 0 |
| T-E1 | `load_ensemble_daily` (synthetisch): je Member genau `days` Tage, rad in MJ/m² im Bereich 0–35 |
| T-B1 | `teff`: bt_mean < 8,3 → 0; bt_mean > 38,9 → 0; bt_mean = 20, bt_max = 25 → 11,7 |
| T-B2 | Abzug über Optimum: bt_max = 40 → bt_diff = (−310,667 + 9,603·40)/24 = 3,061; teff = bt_mean − 3,061 − 8,3 |
| T-B3 | Rindentemperatur: tmax=25, rad=6945 Wh → bt_max = 1,656 + 20,52 + 13,35 + 11,78 = 47,3 (±0,1). **Optional Golden-Test**: CSV mit barrks-Referenzlauf (`tests/data/barrks_phenips_ref.csv`, einmalig in R erzeugt) |
| T-B4 | Konstantwetter tmax=20, tmean=15, rad=0, beschattet: Schwärmbeginn = erster Tag ab 1.4. mit Σ(tmax−8,3) ≥ 140 → 12. Tag (11,7·12 = 140,4) = 12.4.; teff = 6,7/Tag → Gen 1 fertig nach ceil(557/6,7) = 84 Tagen |
| T-B5 | Diapause: keine Brut startet nach dem `diapause_tag`; Geschwisterbrut startet erst bei dev > 0,5 |
| T-B6 | Mortalität: Brut mit dev 0,4 am 31.10. hat `stirbt=True` |
| T-W1 | Massenerhaltung: Σ rain = Σ I + Σ ETa + Σ Versickerung + ΔW (±0,01 mm) |
| T-W2 | Kein Regen, et0 = 4: REW fällt monoton, unter 0,4 verlangsamt (Granier-Reduktion) |
| T-W3 | tmean < 5: ETa = 0 |
| T-Z1 | Konstant tmean = 20 (GT 10/Tag): Reaktivierung am 6. Tag (53 GT), Falter Überwinterer bei 303 GT = Tag 31 |
| T-Z2 | Generationenabstand = gt_generation/10 Tage bei konstant 20 °C |
| T-Z3 | Nach Diapause-Tag keine weitere Eiablage-Generation |
| T-Z4 | Spritzfenster: Junglarven aktiv, tmax 18, Regen 0/0 → „spritzen“; Regen am Folgetag 3 mm → nicht |
| T-Z5 | Kühles Jahr (Mittel 7 °C): höchstens 1 Generation; Sinusjahr Mittel 10 °C: 2 Generationen |
| T-K1 | Jede Zeile von `aktionen_*` hat genau einen `haupt`-Schlüssel aus dem Katalog |
| T-K2 | `kalender_fig` hat 12 × 31 Zellen, 31.4./30.2. leer |
| T-K3 | Rückblick-Stichtag: Aktionen vor dem Stichtag basieren nur auf Ist-Wetter (p ∈ {0,1}) |
| T-S1 | Evalscript enthält B03, B04, B05, B08, B8A, B11, SCL; `_stats_body` alte Aufrufe unverändert |
| T-S2 | `anomalie` auf synthetischen Daten mit Befallsnest 2024: Ampel „Verdacht“ zwischen Juli und September 2024, sonst nicht |
| T-S3 | GeoJSON-Validierung: MultiPolygon akzeptiert, Fläche 0 → Fehler, Koordinaten außerhalb WGS84 → Fehler |

## 12. Leistung

- PHENIPS und Zünsler sind reine NumPy-Schleifen über 365 Tage; 35 Historienjahre + 175 Szenarien ≈ 210 Läufe,
  Ziel < 2 s. Den Generationenteil als Schleife über Tage lassen, die teff-/GT-Berechnung vektorisieren.
- Wasserbilanz über alle Szenarien als Matrix (Szenario × Tag) in einer Tagesschleife.
- Historie einmal je Annahmen-Satz cachen (`st.cache_resource`), aktuelles Jahr mit TTL 3 h.

## 15. Anpassungen für den Brunnhofwald (Angaben von Hubert, 06.11.2026)

Waldstück „Brunnhofwald“, 20,6 ha (`data/wald.geojson`), **Mischwald mit ca. 30 % Fichte**, **keine Pheromonfallen**.
Daraus folgen vier Änderungen gegenüber den Abschnitten 4–6. Claude Code soll sie in den jeweiligen Schritten mit umsetzen.

### 15.1 Satellit: Fichten-Maske statt ganzer Bestand (Schritt 8, +0,5–1 Tag)

Bei 70 % Laubholz bestimmt das Laubholz den Median von NDVI/NDMI. Ein Befallsnest in der Fichte geht darin unter, und der
Laubaustrieb im Mai bzw. die Herbstfärbung im Oktober überlagern jedes Signal. Deshalb wird **nur über Nadelholzpixel**
ausgewertet.

```python
def nadelholz_maske(stueck: dict, client: CDSE, winter: tuple[str, str] = ("01-15", "03-31"),
                    jahre: tuple[int, int] = (2019, 2025), ndvi_schwelle: float = 0.45,
                    min_flaeche_m2: float = 400.0) -> dict    # GeoJSON-MultiPolygon
def lade_oder_erzeuge_maske(stueck: dict, client: CDSE | None) -> dict | None   # Cache data/wald_<id>_nadelholz.geojson
```
1. **Process API** (`/api/v1/process`), Sentinel-2 L2A, Zeitraum laublose Monate mehrerer Jahre, `mosaicking: "ORBIT"`,
   Evalscript: je Pixel Median-NDVI über alle schneefreien, wolkenfreien Szenen (SCL 4/5; Schnee SCL 11 ausschließen).
   Ausgabe GeoTIFF FLOAT32, 10 m, Polygon des Waldstücks als Geometrie.
2. Schwelle: Pixel mit Winter-NDVI ≥ 0,45 gelten als immergrünes Nadelholz (Laubholz ohne Blätter liegt typischerweise
   deutlich darunter). [VERIFIZIEREN] Die Schwelle an Huberts Ortskenntnis prüfen: Maske im UI über dem Luftbild anzeigen
   und den Anteil ausgeben. Er sollte bei etwa 30 % liegen. Einen Regler für die Schwelle anbieten.
3. Vektorisieren mit `rasterio.features.shapes` → `shapely` (kleine Flächen < 400 m² verwerfen, 5 m Innenpuffer).
   Die Maske als `data/wald_brunnhofwald_nadelholz.geojson` speichern. Hubert kann sie in BayernAtlas/QGIS korrigieren.
   Eine vorhandene Datei hat Vorrang, die Maske wird nicht automatisch neu gerechnet.
4. `update_wald()` erhält den Parameter `teilflaeche: Literal["gesamt", "nadelholz", "laubholz"]` und rechnet die
   Statistik für die gewählte Geometrie. Zwei Caches: `sat_wald_<id>_nadelholz.csv`, `sat_wald_<id>_gesamt.csv`.
   Laubholz nur als Vergleich, um Witterungseffekte (Trockenheit trifft beide) von Fichten-spezifischen Abfällen
   (Befall trifft nur Fichte) zu unterscheiden: **Differenz-Index** `ndmi_nadel_q10 − ndmi_laub_q50`, saisonal normiert.
5. Fallback ohne Process API: Copernicus HRL „Dominant Leaf Type“ (10 m, 2018) bzw. Huberts eigene Fichtenflächen als
   GeoJSON. [VERIFIZIEREN] Verfügbarkeit/Download der HRL-Kachel.
6. Hinweis im UI: Bei kleinen, verstreuten Fichtengruppen sind viele Pixel Mischpixel. Befall einzelner Bäume ist kaum
   erkennbar, verlässlich erkennbar sind erst Nester ab etwa 0,1–0,2 ha (ungefähr 10–20 Pixel).

Neue Tests: **T-S4** Maskenanteil bei synthetischem Raster (30 % Pixel über der Schwelle) = 0,30 ± 0,02.
**T-S5** `teilflaeche="nadelholz"` ohne Maske löst eine verständliche Fehlermeldung aus.

### 15.2 Wasserbilanz: Mischbestand (Schritt 4)

Der Bodenwasserspeicher wird von Fichte und Laubholz gemeinsam geleert. Die Fichte wurzelt flacher, deshalb wird der
Stress für die **Fichte** bewertet, die Verdunstung aber für den **Bestand** gerechnet:
```
kc_bestand(Tag) = anteil_fichte · kc_fichte + (1 − anteil_fichte) · kc_laub(Tag)
kc_laub(Tag)    = kc_laub_winter (Nov.–April), linearer Übergang über 3 Wochen zu kc_fichte (Mai–Sept.) [ANNAHME]
Interzeption    = anteil_fichte · Fichte-Werte + (1 − anteil_fichte) · halbe Werte im Winterhalbjahr
```
Ausgabe zusätzlich `rew_fichte_wurzelraum` mit eigener nFK `nfk_fichte_mm = 0,7 · nfk_mm` [ANNAHME] aus derselben
Wasserbilanz (oberer Speicher). **T-W4**: anteil_fichte = 1 reproduziert die reine Fichtenrechnung exakt.

### 15.3 Borkenkäfer-Aktionen ohne Fallen (Schritt 6)

- `BorkenkaeferAnnahmen.pheromonfallen: bool = False`. Die Aktion „Fallen aufstellen“ wird zu
  **„Kontrollsaison beginnt“**, „Fangsysteme leeren“ ist ausgeblendet.
- Ohne eigene Fänge ist **PHENIPS allein der Zeitgeber**. Die Kalibrierung (4.5) läuft deshalb über externe Daten: die
  LWF-Monitoringfallen der nächstgelegenen Station und eigene **Befallsbeobachtungen** (Bohrmehl, Harztropfen) in der
  Beobachtungs-CSV (`typ=befall`). `kalibriere_onset` akzeptiert beide Quellen und vermerkt die Herkunft.
- Bohrmehlkontrolle konzentriert sich auf die **Fichten-Teilflächen aus der Maske** (15.1): Im UI eine kleine Karte mit
  Fichtenflächen und den Bereichen mit Satelliten-Anomalie anzeigen, als Begehungsliste „wo zuerst schauen“.
- Risikohinweis: Im Mischbestand mit 30 % Fichte ist das Massenvermehrungsrisiko geringer als im Reinbestand. Gefährdet
  sind vor allem besonnte Fichtenränder und Fichtengruppen nach Lücken. Deshalb ist **„besonnt“ Standard** für die
  Rindentemperatur, weil der Ränder-Fall die frühesten Termine liefert.

### 15.5 Buchsbaumstandort (Hubert, 06.11.2026)

Standort: UTM 33U 325349 E / 5328853 N = **48,0890 °N, 12,6546 °E** (WGS84), direkt an der Hofstelle bei der Wiese Ost.
Gespeichert als Punkt in `data/buchsbaum.geojson`. Er liegt in derselben ERA5-Zelle wie `STANDORT`, daher ist kein
eigener Wetterabruf nötig. `zuensler()` bekommt `lat` aus dieser Datei, wegen der Tageslänge. Das macht bei 0,004°
Unterschied praktisch nichts aus, ist aber sauber.

**Alter Bestand** (große, dichte Pflanzen):
- Die Raupen überwintern in Gespinsten im dichten Inneren, und dort sitzen auch die Junglarven. Deshalb lautet die
  Aktion **„Kontrolle“** im UI-Hinweistext so: „Zweige im Inneren auseinanderbiegen, nach Gespinsten und grünen
  Kotkrümeln suchen“. Eine Kontrolle nur der Außenseite reicht nicht.
- Bt wirkt nur als Fraßgift. Bei dichten, alten Pflanzen muss die Spritzbrühe ins Innere gelangen, also mit viel Wasser
  und von mehreren Seiten spritzen. Das kommt als Hinweistext an die Aktion „Spritzen Bt“.
- Neuer Schalter `ZuenslerAnnahmen.alter_dichter_bestand: bool = True`. Ist er aktiv, wird die Kontrolle in den
  Junglarvenphasen alle 3–4 Tage statt wöchentlich markiert, weil Befall im Inneren lange unbemerkt bleibt. [ANNAHME]

### 15.4 Aufwand

Gesamt jetzt etwa **8–10 Arbeitstage** (+0,5–1 Tag für die Nadelholzmaske).

## 13. Offene Punkte und Fragen an Hubert

1. **Waldstück**: ERLEDIGT 06.11.2026 – „Brunnhofwald“, 20,6 ha, in `data/wald.geojson`. Mischwald, ca. 30 % Fichte (siehe Abschnitt 15). Offen: Bestandesalter,
   besonnte Ränder (Süd-/Westrand nach Sturm/Hieb)?
2. **Buchsbaum**: ERLEDIGT – 48,0890 °N 12,6546 °E, alter Bestand (siehe 15.5).
3. **Fangsysteme**: ERLEDIGT – keine Pheromonfallen (siehe 15.3).
4. **Literatur besorgen**: Nacambo et al. 2014 Volltext (Entwicklungsschwellen, Gradtage je Stadium);
   Baier et al. 2007 Volltext (Einheit der Strahlung, Tageslängendefinition); Granier et al. 1999 (REW-Schwelle).
5. **Boden im Wald**: nFK des Waldstandorts aus der Standortkarte (Bayerische Forstliche Standortskartierung bzw.
   Übersichtsbodenkarte); bis dahin 120 mm als Annahme.

## 14. Quellen (Stand dieser Prüfung)

- Baier P., Pennerstorfer J., Schopf A. (2007): PHENIPS – A comprehensive phenology model of *Ips typographus*
  (L.) (Col., Scolytinae) as a tool for hazard rating of bark beetle infestation. *Forest Ecology and Management*
  249:171–186. **Parameter abgelesen aus** der R-Implementierung barrks, `R/model-phenips.R`
  (https://rdrr.io/cran/barrks/src/R/model-phenips.R).
- Wermelinger B., Seifert M. (1998): Analysis of the temperature dependent development of the spruce bark beetle
  *Ips typographus*. *J. Appl. Entomol.* 122:185–191 (Entwicklungsfunktion, Optimumabzug).
- Nacambo S., Leuthardt F.L.G., Wan H., Li H., Haye T., Baur B., Weiss R.M., Kenis M. (2014): Development
  characteristics of the box-tree moth *Cydalima perspectalis* and its potential distribution in Europe.
  *J. Appl. Entomol.* 138:14–26. Abstract geprüft über
  https://wiki.pestinfo.org/wiki/Journal_of_Applied_Entomology_(2014)_138,_14-26
- Lemić D., Pajač Živković I., Šuliček M., Virić Gašparić H. (2025): Lifecycle insights as a prerequisite for targeted
  management of *Cydalima perspectalis*. *J. Central European Agriculture* 26(1):150–161,
  https://hrcak.srce.hr/en/329498
- Granier A., Bréda N., Biron P., Villette S. (1999): A lumped water balance model to evaluate duration and intensity
  of drought constraints in forest stands. *Ecological Modelling* 116:269–283. [Werte VERIFIZIEREN]
- Gao B.-C. (1996): NDWI. *Remote Sensing of Environment* 58:257–266.
- Galvão L.S., Formaggio A.R., Tisot D.A. (2005): Discrimination of sugarcane varieties … (DSWI).
  *Remote Sensing of Environment* 94:523–534. [Zitat VERIFIZIEREN]
- Allen R.G. et al. (1998): FAO Irrigation and Drainage Paper 56 (Tageslänge, ET0).
