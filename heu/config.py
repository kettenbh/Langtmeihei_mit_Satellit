"""Generische Annahmen für das MVP.

Alle Werte hier sind bewusst generisch (ein Feld, 2 ha, Kirchweidach, Lkr. Altötting)
und werden später durch betriebsspezifische Daten ersetzt.
"""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


@dataclass(frozen=True)
class Standort:
    name: str = "Kirchweidach (Lkr. Altötting)"
    lat: float = 48.085
    lon: float = 12.645
    hoehe_m: int = 470
    flaeche_ha: float = 2.0


@dataclass(frozen=True)
class Annahmen:
    # --- Zeiträume ---------------------------------------------------------
    trainings_start: int = 1991          # erstes Jahr der Klimatologie
    trainings_ende: int = 2025           # letztes Trainingsjahr (vor 2026)
    simulations_jahr: int = 2026

    # --- Ertragsniveau (Kalibrierziel) --------------------------------------
    # Mittlerer Brutto-Jahresertrag (vor Feldverlusten) eines Bio-Dauergrünlands
    # mit 3-4 Schnitten in Südostbayern. ANNAHME, bitte durch LfL-/Betriebswerte ersetzen.
    ziel_jahresertrag_dt_tm_ha: float = 75.0

    # --- Boden ---------------------------------------------------------------
    nutzbare_feldkapazitaet_mm: float = 130.0   # Lehm/Schluff, Moränen-/Tertiärhügelland

    # --- Schnittregel --------------------------------------------------------
    tsumme_erster_schnitt: float = 750.0   # °Cd ab 1.1. (Tmittel > 0), Schnittreife 1. Aufwuchs
    min_tage_zwischen_schnitten: int = 38
    max_tage_zwischen_schnitten: int = 60
    min_ertrag_kg_tm_ha: float = 1400.0    # erntewürdige Masse über Schnitthöhe
    schnitthoehe_m: float = 0.06
    fruehester_schnitt: str = "04-20"      # MM-DD
    spaetester_schnitt: str = "10-15"
    max_schnitte: int = 5
    max_wartetage: int = 10                # so lange wird auf Heuwetter gewartet

    # --- Heuwetter -----------------------------------------------------------
    trockentage_noetig: int = 3            # Bodenheu; mit Belüftungstrocknung 2
    regen_grenze_mm: float = 1.0           # Tag gilt als trocken unter diesem Wert
    min_et0_mm: float = 2.5                # mittlere Verdunstung im Fenster

    # --- Verluste & Umrechnung -----------------------------------------------
    feldverlust_basis: float = 0.15        # Bröckel-/Atmungsverluste Bodenheu
    feldverlust_je_regentag: float = 0.08  # Zusatzverlust je Regentag im Trocknungsfenster
    tm_gehalt_heu: float = 0.86
    ballengewicht_kg: float = 250.0        # Rundballen Heu, Frischmasse (ANNAHME)

    # --- Unsicherheit --------------------------------------------------------
    modellfehler_sigma: float = 0.10       # log-normaler Strukturfehler je künftigem Schnitt (ANNAHME)
    ziehungen_je_szenario: int = 20

    # --- Futterqualität (Modul 3) ------------------------------------------
    rohasche_g_kg: float = 95.0            # Rohasche XA [g/kg TM] (ANNAHME, aus Futteranalysen ersetzen)
    kleeanteil: float = 0.15               # Leguminosenanteil an der TM (ANNAHME Bio-Dauergrünland)
    n_leguminosen_pct: float = 3.6         # N-Gehalt Klee [% TM] (≈ 225 g XP/kg TM, ANNAHME)
    n_totmasse_pct: float = 1.0            # N-Gehalt abgestorbener Masse [% TM] (ANNAHME)
    belueftung: bool = False               # Heutrocknung mit Belüftung (weniger Bröckelverluste)


STANDORT = Standort()
ANNAHMEN = Annahmen()


@dataclass(frozen=True)
class HitzeAnnahmen:
    """Modul 1: Schwellen nach Zimbelman et al. 2009 (68) und Armstrong 1994 (72/80/90)."""
    schwelle_beginn: float = 68.0
    schwelle_leicht: float = 72.0
    schwelle_mittel: float = 80.0
    schwelle_schwer: float = 90.0
    erholung_thi: float = 65.0             # Nachtstunden unter diesem THI gelten als Erholung
    erholung_stunden: int = 6
    milchverlust_kg_je_thi: float = 0.3    # kg Milch/Kuh/Tag je THI-Punkt des Tagesmittels über 68 (Spanne 0,2–0,4; West 2003, Bouraoui 2002)
    kuehe: int = 50


@dataclass(frozen=True)
class ParasitenAnnahmen:
    """Modul 2: Literaturparameter (siehe heu/parasiten.py für Quellen)."""
    # Leberegel
    oll_regentag_mm: float = 0.25          # Regentag ab 0,01 Zoll (Ollerenshaw)
    oll_monat_max: float = 100.0           # Monatswert wird gedeckelt
    wbgdd_basis_c: float = 10.0            # Malone et al. 1998
    wbgdd_fenster_tage: int = 30
    # Weidefliegen
    # Hornfliege: aus "Lebenszyklus 10–20 Tage je nach Temperatur" (Fitzpatrick & Kaufman, EDIS EENY490)
    # und "Schlupf erst bei Fladentemperatur > 15 °C" (Lysyk & Moon 2001) linear abgeleitet:
    # 20 Tage bei 20 °C, 12 Tage bei 25 °C -> Basis 12,5 °C, 150 Gradtage. ABGELEITET, nicht aus Originaltabelle.
    hornfliege_basis_c: float = 12.5
    hornfliege_gt_generation: float = 150.0
    kopffliege_flug_von: str = "06-01"     # Hydrotaea irritans: eine Generation, Flugzeit Juni–September
    kopffliege_flug_bis: str = "09-30"
    # MDS
    weide_austrieb: str = "05-01"
    weide_abtrieb: str = "10-15"


HITZE = HitzeAnnahmen()
PARASITEN = ParasitenAnnahmen()
