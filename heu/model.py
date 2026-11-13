"""Graswachstumsmodell nach ModVege (Jouven, Carrère & Baumont 2006, Grass Forage Sci. 61:112-124)
mit einfacher Bodenwasserbilanz, regelbasierter Schnittentscheidung und Heuwetter-Logik.

Vier Kompartimente: grün vegetativ (GV), grün reproduktiv (GR), tot vegetativ (DV),
tot reproduktiv (DR); Biomassen in kg TM/ha, Alter in °Cd.
Vereinfachungen gegenüber dem Original sind mit 'MVP:' markiert.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from .config import Annahmen
from .weather import doy365
import datetime as _dt


# ---------------------------------------------------------------------------
# Parameter (Standardwerte aus Jouven et al. 2006 bzw. Calanca et al. 2016)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModVegeParams:
    NI: float = 0.7            # Nährstoffindex (wird kalibriert)
    WHC: float = 130.0         # nutzbare Feldkapazität [mm]
    SLA: float = 0.033         # spezifische Blattfläche [m²/g]
    pLAM: float = 0.62         # Blattanteil an GV
    RUEmax: float = 3.0        # Strahlungsnutzungseffizienz [g TM/MJ PAR]
    T0: float = 4.0
    T1: float = 10.0
    T2: float = 20.0
    T3: float = 40.0
    ST1: float = 600.0         # Beginn reproduktive Phase [°Cd]
    ST2: float = 1200.0        # Ende reproduktive Phase [°Cd]
    minSEA: float = 0.8
    maxSEA: float = 1.2
    LLS: float = 500.0         # Blattlebensdauer [°Cd]
    KGV: float = 0.002
    KGR: float = 0.001
    KlDV: float = 0.001
    KlDR: float = 0.0005
    sigGV: float = 0.4
    sigGR: float = 0.2
    BDGV: float = 850.0        # Lagerungsdichten [g TM/m³]
    BDGR: float = 300.0
    BDDV: float = 500.0
    BDDR: float = 150.0
    maxOMDGV: float = 0.90
    minOMDGV: float = 0.75
    maxOMDGR: float = 0.90
    minOMDGR: float = 0.65
    OMDDV: float = 0.45
    OMDDR: float = 0.40


@dataclass
class State:
    GV: float = 400.0
    GR: float = 0.0
    DV: float = 600.0
    DR: float = 0.0
    aGV: float = 200.0
    aGR: float = 0.0
    aDV: float = 300.0
    aDR: float = 0.0
    WR: float = 130.0
    ST: float = 0.0
    repro_done: bool = False
    n_cuts: int = 0
    last_cut: int = -999
    ready_since: int = -1
    cuts: list = field(default_factory=list)

    def copy(self) -> "State":
        s = replace(self)
        s.cuts = list(self.cuts)
        return s


@dataclass(frozen=True)
class Cut:
    day: int                 # 0-basierter Tag im 365-Tage-Jahr
    brutto_kg: float         # geerntete Masse über Schnitthöhe [kg TM/ha]
    verlust: float           # Feldverlustanteil
    netto_kg: float          # eingefahrenes Heu [kg TM/ha]
    omd: float               # Verdaulichkeit der organischen Masse
    regentage: int           # Regentage im Trocknungsfenster
    erzwungen: bool          # nach Wartezeit ohne volles Heuwetter gemäht
    xp: float = float("nan")       # Rohprotein Grünfutter [g/kg TM] (Modul Futterqualität)
    nel_heu: float = float("nan")  # NEL nach Trocknungsverlusten [MJ/kg TM]
    xp_heu: float = float("nan")   # XP nach Trocknungsverlusten [g/kg TM]
    stadium: str = ""

    @property
    def nel(self) -> float:
        return omd_to_nel(self.omd)


def omd_to_nel(omd: float, xa: float = 95.0) -> float:
    """OMD -> NEL [MJ/kg TM] über DOMD und ME = 0,016·DOMD (MAFF 1975), NEL nach GfE 2001 (siehe heu/qualitaet.py)."""
    from .qualitaet import nel_aus_omd
    return float(nel_aus_omd(omd, xa))


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------
def _f_age(age: float, lls: float) -> float:
    r = age / lls
    if r < 1 / 3:
        return 1.0
    if r < 1:
        return 3.0 * r
    return 3.0


def _f_temp(t: float, p: ModVegeParams) -> float:
    if t <= p.T0 or t >= p.T3:
        return 0.0
    if t < p.T1:
        return (t - p.T0) / (p.T1 - p.T0)
    if t <= p.T2:
        return 1.0
    return (p.T3 - t) / (p.T3 - p.T2)


def _f_par(par: float) -> float:
    return 1.0 if par < 5.0 else max(0.0, 1.0 - 0.0445 * (par - 5.0))


def _f_water(W: float, pet: float) -> float:
    # MVP: vereinfachte Form der ModVege-Wasserstressfunktion; Schwelle steigt mit PET.
    crit = min(0.7, max(0.3, 0.2 + 0.08 * pet))
    return min(1.0, W / crit)


def _sea(st: float, p: ModVegeParams) -> float:
    if st < 200:
        return p.minSEA
    if st < p.ST1 - 200:
        return p.minSEA + (p.maxSEA - p.minSEA) * (st - 200) / (p.ST1 - 400)
    if st < p.ST1 - 100:
        return p.maxSEA
    if st < p.ST2:
        return p.maxSEA - (p.maxSEA - p.minSEA) * (st - p.ST1 + 100) / (p.ST2 - p.ST1 + 100)
    return p.minSEA


def harvestable(s: State, p: ModVegeParams, h: float) -> tuple[float, float, tuple]:
    """Masse über Schnitthöhe h [m] und deren OMD."""
    rGV, rGR, rDV, rDR = h * p.BDGV * 10, h * p.BDGR * 10, h * p.BDDV * 10, h * p.BDDR * 10
    hGV, hGR = max(0.0, s.GV - rGV), max(0.0, s.GR - rGR)
    hDV, hDR = max(0.0, s.DV - rDV), max(0.0, s.DR - rDR)
    tot = hGV + hGR + hDV + hDR
    if tot <= 0:
        return 0.0, 0.0, (rGV, rGR, rDV, rDR)
    oGV = min(p.maxOMDGV, max(p.minOMDGV, p.maxOMDGV - s.aGV * (p.maxOMDGV - p.minOMDGV) / p.LLS))
    oGR = min(p.maxOMDGR, max(p.minOMDGR, p.maxOMDGR - s.aGR * (p.maxOMDGR - p.minOMDGR) / (p.ST2 - p.ST1)))
    omd = (hGV * oGV + hGR * oGR + hDV * p.OMDDV + hDR * p.OMDDR) / tot
    return tot, omd, (rGV, rGR, rDV, rDR)


# ---------------------------------------------------------------------------
# Tagesschritt
# ---------------------------------------------------------------------------
def grow_one_day(s: State, t: float, rad: float, rain: float, pet: float, p: ModVegeParams) -> float:
    """Ein Tag Wachstum, Seneszenz, Abbau, Wasserbilanz. Gibt Bruttozuwachs zurück."""
    tpos = max(t, 0.0)
    s.ST += tpos
    par = 0.48 * rad
    lai = p.SLA * p.pLAM * s.GV * 0.1
    # Wasserbilanz (MVP: Eimer-Modell, AET skaliert mit LAI)
    aet = min(pet * min(1.0, lai / 3.0), s.WR + rain)
    s.WR = min(p.WHC, max(0.0, s.WR + rain - aet))
    W = s.WR / p.WHC

    pgro = par * p.RUEmax * (1.0 - math.exp(-0.6 * lai)) * 10.0
    env = p.NI * _f_temp(t, p) * _f_par(par) * _f_water(W, pet)
    gro = pgro * env * _sea(s.ST, p)

    rep = 0.0
    if not s.repro_done and p.ST1 <= s.ST <= p.ST2:
        rep = min(1.0, max(0.0, 0.25 + 0.75 * (p.NI - 0.35) / 0.65))

    if t > p.T0:
        sGV = p.KGV * s.GV * t * _f_age(s.aGV, p.LLS)
        sGR = p.KGR * s.GR * t * _f_age(s.aGR, p.ST2 - p.ST1)
    elif t < 0:
        sGV = p.KGV * s.GV * abs(t)
        sGR = p.KGR * s.GR * abs(t)
    else:
        sGV = sGR = 0.0
    if t > 0:
        aDV = p.KlDV * s.DV * t * _f_age(s.aDV, p.LLS)
        aDR = p.KlDR * s.DR * t * _f_age(s.aDR, p.ST2 - p.ST1)
    else:
        aDV = aDR = 0.0
    sGV, sGR, aDV, aDR = min(sGV, s.GV), min(sGR, s.GR), min(aDV, s.DV), min(aDR, s.DR)

    gGV, gGR = gro * (1 - rep), gro * rep
    newGV = s.GV - sGV + gGV
    newGR = s.GR - sGR + gGR
    inDV, inDR = (1 - p.sigGV) * sGV, (1 - p.sigGR) * sGR
    newDV = s.DV - aDV + inDV
    newDR = s.DR - aDR + inDR

    s.aGV = (s.GV - sGV) / newGV * (s.aGV + tpos) if newGV > 1e-9 else 0.0
    s.aGR = (s.GR - sGR) / newGR * (s.aGR + tpos) if newGR > 1e-9 else 0.0
    s.aDV = ((s.DV - aDV) * (s.aDV + tpos) + inDV * s.aGV) / newDV if newDV > 1e-9 else 0.0
    s.aDR = ((s.DR - aDR) * (s.aDR + tpos) + inDR * s.aGR) / newDR if newDR > 1e-9 else 0.0
    s.GV, s.GR, s.DV, s.DR = newGV, newGR, newDV, newDR
    return gro


# ---------------------------------------------------------------------------
# Schnittentscheidung
# ---------------------------------------------------------------------------
def dry_window(w: dict, d: int, a: Annahmen) -> tuple[bool, int]:
    n = a.trockentage_noetig
    end = min(d + n, len(w["rain"]))
    rain = w["rain"][d:end]
    et0 = w["et0"][d:end]
    if len(rain) < n:
        return False, n
    regentage = int((rain >= a.regen_grenze_mm).sum())
    ok = regentage == 0 and float(et0.mean()) >= a.min_et0_mm
    return ok, regentage


def is_ready(s: State, d: int, H: float, a: Annahmen) -> bool:
    if s.n_cuts >= a.max_schnitte:
        return False
    if s.n_cuts == 0:
        return s.ST >= a.tsumme_erster_schnitt and H >= a.min_ertrag_kg_tm_ha
    since = d - s.last_cut
    if since >= a.min_tage_zwischen_schnitten and H >= a.min_ertrag_kg_tm_ha:
        return True
    return since >= a.max_tage_zwischen_schnitten and H >= 0.6 * a.min_ertrag_kg_tm_ha


def _md(s: str) -> int:
    m, dd = s.split("-")
    return doy365(_dt.date(2001, int(m), int(dd)))


def do_cut(s: State, d: int, p: ModVegeParams, a: Annahmen, regentage: int, forced: bool) -> Cut:
    H, omd, (rGV, rGR, rDV, rDR) = harvestable(s, p, a.schnitthoehe_m)
    from . import qualitaet as Q
    xp = Q.xp_bestand(s, p, a)
    nel = float(Q.nel_aus_omd(omd, a.rohasche_g_kg))
    nel_h, xp_h = Q.heu_abschlag(nel, xp, regentage, a.belueftung)
    stad = Q.stadium(s.ST, p.ST1, p.ST2, s.n_cuts + 1, None if s.n_cuts == 0 else d - s.last_cut)
    s.GV, s.GR, s.DV, s.DR = min(s.GV, rGV), min(s.GR, rGR), min(s.DV, rDV), min(s.DR, rDR)
    s.repro_done = True
    s.n_cuts += 1
    s.last_cut = d
    s.ready_since = -1
    loss = min(0.6, a.feldverlust_basis + a.feldverlust_je_regentag * regentage)
    c = Cut(d, H, loss, H * (1 - loss), omd, regentage, forced, xp, float(nel_h), float(xp_h), stad)
    s.cuts.append(c)
    return c


def run(w: dict, p: ModVegeParams, a: Annahmen, state: State | None = None,
        start: int = 0, end: int | None = None, trace: list | None = None,
        manage: bool = True) -> State:
    """Simuliert Tage start..end-1 eines 365-Tage-Jahres mit Wetter w (Arrays)."""
    s = state if state is not None else State(WR=p.WHC)
    end = len(w["tmean"]) if end is None else end
    first, last = _md(a.fruehester_schnitt), _md(a.spaetester_schnitt)
    for d in range(start, end):
        if manage and first <= d <= last:
            H, omd, _ = harvestable(s, p, a.schnitthoehe_m)
            if is_ready(s, d, H, a):
                if s.ready_since < 0:
                    s.ready_since = d
                ok, rt = dry_window(w, d, a)
                if ok:
                    do_cut(s, d, p, a, 0, False)
                elif d - s.ready_since >= a.max_wartetage and w["rain"][d] < a.regen_grenze_mm:
                    do_cut(s, d, p, a, rt, True)
            else:
                s.ready_since = -1
        grow_one_day(s, w["tmean"][d], w["rad"][d], w["rain"][d], w["et0"][d], p)
        if trace is not None:
            H, omd, _ = harvestable(s, p, a.schnitthoehe_m)
            trace.append((d, s.GV + s.GR + s.DV + s.DR, H, omd, s.WR / p.WHC, s.ST, s.n_cuts,
                          p.SLA * p.pLAM * s.GV * 0.1))
    return s


def season_total(s: State, netto: bool = True) -> float:
    return sum(c.netto_kg if netto else c.brutto_kg for c in s.cuts)
