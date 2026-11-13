"""ModVege als Ensemble: viele Modelläufe gleichzeitig (NumPy-vektorisiert).

Paket 1 des Plans docs/PLAN_hybridmodell_assimilation.md.

Mathematisch identisch zu heu/model.py (grow_one_day, harvestable, Schnittregel in run),
aber jede Zustandsgröße ist ein Array der Länge N ("Mitglieder"). Jedes Mitglied kann
  - eigene Parameter haben (z. B. NI, RUEmax, WHC aus einer Prior- oder Posterior-Stichprobe),
  - eigenes Wetter haben (z. B. anderes Klimajahr oder gestörtes Wetter für die Assimilation),
  - eigene beobachtete Schnitttage haben (Analysephase, Abschnitt 3.4 des Plans).

Die Skalarversion in heu/model.py bleibt unverändert und ist die Referenz:
tests/test_ensemble.py prüft, dass beide Versionen dieselben Ergebnisse liefern.

Nicht enthalten (bewusst): Futterqualität je Schnitt (XP, NEL nach Trocknung, Stadium).
Die Felder bleiben in `to_states()` leer (nan). Für Qualitätsauswertungen weiter heu/model.py nutzen.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from .config import Annahmen
from .model import Cut, ModVegeParams, State, _md

_PARAM_NAMES = tuple(f.name for f in fields(ModVegeParams))
_STATE_FLOATS = ("GV", "GR", "DV", "DR", "aGV", "aGR", "aDV", "aDR", "WR", "ST")


# ---------------------------------------------------------------------------
# Parameter je Mitglied
# ---------------------------------------------------------------------------
class EnsembleParams:
    """Alle ModVege-Parameter als Arrays der Länge n (Attributzugriff wie ModVegeParams)."""

    def __init__(self, n: int, values: dict[str, np.ndarray]):
        self.n = n
        for k in _PARAM_NAMES:
            setattr(self, k, np.broadcast_to(np.asarray(values[k], dtype=float), (n,)).copy())

    @classmethod
    def from_params(cls, p: ModVegeParams, n: int, **overrides) -> "EnsembleParams":
        """n Kopien von p; einzelne Parameter per Array überschreiben, z. B. NI=np.array([...])."""
        vals = {k: getattr(p, k) for k in _PARAM_NAMES}
        for k, v in overrides.items():
            if k not in vals:
                raise KeyError(f"unbekannter ModVege-Parameter: {k}")
            vals[k] = v
        return cls(n, vals)

    @classmethod
    def from_list(cls, ps: list[ModVegeParams]) -> "EnsembleParams":
        return cls(len(ps), {k: np.array([getattr(p, k) for p in ps]) for k in _PARAM_NAMES})

    def member(self, i: int) -> ModVegeParams:
        return ModVegeParams(**{k: float(getattr(self, k)[i]) for k in _PARAM_NAMES})

    def take(self, idx) -> "EnsembleParams":
        idx = np.asarray(idx)
        return EnsembleParams(len(idx), {k: getattr(self, k)[idx] for k in _PARAM_NAMES})

    def repeat(self, k: int) -> "EnsembleParams":
        """Jedes Mitglied k-mal hintereinander (für Mitglieder × Wetterjahre)."""
        return EnsembleParams(self.n * k, {n: np.repeat(getattr(self, n), k) for n in _PARAM_NAMES})


# ---------------------------------------------------------------------------
# Zustand je Mitglied
# ---------------------------------------------------------------------------
@dataclass
class EnsembleState:
    GV: np.ndarray
    GR: np.ndarray
    DV: np.ndarray
    DR: np.ndarray
    aGV: np.ndarray
    aGR: np.ndarray
    aDV: np.ndarray
    aDR: np.ndarray
    WR: np.ndarray
    ST: np.ndarray
    repro_done: np.ndarray      # bool
    n_cuts: np.ndarray          # int
    last_cut: np.ndarray        # int
    ready_since: np.ndarray     # int
    # Schnittprotokoll: Spalte j = (j+1)-ter Schnitt; nicht belegte Felder: Tag -1, Werte nan
    cut_day: np.ndarray
    cut_brutto: np.ndarray
    cut_verlust: np.ndarray
    cut_netto: np.ndarray
    cut_omd: np.ndarray
    cut_regentage: np.ndarray
    cut_erzwungen: np.ndarray
    cut_beobachtet: np.ndarray  # Schnitt aus Beobachtung (forced_cuts), nicht aus der Regel

    @property
    def n(self) -> int:
        return len(self.GV)

    @classmethod
    def from_state(cls, s: State, n: int, max_cuts: int = 8) -> "EnsembleState":
        """n identische Kopien eines Skalarzustands (inklusive bereits erfolgter Schnitte)."""
        f = {k: np.full(n, float(getattr(s, k))) for k in _STATE_FLOATS}
        es = cls(
            **f,
            repro_done=np.full(n, bool(s.repro_done)),
            n_cuts=np.full(n, int(s.n_cuts)),
            last_cut=np.full(n, int(s.last_cut)),
            ready_since=np.full(n, int(s.ready_since)),
            **_empty_cuts(n, max(max_cuts, len(s.cuts) + 1)),
        )
        for j, c in enumerate(s.cuts):
            es.cut_day[:, j] = c.day
            es.cut_brutto[:, j] = c.brutto_kg
            es.cut_verlust[:, j] = c.verlust
            es.cut_netto[:, j] = c.netto_kg
            es.cut_omd[:, j] = c.omd
            es.cut_regentage[:, j] = c.regentage
            es.cut_erzwungen[:, j] = c.erzwungen
        return es

    @classmethod
    def initial(cls, p: EnsembleParams, max_cuts: int = 8) -> "EnsembleState":
        """Startzustand wie State(WR=WHC) in heu/model.py, für jedes Mitglied."""
        es = cls.from_state(State(), p.n, max_cuts)
        es.WR = p.WHC.copy()
        return es

    def copy(self) -> "EnsembleState":
        return EnsembleState(**{k: v.copy() for k, v in self.__dict__.items()})

    def take(self, idx) -> "EnsembleState":
        idx = np.asarray(idx)
        return EnsembleState(**{k: v[idx].copy() for k, v in self.__dict__.items()})

    def repeat(self, k: int) -> "EnsembleState":
        return EnsembleState(**{n: np.repeat(v, k, axis=0) for n, v in self.__dict__.items()})

    def season_total(self, netto: bool = True) -> np.ndarray:
        v = self.cut_netto if netto else self.cut_brutto
        return np.nansum(v, axis=1)

    def lai(self, p: EnsembleParams) -> np.ndarray:
        return p.SLA * p.pLAM * self.GV * 0.1

    def to_states(self) -> list[State]:
        """Zurück in Skalarzustände (z. B. für bestehende Auswertungen). Qualitätsfelder bleiben leer."""
        out = []
        for i in range(self.n):
            s = State(**{k: float(getattr(self, k)[i]) for k in _STATE_FLOATS},
                      repro_done=bool(self.repro_done[i]), n_cuts=int(self.n_cuts[i]),
                      last_cut=int(self.last_cut[i]), ready_since=int(self.ready_since[i]))
            for j in range(self.cut_day.shape[1]):
                if self.cut_day[i, j] < 0:
                    break
                s.cuts.append(Cut(int(self.cut_day[i, j]), float(self.cut_brutto[i, j]),
                                  float(self.cut_verlust[i, j]), float(self.cut_netto[i, j]),
                                  float(self.cut_omd[i, j]), int(self.cut_regentage[i, j]),
                                  bool(self.cut_erzwungen[i, j])))
            out.append(s)
        return out


def _empty_cuts(n: int, m: int) -> dict:
    return dict(
        cut_day=np.full((n, m), -1, dtype=int),
        cut_brutto=np.full((n, m), np.nan),
        cut_verlust=np.full((n, m), np.nan),
        cut_netto=np.full((n, m), np.nan),
        cut_omd=np.full((n, m), np.nan),
        cut_regentage=np.zeros((n, m), dtype=int),
        cut_erzwungen=np.zeros((n, m), dtype=bool),
        cut_beobachtet=np.zeros((n, m), dtype=bool),
    )


# ---------------------------------------------------------------------------
# Hilfsfunktionen (vektorisiert, identisch zu heu/model.py)
# ---------------------------------------------------------------------------
def _f_age(age, lls):
    r = age / lls
    return np.where(r < 1 / 3, 1.0, np.where(r < 1, 3.0 * r, 3.0))


def _f_temp(t, p: EnsembleParams):
    rise = (t - p.T0) / (p.T1 - p.T0)
    fall = (p.T3 - t) / (p.T3 - p.T2)
    v = np.where(t < p.T1, rise, np.where(t <= p.T2, 1.0, fall))
    return np.where((t <= p.T0) | (t >= p.T3), 0.0, v)


def _f_par(par):
    return np.where(par < 5.0, 1.0, np.maximum(0.0, 1.0 - 0.0445 * (par - 5.0)))


def _f_water(W, pet):
    crit = np.minimum(0.7, np.maximum(0.3, 0.2 + 0.08 * pet))
    return np.minimum(1.0, W / crit)


def _sea(st, p: EnsembleParams):
    up = p.minSEA + (p.maxSEA - p.minSEA) * (st - 200) / (p.ST1 - 400)
    down = p.maxSEA - (p.maxSEA - p.minSEA) * (st - p.ST1 + 100) / (p.ST2 - p.ST1 + 100)
    return np.where(st < 200, p.minSEA,
           np.where(st < p.ST1 - 200, up,
           np.where(st < p.ST1 - 100, p.maxSEA,
           np.where(st < p.ST2, down, p.minSEA))))


def _safe_div(num, den):
    out = np.zeros_like(num, dtype=float)
    np.divide(num, den, out=out, where=den > 1e-9)
    return out


def harvestable(s: EnsembleState, p: EnsembleParams, h: float):
    """Masse über Schnitthöhe h [m] und deren OMD, je Mitglied. Gibt (H, omd, Reste) zurück."""
    rGV, rGR, rDV, rDR = h * p.BDGV * 10, h * p.BDGR * 10, h * p.BDDV * 10, h * p.BDDR * 10
    hGV, hGR = np.maximum(0.0, s.GV - rGV), np.maximum(0.0, s.GR - rGR)
    hDV, hDR = np.maximum(0.0, s.DV - rDV), np.maximum(0.0, s.DR - rDR)
    tot = hGV + hGR + hDV + hDR
    oGV = np.minimum(p.maxOMDGV, np.maximum(p.minOMDGV, p.maxOMDGV - s.aGV * (p.maxOMDGV - p.minOMDGV) / p.LLS))
    oGR = np.minimum(p.maxOMDGR, np.maximum(p.minOMDGR,
                     p.maxOMDGR - s.aGR * (p.maxOMDGR - p.minOMDGR) / (p.ST2 - p.ST1)))
    num = hGV * oGV + hGR * oGR + hDV * p.OMDDV + hDR * p.OMDDR
    omd = np.zeros_like(tot)
    np.divide(num, tot, out=omd, where=tot > 0)
    return tot, omd, (rGV, rGR, rDV, rDR)


# ---------------------------------------------------------------------------
# Tagesschritt
# ---------------------------------------------------------------------------
def grow_ensemble_day(s: EnsembleState, t, rad, rain, pet, p: EnsembleParams, gro_factor=None) -> np.ndarray:
    """Ein Tag Wachstum, Seneszenz, Abbau, Wasserbilanz für alle Mitglieder.

    Wetter: Skalar (gleich für alle) oder Array der Länge n (je Mitglied).
    gro_factor: optionaler multiplikativer Modellfehler auf den Zuwachs (Array der Länge n), für die
    Assimilation (Plan 3.2). Ohne Angabe identisch zu heu/model.grow_one_day.
    Gibt den Bruttozuwachs je Mitglied zurück.
    """
    t = np.asarray(t, dtype=float)
    tpos = np.maximum(t, 0.0)
    s.ST = s.ST + tpos
    par = 0.48 * np.asarray(rad, dtype=float)
    lai = p.SLA * p.pLAM * s.GV * 0.1
    aet = np.minimum(pet * np.minimum(1.0, lai / 3.0), s.WR + rain)
    s.WR = np.minimum(p.WHC, np.maximum(0.0, s.WR + rain - aet))
    W = s.WR / p.WHC

    pgro = par * p.RUEmax * (1.0 - np.exp(-0.6 * lai)) * 10.0
    env = p.NI * _f_temp(t, p) * _f_par(par) * _f_water(W, pet)
    gro = pgro * env * _sea(s.ST, p)
    if gro_factor is not None:
        gro = gro * gro_factor

    rep = np.where(~s.repro_done & (p.ST1 <= s.ST) & (s.ST <= p.ST2),
                   np.minimum(1.0, np.maximum(0.0, 0.25 + 0.75 * (p.NI - 0.35) / 0.65)), 0.0)

    at = np.abs(t)
    sGV = np.where(t > p.T0, p.KGV * s.GV * t * _f_age(s.aGV, p.LLS), np.where(t < 0, p.KGV * s.GV * at, 0.0))
    sGR = np.where(t > p.T0, p.KGR * s.GR * t * _f_age(s.aGR, p.ST2 - p.ST1), np.where(t < 0, p.KGR * s.GR * at, 0.0))
    aDV = np.where(t > 0, p.KlDV * s.DV * t * _f_age(s.aDV, p.LLS), 0.0)
    aDR = np.where(t > 0, p.KlDR * s.DR * t * _f_age(s.aDR, p.ST2 - p.ST1), 0.0)
    sGV, sGR = np.minimum(sGV, s.GV), np.minimum(sGR, s.GR)
    aDV, aDR = np.minimum(aDV, s.DV), np.minimum(aDR, s.DR)

    gGV, gGR = gro * (1 - rep), gro * rep
    newGV = s.GV - sGV + gGV
    newGR = s.GR - sGR + gGR
    inDV, inDR = (1 - p.sigGV) * sGV, (1 - p.sigGR) * sGR
    newDV = s.DV - aDV + inDV
    newDR = s.DR - aDR + inDR

    # Reihenfolge wie im Skalarmodell: aDV/aDR nutzen bereits das neue aGV/aGR
    s.aGV = _safe_div((s.GV - sGV) * (s.aGV + tpos), newGV)
    s.aGR = _safe_div((s.GR - sGR) * (s.aGR + tpos), newGR)
    s.aDV = _safe_div((s.DV - aDV) * (s.aDV + tpos) + inDV * s.aGV, newDV)
    s.aDR = _safe_div((s.DR - aDR) * (s.aDR + tpos) + inDR * s.aGR, newDR)
    s.GV, s.GR, s.DV, s.DR = newGV, newGR, newDV, newDR
    return gro


# ---------------------------------------------------------------------------
# Schnitte
# ---------------------------------------------------------------------------
def dry_windows(rain: np.ndarray, et0: np.ndarray, a: Annahmen) -> tuple[np.ndarray, np.ndarray]:
    """Heuwetter-Prüfung (wie model.dry_window) für alle Tage auf einmal.

    rain/et0: (Tage,) oder (n, Tage). Gibt (ok, regentage) gleicher Form zurück.
    """
    n = a.trockentage_noetig
    rain2, et2 = np.atleast_2d(rain), np.atleast_2d(et0)
    days = rain2.shape[1]
    wet = (rain2 >= a.regen_grenze_mm).astype(int)
    cw = np.concatenate([np.zeros((wet.shape[0], 1), dtype=int), np.cumsum(wet, axis=1)], axis=1)
    ce = np.concatenate([np.zeros((et2.shape[0], 1)), np.cumsum(et2, axis=1)], axis=1)
    d = np.arange(days)
    end = np.minimum(d + n, days)
    full = (end - d) >= n
    regentage = cw[:, end] - cw[:, d]
    et_mean = (ce[:, end] - ce[:, d]) / np.maximum(end - d, 1)
    ok = full & (regentage == 0) & (et_mean >= a.min_et0_mm)
    regentage = np.where(full, regentage, n)
    if np.ndim(rain) == 1:
        return ok[0], regentage[0]
    return ok, regentage


def _ensure_cut_capacity(s: EnsembleState, need: int) -> None:
    m = s.cut_day.shape[1]
    if need <= m:
        return
    extra = _empty_cuts(s.n, need - m)
    for k, v in extra.items():
        setattr(s, k, np.concatenate([getattr(s, k), v], axis=1))


def do_cut(s: EnsembleState, mask: np.ndarray, d: int, p: EnsembleParams, a: Annahmen,
           regentage, erzwungen, beobachtet: bool = False) -> None:
    """Schnitt am Tag d für alle Mitglieder mit mask=True (wie model.do_cut, ohne Qualität)."""
    if not mask.any():
        return
    H, omd, (rGV, rGR, rDV, rDR) = harvestable(s, p, a.schnitthoehe_m)
    regentage = np.broadcast_to(np.asarray(regentage), mask.shape)
    erzwungen = np.broadcast_to(np.asarray(erzwungen, dtype=bool), mask.shape)
    _ensure_cut_capacity(s, int(s.n_cuts[mask].max()) + 1)
    idx = np.nonzero(mask)[0]
    j = s.n_cuts[idx]
    loss = np.minimum(0.6, a.feldverlust_basis + a.feldverlust_je_regentag * regentage[idx])
    s.cut_day[idx, j] = d
    s.cut_brutto[idx, j] = H[idx]
    s.cut_verlust[idx, j] = loss
    s.cut_netto[idx, j] = H[idx] * (1 - loss)
    s.cut_omd[idx, j] = omd[idx]
    s.cut_regentage[idx, j] = regentage[idx]
    s.cut_erzwungen[idx, j] = erzwungen[idx]
    s.cut_beobachtet[idx, j] = beobachtet
    s.GV = np.where(mask, np.minimum(s.GV, rGV), s.GV)
    s.GR = np.where(mask, np.minimum(s.GR, rGR), s.GR)
    s.DV = np.where(mask, np.minimum(s.DV, rDV), s.DV)
    s.DR = np.where(mask, np.minimum(s.DR, rDR), s.DR)
    s.repro_done = s.repro_done | mask
    s.n_cuts = np.where(mask, s.n_cuts + 1, s.n_cuts)
    s.last_cut = np.where(mask, d, s.last_cut)
    s.ready_since = np.where(mask, -1, s.ready_since)


def is_ready(s: EnsembleState, d: int, H: np.ndarray, a: Annahmen) -> np.ndarray:
    since = d - s.last_cut
    first = (s.ST >= a.tsumme_erster_schnitt) & (H >= a.min_ertrag_kg_tm_ha)
    later = ((since >= a.min_tage_zwischen_schnitten) & (H >= a.min_ertrag_kg_tm_ha)) | \
            ((since >= a.max_tage_zwischen_schnitten) & (H >= 0.6 * a.min_ertrag_kg_tm_ha))
    return (s.n_cuts < a.max_schnitte) & np.where(s.n_cuts == 0, first, later)


# ---------------------------------------------------------------------------
# Lauf
# ---------------------------------------------------------------------------
def _col(x: np.ndarray, d: int):
    return x[d] if x.ndim == 1 else x[:, d]


def run_ensemble(w: dict, p: EnsembleParams, a: Annahmen, state: EnsembleState | None = None,
                 start: int = 0, end: int | None = None, manage: bool = True,
                 forced_cuts=None, gro_factor=None, trace: bool = False):
    """Simuliert Tage start..end-1 für alle Mitglieder (Gegenstück zu model.run).

    w: Wetter-Dict mit tmean, rad, rain, et0 als (Tage,) (gemeinsam) oder (n, Tage) (je Mitglied).
    manage: Schnittregel aus heu/model.py anwenden (Prognosephase).
    forced_cuts: beobachtete Schnitttage, Liste (für alle gleich) oder Array (n, K) mit -1 als Lücke.
        An diesen Tagen wird unabhängig von der Regel geschnitten (Analysephase, Plan 3.4).
    gro_factor: optional (Tage, n) oder (n,) Faktor auf den Zuwachs (Modellfehler, Plan 3.2).
    trace: wenn True, wird zusätzlich ein Dict mit Tageswerten (Tage, n) zurückgegeben:
        biomasse, H, omd, W, ST, n_cuts, lai (wie die Spalten 1-7 des Skalar-Trace).

    Der Zustand wird in place verändert und zurückgegeben (plus Trace, falls verlangt).
    """
    s = state if state is not None else EnsembleState.initial(p, max_cuts=a.max_schnitte + 2)
    tm = np.asarray(w["tmean"], dtype=float)
    end = tm.shape[-1] if end is None else end
    rad = np.asarray(w["rad"], dtype=float)
    rain = np.asarray(w["rain"], dtype=float)
    et0 = np.asarray(w["et0"], dtype=float)
    first, last = _md(a.fruehester_schnitt), _md(a.spaetester_schnitt)
    if manage:
        ok_all, rt_all = dry_windows(rain, et0, a)
    if forced_cuts is not None:
        fc = np.asarray(forced_cuts, dtype=int)
        if fc.ndim == 1:
            fc = np.broadcast_to(fc, (s.n, len(fc)))
    gf = None if gro_factor is None else np.asarray(gro_factor, dtype=float)
    rec = {k: [] for k in ("biomasse", "H", "omd", "W", "ST", "n_cuts", "lai")} if trace else None

    for d in range(start, end):
        if forced_cuts is not None and fc.size:
            m = (fc == d).any(axis=1)
            do_cut(s, m, d, p, a, 0, False, beobachtet=True)
        if manage and first <= d <= last:
            H, _, _ = harvestable(s, p, a.schnitthoehe_m)
            ready = is_ready(s, d, H, a)
            s.ready_since = np.where(ready & (s.ready_since < 0), d, s.ready_since)
            s.ready_since = np.where(~ready, -1, s.ready_since)
            ok = np.broadcast_to(_col(ok_all, d), (s.n,))
            rt = np.broadcast_to(_col(rt_all, d), (s.n,))
            r_d = np.broadcast_to(_col(rain, d), (s.n,))
            cut_ok = ready & ok
            cut_forced = ready & ~ok & ((d - s.ready_since) >= a.max_wartetage) & (r_d < a.regen_grenze_mm)
            do_cut(s, cut_ok | cut_forced, d, p, a, np.where(cut_forced, rt, 0), cut_forced)
        g = None
        if gf is not None:
            g = gf[d] if gf.ndim == 2 else gf
        grow_ensemble_day(s, _col(tm, d), _col(rad, d), _col(rain, d), _col(et0, d), p, g)
        if trace:
            H, omd, _ = harvestable(s, p, a.schnitthoehe_m)
            rec["biomasse"].append(s.GV + s.GR + s.DV + s.DR)
            rec["H"].append(H)
            rec["omd"].append(omd)
            rec["W"].append(s.WR / p.WHC)
            rec["ST"].append(s.ST.copy())
            rec["n_cuts"].append(s.n_cuts.copy())
            rec["lai"].append(s.lai(p))
    if trace:
        return s, {k: np.array(v) for k, v in rec.items()}
    return s


def stack_weather(ws: list[dict], keys=("tmean", "rad", "rain", "et0")) -> dict:
    """Liste von Jahres-Wetterdicts -> ein Dict mit (n, Tage)-Arrays (je Mitglied ein Jahr)."""
    return {k: np.stack([np.asarray(w[k], dtype=float) for w in ws]) for k in keys}


# ---------------------------------------------------------------------------
# Störungen für Prior-Ensembles (Grundlage für die Assimilation, Plan 3.2)
# ---------------------------------------------------------------------------
def perturb_params(p: ModVegeParams, n: int, rng: np.random.Generator,
                   log_sigma: dict[str, float]) -> EnsembleParams:
    """Mittelwerttreue log-normale Streuung ausgewählter Parameter, z. B. {"NI": 0.15, "RUEmax": 0.1}."""
    over = {}
    for k, sig in log_sigma.items():
        base = getattr(p, k)
        over[k] = base * rng.lognormal(-0.5 * sig ** 2, sig, n)
    return EnsembleParams.from_params(p, n, **over)


def perturb_state(s: EnsembleState, rng: np.random.Generator, log_sigma: float = 0.3,
                  keys=("GV", "GR", "DV", "DR")) -> EnsembleState:
    """Biomassen je Mitglied log-normal stören (mittelwerttreu). Gibt eine Kopie zurück."""
    out = s.copy()
    for k in keys:
        v = getattr(out, k)
        setattr(out, k, v * rng.lognormal(-0.5 * log_sigma ** 2, log_sigma, s.n))
    return out
