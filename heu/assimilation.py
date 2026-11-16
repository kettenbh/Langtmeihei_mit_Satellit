"""Ensemble-Kalman-Filter für ModVege mit erweitertem Zustand (Plan Abschnitt 3.2 und 3.4).

Zustandsvektor je Mitglied (transformiert, damit nach dem Update nichts negativ wird):
    log(1+GV), log(1+GR), log(1+DV), log(1+DR), logit(WR/WHC), log NI, log RUEmax, ...
Die Alter (aGV ...) und die Temperatursumme ST werden mitgeführt, aber nicht korrigiert.

Ablauf einer Saison (analyse_saison):
  1. Prior-Ensemble: Parameter und Anfangsbiomassen gestört (config.DAKonfig).
  2. Zwischen zwei Aufnahmen läuft jedes Mitglied mit gestörtem Wetter (Strahlung, Regen) und
     einem AR(1)-Modellfehler auf dem Zuwachs. Beobachtete Schnitte werden als Ereignis gesetzt,
     jedes Mitglied mit eigenem Tag aus dem Unsicherheitsfenster.
  3. An einem Aufnahmetag: stochastischer EnKF mit gestörten Beobachtungen (Burgers et al. 1998),
     vorher multiplikative Inflation, Gating gegen Ausreißer, gedämpftes Parameter-Update.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import ensemble as E
from .beobachtung import Beobachtungen, h_lai
from .config import DA, Annahmen, DAKonfig
from .model import ModVegeParams

_BIOMASSE = ("GV", "GR", "DV", "DR")
_EPS_W = 1e-3


# ---------------------------------------------------------------------------
# Transformation Zustand <-> Vektor
# ---------------------------------------------------------------------------
_ALTER = ("aGV", "aGR")


def to_vector(s: E.EnsembleState, p: E.EnsembleParams, params: tuple, alter: bool = False) -> np.ndarray:
    cols = [np.log1p(np.maximum(getattr(s, k), 0.0)) for k in _BIOMASSE]
    w = np.clip(s.WR / p.WHC, _EPS_W, 1 - _EPS_W)
    cols.append(np.log(w / (1 - w)))
    cols += [np.log(getattr(p, k)) for k in params]
    if alter:
        cols += [np.log1p(np.maximum(getattr(s, k), 0.0)) for k in _ALTER]
    return np.stack(cols, axis=1)


def from_vector(X: np.ndarray, s: E.EnsembleState, p: E.EnsembleParams, params: tuple,
                grenzen: tuple, alter: bool = False) -> None:
    for j, k in enumerate(_BIOMASSE):
        setattr(s, k, np.maximum(np.expm1(X[:, j]), 0.0))
    s.WR = p.WHC / (1 + np.exp(-X[:, 4]))
    for i, (k, (lo, hi)) in enumerate(zip(params, grenzen)):
        setattr(p, k, np.clip(np.exp(X[:, 5 + i]), lo, hi))
    if alter:
        # Alter der grünen Kompartimente (°C·d): bestimmen die Verdaulichkeit (OMD) und damit NEL.
        # Über die Ensemble-Kovarianz mit dem LAI korrigiert: niedriger LAI -> eher frisch gemäht -> jünger.
        j0 = 5 + len(params)
        for j, k in enumerate(_ALTER):
            setattr(s, k, np.maximum(np.expm1(X[:, j0 + j]), 0.0))


# ---------------------------------------------------------------------------
# Ein Analyseschritt
# ---------------------------------------------------------------------------
@dataclass
class UpdateInfo:
    tag: int
    y: float
    r: float
    hx_mittel: float
    hx_spread: float
    innovation: float
    verwendet: bool
    grund: str = ""


def enkf_update(X: np.ndarray, hx: np.ndarray, y: float, r: float, rng: np.random.Generator,
                inflation: float = 1.0, gating: float = np.inf, damp: np.ndarray | None = None):
    """Stochastischer EnKF für eine skalare Beobachtung.

    X: (n, d) Zustandsvektoren, hx: (n,) Modellabbild, y: Beobachtung, r: Fehlervarianz.
    damp: (d,) Faktoren je Komponente (1 = volles Update, z. B. 0,4 für Parameter).
    Gibt (X_neu, verwendet, info) zurück; bei Gating bleibt X unverändert (bis auf die Inflation).
    """
    n = X.shape[0]
    xm = X.mean(axis=0)
    X = xm + inflation * (X - xm)
    hm = hx.mean()
    hx = hm + inflation * (hx - hm)
    var_h = hx.var(ddof=1)
    d = y - hm
    info = {"hx_mittel": float(hm), "hx_spread": float(np.sqrt(var_h)), "innovation": float(d)}
    if abs(d) > gating * np.sqrt(var_h + r):
        return X, False, {**info, "grund": "Gating"}
    if var_h <= 0:
        return X, False, {**info, "grund": "kein Spread"}
    A = X - X.mean(axis=0)
    cov_xh = A.T @ (hx - hm) / (n - 1)               # (d,)
    K = cov_xh / (var_h + r)
    if damp is not None:
        K = K * damp
    y_pert = y + rng.normal(0.0, np.sqrt(r), n)
    X = X + np.outer(y_pert - hx, K)
    return X, True, {**info, "grund": ""}


# ---------------------------------------------------------------------------
# Saison-Analyse
# ---------------------------------------------------------------------------
@dataclass
class Analyse:
    zustand: E.EnsembleState
    params: E.EnsembleParams
    updates: list[UpdateInfo]
    trace: dict                                  # Tageswerte (Tage, n), Schlüssel wie run_ensemble
    param_verlauf: dict = field(default_factory=dict)   # name -> (Tage der Updates, (k, n) Werte)
    nachgetragene_schnitte: list = field(default_factory=list)  # (von, bis): aus LAI-Einbruch erschlossen

    @property
    def n(self) -> int:
        return self.zustand.n


def gestoertes_wetter(w: dict, n: int, rng: np.random.Generator, cfg: DAKonfig = DA) -> dict:
    """Wetter je Mitglied: Strahlung und Regen mit mittelwerttreuen log-normalen Tagesstörungen."""
    days = len(w["tmean"])
    def ln(sig):
        return rng.lognormal(-0.5 * sig ** 2, sig, (n, days)) if sig > 0 else np.ones((n, days))
    return {
        "tmean": np.broadcast_to(w["tmean"], (n, days)).copy(),
        "rad": w["rad"] * ln(cfg.wetter_sigma_strahlung),
        "rain": w["rain"] * ln(cfg.wetter_sigma_regen),
        "et0": np.broadcast_to(w["et0"], (n, days)).copy(),
    }


def modellfehler(n: int, days: int, rng: np.random.Generator, cfg: DAKonfig = DA) -> np.ndarray:
    """AR(1)-korrelierter, mittelwerttreuer log-normaler Faktor auf den Tageszuwachs, (Tage, n)."""
    if cfg.modell_sigma <= 0:
        return np.ones((days, n))
    eps = np.zeros((days, n))
    innov = np.sqrt(1 - cfg.modell_ar ** 2)
    for d in range(days):
        prev = eps[d - 1] if d else rng.normal(0, 1, n)
        eps[d] = cfg.modell_ar * prev + innov * rng.normal(0, 1, n)
    sig = cfg.modell_sigma
    return np.exp(sig * eps - 0.5 * sig ** 2)


def prior(p: ModVegeParams, a: Annahmen, rng: np.random.Generator, cfg: DAKonfig = DA,
          n: int | None = None, posterior: E.EnsembleParams | None = None):
    """Prior-Ensemble. Mit `posterior` (Stufe A) werden die Parameter daraus gezogen, sonst um p gestreut."""
    n = n or cfg.mitglieder
    if posterior is not None and posterior.n == n:
        ep = posterior.take(np.arange(n))     # Reihenfolge behalten: Mitglied i hat in jedem Jahr dieselben Parameter
    elif posterior is not None:
        ep = posterior.take(rng.integers(0, posterior.n, n))
    else:
        ep = E.perturb_params(p, n, rng, dict(zip(cfg.parameter, cfg.param_sigma)))
    s0 = E.EnsembleState.initial(ep, max_cuts=a.max_schnitte + 2)
    return ep, E.perturb_state(s0, rng, cfg.zustand_sigma)


def analyse_saison(w: dict, p: ModVegeParams, a: Annahmen, obs: Beobachtungen | None,
                   schnitte: list[int] | np.ndarray | None, ende: int = 365, cfg: DAKonfig = DA,
                   seed: int = 0, schnitt_unsicherheit: int = 0, assimilieren: bool = True,
                   posterior: E.EnsembleParams | None = None, n: int | None = None,
                   schnitt_fenster: list | None = None, schnitt_resampling: int = 30,
                   alter_korrigieren: bool = False, schnitt_nachtragen: bool = False,
                   schnitt_hypothesen: list | None = None, hypothese_anteil: float = 0.5) -> Analyse:
    """Analysephase vom 1. Januar bis Tag `ende` (exklusiv) mit beobachteten Schnitten und S2-LAI.

    schnitte: beobachtete Schnitttage (gemeinsam), jedes Mitglied verschiebt um ±schnitt_unsicherheit.
    schnitt_fenster: alternativ Liste (frueh, spaet) je Schnitt; jedes Mitglied zieht seinen Tag daraus.
    assimilieren=False liefert denselben Lauf ohne Korrektur (Open Loop, gleiche Zufallszahlen).
    schnitt_resampling: Mit Schnittfenstern werden Aufnahmen bis zu so vielen Tagen nach einem Fenster auch
        für ein Resampling der Mitglieder genutzt (Partikelfilter-Schritt vor dem EnKF-Update). Der EnKF kann
        den Schnitttag eines Mitglieds nicht verschieben; das Resampling bevorzugt die Mitglieder, deren
        Schnitttag zum gemessenen Nachwuchs passt (frisch gemäht = niedriger LAI). 0 schaltet es ab.
    alter_korrigieren: Blattalter aGV/aGR mit in den EnKF-Zustand (für die Futterqualität).
    schnitt_nachtragen: Liegt eine Aufnahme weit unter dem Ensemble (Einbruch des LAI) und deckt kein Schnittfenster
        den Zeitraum seit der letzten Aufnahme ab, wurde vermutlich ein Schnitt übersehen (Wolkenlücke, Erkennung
        verfehlt). Dann bekommt jedes Mitglied einen Schnitt an einem zufälligen Tag dieses Zeitraums und der
        Abschnitt wird neu gerechnet, bevor der EnKF die Aufnahme verarbeitet.
    schnitt_hypothesen: Fenster (frueh, spaet) für vermutete, aber nicht erkannte Schnitte (z. B. Termine der
        Schnittregel in einer Wolkenlücke). Nur der Anteil `hypothese_anteil` der Mitglieder mäht darin; die
        Aufnahmen bis 30 Tage nach dem Fenster entscheiden per Resampling, welche Hypothese passt.
    """
    rng = np.random.default_rng(seed)
    ep, s = prior(p, a, rng, cfg, n=n, posterior=posterior)
    n = ep.n
    ww = gestoertes_wetter(w, n, rng, cfg)
    gf = modellfehler(n, len(w["tmean"]), rng, cfg)
    fc = None
    if schnitt_fenster:
        # je Schnitt eigenes Fenster [frueh, spaet] (detect_cuts), Tag je Mitglied gleichverteilt
        fc = np.stack([rng.integers(f, max(f, sp) + 1, n) for f, sp in schnitt_fenster], axis=1)
    elif schnitte is not None and len(schnitte):
        fc = np.asarray(schnitte, dtype=int)[None, :].repeat(n, axis=0)
        if schnitt_unsicherheit:
            fc = fc + rng.integers(-schnitt_unsicherheit, schnitt_unsicherheit + 1, size=fc.shape)
    hyp = [(int(f), int(sp)) for f, sp in (schnitt_hypothesen or [])]
    if hyp:
        neu = np.stack([np.where(rng.random(n) < hypothese_anteil, rng.integers(f, max(f, sp) + 1, n), -1)
                        for f, sp in hyp], axis=1)
        fc = neu if fc is None else np.concatenate([fc, neu], axis=1)
    rs_fenster = [(f, sp + schnitt_resampling) for f, sp in (schnitt_fenster or [])] if schnitt_resampling else []
    rs_fenster += [(f, sp + 30) for f, sp in hyp]
    params = tuple(cfg.parameter)
    damp = np.array([1.0] * 5 + [cfg.param_daempfung] * len(params) + [1.0] * (len(_ALTER) if alter_korrigieren else 0))
    obs_days = [] if obs is None else [(int(t), float(v), float(sg)) for t, v, sg in zip(obs.tag, obs.wert, obs.sigma) if t < ende]
    updates: list[UpdateInfo] = []
    traces = []
    pv = {k: ([], []) for k in params}
    day = 0
    rng_obs = np.random.default_rng(seed + 10_000)
    nachgetragen = []
    abgedeckt = list(schnitt_fenster or []) + hyp + ([(int(t), int(t)) for t in schnitte] if schnitte is not None else [])
    for t_obs, y, sg in obs_days + [(ende, None, None)]:
        day0, s0 = day, None
        if t_obs > day:
            s0 = s.copy()
            s, tr = E.run_ensemble(ww, ep, a, state=s, start=day, end=t_obs, manage=False,
                                   forced_cuts=fc, gro_factor=gf, trace=True)
            traces.append(tr)
            day = t_obs
        if y is None:
            break
        hx = h_lai(s.lai(ep), cfg)
        if schnitt_nachtragen and s0 is not None and assimilieren and t_obs >= 100 and \
                not any(f <= t_obs and sp >= day0 - 5 for f, sp in abgedeckt) and \
                hx.mean() >= 2.5 and y < hx.mean() - max(1.5, 2.5 * np.sqrt(hx.var() + sg ** 2)):
            neu = rng_obs.integers(day0 + 1, t_obs + 1, n) if t_obs > day0 + 1 else np.full(n, t_obs)
            fc = neu[:, None] if fc is None else np.concatenate([fc, neu[:, None]], axis=1)
            s, tr = E.run_ensemble(ww, ep, a, state=s0, start=day0, end=t_obs, manage=False,
                                   forced_cuts=fc, gro_factor=gf, trace=True)
            traces[-1] = tr
            hx = h_lai(s.lai(ep), cfg)
            nachgetragen.append((day0 + 1, t_obs))
            abgedeckt.append((day0 + 1, t_obs))
        if not assimilieren:
            updates.append(UpdateInfo(t_obs, y, sg ** 2, float(hx.mean()), float(hx.std(ddof=1)),
                                      float(y - hx.mean()), False, "Open Loop"))
            continue
        if fc is not None and any(f <= t_obs <= sp for f, sp in rs_fenster):
            r_eff = sg ** 2 + 0.5 ** 2
            lw = -0.5 * (y - hx) ** 2 / r_eff
            wgt = np.exp(lw - lw.max())
            wgt /= wgt.sum()
            if 1.0 / np.sum(wgt ** 2) < 0.5 * n:
                idx = _systematisch(wgt, rng_obs)
                s = s.take(idx)
                ep = ep.take(idx)
                fc = fc[idx]
                gf = gf[:, idx]
                ww = {k: (v[idx] if v.ndim == 2 else v) for k, v in ww.items()}
                traces = [{k: v[:, idx] for k, v in tr_.items()} for tr_ in traces]
                pv = {k: (v[0], [arr[idx] for arr in v[1]]) for k, v in pv.items()}
                hx = hx[idx]
        X = to_vector(s, ep, params, alter_korrigieren)
        X, used, info = enkf_update(X, hx, y, sg ** 2, rng_obs, cfg.inflation, cfg.gating_sigma, damp)
        from_vector(X, s, ep, params, cfg.param_grenzen, alter_korrigieren)
        updates.append(UpdateInfo(t_obs, y, sg ** 2, info["hx_mittel"], info["hx_spread"],
                                  info["innovation"], used, info["grund"]))
        for k in params:
            pv[k][0].append(t_obs)
            pv[k][1].append(getattr(ep, k).copy())
    trace = {k: np.concatenate([t[k] for t in traces]) for k in traces[0]} if traces else {}
    param_verlauf = {k: (np.array(v[0]), np.array(v[1])) for k, v in pv.items()}
    return Analyse(s, ep, updates, trace, param_verlauf, nachgetragen)


def _systematisch(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Systematisches Resampling: Indizes proportional zu den Gewichten w (Summe 1)."""
    n = len(w)
    pos = (rng.random() + np.arange(n)) / n
    return np.minimum(np.searchsorted(np.cumsum(w), pos), n - 1)


def prognose(an: Analyse, w_rest: dict, a: Annahmen, start: int, end: int = 365) -> E.EnsembleState:
    """Weiterrechnen ab `start` mit Schnittregel (Zukunft), Wetter gemeinsam (Tage,) oder je Mitglied."""
    s = an.zustand.copy()
    p = an.params.take(np.arange(an.n))
    return E.run_ensemble(w_rest, p, a, state=s, start=start, end=end, manage=True)
