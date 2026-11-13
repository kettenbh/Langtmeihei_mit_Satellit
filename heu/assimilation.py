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
def to_vector(s: E.EnsembleState, p: E.EnsembleParams, params: tuple) -> np.ndarray:
    cols = [np.log1p(np.maximum(getattr(s, k), 0.0)) for k in _BIOMASSE]
    w = np.clip(s.WR / p.WHC, _EPS_W, 1 - _EPS_W)
    cols.append(np.log(w / (1 - w)))
    cols += [np.log(getattr(p, k)) for k in params]
    return np.stack(cols, axis=1)


def from_vector(X: np.ndarray, s: E.EnsembleState, p: E.EnsembleParams, params: tuple,
                grenzen: tuple) -> None:
    for j, k in enumerate(_BIOMASSE):
        setattr(s, k, np.maximum(np.expm1(X[:, j]), 0.0))
    s.WR = p.WHC / (1 + np.exp(-X[:, 4]))
    for i, (k, (lo, hi)) in enumerate(zip(params, grenzen)):
        setattr(p, k, np.clip(np.exp(X[:, 5 + i]), lo, hi))


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
    if posterior is not None:
        ep = posterior.take(rng.integers(0, posterior.n, n))
    else:
        ep = E.perturb_params(p, n, rng, dict(zip(cfg.parameter, cfg.param_sigma)))
    s0 = E.EnsembleState.initial(ep, max_cuts=a.max_schnitte + 2)
    return ep, E.perturb_state(s0, rng, cfg.zustand_sigma)


def analyse_saison(w: dict, p: ModVegeParams, a: Annahmen, obs: Beobachtungen | None,
                   schnitte: list[int] | np.ndarray | None, ende: int = 365, cfg: DAKonfig = DA,
                   seed: int = 0, schnitt_unsicherheit: int = 0, assimilieren: bool = True,
                   posterior: E.EnsembleParams | None = None, n: int | None = None) -> Analyse:
    """Analysephase vom 1. Januar bis Tag `ende` (exklusiv) mit beobachteten Schnitten und S2-LAI.

    schnitte: beobachtete Schnitttage (gemeinsam), jedes Mitglied verschiebt um ±schnitt_unsicherheit.
    assimilieren=False liefert denselben Lauf ohne Korrektur (Open Loop, gleiche Zufallszahlen).
    """
    rng = np.random.default_rng(seed)
    ep, s = prior(p, a, rng, cfg, n=n, posterior=posterior)
    n = ep.n
    ww = gestoertes_wetter(w, n, rng, cfg)
    gf = modellfehler(n, len(w["tmean"]), rng, cfg)
    fc = None
    if schnitte is not None and len(schnitte):
        fc = np.asarray(schnitte, dtype=int)[None, :].repeat(n, axis=0)
        if schnitt_unsicherheit:
            fc = fc + rng.integers(-schnitt_unsicherheit, schnitt_unsicherheit + 1, size=fc.shape)
    params = tuple(cfg.parameter)
    damp = np.array([1.0] * 5 + [cfg.param_daempfung] * len(params))
    obs_days = [] if obs is None else [(int(t), float(v), float(sg)) for t, v, sg in zip(obs.tag, obs.wert, obs.sigma) if t < ende]
    updates: list[UpdateInfo] = []
    traces = []
    pv = {k: ([], []) for k in params}
    day = 0
    rng_obs = np.random.default_rng(seed + 10_000)
    for t_obs, y, sg in obs_days + [(ende, None, None)]:
        if t_obs > day:
            s, tr = E.run_ensemble(ww, ep, a, state=s, start=day, end=t_obs, manage=False,
                                   forced_cuts=fc, gro_factor=gf, trace=True)
            traces.append(tr)
            day = t_obs
        if y is None:
            break
        hx = h_lai(s.lai(ep), cfg)
        if not assimilieren:
            updates.append(UpdateInfo(t_obs, y, sg ** 2, float(hx.mean()), float(hx.std(ddof=1)),
                                      float(y - hx.mean()), False, "Open Loop"))
            continue
        X = to_vector(s, ep, params)
        X, used, info = enkf_update(X, hx, y, sg ** 2, rng_obs, cfg.inflation, cfg.gating_sigma, damp)
        from_vector(X, s, ep, params, cfg.param_grenzen)
        updates.append(UpdateInfo(t_obs, y, sg ** 2, info["hx_mittel"], info["hx_spread"],
                                  info["innovation"], used, info["grund"]))
        for k in params:
            pv[k][0].append(t_obs)
            pv[k][1].append(getattr(ep, k).copy())
    trace = {k: np.concatenate([t[k] for t in traces]) for k in traces[0]} if traces else {}
    param_verlauf = {k: (np.array(v[0]), np.array(v[1])) for k, v in pv.items()}
    return Analyse(s, ep, updates, trace, param_verlauf)


def prognose(an: Analyse, w_rest: dict, a: Annahmen, start: int, end: int = 365) -> E.EnsembleState:
    """Weiterrechnen ab `start` mit Schnittregel (Zukunft), Wetter gemeinsam (Tage,) oder je Mitglied."""
    s = an.zustand.copy()
    p = an.params.take(np.arange(an.n))
    return E.run_ensemble(w_rest, p, a, state=s, start=start, end=end, manage=True)
