"""Stufe A: Kalibrierung je Feld mit ES-MDA über mehrere Jahre (Plan Abschnitt 3.3).

Ensemble Smoother mit Mehrfachdatenassimilation (Emerick & Reynolds 2013): Ein Parameter-Ensemble läuft
über alle Kalibrierjahre (wahres Wetter, Schnitte erzwungen an den aus dem Satelliten erkannten Terminen),
dann werden alle Beobachtungen auf einmal mit aufgeblähter Fehlervarianz (α·R) assimiliert. Das wird
`len(alphas)`-mal wiederholt, mit Σ 1/α = 1. Damit entspricht das Ergebnis bei linearem Modell genau einer
einzigen Bayes-Analyse, ist bei nichtlinearem Modell aber deutlich robuster.

Freie Größen (log-transformiert, damit alles positiv bleibt):
    NI, RUEmax, WHC, LLS   ModVege-Parameter (Feld-Eigenschaften, über die Jahre konstant)
    lai_a                  Faktor im Beobachtungsoperator S2-LAI ≈ a · LAI_Modell (SNAP-Netz ≠ Modell-LAI)

Daten:
    - S2-LAI aller verwendeten Aufnahmen (Qualitätskontrolle aus beobachtung.aus_s2), Fehler σ_obs aus
      dem Fehlermodell plus ein Modellfehler σ_mod (Struktur, Repräsentativität) in Quadratur;
    - schwacher Prior auf den mittleren Brutto-Jahresertrag (Annahme ziel_jahresertrag),
      solange es keine echten Erträge gibt (σ 10 % log). Er trennt NI·RUEmax vom Operator-Faktor lai_a:
      Das Ertragsniveau ist aus dem LAI allein kaum bestimmt (Niederham 2017–2023: mit σ 20 % landet der
      Ertrag bei 93–105 dt/ha, mit 5 % bei 78–81, der LAI-Fehler in den Prüfjahren bleibt gleich).

Robustheit: Aufnahmen, die in einer Iteration weiter als `gating`·σ vom Ensemble liegen, bekommen für
diese Iteration eine riesige Fehlervarianz (verpasster Schnitt, Dunst). Die Anzahl wird protokolliert.

Schnitte: Jedes Mitglied zieht sein Schnittdatum einmal gleichverteilt aus [frueh, spaet] der Erkennung
(Plan 3.4). Die Ziehung bleibt über die Iterationen gleich, damit sich nur die Parameter ändern.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import beobachtung as B
from . import ensemble as E
from .config import DATA_DIR, Annahmen
from .model import ModVegeParams
from .weather import doy365, year_arrays

PARAMETER = ("NI", "RUEmax", "WHC", "LLS", "lai_a")
PRIOR_SIGMA = {"NI": 0.25, "RUEmax": 0.15, "WHC": 0.3, "LLS": 0.2, "lai_a": 0.3}
GRENZEN = {"NI": (0.2, 1.2), "RUEmax": (1.5, 4.5), "WHC": (40.0, 250.0), "LLS": (250.0, 1000.0), "lai_a": (0.4, 2.5)}
ALPHAS = (9.333, 7.0, 4.0, 2.0)          # Σ 1/α = 1 (Emerick & Reynolds 2013)


@dataclass
class KalibrierDaten:
    jahre: list[int]
    wetter: dict[int, dict]
    obs: dict[int, B.Beobachtungen]                     # verwendete LAI-Aufnahmen je Jahr
    schnitt_fenster: dict[int, list[tuple[int, int]]]   # (frueh, spaet) als Tag 0..364 je Jahr

    @property
    def n_obs(self) -> int:
        return int(sum(len(o) for o in self.obs.values()))


@dataclass
class Ergebnis:
    prior: np.ndarray                    # (n, p) Parameterwerte
    posterior: np.ndarray                # (n, p)
    namen: tuple
    lai_rmse: dict                       # {"prior": .., "posterior": ..} Ensemblemittel gegen S2
    ertrag: dict                         # {"prior": (n, Jahre), "posterior": (n, Jahre)} brutto kg TM/ha
    jahre: list[int]
    verworfen: list[int]                 # Aufnahmen je Iteration aussortiert (Gating)
    sensitivitaet: dict = field(default_factory=dict)
    schnitte: dict = field(default_factory=dict)     # Jahr -> [(Median, 10 %, 90 %) Tag je Schnitt]

    def tabelle(self) -> pd.DataFrame:
        return pd.DataFrame(self.posterior, columns=list(self.namen))

    def zusammenfassung(self) -> pd.DataFrame:
        rows = []
        for j, k in enumerate(self.namen):
            pr, po = self.prior[:, j], self.posterior[:, j]
            rows.append({"Parameter": k, "Prior Median": np.median(pr), "Prior 10–90 %": _q(pr),
                         "Posterior Median": np.median(po), "Posterior 10–90 %": _q(po),
                         "Streuung behalten [%]": 100 * np.std(np.log(po)) / np.std(np.log(pr)),
                         "Sensitivität": self.sensitivitaet.get(k, np.nan)})
        return pd.DataFrame(rows)


def _q(x) -> str:
    lo, hi = np.percentile(x, [10, 90])
    return f"{lo:.3g}–{hi:.3g}"


# ---------------------------------------------------------------------------
# Daten zusammenstellen
# ---------------------------------------------------------------------------
def daten(s2: pd.DataFrame, schnitte: pd.DataFrame, wetter_df: pd.DataFrame, jahre,
          erster_schnitt_frueh: int = 120, erster_schnitt_spaet: int = 135) -> KalibrierDaten:
    """S2-Cache + detect_cuts-Tabelle + Wetter -> Kalibrierdaten. Jahre ohne Wetter, ohne Grünlandnutzung
    oder mit weniger als 4 verwendeten Aufnahmen fallen heraus. Fehlt im Mai jede Aufnahme und ist die
    erste danach schon niedrig (LAI < 2), wird ein Schnittfenster [1. Mai, Tag davor] ergänzt."""
    from . import satellite as SAT
    nutzung = SAT.nutzung_je_jahr(s2)
    tab = B.aus_s2(s2, list(schnitte["datum"]) if len(schnitte) else None)
    out_j, w, ob, sf = [], {}, {}, {}
    for y in jahre:
        wy = year_arrays(wetter_df, y)
        if wy is None or nutzung.get(y) != SAT.NUTZUNG_GRUENLAND:
            continue
        o = B.jahr(tab, y)
        if len(o) < 4:
            continue
        cy = schnitte[schnitte["datum"].dt.year == y] if len(schnitte) else schnitte
        fen = [(doy365(r.frueh.date()), doy365(r.spaet.date())) for r in cy.itertuples()]
        # Erster Aufwuchs ohne Aufnahme im Mai: liegt die erste Aufnahme nach Mitte Mai schon niedrig und
        # ist davor kein Schnitt erkannt, muss der erste Schnitt in der Datenlücke liegen.
        april_okt = o.tag[(o.tag >= 90) & (o.tag <= 304)]
        if len(april_okt) and april_okt[0] >= erster_schnitt_spaet:
            t1 = int(april_okt[0])
            v1 = float(o.wert[o.tag == t1][0])
            if v1 < 2.0 and not any(f < t1 for f, _ in fen):
                fen.append((erster_schnitt_frueh, t1 - 1))
        out_j.append(y)
        w[y], ob[y], sf[y] = wy, o, sorted(fen)
    return KalibrierDaten(out_j, w, ob, sf)


# ---------------------------------------------------------------------------
# Vorwärtsmodell
# ---------------------------------------------------------------------------
def _schnitt_ziehung(d: KalibrierDaten, n: int, rng: np.random.Generator) -> dict[int, np.ndarray]:
    """Schnitttage je Mitglied gleichverteilt aus den Fenstern (für Prüfjahre ohne Kalibrierung)."""
    return schnitte_aus_u(d, rng.uniform(0, 1, (n, sum(len(f) for f in d.schnitt_fenster.values()))))


def schnitte_aus_u(d: KalibrierDaten, U: np.ndarray) -> dict[int, np.ndarray]:
    """U (n, Anzahl Schnitte) in [0, 1] -> Schnitttag je Mitglied: frueh + u·(spaet − frueh)."""
    out, k = {}, 0
    for y in d.jahre:
        fen = d.schnitt_fenster[y]
        if not fen:
            out[y] = np.full((U.shape[0], 0), -1, dtype=int)
            continue
        cols = [np.rint(f + U[:, k + i] * (max(f, s_) - f)).astype(int) for i, (f, s_) in enumerate(fen)]
        out[y] = np.stack(cols, axis=1)
        k += len(fen)
    return out


def vorwaerts(theta: np.ndarray, namen: tuple, d: KalibrierDaten, a: Annahmen, schnitte: dict,
              p0: ModVegeParams | None = None):
    """theta (n, p) -> (Modell-LAI an den Beobachtungstagen (n, m), Brutto-Jahresertrag (n, Jahre))."""
    n = theta.shape[0]
    p0 = p0 or ModVegeParams(WHC=a.nutzbare_feldkapazitaet_mm)
    over = {k: theta[:, j] for j, k in enumerate(namen) if k != "lai_a"}
    lai_a = theta[:, namen.index("lai_a")] if "lai_a" in namen else np.ones(n)
    ep = E.EnsembleParams.from_params(p0, n, **over)
    preds, ertrag = [], []
    for y in d.jahre:
        s0 = E.EnsembleState.initial(ep, max_cuts=a.max_schnitte + 4)
        s, tr = E.run_ensemble(d.wetter[y], ep, a, state=s0, manage=False, forced_cuts=schnitte[y], trace=True)
        preds.append(lai_a[:, None] * tr["lai"][d.obs[y].tag].T)
        ertrag.append(s.season_total(netto=False))
    return np.concatenate(preds, axis=1), np.stack(ertrag, axis=1)


# ---------------------------------------------------------------------------
# ES-MDA
# ---------------------------------------------------------------------------
def prior_ensemble(n: int, rng: np.random.Generator, a: Annahmen, namen=PARAMETER,
                   p0: ModVegeParams | None = None) -> np.ndarray:
    p0 = p0 or ModVegeParams(WHC=a.nutzbare_feldkapazitaet_mm)
    cols = []
    for k in namen:
        mu = 1.0 if k == "lai_a" else getattr(p0, k)
        cols.append(np.clip(mu * rng.lognormal(0.0, PRIOR_SIGMA[k], n), *GRENZEN[k]))
    return np.stack(cols, axis=1)


def esmda(d: KalibrierDaten, a: Annahmen, n: int = 100, seed: int = 0, namen=PARAMETER,
          alphas=ALPHAS, sigma_modell: float = 0.6, ertrag_sigma: float = 0.1, gating: float = 3.5,
          schnitt_schaetzen: bool = True, p0: ModVegeParams | None = None, protokoll=None) -> Ergebnis:
    """ES-MDA über alle Jahre. Mit schnitt_schaetzen=True wird zusätzlich jeder Schnitttermin innerhalb
    seines Erkennungsfensters mitgeschätzt (logit-transformierter Anteil u, Prior ≈ gleichverteilt).
    Das LAI direkt nach dem Schnitt verrät, wie lange der Schnitt zurückliegt."""
    rng = np.random.default_rng(seed)
    X0 = prior_ensemble(n, rng, a, namen, p0)
    p = len(namen)
    n_cut = sum(len(f) for f in d.schnitt_fenster.values())
    U0 = rng.uniform(0.02, 0.98, (n, n_cut))
    y_lai = np.concatenate([d.obs[j].wert for j in d.jahre])
    r_lai = np.concatenate([d.obs[j].sigma for j in d.jahre]) ** 2 + sigma_modell ** 2
    # Pseudo-Beobachtung: log mittlerer Brutto-Jahresertrag (Annahme ist brutto, vor Feldverlusten)
    y = np.append(y_lai, np.log(a.ziel_jahresertrag_dt_tm_ha * 100.0))
    R = np.append(r_lai, ertrag_sigma ** 2)

    def entpacken(Z):
        X = np.exp(Z[:, :p])
        U = 1.0 / (1.0 + np.exp(-Z[:, p:])) if schnitt_schaetzen else U0
        return X, U

    def g(Z):
        X, U = entpacken(Z)
        lai, ert = vorwaerts(X, namen, d, a, schnitte_aus_u(d, U), p0)
        return np.hstack([lai, np.log(np.maximum(ert.mean(axis=1, keepdims=True), 1.0))]), lai, ert

    Z = np.log(X0)
    if schnitt_schaetzen:
        Z = np.hstack([Z, np.log(U0 / (1 - U0))])
    lo = np.r_[np.log([GRENZEN[k][0] for k in namen]), np.full(Z.shape[1] - p, -6.0)]
    hi = np.r_[np.log([GRENZEN[k][1] for k in namen]), np.full(Z.shape[1] - p, 6.0)]
    D, lai0, ert0 = g(Z)
    sens = _sensitivitaet(Z[:, :p], D[:, :-1], namen)
    verworfen = []
    for it, al in enumerate(alphas):
        dm = D.mean(axis=0)
        Rit = al * R.copy()
        aus = np.abs(y - dm) > gating * np.sqrt(D.var(axis=0, ddof=1) + Rit)
        aus[-1] = False
        Rit[aus] = 1e6
        verworfen.append(int(aus.sum()))
        Az = Z - Z.mean(axis=0)
        Ad = D - dm
        Czd = Az.T @ Ad / (n - 1)
        Cdd = Ad.T @ Ad / (n - 1)
        y_pert = y + rng.normal(0.0, 1.0, (n, len(y))) * np.sqrt(Rit)
        K = np.linalg.solve(Cdd + np.diag(Rit), (y_pert - D).T)     # (m, n)
        Z = np.clip(Z + (Czd @ K).T, lo, hi)
        D, lai1, ert1 = g(Z)
        if protokoll:
            protokoll(it + 1, len(alphas))
    X1, U1 = entpacken(Z)
    rmse = lambda L: float(np.sqrt(np.mean((L.mean(axis=0) - y_lai) ** 2)))
    tage = schnitte_aus_u(d, U1)
    schnitte = {y: [(int(np.median(tage[y][:, i])), int(np.percentile(tage[y][:, i], 10)),
                     int(np.percentile(tage[y][:, i], 90))) for i in range(tage[y].shape[1])] for y in d.jahre}
    return Ergebnis(X0, X1, tuple(namen), {"prior": rmse(lai0), "posterior": rmse(lai1)},
                    {"prior": ert0, "posterior": ert1}, list(d.jahre), verworfen, sens, schnitte)


def _sensitivitaet(Z: np.ndarray, D: np.ndarray, namen) -> dict:
    """Einfache globale Sensitivität aus dem Prior-Ensemble: Anteil der Varianz der Modell-LAI, den
    eine lineare Regression auf die (log) Parameter je Parameter erklärt (standardisierte Koeffizienten²,
    gemittelt über alle Beobachtungen). Summe ≈ R² des linearen Modells."""
    Zs = (Z - Z.mean(0)) / Z.std(0)
    Ds = (D - D.mean(0)) / np.maximum(D.std(0), 1e-9)
    beta, *_ = np.linalg.lstsq(Zs, Ds, rcond=None)                # (p, m)
    return {k: float(np.mean(beta[j] ** 2)) for j, k in enumerate(namen)}


# ---------------------------------------------------------------------------
# Speichern und Laden
# ---------------------------------------------------------------------------
def pfad(feld_id: str):
    return DATA_DIR / f"kalibrierung_{feld_id}.csv"


def speichern(erg: Ergebnis, feld_id: str):
    df = erg.tabelle()
    df.to_csv(pfad(feld_id), index=False)
    return pfad(feld_id)


def laden(feld_id: str, a: Annahmen, p0: ModVegeParams | None = None):
    """Posterior-Stichprobe als (EnsembleParams, lai_a-Array) oder None, wenn noch nicht kalibriert.
    Die EnsembleParams lassen sich direkt als `posterior` an assimilation.analyse_saison geben."""
    p = pfad(feld_id)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    p0 = p0 or ModVegeParams(WHC=a.nutzbare_feldkapazitaet_mm)
    over = {k: df[k].to_numpy(dtype=float) for k in df.columns if k != "lai_a"}
    ep = E.EnsembleParams.from_params(p0, len(df), **over)
    return ep, df["lai_a"].to_numpy(dtype=float) if "lai_a" in df else np.ones(len(df))


def prior_fuer_assimilation(feld_id: str, a: Annahmen, cfg=None):
    """(posterior EnsembleParams, DAKonfig mit kalibriertem Operator-Faktor) oder (None, cfg).
    Verwendung: assimilation.analyse_saison(..., posterior=ep, cfg=cfg)."""
    from dataclasses import replace
    from .config import DA
    cfg = cfg or DA
    r = laden(feld_id, a)
    if r is None:
        return None, cfg
    ep, lai_a = r
    return ep, replace(cfg, operator_a=float(np.median(lai_a)))


def kalibrieren_feld(feld_id: str, s2: pd.DataFrame, s1: pd.DataFrame | None, wetter_df: pd.DataFrame,
                     a: Annahmen, jahre=range(2017, 2026), n: int = 200, seed: int = 0, protokoll=None):
    """Ein Aufruf für App und Skript: Daten zusammenstellen, ES-MDA, speichern. Gibt (Ergebnis, Daten)."""
    from . import satellite as SAT
    d = daten(s2, SAT.detect_cuts(s2, s1), wetter_df, jahre)
    if not d.jahre:
        raise ValueError("Keine Grünlandjahre mit genug klaren Aufnahmen und Wetter gefunden.")
    erg = esmda(d, a, n=n, seed=seed, protokoll=protokoll)
    speichern(erg, feld_id)
    return erg, d
