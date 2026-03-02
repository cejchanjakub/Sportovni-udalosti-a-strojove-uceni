# src/line_generator.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

DistName = Literal["poisson", "negbin"]


@dataclass(frozen=True)
class LineRow:
    line: float
    p_over: float
    p_under: float
    odds_over_fair: float
    odds_under_fair: float
    odds_over: float
    odds_under: float


def _safe_prob(x: float, eps: float = 1e-12) -> float:
    return float(min(max(x, eps), 1.0 - eps))


def _apply_margin_two_way(p_over: float, p_under: float, margin: float) -> Tuple[float, float]:
    """
    Přidá margin (overround) tak, aby součet impl. pravděpodobností byl 1 + margin.
    """
    if margin <= 0:
        return p_over, p_under
    scale = (1.0 + margin) / (p_over + p_under)
    return p_over * scale, p_under * scale


def _odds_from_prob(p: float) -> float:
    p = _safe_prob(p)
    return 1.0 / p


def _is_half_line(x: float) -> bool:
    # True pro 0.5, 1.5, 2.5, ...
    return abs((x * 2) - round(x * 2)) < 1e-9 and (round(x * 2) % 2 == 1)


# ---------- Distributions (PMF/CDF) without heavy deps ----------

def _poisson_pmf(k: int, lam: float) -> float:
    # stabilní výpočet přes log
    if k < 0:
        return 0.0
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return float(math.exp(k * math.log(lam) - lam - math.lgamma(k + 1)))


def _poisson_cdf(k: int, lam: float) -> float:
    # P(X <= k)
    if k < 0:
        return 0.0
    s = 0.0
    for i in range(0, k + 1):
        s += _poisson_pmf(i, lam)
    return float(min(max(s, 0.0), 1.0))


def _negbin_params_from_mu_alpha(mu: float, alpha: float) -> Tuple[float, float]:
    """
    NB2 parametrizace: Var = mu + alpha * mu^2
    Převod na (r, p) kde:
      X ~ NegBin(r, p) s podporou {0,1,2,...}
      E[X] = r*(1-p)/p
      Var[X] = r*(1-p)/p^2
    """
    mu = max(mu, 1e-12)
    alpha = max(alpha, 1e-12)
    r = 1.0 / alpha
    p = r / (r + mu)
    return r, p


def _negbin_pmf(k: int, r: float, p: float) -> float:
    # PMF: C(k+r-1, k) * (1-p)^k * p^r
    if k < 0:
        return 0.0
    p = _safe_prob(p)
    log_coeff = math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1)
    return float(math.exp(log_coeff + k * math.log(1.0 - p) + r * math.log(p)))


def _negbin_cdf(k: int, r: float, p: float) -> float:
    if k < 0:
        return 0.0
    s = 0.0
    for i in range(0, k + 1):
        s += _negbin_pmf(i, r, p)
    return float(min(max(s, 0.0), 1.0))


def _quantile_from_cdf(
    cdf_func,
    q: float,
    k_max: int = 200,
) -> int:
    """
    Najde nejmenší k tak, že CDF(k) >= q (diskrétní kvantil).
    """
    q = float(min(max(q, 0.0), 1.0))
    for k in range(0, k_max + 1):
        if cdf_func(k) >= q:
            return k
    return k_max


def _build_cdf(mu: float, dist: DistName, alpha: Optional[float]):
    mu = float(mu)
    if dist == "poisson":
        return lambda k: _poisson_cdf(k, mu)
    if dist == "negbin":
        if alpha is None:
            raise ValueError("Pro dist='negbin' musíš dodat alpha (NB2: Var = mu + alpha*mu^2).")
        r, p = _negbin_params_from_mu_alpha(mu, float(alpha))
        return lambda k: _negbin_cdf(k, r, p)
    raise ValueError(f"Neznámé dist: {dist}")


def _row_from_line(cdf, line: float, margin: float) -> LineRow:
    """
    Linka je x.5 => Under = P(X <= floor(line)), Over = P(X >= ceil(line)).
    """
    k_over = int(math.ceil(line))
    k_under = int(math.floor(line))

    p_under = float(cdf(k_under))
    p_over = float(1.0 - cdf(k_over - 1))  # P(X >= k_over)

    p_over = _safe_prob(p_over)
    p_under = _safe_prob(p_under)

    odds_over_fair = _odds_from_prob(p_over)
    odds_under_fair = _odds_from_prob(p_under)

    p_over_m, p_under_m = _apply_margin_two_way(p_over, p_under, margin)
    odds_over = _odds_from_prob(p_over_m)
    odds_under = _odds_from_prob(p_under_m)

    return LineRow(
        line=float(line),
        p_over=float(p_over),
        p_under=float(p_under),
        odds_over_fair=float(odds_over_fair),
        odds_under_fair=float(odds_under_fair),
        odds_over=float(odds_over),
        odds_under=float(odds_under),
    )


# ---------- Main API (existing: quantile-range grid) ----------

def generate_ou_lines(
    mu: float,
    dist: DistName = "poisson",
    *,
    # negbin only:
    alpha: Optional[float] = None,
    # line grid:
    step: float = 0.5,
    # dynamic range:
    q_low: float = 0.05,
    q_high: float = 0.95,
    pad_lines: int = 2,
    # safety:
    k_max: int = 250,
    # margin (e.g. 0.05 = 5% overround):
    margin: float = 0.0,
) -> List[LineRow]:
    """
    Vrátí seznam linek (Over/Under) pro diskrétní události (góly, rohy, fauly, karty...).
    Linky jsou typu x.5, takže Over = P(X >= ceil(line)), Under = P(X <= floor(line)).

    dist:
      - "poisson": mu = lambda
      - "negbin": mu + alpha (NB2 overdisperze)
    """
    if step <= 0:
        raise ValueError("step musí být > 0")

    mu = float(mu)
    cdf = _build_cdf(mu, dist, alpha)

    # dynamické meze podle kvantilů
    k_lo = _quantile_from_cdf(cdf, q_low, k_max=k_max)
    k_hi = _quantile_from_cdf(cdf, q_high, k_max=k_max)

    # padding
    k_lo = max(0, k_lo - pad_lines)
    k_hi = min(k_max, k_hi + pad_lines)

    # half-line grid: line = k + 0.5
    start_line = max(0.5, (k_lo - 1) + 0.5)
    end_line = (k_hi) + 0.5

    n = int(round((end_line - start_line) / step)) + 1
    lines = [start_line + i * step for i in range(n)]
    lines = [round(x, 2) for x in lines]

    rows: List[LineRow] = []
    for line in lines:
        # dovolíme i jiné kroky, ale defaultně pracujeme s půlkami
        if abs(step - 0.5) < 1e-9 and not _is_half_line(line):
            continue
        rows.append(_row_from_line(cdf, float(line), margin))
    return rows


# ---------- New API (requested: around mean using offsets) ----------

def generate_ou_lines_around_mean(
    mu: float,
    *,
    dist: DistName = "poisson",
    alpha: Optional[float] = None,
    offsets: List[float] = (-1.5, -0.5, 0.5, 1.5),
    min_line: float = 0.5,
    max_line: float = 99.5,
    max_lines: Optional[int] = None,
    margin: float = 0.0,
) -> List[LineRow]:
    """
    Dynamické linky kolem mean:
      center = round(mu)
      candidates = center + offset, kde offsety typicky [-1.5, -0.5, 0.5, 1.5]
    Držíme pouze půlkové linky (.5) a ořízneme do [min_line, max_line].

    Vhodné pro goals/corners/cards/shots – liší se jen konfigurací offsets a min/max.
    """
    mu = float(mu)
    cdf = _build_cdf(mu, dist, alpha)

    center = round(mu)
    candidates: List[float] = []
    for off in offsets:
        L = float(center + off)
        if L < min_line or L > max_line:
            continue
        if not _is_half_line(L):
            continue
        candidates.append(L)

    # dedupe + sort
    candidates = sorted(set(round(x, 2) for x in candidates))

    # omez počet linek: vyber nejbližší k mu, pak seřaď
    if max_lines is not None and len(candidates) > max_lines:
        candidates = sorted(candidates, key=lambda x: abs(x - mu))[: int(max_lines)]
        candidates = sorted(candidates)

    return [_row_from_line(cdf, line, margin) for line in candidates]


def pick_main_line(lines: List[LineRow], target_p_over: float = 0.5) -> LineRow:
    """
    Vybere “hlavní” linku: tu, kde je p_over nejblíž cíli (typicky 0.5).
    """
    if not lines:
        raise ValueError("lines je prázdné")
    t = float(target_p_over)
    return min(lines, key=lambda r: abs(r.p_over - t))


def to_dicts(lines: List[LineRow]) -> List[Dict[str, float]]:
    return [
        {
            "line": r.line,
            "p_over": r.p_over,
            "p_under": r.p_under,
            "odds_over_fair": r.odds_over_fair,
            "odds_under_fair": r.odds_under_fair,
            "odds_over": r.odds_over,
            "odds_under": r.odds_under,
        }
        for r in lines
    ]