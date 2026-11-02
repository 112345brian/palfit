"""Invert the Hall model: observed weights -> the PAL that reproduces them."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

from palfit._model import (
    Trajectory,
    simulate,
    validate_intake,
    validate_person,
)

_INVPHI = (math.sqrt(5) - 1) / 2


@dataclass(frozen=True)
class FitResult:
    pal: float
    maintenance_kcal: float  # mean model expenditure over the last ``tail_days``
    sse_kg2: float
    n_obs: int
    hit_lower: bool  # optimum pinned at the lower bound: the fit is not trustworthy
    hit_upper: bool
    trajectory: Trajectory


def _observed(observed_kg: Sequence[Optional[float]], n: int) -> list[Optional[float]]:
    if len(observed_kg) != n:
        raise ValueError("intake_kcal and observed_kg must be the same length")
    out: list[Optional[float]] = []
    for i, v in enumerate(observed_kg):
        if v is None:
            out.append(None)
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise ValueError(f"observed_kg[{i}] must be None or a positive finite number, got {v!r}")
        out.append(float(v))
    return out


def fit_pal(
    *,
    height_m: float,
    age_years: float,
    sex: str,
    start_weight_kg: float,
    intake_kcal: Sequence[float],
    observed_kg: Sequence[Optional[float]],
    fat0_kg: Optional[float] = None,
    carb_fraction: float = 0.5,
    baseline_carb_fraction: float = 0.5,
    lo: float = 1.1,
    hi: float = 2.5,
    tol: float = 1e-4,
    tail_days: int = 7,
) -> FitResult:
    """Golden-section search for the PAL whose simulated weight best matches ``observed_kg``.

    ``observed_kg[i]`` is ``None`` on days without a reading. ``intake_kcal`` must
    have no gaps. At least two observations are required; with none the error
    surface is flat and a search would silently return a bound.

    ``hit_lower`` / ``hit_upper`` are set when the optimum lies within ``10 * tol``
    of a bound; treat such a result as "the data cannot be explained", not as an
    estimate.
    """
    if not 0 < tol < 1:
        raise ValueError("tol must be in (0, 1)")
    if not 0 < lo < hi:
        raise ValueError("need 0 < lo < hi")
    if not isinstance(tail_days, int) or tail_days < 1:
        raise ValueError("tail_days must be a positive integer")
    validate_person(height_m, age_years, sex, start_weight_kg, fat0_kg)
    validate_intake(intake_kcal)
    obs = _observed(observed_kg, len(intake_kcal))
    n_obs = sum(o is not None for o in obs)
    if n_obs < 2:
        raise ValueError(f"need at least 2 observed weights, got {n_obs}")

    def run(pal: float) -> Trajectory:
        return simulate(
            height_m=height_m,
            age_years=age_years,
            sex=sex,
            start_weight_kg=start_weight_kg,
            pal=pal,
            intake_kcal=intake_kcal,
            fat0_kg=fat0_kg,
            carb_fraction=carb_fraction,
            baseline_carb_fraction=baseline_carb_fraction,
        )

    def sse(traj: Trajectory) -> float:
        total = sum((s - o) ** 2 for s, o in zip(traj.weight_kg, obs) if o is not None)
        return total if math.isfinite(total) else math.inf

    def cost(pal: float) -> float:
        return sse(run(pal))

    a, b = lo, hi
    c = b - _INVPHI * (b - a)
    d = a + _INVPHI * (b - a)
    fc, fd = cost(c), cost(d)
    while abs(b - a) > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - _INVPHI * (b - a)
            fc = cost(c)
        else:
            a, c, fc = c, d, fd
            d = a + _INVPHI * (b - a)
            fd = cost(d)
    pal = (a + b) / 2
    traj = run(pal)
    tail = traj.expenditure_kcal[-tail_days:]
    eps = 10 * tol
    return FitResult(
        pal=pal,
        maintenance_kcal=sum(tail) / len(tail),
        sse_kg2=sse(traj),
        n_obs=n_obs,
        hit_lower=pal <= lo + eps,
        hit_upper=pal >= hi - eps,
        trajectory=traj,
    )
