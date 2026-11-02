"""Hall adult body-weight ODE, forward simulation.

Equations and constants are transcribed from INSP-RH/bw ``src/adult_weight.cpp``
(MIT, see NOTICE). Intake forcing follows upstream exactly: upstream reads the
intake change as ``EIchange[floor(t / dt)]``, so within the RK4 step from day
i-1 to day i the stages k1-k3 (t = i-1, i-1/2) see day i-1's value and only k4
(t = i) sees day i's.

Deliberate deviation from upstream: when the baseline-day carbohydrate intake
is zero, the carb/sodium coupling term of the ECF equation is dropped instead
of dividing by zero (upstream and ``hallmodel`` return NaN there).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

# Units: energy kcal, mass kg, sodium mg. Verbatim from Adult::getParameters().
ROG = 4206.501  # kcal/kg glycogen
NA_CONST = 3220.0
ZETA_NA = 3000.0
ZETA_CI = 4000.0
ROF = 9440.727  # kcal/kg fat
ROL = 1816.444  # kcal/kg lean tissue
GAMMA_F = 3.107075
GAMMA_L = 21.98853
ETA_F = 179.2543
ETA_L = 229.4455
BETA_TEF = 0.1
BETA_AT = 0.14
TAU_AT = 14.0  # days
C_CONST = 10.4 * (ROL / ROF)
ALFA1 = -(1.0 + ETA_L / ROL) * C_CONST
ALFA2 = -(1.0 + ETA_F / ROF)
G_BASE_KG = 0.5
GLYCOGEN_WATER = 3.7  # kg of glycogen + bound water per kg glycogen

# Mifflin-St Jeor, as in Adult::getRMR.
_RMR_BW, _RMR_HT, _RMR_AGE, _RMR_M, _RMR_F = 9.99, 625.0, 4.92, 5.0, 161.0

SEXES = ("male", "female")


@dataclass(frozen=True)
class Trajectory:
    """Day-indexed model output; index 0 is the initial condition."""

    pal: float
    rmr_kcal: float
    baseline_expenditure_kcal: float  # RMR * PAL, the steady-state expenditure
    weight_kg: tuple[float, ...]
    fat_kg: tuple[float, ...]
    lean_kg: tuple[float, ...]
    glycogen_kg: tuple[float, ...]  # glycogen only; its bound water is in weight_kg
    ecf_kg: tuple[float, ...]
    adaptive_thermogenesis_kcal: tuple[float, ...]
    expenditure_kcal: tuple[float, ...]


def _finite(name: str, x: float) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError(f"{name} must be a finite number, got {x!r}")
    return float(x)


def validate_person(
    height_m: float,
    age_years: float,
    sex: str,
    start_weight_kg: float,
    fat0_kg: Optional[float],
) -> None:
    if sex not in SEXES:
        raise ValueError(f"sex must be one of {SEXES}, got {sex!r}")
    if not 0.5 <= _finite("height_m", height_m) <= 2.5:
        raise ValueError(f"height_m must be in metres (0.5-2.5), got {height_m}")
    if _finite("age_years", age_years) < 18:
        raise ValueError("this is the adult model: age_years must be >= 18")
    if _finite("start_weight_kg", start_weight_kg) <= 0:
        raise ValueError("start_weight_kg must be positive")
    if fat0_kg is not None and not 0 < _finite("fat0_kg", fat0_kg) < start_weight_kg:
        raise ValueError("fat0_kg must be between 0 and start_weight_kg")


def validate_intake(intake_kcal: Sequence[float]) -> None:
    if len(intake_kcal) < 2:
        raise ValueError("need at least 2 days of intake to integrate")
    for i, v in enumerate(intake_kcal):
        if v is None:
            raise ValueError(f"intake_kcal[{i}] is None: fill intake gaps before calling")
        if _finite(f"intake_kcal[{i}]", v) < 0:
            raise ValueError(f"intake_kcal[{i}] is negative")


def rmr_kcal(weight_kg: float, height_m: float, age_years: float, sex: str) -> float:
    base = _RMR_BW * weight_kg + _RMR_HT * height_m -_RMR_AGE * age_years
    return base + _RMR_M if sex == "male" else base - _RMR_F


def _ecf_init(bw: float, ht: float, age: float, sex: str) -> float:
    """Silva's equation, Adult::getECFinit."""
    if sex == "male":
        return 0.025 * age + 9.57 * ht + 0.191 * bw - 12.4
    return -4.0 + 5.98 * ht + 0.167 * bw


def _baseline_fat_kg(bw: float, ht: float, age: float, sex: str) -> float:
    """Adult::getBaselineMass; used only when no measured fat mass is given."""
    bmi = bw / ht**2
    if sex == "male":
        pct = 0.14 * age + 37.31 * math.log(bmi) - 103.94
    else:
        pct = 0.14 * age + 39.96 * math.log(bmi) - 102.01
    return bw * pct / 100.0


def simulate(
    *,
    height_m: float,
    age_years: float,
    sex: str,
    start_weight_kg: float,
    pal: float,
    intake_kcal: Sequence[float],
    fat0_kg: Optional[float] = None,
    carb_fraction: float = 0.5,
    baseline_carb_fraction: float = 0.5,
) -> Trajectory:
    """Run the model forward, one step per entry of ``intake_kcal``.

    ``intake_kcal[0]`` is the baseline (energy-balance) intake. The model starts
    on day 0 at ``start_weight_kg`` with zero adaptive thermogenesis and
    expenditure equal to ``RMR * pal``.
    """
    validate_person(height_m, age_years, sex, start_weight_kg, fat0_kg)
    validate_intake(intake_kcal)
    if not 0.0 < _finite("pal", pal) < 5.0:
        raise ValueError(f"pal must be in (0, 5), got {pal}")
    for nm, v in (("carb_fraction", carb_fraction), ("baseline_carb_fraction", baseline_carb_fraction)):
        if not 0.0 <= _finite(nm, v) <= 1.0:
            raise ValueError(f"{nm} must be in [0, 1]")

    bw0, ht, age = float(start_weight_kg), float(height_m), float(age_years)
    n = len(intake_kcal)
    ei0 = float(intake_kcal[0])

    rmr = rmr_kcal(bw0, ht, age, sex)
    ecf_init = _ecf_init(bw0, ht, age, sex)
    fat_base = float(fat0_kg) if fat0_kg is not None else _baseline_fat_kg(bw0, ht, age, sex)
    lean_base = bw0 - (ecf_init + fat_base + GLYCOGEN_WATER * G_BASE_KG)
    if lean_base <= 0 or fat_base <= 0:
        raise ValueError("inputs imply non-positive fat or lean mass; check units and fat0_kg")
    delta = ((1.0 - BETA_TEF) * pal - 1.0) * rmr / bw0
    k_const = rmr * pal - GAMMA_L * lean_base - GAMMA_F * fat_base - delta * bw0
    cib = baseline_carb_fraction * ei0
    kg_const = cib / (G_BASE_KG**2)
    delta_ei = [float(v) - ei0 for v in intake_kcal]

    def fat_of(lean: float) -> float:
        return fat_base * math.exp(ROL * (lean - lean_base) / (ROF * C_CONST))

    def d_g(g: float, dei: float) -> float:
        return (carb_fraction * (ei0 + dei) - kg_const * g * g) / ROG

    def d_at(at: float, dei: float) -> float:
        return (BETA_AT * dei - at) / TAU_AT

    def d_ecf(ecf: float, dei: float) -> float:
        ci = carb_fraction * (ei0 + dei)
        ci_term = ZETA_CI * (1.0 - ci / cib) if cib else 0.0
        return (-ZETA_NA * (ecf - ecf_init) - ci_term) / NA_CONST

    def d_l(lean: float, g: float, at: float, ecf: float, dei: float) -> float:
        fat = fat_of(lean)
        weight = lean + fat + ecf + GLYCOGEN_WATER * g
        r3 = k_const + delta * weight + BETA_TEF * dei + at - (ei0 + dei) + ROG * d_g(g, dei)
        r = (r3 + GAMMA_L * lean + GAMMA_F * fat) / (ALFA1 + ALFA2 * fat)
        return r * (C_CONST / ROL)

    def expenditure(lean: float, fat: float, ecf: float, g: float, at: float, dei: float) -> float:
        weight = lean + fat + ecf + GLYCOGEN_WATER * g
        return k_const + GAMMA_L * lean + GAMMA_F * fat + delta * weight + BETA_TEF * dei + at

    def rk4(f, y: float, dei_a: float, dei_b: float) -> float:
        # k1-k3 evaluated at t = i-1 (+ half step): previous day's forcing; k4 at t = i.
        k1 = f(y, dei_a)
        k2 = f(y + 0.5 * k1, dei_a)
        k3 = f(y + 0.5 * k2, dei_a)
        k4 = f(y + k3, dei_b)
        return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

    at = [0.0] * n
    ecf = [ecf_init] * n
    g = [G_BASE_KG] * n
    lean = [lean_base] * n
    fat = [fat_base] * n
    bw = [bw0] * n
    ee = [0.0] * n
    ee[0] = expenditure(lean[0], fat[0], ecf[0], g[0], at[0], delta_ei[0])

    for i in range(1, n):
        da, db = delta_ei[i - 1], delta_ei[i]
        at[i] = rk4(d_at, at[i - 1], da, db)
        ecf[i] = rk4(d_ecf, ecf[i - 1], da, db)
        g[i] = rk4(d_g, g[i - 1], da, db)

        g_avg = 0.5 * (g[i] + g[i - 1])
        at_avg = 0.5 * (at[i] + at[i - 1])
        ecf_avg = 0.5 * (ecf[i] + ecf[i - 1])
        y = lean[i - 1]
        k1 = d_l(y, g[i - 1], at[i - 1], ecf[i - 1], da)
        k2 = d_l(y + 0.5 * k1, g_avg, at_avg, ecf_avg, da)
        k3 = d_l(y + 0.5 * k2, g_avg, at_avg, ecf_avg, da)
        k4 = d_l(y + k3, g[i], at[i], ecf[i], db)
        lean[i] = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

        fat[i] = fat_of(lean[i])
        bw[i] = fat[i] + lean[i] + ecf[i] + GLYCOGEN_WATER * g[i]
        ee[i] = expenditure(lean[i], fat[i], ecf[i], g[i], at[i], db)

    return Trajectory(
        pal=float(pal),
        rmr_kcal=rmr,
        baseline_expenditure_kcal=rmr * pal,
        weight_kg=tuple(bw),
        fat_kg=tuple(fat),
        lean_kg=tuple(lean),
        glycogen_kg=tuple(g),
        ecf_kg=tuple(ecf),
        adaptive_thermogenesis_kcal=tuple(at),
        expenditure_kcal=tuple(ee),
    )
