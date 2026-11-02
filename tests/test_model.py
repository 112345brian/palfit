import math

import pytest

import palfit
from palfit import _model

P = dict(height_m=1.78, age_years=32, sex="male", start_weight_kg=85.0)


def sim(intake, pal=1.6, **kw):
    return palfit.simulate(pal=pal, intake_kcal=intake, **{**P, **kw})


def test_steady_state_is_invariant():
    # Steady only when baseline intake equals RMR * PAL; the model does not assume it.
    maintenance = sim([2500.0] * 2).baseline_expenditure_kcal
    t = sim([maintenance] * 40)
    assert max(abs(w - 85.0) for w in t.weight_kg) < 1e-9
    assert all(abs(e - t.baseline_expenditure_kcal) < 1e-6 for e in t.expenditure_kcal)


def test_deficit_loses_weight_monotonically_and_stays_positive():
    t = sim([2500.0] + [1800.0] * 89)
    assert all(b < a for a, b in zip(t.weight_kg, t.weight_kg[1:]))
    assert min(t.fat_kg) > 0 and min(t.lean_kg) > 0


def test_energy_balance_closes_within_tissue_efficiency_losses():
    # Stored energy change / (intake - expenditure) is ~0.96-0.97: the remainder is the
    # model's tissue deposition/turnover efficiency terms. Guard against gross drift only.
    intake = [2500.0] + [2200.0] * 89
    t = sim(intake)
    stored = lambda i: (
        _model.ROF * t.fat_kg[i] + _model.ROL * t.lean_kg[i] + _model.ROG * t.glycogen_kg[i]
    )
    net = sum(intake[i] - t.expenditure_kcal[i] for i in range(1, 90))
    assert 0.93 < (stored(89) - stored(0)) / net < 1.0


def test_carb_surge_moves_glycogen_and_water_not_fat():
    t = sim([2500.0] * 5 + [3400.0] * 2 + [2500.0] * 23)
    assert max(t.glycogen_kg) - 0.5 > 0.01
    assert max(t.fat_kg) - t.fat_kg[0] < 0.1


def test_forcing_split_matches_upstream_convention():
    # Pinned to hallmodel 2026.10.21.16 (adult_weight, same inputs). Using day i's intake in
    # all four RK4 stages, as an earlier port did, gives a visibly different day-5 weight.
    t = sim([2500.0] * 5 + [3400.0] * 2 + [2500.0] * 23)
    assert t.weight_kg[5] == pytest.approx(84.87534667, abs=1e-6)


def test_zero_baseline_intake_is_finite():
    t = sim([0.0] + [2500.0] * 29)
    assert all(math.isfinite(x) for x in t.weight_kg + t.expenditure_kcal)


@pytest.mark.parametrize(
    "kw",
    [
        dict(sex="other"),
        dict(sex=None),
        dict(height_m=178.0),
        dict(age_years=12),
        dict(start_weight_kg=-1),
        dict(start_weight_kg=float("nan")),
        dict(fat0_kg=90.0),
        dict(pal=0.0),
        dict(pal=float("inf")),
        dict(carb_fraction=1.5),
    ],
)
def test_bad_person_inputs_raise(kw):
    with pytest.raises(ValueError):
        sim([2500.0] * 5, **kw)


@pytest.mark.parametrize(
    "intake",
    [[2500.0], [], [2500.0, None, 2500.0], [2500.0, float("nan")], [2500.0, -5.0], [2500.0, "x"]],
)
def test_bad_intake_raises(intake):
    with pytest.raises(ValueError):
        sim(intake)


def test_measured_fat_is_respected():
    t = sim([2500.0] * 3, fat0_kg=15.0)
    assert t.fat_kg[0] == 15.0
