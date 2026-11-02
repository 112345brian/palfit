import math
import random
import statistics

import pytest

import palfit

H = dict(height_m=1.78, age_years=32, sex="male", start_weight_kg=85.0)


def swing_intake(days):
    return [2500.0] + [2500.0 + (300.0 if (i // 7) % 2 else -300.0) for i in range(1, days)]


def noisy(truth, sd, seed, keep=0.67):
    r = random.Random(seed)
    return [t + r.gauss(0, sd) if r.random() < keep else None for t in truth]


def test_noise_free_recovery():
    ei = swing_intake(90)
    truth = palfit.simulate(pal=1.72, intake_kcal=ei, **H)
    res = palfit.fit_pal(intake_kcal=ei, observed_kg=list(truth.weight_kg), **H)
    assert res.pal == pytest.approx(1.72, abs=2e-4)
    assert res.maintenance_kcal == pytest.approx(sum(truth.expenditure_kcal[-7:]) / 7, rel=2e-3)
    assert not (res.hit_lower or res.hit_upper)


def test_recovery_study_with_gaps_and_noise():
    # 90 days, 0.15 kg noise, a third of days missing, 20 seeds. Measured on this
    # configuration: bias ~0.0002, RMSE ~0.0015, worst ~0.003 PAL.
    ei = swing_intake(90)
    truth = palfit.simulate(pal=1.72, intake_kcal=ei, **H)
    errs = [
        palfit.fit_pal(intake_kcal=ei, observed_kg=noisy(truth.weight_kg, 0.15, s), **H).pal - 1.72
        for s in range(20)
    ]
    assert abs(statistics.mean(errs)) < 0.003
    assert math.sqrt(statistics.mean(e * e for e in errs)) < 0.006
    assert max(abs(e) for e in errs) < 0.012


def test_impossible_trajectory_flags_a_bound():
    res = palfit.fit_pal(intake_kcal=[2500.0] * 60, observed_kg=[85.0] + [60.0] * 59, **H)
    assert res.hit_lower or res.hit_upper


def test_ordinary_fit_flags_nothing():
    ei = swing_intake(60)
    truth = palfit.simulate(pal=1.5, intake_kcal=ei, **H)
    res = palfit.fit_pal(intake_kcal=ei, observed_kg=noisy(truth.weight_kg, 0.1, 3), **H)
    assert not (res.hit_lower or res.hit_upper)


def test_all_none_raises_instead_of_returning_a_bound():
    with pytest.raises(ValueError, match="at least 2 observed"):
        palfit.fit_pal(intake_kcal=[2500.0] * 10, observed_kg=[None] * 10, **H)


def test_single_observation_raises():
    with pytest.raises(ValueError):
        palfit.fit_pal(intake_kcal=[2500.0] * 10, observed_kg=[85.0] + [None] * 9, **H)


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        palfit.fit_pal(intake_kcal=[2500.0] * 10, observed_kg=[85.0] * 9, **H)


@pytest.mark.parametrize("bad", [float("nan"), -1.0, 0.0, "85", True])
def test_bad_observation_values_raise(bad):
    obs = [85.0] * 10
    obs[4] = bad
    with pytest.raises(ValueError):
        palfit.fit_pal(intake_kcal=[2500.0] * 10, observed_kg=obs, **H)


def test_none_in_intake_raises():
    ei = [2500.0] * 10
    ei[3] = None
    with pytest.raises(ValueError, match="intake gaps"):
        palfit.fit_pal(intake_kcal=ei, observed_kg=[85.0] * 10, **H)


def test_result_is_immutable():
    ei = swing_intake(20)
    res = palfit.fit_pal(intake_kcal=ei, observed_kg=list(palfit.simulate(pal=1.6, intake_kcal=ei, **H).weight_kg), **H)
    with pytest.raises(Exception):
        res.pal = 2.0  # type: ignore[misc]
    assert len(res.trajectory.weight_kg) == 20


def test_inputs_are_not_mutated():
    ei = swing_intake(20)
    obs = [85.0] * 20
    ei_copy, obs_copy = list(ei), list(obs)
    palfit.fit_pal(intake_kcal=ei, observed_kg=obs, **H)
    assert ei == ei_copy and obs == obs_copy


def test_accepts_tuples_and_ints():
    ei = tuple(int(x) for x in swing_intake(20))
    obs = tuple(int(w) for w in [85] * 20)
    assert palfit.fit_pal(intake_kcal=ei, observed_kg=obs, **H).n_obs == 20
