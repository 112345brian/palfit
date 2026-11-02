"""Dev-only parity against the compiled `hallmodel` package (pip install palfit[parity])."""

import pytest

np = pytest.importorskip("numpy")
hallmodel = pytest.importorskip("hallmodel")

import palfit  # noqa: E402

SCENARIOS = {
    "steady": ("male", 85.0, 1.78, 32, 1.6, [2500.0] * 30),
    "carb_surge": ("male", 85.0, 1.78, 32, 1.6, [2500.0] * 5 + [3400.0] * 2 + [2500.0] * 23),
    "cut": ("male", 85.0, 1.78, 32, 1.6, [2500.0] + [1900.0] * 59),
    "bulk_female": ("female", 62.0, 1.68, 28, 1.6, [2200.0] + [2700.0] * 59),
    "high_pal": ("male", 95.0, 1.85, 41, 2.1, [3200.0] * 10 + [2600.0] * 40),
    "long_refeed": ("female", 70.0, 1.65, 50, 1.4, [1500.0] * 30 + [2400.0] * 60),
    "noisy_week": ("male", 78.0, 1.80, 25, 1.7, [2400, 3100, 1900, 2600, 3500, 2000, 2450] * 8),
}

# Exact agreement is expected (same equations, same forcing); the bound is far below
# anything that would matter and well above float round-off.
TOL_KG = 1e-9
TOL_KCAL = 1e-6


def lib_run(sex, bw, ht, age, pal, intake):
    ei = np.array(intake, float)
    return hallmodel.adult_weight(
        bw, ht, age, sex,
        EIchange=(ei - ei[0]).reshape(1, -1), EI=float(ei[0]), PAL=pal,
        pcarb_base=0.5, pcarb=0.5, days=len(ei) - 1, dt=1.0,
    )


@pytest.mark.parametrize("name", SCENARIOS)
def test_simulate_matches_hallmodel(name):
    sex, bw, ht, age, pal, intake = SCENARIOS[name]
    lib = lib_run(sex, bw, ht, age, pal, intake)
    t = palfit.simulate(
        height_m=ht, age_years=age, sex=sex, start_weight_kg=bw, pal=pal, intake_kcal=intake
    )
    pairs = {
        "Body_Weight": t.weight_kg,
        "Fat_Mass": t.fat_kg,
        "Lean_Mass": t.lean_kg,
        "Glycogen": t.glycogen_kg,
        "Extracellular_Fluid": t.ecf_kg,
        "Adaptive_Thermogenesis": t.adaptive_thermogenesis_kcal,
    }
    for key, mine in pairs.items():
        diff = np.abs(np.array(mine) - np.array(lib[key]).ravel()).max()
        assert diff < TOL_KG, (key, diff)
    ee = np.abs(np.array(t.expenditure_kcal) - np.array(lib["Total_Expenditure"]).ravel()).max()
    assert ee < TOL_KCAL, ee


def test_fit_matches_a_fit_wrapped_around_hallmodel():
    sex, bw, ht, age, _, _ = SCENARIOS["steady"]
    ei = [2500.0] + [2500.0 + (300.0 if (i // 7) % 2 else -300.0) for i in range(1, 60)]
    truth = lib_run(sex, bw, ht, age, 1.72, ei)["Body_Weight"].ravel()
    mine = palfit.fit_pal(
        height_m=ht, age_years=age, sex=sex, start_weight_kg=bw, intake_kcal=ei,
        observed_kg=[float(x) for x in truth],
    )
    assert mine.pal == pytest.approx(1.72, abs=2e-4)
