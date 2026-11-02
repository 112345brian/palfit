# palfit

Fit a **physical activity level (PAL)**, and from it a maintenance-calorie estimate, to a daily intake log and a gappy body-weight log, using Kevin Hall's adult dynamic body-weight model. Pure Python, no runtime dependencies.

The forward model is a port of the R package [INSP-RH/bw](https://github.com/INSP-RH/bw) (MIT). The compiled PyPI package `hallmodel` offers the same forward model; **neither it nor `bw` fits PAL to data**. That inversion is what this package adds. See `NOTICE`.

Source and issues: https://github.com/112345brian/palfit

## Install

```bash
pip install palfit
```

## Use

```python
import palfit

result = palfit.fit_pal(
    height_m=1.78, age_years=32, sex="male",
    start_weight_kg=85.0,
    intake_kcal=intake,        # one value per day, no gaps; intake[0] is the baseline day
    observed_kg=weights,       # same length; None on days without a reading
)

if result.hit_lower or result.hit_upper:
    print("PAL pinned at a bound: the weights cannot be explained by this model")
print(result.pal, result.maintenance_kcal)
result.trajectory.fat_kg, .lean_kg, .glycogen_kg, .ecf_kg, .weight_kg, .expenditure_kcal
```

* `fit_pal(*, height_m, age_years, sex, start_weight_kg, intake_kcal, observed_kg, fat0_kg=None, carb_fraction=0.5, baseline_carb_fraction=0.5, lo=1.1, hi=2.5, tol=1e-4, tail_days=7) -> FitResult`
* `simulate(*, height_m, age_years, sex, start_weight_kg, pal, intake_kcal, fat0_kg=None, carb_fraction=0.5, baseline_carb_fraction=0.5) -> Trajectory`
* `FitResult`: `pal`, `maintenance_kcal`, `sse_kg2`, `n_obs`, `hit_lower`, `hit_upper`, `trajectory`. Frozen dataclasses of tuples.
* Raises `ValueError` for: unknown sex, age under 18, height not in metres, non-finite or negative values, `None` in intake, length mismatch, fewer than 2 observed weights, `fat0_kg` outside (0, weight).
* No file, database, network or global state; inputs are never mutated.

## What PAL means here

PAL = total daily energy expenditure divided by resting metabolic rate (RMR, Mifflin-St Jeor from the **start weight**). `fit_pal` searches PAL in `[lo, hi]` (golden section) to minimise the squared difference between simulated and observed weights.

* `maintenance_kcal` is the model's expenditure averaged over the **last `tail_days` of the simulated period**, which includes the model's adaptive-thermogenesis and thermic-effect terms for that period. It is not simply `RMR * PAL` (that value is `trajectory.baseline_expenditure_kcal`, the steady-state figure at the start).
* The model does **not** assume day-0 intake equals expenditure. Day 0 is the model's "energy balance" reference for deviations (`intake - intake[0]`), and PAL is free. Pick a typical day as `intake[0]`, not a fast day or a feast.
* Glycogen is reported as glycogen only; its bound water (3.7 kg per kg glycogen) is inside `weight_kg`.

## Limitations (read these)

* **Garbage in, garbage out on intake.** Logged intake that is 10% low or high moves the estimated maintenance by about -240 / +240 kcal in the test case below. The fit cannot tell a bad log from a different metabolism.
* **The start weight is treated as exact.** A start weight off by 0.5 kg shifted the fit by about 0.03 PAL (about 50 kcal) in the test case. Use a smoothed or averaged start value.
* **Short windows are noisy.** In the synthetic study, 28 days with 0.4 kg noise gave an RMSE of about 0.027 PAL (roughly 50 kcal).
* **Flat error surface is not detected.** With almost no intake variation or a very short window, many PALs fit about equally well and the optimum is poorly determined. Only hitting a bound is flagged.
* Adult model only; daily steps only; sodium input not supported; no uncertainty interval.
* Not medical advice. The model is a published population-level approximation; this package has been tested against that model, **not against measured individuals**.

## What is verified, and how

Everything below was run on macOS arm64 with `hallmodel==2026.10.21.16`. "Not verified" items are listed explicitly.

| Claim | Evidence |
|---|---|
| `simulate` equals `hallmodel.adult_weight` on 7 scenarios (steady, carb surge, cut, female bulk, high PAL, long refeed, noisy week) | Test tolerances 1e-9 kg and 1e-6 kcal on all series; measured worst case on the carb-surge scenario: 0.0 kg weight, 4.5e-13 kcal expenditure |
| Intake forcing matches upstream C++ (`EIchange[floor(t/dt)]`: RK4 stages k1-k3 use the previous day's value, k4 the current day's) | Read from `adult_weight.cpp`; a port using the current day in all stages differed from `hallmodel` by up to 0.43 kg and 12 kcal before the change |
| PAL recovery on model-generated data (truth 1.72, 90 days, 0.15 kg noise, a third of days missing, 30 seeds) | bias 0.0002, RMSE 0.0015, worst error 0.003 PAL |
| Same, 28 days / 56 days, 0.15 kg noise | RMSE 0.0102 / 0.0036 PAL |
| Energy balance closes to within the model's tissue-efficiency losses | stored/net ratio 0.96-0.97 over 30-365 days (a test guards 0.93-1.0) |
| Zero baseline intake gives finite output | tested; this deliberately differs from upstream/`hallmodel`, which return NaN |

**Not verified:** agreement with the R package itself (R was not run; the claim rests on the C++ source and `hallmodel`); correctness of `hallmodel`'s own parity claim (its upstream "contract" branch is not publicly visible); behaviour on Python 3.11 and on non-macOS platforms; real-world accuracy against measured expenditure (doubly labelled water or similar); PAL uncertainty estimates; unimodality of the error surface over `[1.1, 2.5]` beyond the cases above.

## Development

```bash
pip install -e '.[test]'          # unit tests, no network or data needed
pip install -e '.[parity]'        # adds numpy + hallmodel for the dev-only parity tests
pytest
```

## Before release

Run `python tools/check_release.py`; it fails while any placeholder marker remains in `LICENSE`, `NOTICE`, `README.md` or `pyproject.toml`, or if `authors` is missing. Re-check that the distribution name is unclaimed on PyPI immediately before uploading.

## Licence

MIT. Model code derived from INSP-RH/bw (MIT, Instituto Nacional de Salud Pública); see `LICENSE` and `NOTICE`.
