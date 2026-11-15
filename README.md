# Lab fitting script

A Jupyter notebook (`lab_fitting_scrip2025.ipynb`) for fitting physics-lab data with [iminuit](https://scikit-hep.org/iminuit/):
enter `x, y, dx, dy`, write the model function, run the regressor, and get the fit parameters with their uncertainties, a plot, and goodness-of-fit.

## Quick start

```python
measured_x = np.array([5.1, 10.5, 16.2]) * 1e-3
measured_y = np.array([1.8, 3.6, 5.4])
x_uncertainties = np.ones(measured_x.size) * 0.1e-3
y_uncertainties = np.array([0.01, 0.02, 0.03])

linear_fun = lambda x, a, b: a + x * b

reg = Chi2Reg(linear_fun, measured_x, measured_y, x_uncertainties, y_uncertainties)
opt = Minuit(reg, a=1, b=1)
opt.migrad()
reg.show(opt, x_title="t [ms]", y_title="d [m]")
```

## Regressors

| Class | Cost function | Uses uncertainties |
|---|---|---|
| `Chi2Reg` | χ² = Σ (y − f(x))² / (σ_y² + (f′(x)·σ_x)²) — effective-variance method | `dy` (required), `dx` (optional) |
| `LeastSquaresReg` | Σ (y − f(x))² | no (unweighted) |

## Changes (Nov 2026)

### 1. Uncertainty formatting – `fmt_pm(val, err, n=2, latex=True)`
Before, `show(..., sig_digits=5)` printed a fixed number of **decimal places**, so a parameter such as `1e-7` showed as `0.00000 ± 0.00000` and large values had meaningless digits.
Now the error is rounded to `sig_digits` **significant figures** (default 2) and the value is rounded to the same decimal place, as is standard in lab reports. Values outside 10⁻²…10³ get a common power-of-ten factor:

```
658776096.8 ± 12345678   ->  (6.59 ± 0.12)×10^8
-0.000889   ± 0.0000123  ->  (-8.89 ± 0.12)×10^-4
1.23456     ± 0.0234     ->  1.235 ± 0.023
```
Use `fmt_pm(v, e, latex=False)` for `print()`.

### 2. Fit-validity check – `check_fit(opt)`
iminuit warnings are still silenced (`warnings.simplefilter("ignore")`), but if Minuit's minimum is **not valid** a warning with the reason is printed (call limit reached, EDM above max / not converged, parameter at limit, inaccurate or forced covariance). Nothing is printed for a good fit.
It runs automatically inside `show`, `predict` and `inverse_predict`, and returns `True/False` if you want to use it yourself.

### 3. Stable derivative and sorted data
- `df/dx` in the effective-variance χ² is now a **central difference** `[f(x+h) − f(x−h)] / 2h` (error O(h²) instead of O(h) for the old forward difference).
- `h = 1e-6 × (x_max − x_min)`. Previously `h = (x[-1] − x[0]) / 10000`, which was negative or zero for unsorted data.
- Data is converted to float arrays and **sorted by x if needed** (x, y, dx, dy kept together; a note is printed when sorting happens).
- Also: `np.sum` instead of Python `sum`, Python lists are accepted, and the deprecated `iminuit.util.make_func_code` was removed.

### 4. New: interpolation / extrapolation with uncertainty – `reg.predict(opt, x_new)`
Returns `(y, sigma_y)` for a number or array `x_new`. `sigma_y` is propagated from the **full covariance matrix** of the parameters (correlations included):

  σ_y² = J C Jᵀ,  J_i = ∂f/∂p_i (numerical, central difference; fixed parameters are skipped)

```python
y0, dy0 = reg.predict(opt, 0.0)               # extrapolate to x = 0
print(fmt_pm(y0, dy0, latex=False))
```
`show()` has two new options:
- `x_range=(xmin, xmax)` – draw the fit beyond the data; the extrapolated part is dashed.
- `show_band=True` – draw the ±1σ confidence band of the fit.

### 5. New: inverse prediction – `reg.inverse_predict(opt, y_target, dy_target=0, x_range=None)`
Finds **all** x in `x_range` (default: the data range) where `f(x) = y_target`, using a scan and `scipy.optimize.brentq`. Returns arrays `(x, sigma_x)` with

  σ_x = √(σ_f(x)² + σ_target²) / |f′(x)|

A note is printed if f′(x) ≈ 0 at a solution (an extremum, where σ_x isn't meaningful), and a message if no solution is found (try a wider `x_range`).

```python
x_sol, dx_sol = reg.inverse_predict(opt, y_target=7.0, dy_target=0.05, x_range=(0, 30e-3))
```

## Requirements
`numpy`, `pandas`, `matplotlib`, `iminuit` (tested with 2.25), `scipy`, `scikit-learn`.
