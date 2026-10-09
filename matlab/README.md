# MATLAB version of the workshop notebooks

The six Jupyter notebooks of this repository as MATLAB scripts (`.m` files with `%%` sections), written for the
MATLAB port of PySINDy, `pysindy_matlab` (package `+ps`, base MATLAB only, no toolboxes).

| notebook | script | |
|---|---|---|
| `1_SINDy_ODE.ipynb` | `SINDy_1_ODE.m` | Lorenz system: derivatives, library, sequential thresholded least squares, forecast; then the same with `ps.SINDy` |
| `2_SINDy_challenges.ipynb` | `SINDy_2_challenges.m` | noise, the threshold lambda, polynomial vs. trigonometric library (pendulum) |
| `3_SINDy_PDE.ipynb` | `SINDy_3_PDE.m` | Kuramoto-Sivashinsky equation from `data/KS_data.mat` |
| `4_SINDy_model_selection.ipynb` | `SINDy_4_model_selection.m` | lambda path, k-fold cross-validation, AIC, stability selection |
| `5_Ensemble-SINDy.ipynb` | `SINDy_5_Ensemble.m` | E-SINDy: bagging, noise study, coefficient distributions |
| `6_weak_SINDy_PDE.ipynb` | `SINDy_6_weak_PDE.m` | weak-form SINDy on noisy KS data |

`utils/` replaces `utils.py` (`plot_sindy.m`, `plot_pendulum.m`, `plot_sindy_pendulum.m`), holds the two right-hand sides
(`lorenz_system.m`, `pendulum_ode.m`) and `setup_pysindy.m`.

## Run

Open a script and press Run, or type its name. The scripts find their own folder (`utils`, `../data`), so the Current
Folder does not matter. `setup_pysindy.m` looks for the port in `$PYSINDY_MATLAB`, then `~/data/GitHub/pysindy_matlab`;
edit it if the port is somewhere else. The scripts write no files.

## What differs from the notebooks

* **Random numbers** (noise, folds, subsamples, bootstrap samples, subdomains) come from MATLAB's generator, not numpy's,
  so noisy results agree with the notebooks' printed numbers in character, not in the last digits. Everything that does not
  depend on random draws is reproduced to the printed precision (below). `np.std(x)` is the population standard deviation:
  `std(x(:), 1)`.
* **Integrator**: `ode45` (Runge-Kutta 4(5), as `solve_ivp`'s RK45) with `RelTol = AbsTol = 1e-12`. A chaotic trajectory
  can differ from SciPy's in the last digits after t = 10.
* **Cross-validation** (`4`): scikit-learn's `KFold` is replaced by a random permutation cut into 10 folds; `cvpartition`
  would need a toolbox.
* **PDE libraries** (`3`, `6`): the notebooks call `PDELibrary(function_library=..., ...)` and `WeakPDELibrary`, which the
  PySINDy version the port follows (v2.1) no longer has, so these two notebooks need an older PySINDy. In the port, and in PySINDy v2.1, a `PDELibrary` holds the
  derivatives, products are written `poly * pde`, and the weak form is `ps.WeakSINDy`.
* Python's private calls such as `ps.FiniteDifference(axis=1)._differentiate(u, t=dt)` are `ps.differentiation.FiniteDifference('axis', 2).call(u, dt)`
  (MATLAB axes count from 1: Python's axis 0 is MATLAB's 1).

## Checked against the notebooks' own output

| script | compared with | result |
|---|---|---|
| `1` | coefficient matrix and the two PySINDy models printed in the notebook | identical to the printed digits |
| `2` | the two printed Lorenz models (with numpy's noise handed to the script) and the two pendulum models | identical |
| `3` | `-0.994 u_11 - 0.996 u_1111 - 0.992 u u_1` | identical |
| `4` | with numpy's noise: the number of terms of all 50 models of the lambda path, and the final stability-selection model (Python PySINDy v2.1 as reference); cross-validated threshold | 50 of 50 equal; coefficients 3e-7; threshold 0.3728 as in the notebook |
| `5` | with numpy's noise: the SINDy columns of the printed table | identical (E-SINDy columns are random, same trend) |
| `6` | both printed PDEs | identical to the printed digits |

"With numpy's noise" means a test in which `randn` was replaced by the numbers numpy draws with the notebook's seeds; the
scripts here use `randn` and `rng` normally.
