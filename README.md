
## Experiment 5: Parameter Sweep Runner

`exp5_sweeps.py` provides a configurable CLI for running one-dimensional parameter sweeps across the unstructured, linear, and causal bandit suites described in the accompanying report. Each sweep executes the registered policies with a shared random seed list and stores both JSON summaries and PDF plots under `results/exp5_sweeps/<family>/` by default.

### Usage

```bash
python -m experiments.exp5_sweeps <family> [options]
````

Where `<family>` selects one of:

* `unstructured`
* `linear`
* `causal`

---

### Unstructured bandits

```bash
python -m experiments.exp5_sweeps unstructured [--sweeps ...]
```

* `--sweeps` (default: `sigma gap arms heavy drift`): names of the sweeps to execute.
* `--sigma-levels`, `--gap-levels`, `--arm-levels`, `--nu-levels`, `--drift-levels`: grids for observation noise, gap size, number of arms, Student-(t) degrees of freedom, and drift magnitude, respectively.
* `--sigma`, `--gap`, `--arms`, `--nu`, `--drift`: baseline parameters used when a sweep is not active.
* `--horizon`, `--seeds`, `--results-dir`: common run length, random seeds, and output directory.

Each sweep reports mean regret and 95% confidence intervals per policy; the JSON metadata records the sweep property and concrete values.

---

### Linear bandits

```bash
python -m experiments.exp5_sweeps linear [--sweeps ...]
```

* `--sweeps` (default: `sigma dimension conditioning signal misspec context`).
* `--sigma-levels`, `--dim-levels`, `--condition-levels`, `--signal-levels`, `--misspec-levels`, `--context-levels`: grids for the corresponding properties (observation noise, feature dimension, feature conditioning, signal norm, misspecification strength, context exposure).
* `--signal`: baseline (\lVert \theta^* \rVert_2) used when not sweeping signal magnitude.
* `--feature-seed`: RNG seed for feature matrices so repeated runs reuse identical contexts.
* Shared options: `--horizon`, `--seeds`, `--results-dir`.

---

### Causal bandits

```bash
python -m experiments.exp5_sweeps causal [--sweeps ...]
```

* `--sweeps` (default: `confounding overlap backdoor noise misspec latent`).
* `--conf-levels`, `--tau-levels`, `--backdoor-levels`, `--outcome-noise-levels`, `--eta-levels`, `--latent-levels`: grids for confounding strength, propensity slope, back-door set size, outcome noise, structural misspecification, and latent confounding strength.
* Baseline SCM parameters (`--backdoor-size`, `--conf-t`, `--conf-y`, `--tau`, `--sigma-z`, `--sigma-t`, `--sigma-y`, `--nonlinear-eta`, `--latent-t`, `--latent-y`) configure the data-generating process when a sweep is inactive.
* Estimation controls: `--refit-every` (policy refit cadence), `--clip` (propensity weight clipping), `--obs-samples` / `--obs-seed` (size and seed for observational datasets), `--scm-seed` (interventional SCM seed).
* Shared options: `--horizon`, `--seeds`, `--results-dir`.

---

### Outputs and logging

For every sweep, the runner:

1. Evaluates all registered policies on the specified grid using the provided seeds.
2. Logs per-point summary lines to stdout.
3. Writes `<tag>_summary.json` files containing mean regret, confidence intervals, and sweep metadata.
4. Emits companion PDF plots (e.g., `sigma_curve.pdf`) comparing algorithms over the swept property.

Override `--results-dir` to redirect outputs; existing directories are created on demand.
