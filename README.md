
## Installation

- **Python:** 3.9+ (tested on CPython).
- **Dependencies:** `numpy`, `tqdm`, `matplotlib`; `scipy` is optional but enables smoother gradients/statistics in the analysis CLI.
- **Editable install (recommended):**

  ```bash
  python -m pip install --upgrade pip
  python -m pip install -e .
  ```

Run the commands below from the repository root (so `perfcba` is importable) or inside the environment created by the editable install.

## Causal bandit utilities

The repository now includes helpers for constructing discrete causal bandit
environments matching the setup in “Graph Learning is Suboptimal in Causal
Bandits”:

- `experiments.causal_envs` samples random acyclic SCMs, enumerates intervention
  spaces, and exposes lightweight dataclasses describing the reward node and its
  parent set.
- `experiments.causal_setup` wraps the SCMs into the new
  `Bandit.CausalInterventionalBandit` class and plugs them into the generic
  `Experiment` harness, so any existing policy can be evaluated in this causal
  setting with only a few lines of code.

### SCM generation modes

`experiments.causal_envs.CausalBanditConfig` exposes an ``scm_mode`` switch so
you can choose between the original Beta–Dirichlet sampler (`beta_dirichlet`)
and a `reference` mode that mirrors the official
[Unknown-Graph-Causal-Bandits](https://github.com/ban-epfl/Unknown-Graph-Causal-Bandits)
environment (Dirichlet reward prior plus the `parent_effect` mixtures used in
the paper). The tau-study drivers accept `--scm-mode` and `--parent-effect`, so
you can, for example, run `--scm-mode reference --parent-effect 0.7` to match
the paper’s default setting.

## Tau-scheduled study CLI

Run the full budget-allocation study via:

```bash
python -m perfcba.experiments.run_tau_study \
  --vary graph_density \
  --n 50 --ell 2 --k 2 --m 2 --T 10000 \
  --seeds 0:19 \
  --scheduler interleaved \
  --output-dir results/tau_study/graph_density
```

The CLI sweeps `tau`, generates SCMs via `CausalBanditConfig`, interleaves
structure learning with exploitation, records JSONL summaries, and emits
heatmaps under the specified output directory (e.g.,
`results/tau_study/graph_density/`). Each JSONL row captures the seed, knob
value, scheduler, tau, and metrics, so downstream tools can consume the file
directly.

`--structure-backend` selects which parent learner is wrapped by the schedulers.
`budgeted_raps` (default) reuses the official `RAPSUCB` implementation via
`perfcba.budgeted_raps`, while `proxy` keeps the lightweight heuristic learner
for ablations.  When the budgeted backend is active the CLI also honors
`--algo-eps`, `--algo-delta`, and `--raps-delta`, mirroring the finite-sample
constants from the reference code.  You can sweep those gaps directly via
`--vary algo_eps` or `--vary algo_delta`, optionally overriding the default
grids with `--algo-eps-grid` / `--algo-delta-grid`.

### Single-knob performance curves

For “fixed environment” studies—e.g., hold graph density, alphabet, and reward
priors constant while sweeping `tau`—run:

```bash
python -m perfcba.experiments.run_tau_study \
  --vary tau \
  --tau-grid 0.0 0.1 0.2 0.4 0.6 0.8 1.0 \
  --n 50 --ell 2 --k 2 --m 2 --T 10000 \
  --graph-grid 0.3 --parent-grid 2 --node-grid 50 \
  --algo-eps 0.05 --algo-delta 0.05 \
  --seeds 0:19 \
  --scheduler interleaved \
  --output-dir results/tau_study/tau_sweep_fixed_env
```

When sweeping `--vary intervention_size`, pass `--intervention-grid <values>` (including optional `start[:step]:stop` ranges) to control the exact `m` settings such as `{1, 2, 3, 4, 5}`.

Switch `--vary algo_eps` or `--vary algo_delta` (with matching grids)
to study the ancestral/reward-gap knobs at a fixed `tau`. After the sweep,
turn the JSONL output into a line plot showing cumulative regret, time to
optimality, simple regret, and optimal-action rate versus the knob:

```bash
python -m perfcba.experiments.analysis \
  --results results/tau_study/tau_sweep_fixed_env/results.jsonl \
  --vary tau \
  --metrics cumulative_regret tto simple_regret optimal_rate \
  --plot-mode line \
  --out-dir results/tau_study/tau_sweep_fixed_env/analysis
```

Line plots collapse the tau/knob grid down to the swept knob and plot all
requested metrics on shared axes (distinct colors per metric). Omit the flag—or
pass `--plot-mode heatmap` explicitly—to retain the original heat-map output
with gradient overlays.

Additional knobs introduced in this revision:

- `--hybrid-arms` enables “mixed control” exploitation arms that append
  high-priority non-parents when fewer than `m` parents are known. Use
  `--hybrid-max-fillers` / `--hybrid-max-hybrid-arms` to keep the enumeration
  bounded on large graphs.
- `--structure-mc-samples`, `--arm-mc-samples`, and
  `--optimal-mean-mc-samples` bubble up the Monte Carlo budgets that were
  previously only controllable via the `--small`/`--very-small` presets, so
  experiments with tiny reward gaps can dial up accuracy explicitly.
- `--scm-epsilon` / `--scm-delta` bound every conditional probability away from
  zero and one (reward CPTs use the tighter of the two), making it easy to
  enforce robust SCM generation regimes.

To probe reward noise, you can either set `--reward-logit-scale <value>` for a
single configuration or sweep it via `--vary arm_variance`. The plots now
annotate this axis as “Reward Logit Scale,” which rescales Bernoulli logits
before sampling rewards: values below 1 push arm means toward 0.5 (higher
variance), while values above 1 make the arms sharper and reduce variance.

### Regret-curve driver

Use the reduced driver to focus on classical cumulative regret curves for a
single `tau` while reusing the same artifact format:

```bash
python -m perfcba.experiments.run_tau_regret_curve \
  --tau 0.2 \
  --n 20 --ell 2 --k 2 --m 2 --T 20000 \
  --seeds 0:9 \
  --artifact-dir results/tau_regret/tau_0.2 \
  --plot-path results/tau_regret/tau_0.2/regret_curve.png
```

If `--plot-only` is supplied, the script skips simulation and reads the existing
artifacts produced by either CLI.

### Post-processing and hypothesis testing

Once a sweep finishes, plug the JSONL output into the analysis CLI to generate
annotated heat maps, gradient flow overlays, and regression-based hypothesis
tests:

```bash
python -m perfcba.experiments.analysis \
  --results results/tau_study/graph_density/results.jsonl \
  --vary graph_density \
  --n 50 \
  --metrics cumulative_regret tto simple_regret optimal_rate \
  --out-dir results/tau_study/graph_density/analysis
```

Key implementation details:

- gradients are evaluated in the transformed density axis
  (`g(p) = log(n * p)`) to respect the geometric spacing of the grid, but
  plotted against the raw densities;
- bootstrap resampling over seeds filters the quiver overlay so only
  statistically reliable arrows remain opaque;
- the interaction test fits `metric ~ tau + g(p) + tau * g(p)` with robust
  covariance (clustered if `instance_id` is provided, otherwise HC3), and the
  Markdown/JSON reports also collect the Spearman column trends and the
  slope-versus-density meta-regression; and
- smoothing only affects the flow visualization – statistical tests always use
  the unsmoothed per-seed values, and you can disable smoothing via
  `--no-smooth` for audits.

Each run captures reproducibility breadcrumbs (CLI invocations, grids, bootstrap
settings) inside `tests_<metric>.json`, which makes downstream reporting
straightforward.

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
