# FedTrust

**Adaptive Trust-Weighted Knowledge Distillation for Heterogeneous Multi-Campus Federated Learning**

A working multi-node implementation in Python/Flask/PyTorch. Every campus is its own Flask service
that holds private data and a local model. A coordinator (also Flask) runs the federated rounds over
HTTP and serves a web console for configuring, running and analysing experiments.

```
                      ┌──────────────────────────────────────────────┐
                      │  Coordinator  (Flask, port 5000)             │
                      │  web console · REST API · SQLite · scheduler │
                      │  entropy → trust → trust-weighted teacher    │
                      └──────┬───────────────┬───────────────┬───────┘
         HTTP + token        │               │               │      only noise-protected
   (predictions on the       │               │               │      anchor predictions
    public anchor set)       ▼               ▼               ▼      and the teacher travel
                   ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
                   │ Campus A :5001│ │ Campus B :5002│ │ Campus C :5003│  …
                   │ private data  │ │ private data  │ │ private data  │
                   │ MLP           │ │ small CNN     │ │ wide MLP      │  ← different architectures
                   └───────────────┘ └───────────────┘ └───────────────┘
```

Private data and model weights never leave a campus.

---

## 1. Quick start

Requires Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# optional, much smaller CPU-only PyTorch:
#   pip install torch --index-url https://download.pytorch.org/whl/cpu

python scripts/start_cluster.py --campuses 4
```

Open **http://127.0.0.1:5000**. The script starts four campus services (ports 5001–5004) and the
coordinator, and registers the campuses. Then:

1. **Experiments → New experiment** → pick the preset *Proposed method* → *Queue experiment*.
2. Watch accuracy, trust, λ, teacher quality and communication update round by round.
3. **Studies** → *Methods: proposed vs baselines* → 3 seeds → you get the comparison table for your report.

Stop everything with `Ctrl+C`.

Each campus also has its own small status page (e.g. http://127.0.0.1:5001).

### Running the pieces by hand (what you would do on separate machines)

```bash
# on each campus machine
export FEDTRUST_TOKEN="a-long-random-secret"
python -m campus.app --name "Campus A" --host 0.0.0.0 --port 5001 \
       --public-url http://campus-a.example.edu:5001 --coordinator http://coordinator.example.edu:5000

# on the coordinator machine
export FEDTRUST_TOKEN="a-long-random-secret"          # same secret
export COORDINATOR_ADMIN_PASSWORD="choose-one"         # optional: password for the web UI
python -m coordinator.app --host 0.0.0.0 --port 5000
```

With `--coordinator`, a campus registers itself and appears on the **Campuses** page. Otherwise add it by URL.

### Docker

```bash
cp .env.example .env        # set FEDTRUST_TOKEN
docker compose up --build   # coordinator + 4 campus containers, then open http://localhost:5000
```

> Note: the Docker files were written but could not be run in the environment this project was built in.
> The Python services themselves were tested as separate processes.

### Headless runs and figures

```bash
python scripts/run_study.py modes   --seeds 42 43 44
python scripts/run_study.py privacy --epsilons 2 5 10 25 50 off
python scripts/run_study.py noniid  --alphas 0.1 0.3 1 10
python scripts/make_figures.py --modes-study <study_id> --privacy-study <study_id> --out figures/
```

### Tests

```bash
python -m pytest tests -q        # 39 tests, ~100 s: unit tests + end-to-end with real campus processes
```

---

## 2. What each round does

| # | Step | Where | Endpoint |
|---|------|-------|----------|
| 1 | Local training on private data | campus | `POST /api/train` |
| 2 | Predict on the public **anchor** set; add Gaussian noise (DP) | campus | `POST /api/predict` |
| 3 | Normalised entropy → **trust** per campus; per-sample weights; **trust-weighted teacher** | coordinator | – |
| 4 | **Adaptive distillation**: `CE(local) + λₖ·T²·KL(teacher‖student)` on anchor samples | campus | `POST /api/distill` |
| 5 | Evaluate on a common held-out test set | campus | `POST /api/evaluate` |

Formulas (also on the **Method** page of the console):

* trust: `Tₖ = softmax(−H̄ₖ/τ)`, smoothed across rounds with an EMA
* per-sample weights: `wₖ(x) ∝ Tₖ · exp(−Hₖ(x)/τₛ)`; teacher `q(x) = Σₖ wₖ(x) pₖ(x)`
* adaptive weight: `λₖ = Q · λmax · (floor + (1−floor)(1 − Tₖ/maxⱼTⱼ))`
  where `Q` is the teacher's consensus confidence, so a less trusted campus leans more on the teacher
  and an unreliable teacher is listened to less.

Entropies are divided by the entropy that pure privacy noise would produce (a post-processing step that
costs no privacy), so they stay comparable across privacy levels.

### Methods you can run (the `mode` setting)

| Mode | Teacher weights | Distillation weight | Purpose |
|------|-----------------|---------------------|---------|
| `trust_adaptive` | trust-weighted | adaptive | **proposed method** |
| `trust_fixed` | trust-weighted | fixed | ablation: effect of adaptive λ |
| `uniform_fd` | uniform | fixed | baseline (FedMD-style averaging) |
| `local_only` | – | – | baseline: no collaboration, same epochs |

### Mapping to the review deck

| Deck item | Where it lives |
|-----------|----------------|
| Campus data partitioning (non-IID) | `common/data.py` (`dirichlet_partition`), α slider in the UI |
| Local model training | `campus/node.py::train_local` |
| Anchor prediction generation | `campus/node.py::predict_anchor` |
| Entropy-based trust estimation | `common/trust.py`, `coordinator/orchestrator.py` |
| Privacy protection (differential privacy) | `common/privacy.py` |
| Trust-weighted global teacher | `common/trust.py::aggregate_teacher` |
| Adaptive knowledge distillation | `campus/node.py::distill`, `common/trust.py::adaptive_lambda` |
| Federated round management | `coordinator/orchestrator.py` |
| Performance evaluation, comm. cost, privacy trade-off | experiment pages, Studies, `scripts/make_figures.py` |
| Non-IID, heterogeneous models, privacy levels, fixed vs adaptive | Studies (`noniid`, `privacy`, `modes`) |

---

## 3. Differential privacy: what is and is not claimed

Before a campus releases its softmax predictions it adds Gaussian noise calibrated with the **analytic
Gaussian mechanism** (Balle & Wang 2018) for a chosen (ε, δ) and L2-sensitivity √2. Clipping to the simplex
afterwards is post-processing. Across rounds the console reports the tighter of basic and advanced composition.

The guarantee is **per released prediction vector, per round** (output perturbation). It is not a
record-level guarantee for each training example the way DP-SGD would give. Describe ε in your report as a
tunable privacy–utility dial for the shared predictions. The default ε = 50 is weak privacy, chosen so the
effect of trust weighting is visible. Run the privacy sweep to show the whole trade-off.

---

## 4. Example results (from the build environment)

Digits dataset, 4 campuses (MLP, small CNN, wide MLP, deep CNN; the last has 35 % label noise),
Dirichlet α = 0.5, 8 rounds. Single machine, CPU. Reproduce with the Studies page.

Method comparison at ε = 50, 3 seeds (mean ± sd of final mean campus accuracy):

| Method | Accuracy | Weakest campus |
|--------|----------|----------------|
| Local only | 69.6 ± 2.8 | 58.1 |
| Uniform FD | 77.0 ± 2.1 | 67.0 |
| Trust + fixed λ | 79.1 ± 1.3 | 68.8 |
| Proposed (trust + adaptive λ) | 78.8 ± 2.3 | 68.6 |

Privacy sweep, 2 seeds, proposed method vs uniform FD:

| ε per release | 2 | 5 | 10 | 25 | 50 | no DP |
|---------------|---|---|----|----|----|-------|
| Proposed | 68.4 | 69.8 | 71.0 | 74.6 | 78.5 | **87.7** |
| Uniform FD | 67.3 | 69.0 | 71.2 | 75.0 | 76.8 | 83.6 |

How to read this honestly:

* Collaboration clearly beats local training (+7 to +18 points).
* Accuracy falls smoothly as privacy gets stronger: a clear privacy–accuracy trade-off.
* **Trust weighting beats uniform averaging when the shared predictions are informative** (no DP: +4 points;
  ε = 50: about +2). With very strong noise (ε ≤ 25) the two are statistically indistinguishable.
* In these runs **adaptive λ was not better than a fixed λ** (78.8 vs 79.1, well within the spread). If your
  report claims an adaptive benefit, run more seeds and tune `lambda_max`, `lambda_floor`, `tau`.
* These are small-scale results on 1,797 images. The seeds differ in data partition, so spread is large.
  Use at least 5 seeds for any number you publish.

`example_results/` contains figures generated from these runs.

---

## 5. Configuration reference

All settings are available in the UI and as JSON on `POST /api/experiments`.

| Setting | Default | Meaning |
|---------|---------|---------|
| `mode` | `trust_adaptive` | see the table above |
| `dataset` | `digits` | `digits` (bundled, offline), `mnist`, `fashion_mnist` (download via OpenML on first use) |
| `alpha` | 0.5 | Dirichlet concentration; small = strongly non-IID |
| `n_rounds`, `warmup_epochs`, `local_epochs`, `distill_epochs` | 10, 5, 2, 2 | schedule |
| `anchor_per_round` | 200 | anchor samples shared each round (drives communication cost) |
| `dp.enabled`, `dp.epsilon`, `dp.delta`, `dp.sigma` | on, 50, 1e-5, – | give `sigma` instead of `epsilon` to set noise directly |
| `distill.temperature`, `lambda_max`, `lambda_floor`, `lambda_fixed` | 2, 1, 0.4, 0.7 | distillation |
| `trust.tau`, `tau_sample`, `ema` | 0.15, 0.5, 0.5 | trust sharpness and smoothing |
| per campus: `model`, `label_noise`, `data_fraction` | – | `mlp_small`, `mlp_wide`, `cnn_small`, `cnn_deep`, `linear`; data quality and size |

## 6. REST API (coordinator)

| Method & path | Purpose |
|---------------|---------|
| `GET /api/meta` | defaults, modes, models, scheduler state |
| `GET/POST /api/nodes`, `DELETE /api/nodes/<id>` | campus registry (with live status) |
| `POST /api/nodes/register` | self-registration (needs `X-FedTrust-Token`) |
| `GET/POST /api/experiments` | list / queue an experiment |
| `GET /api/experiments/<id>` | config, status, summary, every round |
| `POST /api/experiments/<id>/cancel`, `DELETE …` | cancel / delete |
| `GET /api/experiments/<id>/export.csv` | per-round, per-campus metrics |
| `GET /api/compare?ids=a,b` | several experiments at once |
| `GET/POST /api/studies`, `GET /api/studies/<id>`, `…/export.csv`, `POST …/cancel`, `DELETE …` | grids of runs with aggregation |

Campus node API (all under `/api/`, token required except `/health`): `info`, `init`, `train`, `predict`,
`distill`, `evaluate`.

## 7. Behaviour worth knowing

* **Fault tolerance:** if a campus fails during a round, that round continues with the others and the error is
  shown on the experiment page. If fewer than two campuses answer, the run fails with a clear message.
* **Reproducibility:** every node derives the same splits from `(dataset, seed, fractions)`; runs with the same
  seed give the same partitions. Training uses PyTorch RNG seeded per campus, so results are close but not
  guaranteed bit-identical across hardware.
* **Evaluation oracle:** the coordinator loads the dataset to score *teacher accuracy* against anchor labels.
  Those labels are never sent to campuses and are not used for training or trust.
* **Security:** services authenticate each other with a shared token (`FEDTRUST_TOKEN`); the web UI can be
  protected with `COORDINATOR_ADMIN_PASSWORD`. Use HTTPS (a reverse proxy) before exposing any of this
  beyond a trusted network. The built-in Flask server is for demos; for production use `gunicorn`.
* **Scheduler:** one experiment runs at a time because campuses are shared; others wait in the queue.

## 8. Limitations

* Entropy-based trust rewards confidence. A campus that is confidently wrong is over-trusted.
  The "trust vs released accuracy" correlation shown per run tells you how well trust tracked quality.
* Under strong privacy noise, entropy differences between campuses shrink, so trust approaches uniform.
* The anchor set is assumed public and identical for all campuses.
* Only classification on image-shaped inputs is implemented. MNIST/Fashion-MNIST loaders are included but were
  not exercised in the build environment (they need internet access).

## 9. Project layout

```
campus/        campus node: Flask app (app.py), training/prediction/distillation logic (node.py)
coordinator/   Flask app (app.py), round orchestration (orchestrator.py), scheduler + studies (studies.py),
               SQLite layer (db.py), templates/, static/ (CSS, JS, bundled Chart.js and fonts)
common/        data & partitioning, model zoo, differential privacy, trust/teacher maths, config, wire format
scripts/       start_cluster.py, run_study.py, make_figures.py
tests/         unit tests (privacy, trust, data, config) and end-to-end tests with real processes
docs/          example result figures
```

Bundled third-party assets: Chart.js 4.4.7 (MIT), Source Sans 3 and Source Serif 4 (SIL OFL), with licences
next to the files.
