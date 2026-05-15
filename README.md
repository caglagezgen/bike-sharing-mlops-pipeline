# Bike Sharing Demand MLOps Pipeline
End-to-end MLOps pipeline for the Kaggle Bike Sharing Demand dataset. Implements data ingestion, preprocessing, model training/testing, semantic model versioning, drift monitoring, CI, CD, and continuous training. The model is served via a FastAPI service deployed as a Docker container to a Kind Kubernetes cluster on a GCP VM.

## Machine Learning Problem Overview

Bike sharing systems automate the entire rental lifecycle — membership, pickup, and return — through a network of kiosk locations across a city. With over 500 programmes worldwide, these systems generate rich operational data that makes them well-suited to demand forecasting.

**Problem type:** Supervised regression  
**Target variable:** `count` — total hourly bike rentals (casual + registered riders)  
**Input features:** datetime, season, weather conditions, temperature, humidity, windspeed, and holiday flags

The core challenge is predicting hourly rental demand accurately enough that operators can pre-position bikes and plan maintenance without disrupting service. Demand is highly non-linear — it spikes during commute hours (08:00, 17:00–18:00), varies by season, and drops sharply in poor weather — making it a strong candidate for gradient-boosted tree models that can capture these interaction effects without explicit feature crosses.

An XGBoost regressor is trained on two years of historical hourly data (2011–2012) from the Capital Bikeshare system in Washington D.C. The model is evaluated using RMSLE, which penalises under-prediction proportionally — operationally important because running out of bikes at a busy station is costlier than having idle ones.

Dataset: https://www.kaggle.com/competitions/bike-sharing-demand

## Why FastAPI (instead of Flask)

FastAPI was selected because it provides strong request validation through Pydantic models, automatic OpenAPI/Swagger docs for demoing and testing the API, and native async support if the service grows. This reduces boilerplate for input checks and gives clear contracts for ML inference inputs and outputs, which is valuable in production and for assessment demonstrations.

## Pipeline Stages

| Stage | Implementation |
|---|---|
| Data Acquisition | `src/data/ingest.py` — Kaggle CLI download |
| Preprocessing | `src/data/preprocess.py` — feature engineering, SHA256 versioning |
| Model Training | `src/models/train.py` — XGBoost, MLflow tracking |
| Model Versioning | `src/models/version.py` — semantic versioning (major.minor.patch) |
| Drift Monitoring | `monitoring/drift.py` — KS test, Chi2, PSI |
| Serving | `app/app.py` — FastAPI, `/health` + `/predict` |
| CI | `.github/workflows/ci.yml` — pytest on PRs |
| CD | `.github/workflows/deploy.yml` — Docker → Artifact Registry → K8s |
| Continuous Training | `.github/workflows/ct.yml` — retrain on new data |

## Evaluation Metrics

Primary metric for the Kaggle competition is RMSLE:

$$\sqrt{\frac{1}{n} \sum_{i=1}^n (\log(p_i + 1) - \log(a_i + 1))^2}$$

RMSLE is preferred over RMSE for count data because it:
- Penalises **under-prediction** more heavily than over-prediction on a proportional scale (for a bike-sharing operator, running out of bikes is costlier than having idle ones)
- Is scale-invariant across the wide demand range (1–900+ bikes/hour)

Additional business-interpretable metrics logged per run:

| Metric | Formula | Why It Matters |
|---|---|---|
| MAE | $\frac{1}{n}\sum|y_i - \hat{y}_i|$ | Average error in bike units; robust to outliers (linear penalty vs quadratic) |
| MAPE | $\frac{100}{n}\sum\frac{|y_i - \hat{y}_i|}{y_i}$ | Percentage error; comparable across low/high demand hours. Zero-guarded: hours with `count=0` are excluded from the mean to avoid division-by-zero |
| R² | $1 - \frac{\sum(y_i-\hat{y}_i)^2}{\sum(y_i-\bar{y})^2}$ | Proportion of demand variance explained relative to a naive mean predictor. Target benchmark: **R² ≥ 0.95** |

See [Bike Sharing Demand Prediction](https://medium.com/@muhammadaris10/bike-sharing-demand-prediction-fc692d90b5b3) for full analysis.

## Repository Structure

```
├── .github/workflows/       CI, CD, ingest, train, CT pipelines
├── app/
│   └── app.py               FastAPI inference service
├── configs/
│   └── model_config.yaml    Hyperparameters and training config
├── data/
│   ├── bronze/              New data drop zone (triggers CT)
│   ├── processed/           Preprocessed features
│   └── raw/                 Downloaded Kaggle data
├── docs/                    Assessment report
├── k8s/
│   ├── deployment.yaml      Kubernetes Deployment
│   └── service.yaml         NodePort Service (port 30080)
├── monitoring/
│   └── drift.py             KS, Chi2, PSI drift detection
├── notebooks/               EDA, training, optimisation, drift analysis
├── scripts/
│   ├── ingest.sh
│   ├── preprocess.sh
│   └── train.sh
├── src/
│   ├── config.py            Path constants
│   ├── data/
│   │   ├── ingest.py        Kaggle download + credential handling
│   │   └── preprocess.py    Feature engineering + data profiling
│   ├── features/
│   │   └── engineering.py   Feature definitions and transformations
│   └── models/
│       ├── train.py         XGBoost training + MLflow logging
│       └── version.py       Semantic model version manager
└── tests/                   Unit tests
```

## Feature Engineering

### Features Used

| Feature | Type | Notes |
|---|---|---|
| `season` | Categorical | 1=Winter, 2=Spring, 3=Summer, 4=Fall |
| `holiday` | Binary | |
| `weather` | Categorical | 1=Clear → 4=Heavy Rain. Winsorized at 1% bounds |
| `temp` | Continuous | Normalised temperature |
| `humidity` | Continuous | Winsorized at 1% bounds |
| `windspeed` | Continuous | Winsorized at 1% bounds; highest outlier rate (1.91%) |
| `year` | Derived | Captures year-on-year demand growth trend (2011→2012) |
| `month` | Derived | Seasonal rhythm |
| `hour` | Derived | **Strongest single predictor** — demand peaks at commute hours 08:00 and 17:00–18:00 |
| `dayofweek` | Derived | Encodes weekly rhythm; more granular than the binary `workingday` flag |

### Why These Features Were Dropped

**`atemp` (apparent/feels-like temperature)**

Pearson correlation with `temp` = **0.99**. Including both columns causes multicollinearity: the model cannot isolate the individual contribution of either variable, coefficient/importance estimates become unstable, and SHAP values are split arbitrarily between two near-identical features. In gradient boosted trees, correlated features compete for the same splits, wasting tree capacity. Dropping `atemp` retains the same thermal signal without the collinearity cost.

**`workingday`**

A binary flag (0/1) that is entirely derivable from `dayofweek` — any day where `dayofweek` ∈ {0,1,2,3,4} and `holiday` = 0 is a working day. Because `dayofweek` encodes strictly more information (seven distinct values vs two), `workingday` is a redundant feature that adds no predictive power beyond what `dayofweek` already provides. Including it would introduce near-collinearity with an existing feature.

**`day` (day of month)**

Day of month has no meaningful cyclical relationship with bike demand. A rider on the 3rd of the month does not behave differently from one on the 17th. Feature importance analysis confirmed `day` as a near-zero contributor. Removing it reduces model dimensionality and avoids the risk of spurious overfitting to calendar artefacts in the training window.

**`is_weekend`**

A deterministic function of `dayofweek`: `is_weekend = (dayofweek >= 5)`. Including a feature that is a linear transformation of an existing column creates a linearly dependent feature — the feature matrix becomes rank-deficient, which adds no new predictive information and increases noise in the split-selection process.

**`casual` and `registered`**

These columns sum to the target: `count = casual + registered`. Including them would cause **target leakage** — the model would learn a trivial identity mapping and achieve near-zero training error, but would completely fail at inference time when only raw inputs are available.

### Outlier Handling: Winsorization

Winsorization clips values at the 1st and 99th percentile rather than deleting rows. This is preferred over row removal for two reasons:

1. **Preserves dataset size.** Gradient boosting is data-hungry; removing rows for outliers reduces the information available for learning low-frequency patterns (e.g., extreme weather conditions).
2. **Stabilises the loss function.** RMSLE and MSE penalise large residuals quadratically. A single extreme outlier can dominate the gradient signal and pull tree splits away from the main data distribution. Capping the tails ensures gradients remain proportional to the typical error magnitude.

The three targeted columns were selected based on IQR analysis:
- `windspeed`: 1.91% of rows outside the IQR fence (highest in the dataset)
- `humidity`: meaningful tail with extreme low-humidity events
- `weather`: categorical, but extreme codes (4 = Heavy Rain) occur rarely enough to be outlier-like in frequency

### Target Transformation: log1p

Bike rental counts are right-skewed with heavy tails at peak commute hours. Applying `log1p(count)` before training:
- Makes the residual distribution more symmetric, which aligns with the squared-error loss assumption
- Directly optimises RMSLE: training on `log(count)` with MSE loss is mathematically equivalent to minimising RMSLE on the original scale, since RMSLE measures error in log space
- Reduces the leverage of extreme peak-hour values that would otherwise dominate gradient updates

## Branching Strategy (GitFlow-lite)

| Branch | Purpose |
|---|---|
| `main` | Production-ready releases only |
| `develop` | Integration branch — all features merge here first |
| `feature/*` | New features and experiments |
| `release/*` | Pre-release hardening and version bumps |
| `hotfix/*` | Emergency production fixes |

Pull requests target `develop`. Releases are merged into `main` via a release branch.

**Workflow alignment**
- CI runs on PRs to `main` and `develop`, and on pushes to `develop`.
- Training runs on pushes to `develop` and `main` (plus scheduled runs).
- Deployment auto-triggers only from successful training on `main`.

**How to work**
1. Create `feature/*` from `develop`.
2. Open a PR into `develop` (CI + training validate the change).
3. Cut `release/*` from `develop` when ready, then merge to `main`.
4. A `main` merge triggers training; a successful run triggers deploy.
5. For urgent fixes, branch `hotfix/*` from `main`, merge back into `main` and `develop`.

**Branch protections (GitHub settings)**
- Protect `main` and `develop` (no direct pushes).
- Require PR reviews (1 minimum) and status check `CI` before merge.
- Restrict who can push to `main` (or allow only admins).
- Optional: require linear history and signed commits.

**Release tagging**
- Tag `main` merges with `vMAJOR.MINOR.PATCH`.
- Record the deployed image tag and model version in the release notes for traceability.

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 1. Configure Kaggle credentials

Place `~/.kaggle/kaggle.json` with your API token, or export:

```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_key
```

### 2. Download and Preprocess

```bash
python -m src.data.ingest --competition bike-sharing-demand
python -m src.data.preprocess
```

Or use the convenience scripts:

```bash
bash scripts/ingest.sh --competition bike-sharing-demand
bash scripts/preprocess.sh
```

### 3. Train and Track

```bash
python -m src.models.train
mlflow ui --backend-store-uri ./mlruns
```

### 4. Model Versioning

```bash
# View version history
python -m src.models.version history

# Diff two versions
python -m src.models.version diff 0.0.1 0.0.2

# Current version
python -m src.models.version current
```

### 5. Run the API

```bash
uvicorn app.app:app --host 0.0.0.0 --port 8080
# Swagger UI: http://localhost:8080/docs
```

Example prediction request:

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "datetime": "2011-01-01 00:00:00",
    "season": 1, "holiday": 0,
    "weather": 1, "temp": 9.84,
    "humidity": 81, "windspeed": 0.0
  }'
```

### 6. Run Tests

```bash
pytest tests/ -v
```

## Docker

```bash
docker build -t bike-sharing-api:latest .
docker run -p 8080:8080 bike-sharing-api:latest
```

## Kubernetes (Kind on GCP VM)

1. Create a GCP VM and install Docker, Kind, and kubectl.
2. Create a Kind cluster: `kind create cluster`
3. Register a self-hosted GitHub Actions runner on the VM with labels: `self-hosted`, `kind`, `gcp`.
4. Create the Artifact Registry pull secret:

```bash
kubectl create secret docker-registry artifact-registry \
  --docker-server=REGION-docker.pkg.dev \
  --docker-username=_json_key \
  --docker-password="$(cat /path/to/gcp-sa-key.json)" \
  --docker-email=you@example.com
```

5. Deploy manually or trigger the CD workflow:

```bash
IMAGE_URI=REGION-docker.pkg.dev/PROJECT_ID/REPO/bike-sharing-api:latest
IMAGE_URI="$IMAGE_URI" envsubst < k8s/deployment.yaml | kubectl apply -f -
kubectl apply -f k8s/service.yaml
kubectl rollout status deployment/bike-sharing-api
```

Service is exposed on NodePort `30080`.

## GitHub Actions Workflows

| Workflow | Trigger | Purpose |
|---|---|---|
| `ci.yml` | PR to `main`/`develop` | Run unit tests |
| `ingest_preprocess.yml` | Manual / called | Download + preprocess data |
| `train.yml` | Manual / weekly cron | Full train pipeline |
| `ct.yml` | Push to `data/bronze/**` | Retrain on new data |
| `deploy.yml` | After train / manual | Build image, push to GAR, deploy to K8s |

## Continuous Training Trigger

Drop new data into `data/bronze/` and push to trigger automatic retraining:

```bash
cp new_data.csv data/bronze/train.csv
git add data/bronze/train.csv
git commit -m "chore: add new training batch"
git push origin develop
```

## Required GitHub Secrets

| Secret | Description |
|---|---|
| `KAGGLE_USERNAME` | Kaggle account username |
| `KAGGLE_KEY` | Kaggle API key |
| `GCP_PROJECT_ID` | GCP project ID |
| `GAR_REGION` | Artifact Registry region (e.g. `europe-west1`) |
| `GAR_REPOSITORY` | Artifact Registry repository name |
| `GCP_SA_KEY` | Service account JSON key with Artifact Registry access |

## Hyperparameter Configuration

Model hyperparameters are externalised in `configs/model_config.yaml`.

```yaml
model:
  params:
    n_estimators: 1300
    max_depth: 5
    learning_rate: 0.05
    subsample: 0.9
    colsample_bytree: 0.9
```

### Why These Values

**`learning_rate = 0.05` and `n_estimators = 1300`**

Gradient boosting constructs an ensemble of trees sequentially, where each tree corrects the residuals of the previous ones. The learning rate $\eta$ scales each tree's contribution:

$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$

A smaller $\eta$ shrinks each update, acting as a regulariser — the model moves more carefully along the loss surface and is less likely to overfit to individual training examples. The trade-off is that more trees are needed to reach the same training loss. The general rule is: halving the learning rate roughly doubles the optimal $M$ (number of trees). Starting from a baseline of $\eta=0.1, M=300$, reducing to $\eta=0.05$ calls for approximately $M\approx1300$ trees. The tuning experiments in the reference analysis confirmed this yielded the best MAPE improvement (from ~46% to ~24%).

**`max_depth = 5`**

Tree depth controls model complexity. Depth $d$ allows up to $2^d = 32$ leaf nodes, which is sufficient to capture multi-way interactions such as `hour × season × weather`. Shallower trees act as high-bias/low-variance weak learners — the boosting ensemble compensates for bias through additive combination. Experiments with `max_depth=6` showed marginal accuracy gain but higher validation variance, a sign of overfitting to training noise.

**`subsample = 0.9`**

At each boosting round, 90% of training rows are sampled without replacement (stochastic gradient boosting, Friedman 2002). This introduces a bagging effect: each tree sees a slightly different view of the data, which reduces correlation between consecutive trees and lowers ensemble variance. It also speeds up training by ~10%.

**`colsample_bytree = 0.9`**

Each tree is trained on a random 90% subset of features (column sampling). This decorrelates trees further — analogous to the random subspace method in Random Forests — and prevents dominant features like `hour` from appearing in every tree split, making the ensemble more robust to feature-level noise.

**`objective = reg:squarederror`**

MSE loss on the log-transformed target is equivalent to minimising RMSLE on the original count scale. This directly aligns the training objective with the Kaggle competition metric.

## Data Versioning Artifacts

Each run writes to `artifacts/`:

| File | Contents |
|---|---|
| `dataset_meta.json` | Source, SHA256 hash, row counts, timestamps |
| `data_profile.json` | Column-level stats (min/max/mean/std/nulls) |
| `feature_config.json` | Feature column list used for training |
| `metrics.json` | Validation RMSLE and row count |
| `model_meta.json` | Training timestamp and feature columns |
| `model_versions.json` | Full semantic version history |

## References

- Will Cukierski. Bike Sharing Demand. https://kaggle.com/competitions/bike-sharing-demand, 2014.

