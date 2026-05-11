# MLOps Assessment Report

## Title
Bike Sharing Demand MLOps Pipeline

## Author
Cagla Gezgen

## Date
20 MAY 2026

## 1. Introduction
This project delivers an end-to-end MLOps pipeline for the Kaggle Bike Sharing Demand dataset. It covers data acquisition, preprocessing, model training/testing, deployment, CI/CD, and continuous training, with a FastAPI inference service deployed as a Docker container on a Kubernetes cluster.

## 2. Artefact Overview
- Dataset: Kaggle Bike Sharing Demand
- Target: hourly bike rental count
- Model: XGBoost regressor
- Serving: FastAPI with /health and /predict endpoints
- Tracking: MLflow for metrics and model artifacts

## 3. Branching Strategy
GitFlow is used:
- main: production-ready releases
- develop: integration branch
- feature/*: new features and experiments
- release/*: release stabilization
- hotfix/*: production fixes

## 4. Pipeline Stages
### 4.1 Data Acquisition and Preprocessing
- Kaggle API download with secure credentials.
- Feature engineering on timestamps (year, month, day, hour, etc.).
- Outputs:
  - Processed dataset
  - Feature configuration
  - Dataset metadata and profile artifacts

### 4.2 Model Training and Testing
- Train XGBoost model on processed data.
- Evaluation metric: RMSLE
- Artifacts:
  - model.joblib
  - metrics.json
  - model_meta.json
  - dataset_meta.json
  - data_profile.json

### 4.3 Model Deployment
- Containerized API using Docker.
- Kubernetes Deployment and Service on Kind (GCP VM).
- Image pulled from Google Artifact Registry.

### 4.4 Continuous Integration
- GitHub Actions workflow runs tests on pull requests and develop branch pushes.
- Includes unit and API tests.

### 4.5 Continuous Delivery
- Build and push Docker image to Artifact Registry.
- Deploy to Kind on GCP VM via self-hosted runner.

### 4.6 Continuous Training / Continuous Monitoring
- Retrain on new data changes in data/bronze.
- MLflow logs metrics and artifacts.
- Data profile artifacts provide monitoring evidence and dataset drift checkpoints.

## 5. Test Use Cases
- API prediction endpoint returns predictions for valid input.
- Health endpoint reports service status.
- Feature engineering adds required time features.

## 6. CI/CD and CT/CM Workflows
- Ingest and Preprocess: downloads Kaggle data and creates processed artifacts.
- Train Model: trains, logs to MLflow, uploads artifacts.
- Continuous Training: retrains when new data lands in data/bronze.
- Deploy: builds and pushes image to Artifact Registry, deploys to Kubernetes.

## 7. Deployment Architecture
- GCP VM hosts a Kind cluster and a self-hosted GitHub Actions runner.
- Kubernetes Service exposes the API via NodePort.
- Docker images are stored in Google Artifact Registry.

## 8. Secrets and Access
- Kaggle credentials via GitHub Actions secrets.
- GCP service account key for Artifact Registry access.

## 9. References
- Kaggle Bike Sharing Demand: https://www.kaggle.com/competitions/bike-sharing-demand
- MLflow: https://mlflow.org/
- FastAPI: https://fastapi.tiangolo.com/
- Google Artifact Registry: https://cloud.google.com/artifact-registry
- Kubernetes: https://kubernetes.io/
