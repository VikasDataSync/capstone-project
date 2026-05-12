# End-to-End MLOps Sentiment Analysis Platform

Production-style MLOps project that takes a sentiment model from data ingestion to CI/CD-driven deployment, with experiment tracking, model registry, automated promotion, and application monitoring.

## Why this project is employer-relevant

- Built as a complete **ML lifecycle system**, not just a notebook model.
- Demonstrates **data + ML + software + DevOps integration** in one repo.
- Uses real-world tooling expected in MLOps roles: **DVC, MLflow, Flask, Docker, GitHub Actions, AWS, Kubernetes, Prometheus**.
- Includes both **offline model quality gates** and **online service observability**.

## Business problem

Classify text reviews into positive/negative sentiment and operationalize the model in a reproducible, deployable, and monitorable pipeline.

## Architecture at a glance

1. **Data ingestion** from S3 into local raw datasets.
2. **Data preprocessing** (text normalization, cleaning).
3. **Feature engineering** with Bag-of-Words vectorization.
4. **Model training** using Logistic Regression.
5. **Model evaluation** with metrics + MLflow logging.
6. **Model registration** and stage transition in MLflow Model Registry.
7. **Promotion workflow** to move staging model to production.
8. **Serving layer** via Flask API/UI.
9. **Observability** with Prometheus-compatible custom metrics.
10. **Container + deployment** using Docker + Kubernetes manifest.

## Tech stack

| Area | Tools |
|---|---|
| Language | Python 3.10 |
| ML/Data | pandas, scikit-learn, nltk |
| Experiment tracking | MLflow + DagsHub |
| Pipeline orchestration | DVC |
| API serving | Flask + Gunicorn |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Cloud/Infra | AWS (S3, ECR, EKS) + Kubernetes |
| Monitoring | Prometheus client metrics |
| Testing | unittest |

## Model quality snapshot

Latest tracked evaluation (`reports/metrics.json`):

- Accuracy: **0.592**
- Precision: **0.6207**
- Recall: **0.5538**
- AUC: **0.6628**

This repository also contains a model performance gate in tests (`tests/test_model.py`) with threshold-based assertions.

## Repository highlights

- `dvc.yaml`: Multi-stage reproducible ML pipeline.
- `src/data/`: Ingestion + preprocessing.
- `src/features/`: Feature engineering and vectorizer persistence.
- `src/model/`: Training, evaluation, registry integration.
- `scripts/promote_model.py`: Controlled model stage promotion.
- `flask_app/app.py`: Inference app + `/metrics` endpoint for Prometheus scraping.
- `.github/workflows/ci.yaml`: CI/CD automation for pipeline, tests, container build, and deploy flow.
- `deployment.yaml`: Kubernetes deployment/service definition.

## Local setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Run the ML pipeline

```bash
dvc repro
```

This executes stages defined in `dvc.yaml`:
`data_ingestion -> data_preprocessing -> feature_engineering -> model_building -> model_evaluation -> model_registration`

## Run tests

```bash
python -m unittest tests/test_model.py
python -m unittest tests/test_flask_app.py
```

## Run the Flask app

```bash
python flask_app/app.py
```

App endpoints:

- `/` : UI
- `/predict` : sentiment inference
- `/metrics` : Prometheus metrics

## Docker run

```bash
docker build -t capstone-app:latest .
docker run -p 8888:5000 -e CAPSTONE_TEST=<DAGSHUB_TOKEN> capstone-app:latest
```

## CI/CD workflow (GitHub Actions)

Pipeline triggers on push and automates:

1. Dependency install
2. DVC pipeline execution
3. Model tests
4. Model promotion
5. Flask app tests
6. Docker image build/tag/push
7. Kubernetes deployment steps

## Required environment variables/secrets

Use GitHub Secrets/Variables or local env configuration. Do not hardcode credentials.

- `CAPSTONE_TEST` (DagsHub token for MLflow auth)
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`
- `AWS_ACCOUNT_ID`
- `ECR_REPOSITORY`

## Security note

This repo is now sanitized to use placeholders for sensitive values. Keep all credentials in secure secret stores and rotate tokens/keys immediately if exposed.

## What this project demonstrates to employers

- End-to-end ownership of an ML system from ingestion to deployment.
- Ability to productionize ML with reproducibility and governance.
- Practical CI/CD and cloud deployment exposure.
- Strong understanding of monitoring, testing, and operational reliability in ML applications.
