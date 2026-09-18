# Arabic Sentiment MLOps

End-to-end MLOps project for Arabic sentiment classification using modern machine learning engineering practices.

The project covers the complete ML lifecycle:

- Data preparation and preprocessing
- Model training and evaluation
- Experiment tracking
- Data and pipeline versioning
- Model serving
- REST API deployment
- Containerization
- Automated testing
- Load testing
- Production monitoring and drift detection


# Project Components

## Data Pipeline

- Data preprocessing and cleaning
- Dataset preparation
- Train / validation / test splitting
- Data versioning using DVC


## Model Training

The project includes:

- Baseline model evaluation
- Transformer-based Arabic sentiment classification
- Model evaluation metrics
- Model artifact management


## Experiment Tracking

MLflow is used for:

- Tracking training experiments
- Logging parameters
- Logging evaluation metrics
- Managing trained model artifacts


## Data Versioning

DVC is used for:

- Dataset version control
- Pipeline reproducibility
- Tracking data changes

Run:

```bash
dvc status

Reproduce pipeline:

dvc repro
Model Serving

The trained model is served using BentoML and exposed through a REST API.

BentoML Service

Start BentoML service:

bentoml serve src/arabic_sentiment/bento_service.py
FastAPI REST API

FastAPI provides the production inference API.

Available endpoints:

Health Check
GET /health

Example response:

{
  "status": "healthy",
  "device": "cuda",
  "model_version": "1"
}
Prediction
POST /predict

Request:

{
  "text": "الخدمة ممتازة والتجربة رائعة"
}

Response:

{
  "label": "positive",
  "confidence": 0.98,
  "model_version": "1"
}

Interactive API documentation:

/docs
Docker Deployment

The project is containerized using Docker.

Build and start services:

docker compose up --build

Check running services:

docker compose ps
Testing

Automated testing is implemented using Pytest.

Run tests:

pytest -q

Validated components:

API functionality
Model loading
Prediction pipeline
Service health checks
Load Testing

Locust is included for API load testing.

The project contains:

Locust configuration
Load testing scripts
Testing artifacts
Production Monitoring

The project includes production monitoring for detecting data drift.

Monitoring features:

Text length distribution monitoring
Sentiment distribution monitoring
Population Stability Index (PSI)
Evidently data drift reports

Run monitoring:

python scripts/monitoring/run_monitoring.py

Generated monitoring reports:

reports/monitoring/

├── monitoring_metrics.json
├── evidently_data_drift.html
└── monitoring_summary.md
Drift Detection Rule

Population Stability Index (PSI):

PSI <= 0.10 : Low change
0.10 - 0.25 : Moderate change
PSI > 0.25 : Significant drift
CI/CD

Continuous integration workflow validates:

Environment setup
Automated tests
Project consistency
Project Structure
arabic-sentiment-mlops

├── src/
│   └── arabic_sentiment/
│       ├── api.py
│       ├── bento_service.py
│       ├── model.py
│       ├── preprocessing.py
│       └── config.py
│
├── scripts/
│   ├── prepare_data.py
│   ├── train_baseline.py
│   ├── train_mlflow.py
│   ├── evaluate_baseline.py
│   └── monitoring/
│       └── run_monitoring.py
│
├── reports/
│   └── monitoring/
│
├── tests/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── dvc.yaml
├── mlflow.db
└── README.md
Final Validation

The following components have been implemented and validated successfully:

✓ Data preprocessing pipeline
✓ Model training and evaluation
✓ MLflow experiment tracking
✓ DVC data and pipeline versioning
✓ BentoML model serving
✓ FastAPI REST API
✓ Docker deployment
✓ Automated testing
✓ Load testing
✓ Production monitoring
✓ Data drift detection using PSI and Evidently

Project Status

Implementation completed.

The project is ready for final production review and submission.