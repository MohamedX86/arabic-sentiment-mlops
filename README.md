# Arabic Sentiment MLOps

End-to-end MLOps project for Arabic sentiment classification.

## Main Components

- Data preprocessing and preparation
- Model training and evaluation
- MLflow experiment tracking
- DVC data and pipeline versioning
- BentoML model serving
- REST API
- Docker
- Pytest automated testing
- Locust load testing

## Model Serving

Start the BentoML service:

```bash
bentoml serve src/arabic_sentiment/bento_service.py

Prediction endpoint:

POST /predict

Health endpoint:

GET /health
Docker
docker compose up --build

Check the service:

docker compose ps
MLflow

MLflow is used to track experiments, training runs, evaluation metrics, and the trained model.

DVC

Check the pipeline:

dvc status
dvc repro
Testing

Run:

pytest -q

The project test suite has been validated successfully.

Load Testing

Locust is included for API load testing and the project contains the corresponding load-test artifacts.

Final Validation

Validated components include:

MLflow experiments and model
DVC pipeline
BentoML service
/health endpoint
/predict endpoint
Docker service
Automated tests
Load testing
Status

Project implementation and validation completed. Final production review is the remaining submission step.
