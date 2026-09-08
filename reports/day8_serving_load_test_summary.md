# Day 8 — Production Serving & Load Testing Summary

## BentoML Serving

Production serving was implemented using BentoML.

- Service name: `arabic_sentiment_service`
- BentoML version: `1.4.39`
- Endpoint: `POST /predict`
- Host: `http://localhost:3000`
- Model version: `1`

The BentoML service successfully loaded the deployed AraBERT sentiment model and returned predictions in the expected response format:

```json
{
  "label": "positive",
  "confidence": 0.8107508420944214,
  "model_version": "1"
}

The endpoint returned HTTP status 200.

Locust Load Test

Load testing was performed using Locust against the BentoML /predict endpoint.

Configuration:

Users: 5
Spawn rate: 1 user/second
Duration: 30 seconds
Host: http://localhost:3000
Endpoint: POST /predict

Observed results:

Total requests: 35
Failures: 0
Failure rate: 0.00%
Average response time: approximately 350 ms
Median response time: approximately 63 ms
Maximum response time: approximately 2132 ms
p95 response time: approximately 2100 ms
Throughput: approximately 2.70 requests/second
Generated Reports

Locust generated the following report files:

reports/locust_stats.csv
reports/locust_stats_history.csv
reports/locust_failures.csv
reports/locust_exceptions.csv
Result

The BentoML inference service successfully handled the load test without request failures.

The production serving and load-testing requirements for Day 8 were validated successfully.