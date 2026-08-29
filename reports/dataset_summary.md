# Dataset Summary

## Dataset

**Name:** PROD.csv  
**Domain:** Arabic e-commerce product reviews  
**Source:** Souq.com product reviews  
**Total samples:** 4,272

## Schema

| Column | Description |
|---|---|
| `text` | Arabic product review text |
| `polarity` | Sentiment label |

## Label Mapping

| Polarity | Sentiment |
|---:|---|
| `-1` | Negative |
| `0` | Neutral |
| `1` | Positive |

## Class Distribution

- Negative: 863
- Neutral: 308
- Positive: 3,101

## Data Quality Checks

- Missing values: 0
- Duplicate rows: 0
- Duplicate texts: 0

## Observations

The dataset is highly imbalanced, with the Positive class representing the majority of samples.

To reduce the impact of class imbalance:

- Train/validation/test splits will be stratified.
- Macro F1 will be treated as a key evaluation metric.
- Accuracy will not be used alone to judge model quality.
- Class weighting or sampling strategies may be evaluated during training experiments.

## Final Decision

The dataset is approved as the primary dataset for the Arabic Sentiment MLOps project because it:

- Matches the required Arabic product review domain.
- Contains all three required sentiment classes.
- Has no missing values.
- Has no duplicate reviews.