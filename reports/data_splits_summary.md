# Data Splits Summary

## Split Strategy

The processed dataset was split using stratified sampling to preserve the original class distribution across all subsets.

Split ratios:

- Train: 80%
- Validation: 10%
- Test: 10%

Random seed:

- `42`

## Sample Counts

- Train: 3,417
- Validation: 427
- Test: 428

Total:

- 4,272 samples

## Class Distribution

### Train

- Positive: 2,481
- Negative: 690
- Neutral: 246

### Validation

- Positive: 310
- Negative: 86
- Neutral: 31

### Test

- Positive: 310
- Negative: 87
- Neutral: 31

## Notes

The dataset is imbalanced, with the Positive class being dominant.

Stratified splitting was used to maintain approximately the same class proportions in the train, validation, and test sets.

Macro F1 will be used as a key evaluation metric during training and experiment tracking.