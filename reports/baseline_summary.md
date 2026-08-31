# Baseline Model Summary

## Model

- Model: `aubmindlab/bert-base-arabertv02`
- Task: Arabic sentiment classification
- Classes:
  - Negative
  - Neutral
  - Positive

## Training Configuration

- Epochs: 2
- Batch size: 8
- Learning rate: 2e-5
- Max sequence length: 128
- Device: CUDA
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU

## Validation Results

- Accuracy: 0.8525
- Macro F1: 0.5864

## Test Results

- Accuracy: 0.8411
- Macro F1: 0.5563

## Per-Class Performance

### Negative

- Precision: 0.8000
- Recall: 0.7356
- F1-score: 0.7665
- Support: 87

### Neutral

- Precision: 0.0000
- Recall: 0.0000
- F1-score: 0.0000
- Support: 31

### Positive

- Precision: 0.8555
- Recall: 0.9548
- F1-score: 0.9024
- Support: 310

## Confusion Matrix

```text
[[ 64   0  23]
 [  4   0  27]
 [ 12   2 296]]
Key Observation

The baseline model achieves relatively high overall accuracy, but its macro F1 score is much lower because the model fails to correctly classify the Neutral class.

This confirms that accuracy alone is not sufficient for evaluating this imbalanced dataset.

Future experiments should focus on improving minority-class performance, especially Neutral sentiment.

Possible strategies to evaluate in later experiments include:

Class-weighted loss
Sampling strategies
Different learning rates
More training epochs
Different maximum sequence lengths
Alternative pretrained Arabic transformer models
Baseline Decision

This model will be used as the initial baseline for future MLflow experiments.

Model selection in later stages will prioritize Macro F1 alongside Accuracy.