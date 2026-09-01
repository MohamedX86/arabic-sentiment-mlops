# MLflow Experiments Summary

## Experiment

- Experiment name: `arabic-sentiment-arabert`
- Tracking tool: MLflow
- Model family: AraBERT
- Task: Arabic sentiment classification
- Main selection metric: Macro F1
- Secondary metric: Accuracy

## Runs Summary

| Run | Configuration | Accuracy | Macro F1 |
|---|---|---:|---:|
| baseline_mlflow | LR=2e-5, BS=8, Epochs=2, MaxLen=128, Weighted Loss=False | 0.8525 | 0.5864 |
| weighted_loss | LR=2e-5, BS=8, Epochs=2, MaxLen=128, Weighted Loss=True | 0.7892 | 0.6478 |
| weighted_lr_1e5 | LR=1e-5, BS=8, Epochs=2, MaxLen=128, Weighted Loss=True | 0.8220 | 0.6621 |
| weighted_lr_1e5_ep3 | LR=1e-5, BS=8, Epochs=3, MaxLen=128, Weighted Loss=True | 0.7494 | 0.6177 |
| weighted_lr_5e6 | LR=5e-6, BS=8, Epochs=2, MaxLen=128, Weighted Loss=True | 0.7822 | 0.6331 |
| best_weighted_lr_1e5_logged | LR=1e-5, BS=8, Epochs=2, MaxLen=128, Weighted Loss=True | 0.8220 | 0.6621 |

## Best Configuration

The best-performing configuration based on Macro F1 was:

- Run: `weighted_lr_1e5`
- Learning rate: `1e-5`
- Batch size: `8`
- Epochs: `2`
- Max sequence length: `128`
- Weighted loss: `True`

Results:

- Validation Accuracy: `0.8220`
- Validation Macro F1: `0.6621`

## Baseline Comparison

Baseline:

- Accuracy: `0.8525`
- Macro F1: `0.5864`

Best experiment:

- Accuracy: `0.8220`
- Macro F1: `0.6621`

The weighted-loss configuration improved Macro F1 substantially compared with the baseline.

Macro F1 improvement:

- Absolute improvement: `0.0757`
- Relative improvement: approximately `12.9%`

The reduction in overall accuracy was accepted because the main objective was to improve performance across minority classes rather than optimize only for the dominant Positive class.

## Experiment Findings

1. Weighted loss improved class-balanced performance.
2. Reducing the learning rate from `2e-5` to `1e-5` improved Macro F1.
3. Increasing training from 2 to 3 epochs reduced performance.
4. Reducing the learning rate further to `5e-6` did not improve over `1e-5`.
5. `1e-5`, 2 epochs, and weighted loss provided the best balance observed in the experiment set.

## Model Logging

The best configuration was reproduced in:

- Run: `best_weighted_lr_1e5_logged`

The model was successfully logged in MLflow.

Logged model URI:

- `models:/m-223ebb91d31b40a19795d7ee6e70c203`

## Model Registry

The logged model was registered in MLflow Model Registry as:

- Registered model: `arabic-sentiment-arabert`
- Version: `1`

## Final Decision

The MLflow registered model `arabic-sentiment-arabert`, Version 1, is the current selected model for the next stages of the MLOps pipeline.

Future evaluation and deployment work should use this registered model unless a later experiment produces a better Macro F1 score.