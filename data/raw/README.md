# Raw Data

## Candidate Dataset

**Large Arabic Resources for Sentiment Analysis — PROD.csv**

- Domain: Arabic product reviews
- Source platform: Souq.com
- Number of reviews: 4,272
- Intended use: Arabic sentiment analysis
- Repository: https://github.com/hadyelsahar/large-arabic-sentiment-analysis-resouces

## Validation Status

Approved as the primary dataset for this project.

Validation results:

- Total reviews: 4,272
- Columns: `text`, `polarity`
- Labels:
  - `-1` = Negative
  - `0` = Neutral
  - `1` = Positive
- Missing values: 0
- Duplicate rows: 0
- Duplicate texts: 0
- Class distribution:
  - Negative: 863
  - Neutral: 308
  - Positive: 3,101
- Note: the dataset is imbalanced, with the Positive class being dominant. Training and evaluation will therefore use stratified splits and macro-F1 as a key metric.