# RipeNet

RipeNet is an end-to-end computer vision suite for fruit species identification, ripeness classification, and shelf-life estimation using deep learning. The system utilizes multiple specialized models to provides a comprehensive analysis of fruit quality and remaining viability.

The core innovation of RipeNet is its transition from discrete classification to continuous regression, modeling shelf-life as a time-based value in days.

---

## Technical Overview

RipeNet is structured as a triple-expert system:

1.  **Identity Model**: An EfficientNet-B0 classifier that identifies the fruit species (Apple, Banana, Orange, Papaya) to select appropriate biological decay parameters.
2.  **Classification Model**: A secondary check for discrete ripeness stages (Unripe, Fresh/Ripe, Rotten) to validate visual status.
3.  **Regression Model**: The primary engine that predicts the exact remaining shelf-life in days by analyzing surface features, color distribution, and texture degradation.

The system is trained on a custom agricultural dataset and achieves high precision in the transition between ripeness stages.

---

## Key Features

- **Multi-Model Pipeline**: Automated flow from species identification to shelf-life prediction.
- **Continuous Regression**: Predicts shelf-life in days instead of static categories.
- **Pre-trained Backbones**: Leverage transfer learning with EfficientNet-B0 for high accuracy and low latency.
- **Natural Language Inference**: Generates varied, human-readable status reports.
- **Metric Verification**: Validated using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

---

## Project Structure

```
.
├── saved_models/           # Pre-trained model weights (.pth)
│   ├── best_model.pth             # Classification model
│   ├── best_identity_model.pth    # Species identification model
│   └── best_regression_model.pth  # Shelf-life regression model
│
├── src/                    # Main inference and classification logic
│   ├── dataset.py          # Data loading for classification
│   ├── model.py            # Classification architecture
│   ├── train.py            # Training script for classification
│   ├── evaluate.py         # Metrics reporting
│   └── predict.py          # Unified inference script (FruitGPT engine)
│
├── src_identity/           # Fruit species identification
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
│
├── src_regression/         # Shelf-life trajectory prediction
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
│
├── requirements.txt        # Environment dependencies
├── generate_labels.py      # Automated labeling from directory structure
├── split_data.py           # Dataset partitioning (Train/Val/Test)
└── update_days_remaining.py # Biological decay modeling logic
```

---

## Data Labeling Strategy

The regression targets are derived from fruit-specific decay curves. The model learns to map visual degradation to a temporal scale.

| Fruit   | Unripe Stage (Days) | Fresh Stage (Days) | Rotten Stage (Days) |
|---------|---------------------|--------------------|---------------------|
| Apple   | 10.0                | 5.0                | 2.0                 |
| Banana  | 6.0                 | 3.0                | 1.0                 |
| Orange  | 8.0                 | 4.0                | 2.0                 |
| Papaya  | 6.0                 | 3.0                | 1.0                 |

*Values represent average viability timeframes under standard conditions.*

---

## Evaluation Results

Recent test set performance demonstrates the system's robustness:

- **Classification Accuracy**: 92.6%
- **Identity Model Accuracy**: 93.6%
- **Regression Mean Absolute Error (MAE)**: 0.74 days
- **Regression RMSE**: 1.07 days

This signifies that the system predicts shelf-life within a margin of error of approximately 18 hours.

---

## Usage

### Environment Setup
```bash
pip install -r requirements.txt
```

### Automated Inference
To analyze an image and generate a shelf-life report:
```bash
python src/predict.py path/to/image.jpg
```

The script will automatically execute the identity model, followed by the classification and regression models, to produce a unified result.

---

## Why Regression?

| Stage-based Classification | Time-based Regression |
|----------------------------|-----------------------|
| Discrete labels (Ripe/Rotten)| Continuous days remaining |
| No granularity within stages | Captures progression of decay |
| Informational only          | Actionable for logistics/inventory |

---

## Dependencies

- PyTorch
- Torchvision
- Pandas
- Scikit-learn
- Pillow (PIL)
- NumPy

---

## Author

Alex (alexcj10)

This system was developed as a case study in applying Deep Learning to solve food waste and quality control challenges in the agricultural sector.


