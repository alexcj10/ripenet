<div align="center">
  <h1><img src="assets/apple.svg" width="38" style="vertical-align: middle;"> RipeNet</h1>
</div>

RipeNet is an end-to-end computer vision suite for fruit species identification, ripeness classification, and shelf-life estimation using deep learning. The system utilizes multiple specialized models to provide a comprehensive analysis of fruit quality and remaining viability.

The core innovation of RipeNet is its transition from discrete classification to continuous regression, modeling shelf-life as a time-based value in days.

---

## 🚀 Live Demo

- **Web Application**: [ripenet.vercel.app](https://ripenet.vercel.app)
- **Backend API**: [huggingface.co/spaces/alexcj10/ripenet-backend](https://huggingface.co/spaces/alexcj10/ripenet-backend)
- **CLI Tool**: Included in this repository! (See below for setup)

---

## 🌎 How to Use RipeNet

You can use RipeNet's AI in two ways, depending on your setup.

### Option 1: Premium CLI (Best Experience)
Ideal if you have **Python** installed. Provides beautiful terminal output, batch scanning, and the official `ripenet` command.

1. **Install**:
   ```bash
   pip install .
   ```
2. **Analyze**:
   ```bash
   ripenet scan "path/to/fruit.jpg"
   ```

### Option 2: Instant API (Zero Install)
Works on **any laptop** (Mac, Windows, Linux) without installing Python or anything else.

1. **Analyze immediately**:
   ```bash
   curl -F "file=@fruit.jpg" https://alexcj10-ripenet-backend.hf.space/predict
   ```

> [!TIP]
> **Windows Users**: If the `curl` command acts unexpectedly in PowerShell, try using `curl.exe` instead.
> [!TIP]
> **Troubleshooting `curl: (26)`**: If you see this error, it's often due to complex characters (commas, colons, etc.) in the filename. **Rename your file** to something simple like `fruit.png` and try again.
>
> **Paths with Spaces**: If your folder names have spaces (e.g., `unripe orange`), always wrap the path in double quotes: 
> `"C:\My Fruits\fresh apple.jpg"`
>
> **Local Files Only**: The CLI and API expect a file saved on your computer. If you have an online image, download it first before scanning!

---

## 💻 Compatibility Matrix

| Environment | **Premium CLI** | **Instant API** |
| :--- | :---: | :---: |
| **Windows** | ✅ | ✅ (`curl.exe`) |
| **macOS** | ✅ | ✅ |
| **Linux** | ✅ | ✅ |
| **Dependencies** | Python 3.7+ | None |

---

## Technical Overview

RipeNet is structured as a triple-expert system:

1.  **Identity Model**: An EfficientNet-B0 classifier that identifies the fruit species (Apple, Banana, Orange, Papaya) to select appropriate biological decay parameters.
2.  **Classification Model**: A secondary check for discrete ripeness stages (Unripe, Fresh/Ripe, Rotten) to validate visual status.
3.  **Regression Model**: The primary engine that predicts the estimated remaining shelf-life in days by analyzing surface features, color distribution, and texture degradation.

The system is deployed using a decoupled architecture:
- **Frontend**: React (Vite) hosted on Vercel with high-performance Framer Motion animations.
- **Backend**: FastAPI (Python) hosted on Hugging Face Spaces (Docker) leveraging 16GB RAM for rapid model inference.

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
├── api/                    # Backend API (Hugging Face / Docker)
│   ├── main.py             # FastAPI Application logic
│   ├── Dockerfile          # Container configuration
│   └── requirements.txt    # CPU-optimized dependencies
│
├── frontend/               # React Web Application (Vercel)
│   ├── src/                # UI components and API logic
│   ├── App.jsx             # Main application flow
│   └── App.css             # Premium styling and glassmorphism
│
├── saved_models/           # Pre-trained model weights (.pth)
│   ├── best_model.pth             # Classification model
│   ├── best_identity_model.pth    # Species identification model
│   └── best_regression_model.pth  # Shelf-life regression model
│
├── src/                    # Training and logic for Classification
├── src_identity/           # Training and logic for Identity
├── src_regression/         # Training and logic for Regression
├── .gitignore              # Project-wide git ignore rules
└── requirements.txt        # Local environment dependencies
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

- [x] PyTorch
- [x] Torchvision
- [x] Pandas
- [x] Scikit-learn
- [x] Pillow (PIL)
- [x] NumPy 

---

## Author

Alex (alexcj10)

This system was developed as a case study in applying Deep Learning to solve food waste and quality control challenges in the agricultural sector.





