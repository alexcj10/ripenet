<p align="center">
  <img src="assets/rip.svg" alt="RipeNet Logo" width="500">
</p>

RipeNet is an end-to-end computer vision suite for fruit species identification, ripeness classification, and shelf-life estimation using deep learning. The system utilizes multiple specialized models to provide a comprehensive analysis of fruit quality and remaining viability.

The core innovation of RipeNet is its transition from discrete classification to continuous regression, modeling shelf-life as a time-based value in days.

---

## Live Demo

- **Web Application**: [ripenet.vercel.app](https://ripenet.vercel.app)
- **Backend API**: [huggingface.co/spaces/alexcj10/ripenet-backend](https://huggingface.co/spaces/alexcj10/ripenet-backend)
- **CLI Tool**: Included in this repository! (See below for setup)

---

## Command Line & API Access
<img width="669" height="436" alt="image" src="https://github.com/user-attachments/assets/9e91e189-6daf-40ad-ae28-f8b24cd5d0ea" />
<br>

For **Developers** and **Power Users** who prefer the terminal, RipeNet offers two advanced access methods:

### Option 1: Premium CLI (Best Experience)
**Requires**: Python 3.7+

This is the official terminal tool for power users. **Choose ONE of the two methods below** to install the `ripenet` command:

#### Method A: Fast Install (Requires Git)
Best for users who want to install directly without manual downloads.
```bash
pip install git+https://github.com/alexcj10/ripenet.git
```

#### Method B: Manual Install (For Developers)
Best if you have already downloaded/cloned the code manually.
1. Open your terminal **inside the project folder**.
2. Run: `pip install .`

---

### **Done!** 
After running **either** A or B, the `ripenet` command is now live on your machine! You can now run these from any terminal window:
- **Scan a single image**: `ripenet scan "path/to/fruit.jpg"`
- **Batch scan a folder**: `ripenet batch "path/to/folder"`
- **Check System & API Status**: `ripenet info`

> [!IMPORTANT]
> You only need to install once. Once installed, you can use `ripenet` from any folder forever (no need to be inside the project folder anymore).
>
> **Windows/PowerShell User?**
> If you see `ripenet: The term is not recognized`, run this command to fix your path:
> ```powershell
> $env:PATH += ";C:\Users\$env:USERNAME\AppData\Roaming\Python\Python313\Scripts"
> ```

<img width="845" height="406" alt="image" src="https://github.com/user-attachments/assets/885a9f10-3090-4b66-a36f-9afa8090e522" />

---

### Option 2: Instant API (Zero Install)
**Requires**: Nothing but a terminal!
Works on **any laptop** (Mac, Windows, Linux) immediately.

1. **Run the One-Liner**:
   ```bash
   curl -F "file=@fruit.jpg" https://alexcj10-ripenet-backend.hf.space/predict
   ```
<img width="1046" height="104" alt="image" src="https://github.com/user-attachments/assets/7b3485cc-4f49-4240-bc83-63ce82a131a3" />
<br>

> [!TIP]
> **Windows Users**: If the `curl` command acts unexpectedly in PowerShell, try using **`curl.exe`** instead.
>
> **Troubleshooting `curl: (26)`**: If you see this error, it's often due to complex characters (commas, colons, etc.) in the filename. **Rename your file** to something simple like `fruit.png` and try again.
>
> **Paths with Spaces**: If your folder names have spaces (e.g., `unripe orange`), always wrap the path in double quotes: 
> `"C:\My Fruits\fresh apple.jpg"`
>
> **Local Files Only**: The CLI and API expect a file saved on your computer. If you have an online image, download it first before scanning!

---

## Compatibility Matrix

| Environment | **Premium CLI** | **Instant API** |
| :--- | :---: | :---: |
| **Windows** | Supported | Supported (`curl.exe`) |
| **macOS** | Supported | Supported |
| **Linux** | Supported | Supported |
| **Dependencies** | Python 3.7+ | None |

---

## Technical Overview

1.  **Identity Model**: A deep learning classifier that identifies the fruit species (supported: Apple, Banana, Mango, Orange, Papaya, Pineapple) to select appropriate biological decay parameters.
2.  **Classification Model**: A secondary check for discrete ripeness stages (Unripe, Fresh/Ripe, Rotten) to validate visual status.
3.  **Regression Model**: The primary engine that predicts estimated remaining shelf-life in days by analyzing surface features, color distribution, and texture degradation.

Note: RipeNet V2 (Production) utilizes a Multi-Task Learning (MTL) architecture where a single EfficientNet-B0 backbone processes identity, classification, and regression simultaneously. This reduces inference latency by over 50% compared to the original sequential architecture.

The system is deployed using a decoupled architecture:
- **Frontend**: React (Vite) hosted on Vercel with high-performance Framer Motion animations.
- **Backend**: FastAPI (Python) hosted on Hugging Face Spaces (Docker) leveraging 16GB RAM for rapid model inference.

---

## Web Application & Local Storage

The RipeNet Web Application is designed for efficiency and privacy, utilizing local browser storage for session persistence.

- **Fruit Vault (History)**: The application automatically stores your last **10 scans** locally in the browser's "Fruit Vault" for quick reference.
- **Privacy First**: All history data is saved in your browser's `IndexedDB`. No images are stored on our servers after analysis.
- **Storage Capacity**: Validated for high-resolution images (up to 10MB+ per scan). The local storage system can comfortably handle over 100MB of historical scan data.

---

## Key Features

- **Multi-Model Pipeline**: Automated flow from species identification to shelf-life prediction.
- **Continuous Regression**: Predicts shelf-life in days instead of static categories.
- **Pre-trained Backbones**: Leverage transfer learning with EfficientNet-B0 for high accuracy and low latency.
- **Natural Language Inference**: Generates varied, human-readable status reports.
- **Metric Verification**: Validated using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

---

## Dataset

The complete RipeNet v2.0 image dataset and labels (required for training and evaluation) can be downloaded from Kaggle:
[RipeNet 2.0 Fruit Dataset](https://www.kaggle.com/datasets/alexcj10/ripenet-2-0-fruit-dataset)

---

## Project Structure

```
.
├── api/                    # Backend API (Hugging Face / Docker)
│   ├── main.py             # FastAPI Application logic
│   ├── Dockerfile          # Container configuration
│   └── requirements.txt    # CPU-optimized dependencies
│
├── data/                   # Dataset Directory (Ignored in Git, hosted on Kaggle)
│   ├── raw/                # Image folders (train, train_selected, RipeNet 2.0)
│   └── labels/             # CSV ground truth files
│
├── frontend/               # React Web Application (Vercel)
│   ├── src/                # UI components and API logic
│   ├── App.jsx             # Main application flow
│   └── App.css             # Premium styling and glassmorphism
│
├── ml/                     # Modular Machine Learning Code
│   ├── core/               # Training and logic for Multi-Task Model (V2)
│   ├── identity/           # Training and logic for Identity Model
│   └── regression/         # Training and logic for Regression Model
│
├── reports/                # Benchmarking and evaluation reports
│   ├── metrics/            # CSV summaries and validation data
│   ├── plots/              # Visualization of model performance
│   └── evaluate_models.py  # Professional benchmarking script
│
├── saved_models/           # Pre-trained model weights (.pth)
│   ├── best_model.pth             # Classification model (V1)
│   ├── best_identity_model.pth    # Species identification model (V1)
│   ├── best_regression_model.pth  # Shelf-life regression model (V1)
│   └── ripenet_v2_mtl.pth         # Multi-Task Learning model (V2 Production)
│
├── scripts/                # Utility scripts for data preparation
│   ├── prepare_ripenet_v2.py   # Dataset preparation for V2
│   ├── verify_v2_pipeline.py   # Validation script for MTL architecture
│   └── ...                     # Other data manipulation scripts
│
├── ripenet_cli.py          # Entry point for the Premium CLI
├── .gitignore              # Project-wide git ignore rules
└── requirements.txt        # Local environment dependencies
```

---

## Data Labeling Strategy

Regression targets are derived from fruit-specific decay curves. The systems use distinct labeling strategies to optimize for either discrete experts (Model 1) or multi-task correlation (Model 2).

### Model 1 Configuration (Original Sequential)
Labeling strategy focused on standard viability windows.

| Fruit   | Unripe Stage (Days) | Fresh Stage (Days) | Rotten Stage (Days) |
|---------|---------------------|--------------------|---------------------|
| Apple   | 10.0                | 5.0                | 2.0                 |
| Banana  | 6.0                 | 3.0                | 1.0                 |
| Orange  | 8.0                 | 4.0                | 2.0                 |
| Papaya  | 6.0                 | 3.0                | 1.0                 |

### Model 2 Configuration (MTL Production)
Labeling strategy optimized for 6 fruit species with robust boundary modeling.

| Fruit     | Unripe Stage (Days) | Fresh Stage (Days) | Rotten Stage (Days) |
|-----------|---------------------|--------------------|---------------------|
| Apple     | 10.0                | 5.0                | 0.0                 |
| Banana    | 6.0                 | 3.0                | 0.0                 |
| Mango     | 7.0                 | 3.0                | 0.0                 |
| Orange    | 8.0                 | 5.0                | 0.0                 |
| Papaya    | 5.0                 | 2.0                | 0.0                 |
| Pineapple | 6.0                 | 3.0                | 0.0                 |

Additional Refinements in Model 2:
- **Spoilage Modeling**: For very spoiled or black fruit, targets are set to a random range between -1.0 and -3.0 to force the model to learn expiration depth beyond the zero-day threshold.
- **Data Jittering**: A small variation (e.g., +/- 0.2) is added to global labels to enhance sensitivity and prevent the model from over-fitting to fixed integer values.

---

## Benchmarking and Validation

To validate the production readiness of RipeNet V2, a professional benchmark was conducted against the original V1 baseline using a held-out test set.

### Methodology
Testing was performed on an independent, held-out benchmark set of 120 images (10 images per class for the 4 primary fruits across 3 ripeness stages). 

#### Data Volume & Training
*   **RipeNet V1**: Trained on **6,200 images** (4,340 Training / 1,860 Validation & Test).
*   **RipeNet V2 (MTL)**: Trained on a refined, high-quality dataset of **9,537 images** (7,628 Training / 1,909 Validation) — an ~86% increase in training data volume compared to V1.

#### Label Alignment Strategy
To ensure a **100% fair and representative comparison**, the benchmarking process utilizes a **Label Alignment Layer**. While V1 and V2 models were trained using different labeling strategies (e.g., V2 incorporates negative values to model 'spoilage depth'), the evaluation script normalizes all predictions to a unified positive scale.

#### Benchmark Ground Truth
To ensure reproducibility, the 120-image benchmark set (covering Apple, Banana, Orange, and Papaya) utilized the following interpretable ground truth labels:

| Fruit   | Unripe (Label 0) | Ripe (Label 1) | Rotten (Label 2) |
|---------|------------------|----------------|------------------|
| Apple   | 10 days          | 5 days         | 2 days           |
| Banana  | 6 days           | 3 days         | 1 day            |
| Orange  | 8 days           | 4 days         | 2 days           |
| Papaya  | 6 days           | 3 days         | 1 day            |

![Benchmark Comparison](assets/benchmark_comparison.png)

| Metric | RipeNet V1 (Baseline) | RipeNet V2 (MTL) | Development ROI |
| :--- | :--- | :--- | :--- |
| **Fruit Identity Accuracy** | 69.17% | **77.50%** | +12.0% Improvement |
| **Stage Accuracy** | 65.83% | **70.83%** | +7.6% Improvement |
| **Error (MAE)** | 1.70 Days | **1.53 Days** | -10.3% Error Reduction |
| **Avg Latency** | 83.02 ms | **35.50 ms** | **~2.3x Faster** |

The V2 Multi-Task Backbone achieves faster inference by sharing feature extraction across all three task heads in a single forward pass.

---

## Evaluation Results

Performance metrics recorded during the final validation phases:

### RipeNet V1 (Original Metrics)
- **Classification Accuracy**: 92.6%
- **Identity Model Accuracy**: 93.6%
- **Regression Mean Absolute Error (MAE)**: 0.74 days

### RipeNet V2 (Production MTL Metrics)
- **Fruit Identification Accuracy**: 93.3%
- **Ripeness Stage Accuracy**: 80.4%
- **Regression Mean Absolute Error (MAE)**: 1.26 days

Note: RipeNet V2 training was conducted over 15 epochs with an un-frozen EfficientNet backbone, allowing for higher fidelity feature representations tailored to the 6-fruit classification task.

---

## Analysis of Metric Variance: Internal Validation vs. External Benchmarking

A distinction must be made between internal validation metrics recorded during model development and the results of the independent external benchmark.

During the internal training and validation phase, performance was measured against the models' respective original data splits:
- **RipeNet V1 Internal MAE**: 0.74 Days
- **RipeNet V2 Internal MAE**: 1.26 Days

The **External Benchmark** presented in this report was conducted on an entirely independent, balanced test set of 120 images to assess architectural robustness and generalization:
- **RipeNet V1 External MAE**: 1.70 Days
- **RipeNet V2 External MAE**: 1.53 Days

### Interpretation of Generalization Performance
While RipeNet V1 achieved higher precision on its internal split, the external benchmark reveals a significant increase in error when exposed to unseen data (0.74 to 1.70). This suggests a higher degree of architectural overfitting to its original training distribution.

In contrast, the Multi-Task Learning (MTL) design of RipeNet V2 demonstrates superior generalization. By achieving a lower regression error (1.53 MAE) and higher identification accuracy on the independent benchmark set, RipeNet V2 proves more robust for real-world deployment. The benchmark results provide a high-confidence estimation of performance in production environments where data variety is higher than in controlled training sets.

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

The script utilizes the RipeNet V2 Multi-Task Learning (MTL) architecture to simultaneously identify species, classify ripeness, and predict shelf-life in a single forward pass.

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




















