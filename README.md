# 🫀 HeartGuard MLOps

> End-to-end machine learning pipeline for heart disease classification — built with production-grade MLOps practices.

[![DVC](https://img.shields.io/badge/DVC-Pipeline-945DD6?logo=dvc)](https://dvc.org)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow)](https://mlflow.org)
[![DAGsHub](https://img.shields.io/badge/DAGsHub-Experiment%20Store-FC6D26)](https://dagshub.com/asadullahcreative)
[![Streamlit](https://img.shields.io/badge/Streamlit-Demo-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-22c55e)](LICENSE)

---

## Overview

HeartGuard is a fully automated, reproducible MLOps pipeline that predicts the presence of heart disease from clinical features. The project covers the complete ML lifecycle — from raw data ingestion through model deployment — using industry-standard tooling.

---

## Pipeline Architecture

```
Data Ingestion  →  Data Validation  →  Data Transformation
                                              ↓
                                      Model Training
                                              ↓
                                   Model Evaluation (MLflow)
                                              ↓
                                   Streamlit Deployment
```

---

## Tech Stack

| Layer | Tool |
|-------|------|
| Pipeline Orchestration | DVC |
| Experiment Tracking | MLflow + DAGsHub |
| Model | Logistic Regression (scikit-learn) |
| Config Management | YAML + Python dataclasses |
| Demo App | Streamlit |
| CI/CD | GitHub Actions |
| Environment | Docker |

---

## Project Structure

```
HeartGuard-MLOps/
├── .github/workflows/    # CI/CD pipeline
├── config/               # YAML configuration
├── src/mlProject/        # Core pipeline modules
│   ├── components/       # Stage implementations
│   ├── pipeline/         # Stage runners
│   ├── entity/           # Dataclasses
│   └── utils/            # Helpers
├── dvc.yaml              # DVC pipeline definition
├── params.yaml           # Model hyperparameters
├── schema.yaml           # Data schema & target column
├── app.py                # Streamlit demo
└── Dockerfile            # Container setup
```

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/AsadullahShehbaz/HeartGuard-MLOps.git
cd HeartGuard-MLOps

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run full pipeline (skips unchanged stages automatically)
dvc repro

# 4. Launch demo app
streamlit run app.py
```

---

## MLflow Experiment Tracking

Experiments are tracked and stored on DAGsHub. Metrics logged per run:

- Accuracy
- Precision  
- Recall

```python
# .env setup required
MLFLOW_TRACKING_URI=https://dagshub.com/asadullahcreative/<repo>.mlflow
MLFLOW_TRACKING_USERNAME=asadullahcreative
MLFLOW_TRACKING_PASSWORD=<your_token>
```

---

## Dataset

**Heart Disease Dataset** — 13 clinical features including age, blood pressure, cholesterol, ECG results, and thallium stress test results. Target: `Heart Disease` (Presence / Absence).

---

## Author

**Asadullah Shehbaz** — AI/ML Engineer

[![Kaggle](https://img.shields.io/badge/Kaggle-Datasets%20Grandmaster-20BEFF?logo=kaggle)](https://www.kaggle.com/asadullahcreative)
[![GitHub](https://img.shields.io/badge/GitHub-AsadullahShehbaz-181717?logo=github)](https://github.com/AsadullahShehbaz)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin)](https://www.linkedin.com/in/asadullah-shehbaz-18172a2bb/)
[![Fiverr](https://img.shields.io/badge/Fiverr-Hire%20Me-1DBF73?logo=fiverr)](https://www.fiverr.com/asadullah_92)

---

## License

MIT © Asadullah Shehbaz
