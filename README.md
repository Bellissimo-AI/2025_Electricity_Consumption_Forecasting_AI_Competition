# 2025 Electricity Consumption Forecasting AI Competition

## Awards
- **2nd place (Excellence Award)** out of 934 teams, *Team SKKU brAIn*
- **Top 0.2%**
- **Korea Energy Agency (KEA)**
- [Competition link](https://dacon.io/competitions/official/236531/overview/description)
- [Competition code release and presentation materials link](https://dacon.io/competitions/official/236531/codeshare/12766?page=1&dtype=recent)

---

## Electricity consumption forecasting pipeline

> **If you encounter any issues, please refer to `SKKU_brAIn.ipynb`.**  
> This repository modularizes the original notebook into a reproducible script-based pipeline.

## Overview
- **Task**: Forecast building-level hourly electricity consumption (kWh) and generate a submission file  
- **Key ideas**: Seasonal (summer) features + clustering-based decomposition + group-wise XGBoost models + weighted ensemble + holiday-based post-processing  
- **Outputs**: `outputs/final.csv` (submission file), intermediate results saved under `outputs/answer_*.csv`  
- **Reference**: `SKKU_brAIn.ipynb` is the original competition notebook

---

## Repository structure
```
.
├─ data/ # train/test/building_info, sample_submission
├─ config.py # global configs (paths, seed, KFold splits)
├─ preprocessing.py # preprocessing (feature engineering, outliers, clustering)
├─ modeling.py # XGBoost training (group-wise & global)
├─ ensemble.py # weighted ensemble → outputs/final_ensemble.csv
├─ postprocessing.py # holiday-based post-processing → outputs/final.csv
├─ main.py # end-to-end pipeline runner
├─ utils.py # metrics & custom objectives
├─ requirements.txt # dependencies
└─ SKKU_brAIn.ipynb # original competition notebook
```

---

## Requirements
- Python 3.10+
- NVIDIA GPU recommended (XGBoost `gpu_hist`)

Install:
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
---

## Data

The following files must be placed in the `data/` directory:

- `train.csv`  
- `test.csv`  
- `building_info.csv`  
- `sample_submission.csv`  
- `outlier (4).xlsx` (used in preprocessing)

> On Colab, the code automatically mounts Google Drive at `/content/drive/MyDrive/Colab Notebooks`.  
> Otherwise, it defaults to local `./data`.

---

## Configuration

`config.py`:

```python
RANDOM_SEED = 42
KFOLD_SPLITS = 10
DATA_DIR = "./data"
OUTPUT_DIR = "./outputs"
```

---

## How to run

### End-to-end pipeline

Run:
```bash
python main.py
```

**Steps:**
1. Train 8 models: {summer=0,1} × {type, number, cluster, global}  
2. Weighted ensemble → `outputs/final_ensemble.csv`  
3. Holiday-based adjustment → `outputs/final.csv` (submission file)

### Run partial steps

Re-run **ensemble only**:
```bash
python -c "from ensemble import weighted_ensemble; weighted_ensemble()"
```

Re-run **post-processing only**:
```bash
python -c "from postprocessing import apply_holiday_adjustment; apply_holiday_adjustment()"
```

---

## Modeling

### Group-wise training (`train_xgb`)
- Models trained per group (`building_type`, `building_number`, or `cluster`)  
- 10-fold CV, log-transform target during training → exp-transform at prediction  

### Global training (`train_global`)
- One-hot encoding of categorical features  
- Single global model  

### Loss/metrics
- Custom objective: `weighted_mse(alpha=3)`  
- Validation metric: `custom_smape`  
- Report metric: `smape`  

### Key hyperparameters
- `learning_rate=0.05`, `n_estimators=5000`, `subsample=0.7`, `colsample_bytree=0.5`, `min_child_weight=3`  
- `max_depth` varies by group  
- `tree_method="gpu_hist"`, `early_stopping_rounds=100`  

---

## Features
- **Datetime features**: hour, day, month, day_of_week + Fourier (sin/cos) encodings  
- **Daily temperature stats**: max, mean, min, range  
- **Indices**: THI (Temperature-Humidity Index), WCT (Wind Chill Temperature), CDH (Cooling Degree Hours, threshold=26°C)  
- **Holiday flag**: weekends + specific holidays  
- **Summer-specific signals**: sin/cos seasonal cycles for June–September  
- **Cluster features**: KMeans (k=5) on pivoted day-of-week × hour consumption  

---

## Ensemble
`ensemble.weighted_ensemble()` performs:

- **Base combination**  
  - Summer: `type*0.2 + number*0.3 + global*0.5`  
  - No-summer: `type*0.25 + number*0.25 + global*0.5`  
  - Base: `(summer*0.8 + nosummer*0.2) * 0.8`

- **Cluster combination**  
  - `summer_cluster*0.7 + nosummer_cluster*0.3`

- **Final result**  
  - `final = base + cluster`  
  - Negative values clipped to 0  
  - Output: `outputs/final_ensemble.csv`

---

## Post-processing
`postprocessing.apply_holiday_adjustment()`:
- Implements original competition holiday replacement rules  
- Replaces specific building/time predictions with trimmed means of historical values  
- Final output: `outputs/final.csv`

---

## Outputs
- **Intermediate**:  
  - `outputs/answer_type_summer{0|1}.csv`  
  - `outputs/answer_number_summer{0|1}.csv`  
  - `outputs/answer_global_summer{0|1}.csv`  
  - `outputs/answer_cluster_summer{0|1}.csv`  
- **Ensemble**: `outputs/final_ensemble.csv`  
- **Final submission**: `outputs/final.csv`

---

## Reproducibility
- Seeds fixed: Python `random`, NumPy, XGBoost (`random_state=42`)  
- KFold: 10 splits, shuffle=True, random_state=42  
> Small numerical differences may occur across hardware/driver versions.

---

## Troubleshooting
- **No GPU**: change `tree_method="hist"` and remove `gpu_id`  
- **Schema mismatch**: ensure one-hot encoded columns are reindexed properly  
- **Encoding issues**: CSV files assumed to be `utf-8-sig`  
- **Negative predictions**: automatically clipped to 0 in ensemble step  

---

## Acknowledgements
- Developed using official competition data and task definition  
- Modularized from the original `SKKU_brAIn.ipynb` into a reproducible pipeline  
- **If issues arise, please check `SKKU_brAIn.ipynb`**
