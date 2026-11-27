# Predicting 30-Day Readmission of Diabetic Patients

CISC 5790 Data Mining — course project (Fordham University, Fall 2026).

Binary classification on the [Diabetes 130-US Hospitals (1999–2008)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) dataset:
predict whether a patient is readmitted within 30 days of a hospital stay (`readmitted == "<30"`).

## Data

`archive.zip` contains the raw data (`diabetic_data.csv`, 101,766 hospital stays × 50 columns) and `description.pdf`.
Each row is one hospital stay (encounter); a patient can have several stays.

## Requirements

Python 3 with `pandas`, `numpy`, `scikit-learn >= 1.0`.

## Preprocessing

```bash
python3 preprocess.py                    # reads archive.zip, writes data/processed/
python3 preprocess.py --seed 42 --out data/processed
python3 preprocess.py --first-encounter  # keep only each patient's first stay
```

Steps:
1. **Cleaning** – remove stays ending in death/hospice (cannot be readmitted) and invalid gender; drop `weight` (97% missing) and 12 near-constant medication columns.
2. **Missing values** – missing `race`, `payer_code`, `medical_specialty` and "not available" ID codes become an `Unknown` category; `None` in lab results (test not performed) is kept as its own category.
3. **Feature engineering** – ICD-9 diagnosis codes grouped into 9 disease groups, age bracket → numeric, log-transformed visit counts, `total_visits`, `num_meds_taken`, `num_med_changes`.
4. **Split** – 70/15/15 train/val/test, **grouped by patient** (no patient in more than one split) and stratified on the target.
5. **Encoding** – one-hot encoding (rare levels → `Other`) and standardization, fitted on the training set only.

Class imbalance (~11% positive) is handled in the modelling step, on the training set only.

### Output (`data/processed/`)

| Path | Contents |
|---|---|
| `clean/{train,val,test}.csv` | Cleaned, human-readable data (for EDA / report) |
| `model_ready/{train,val,test}.csv` | One-hot encoded + scaled features for models. Drop `encounter_id`, `patient_nbr`; target is `readmitted_30` |
| `summary.json` | Split sizes, positive rates, feature names |
| `encoder.json` | Category levels and scaling statistics learned from train |

## Reference

Strack, B. et al. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records.* BioMed Research International.
