#!/usr/bin/env python3
"""
Preprocessing + feature engineering for the Diabetes 130-US Hospitals dataset.

Task: predict whether a diabetic patient is readmitted within 30 days
      (readmitted == "<30" -> 1, otherwise 0).

Pipeline
  1. Load raw data (directly from archive.zip or from a CSV).
  2. Clean: drop unusable columns/rows, mark hidden missing values.
  3. Feature engineering (row-wise, no information shared between rows).
  4. Split into train / val / test (70 / 15 / 15), grouped by patient so the
     same patient never appears in two splits, and stratified on the target.
  5. Fit encoders/scaler on TRAIN ONLY, then apply them to val and test.
  6. Save clean (human-readable) and model-ready (encoded + scaled) splits.

Class imbalance is NOT handled here (no resampling): resampling must only be
applied to the training data inside the modelling step.

Usage
  python3 preprocess.py                         # defaults
  python3 preprocess.py --input archive.zip --out data/processed --seed 42
  python3 preprocess.py --first-encounter       # keep one stay per patient
"""

import argparse
import json
import os
import zipfile

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

TARGET = "readmitted_30"

# Discharge to "expired" (11, 19, 20, 21) or hospice (13, 14): these patients
# cannot realistically be readmitted, so they are removed (as in Strack et al. 2014).
EXPIRED_OR_HOSPICE = {11, 13, 14, 19, 20, 21}

# ID codes that mean "missing / not available / unknown" (see IDs_mapping in description.pdf).
UNKNOWN_ADMISSION_TYPE = {5, 6, 8}
UNKNOWN_DISCHARGE = {18, 25, 26}
UNKNOWN_ADMISSION_SOURCE = {9, 15, 17, 20, 21}

ALL_MEDS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
]
# >99.9% "No" (examide/citoglipton are 100% "No"): carry ~no information.
NEAR_CONSTANT_MEDS = [
    "chlorpropamide", "acetohexamide", "tolbutamide", "miglitol", "troglitazone",
    "tolazamide", "examide", "citoglipton", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
]
KEPT_MEDS = [m for m in ALL_MEDS if m not in NEAR_CONSTANT_MEDS]

VISIT_COLS = ["number_outpatient", "number_emergency", "number_inpatient"]

NUMERIC_FEATURES = [
    "age_num", "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_diagnoses",
    "log_number_outpatient", "log_number_emergency", "log_number_inpatient",
    "log_total_visits", "num_meds_taken", "num_med_changes",
]
BINARY_FEATURES = ["gender_male", "change", "diabetesMed"]
CATEGORICAL_FEATURES = [
    "race", "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "payer_code", "medical_specialty", "max_glu_serum", "A1Cresult",
    "diag_1_group", "diag_2_group", "diag_3_group",
] + KEPT_MEDS

# Categories with fewer than this many TRAIN rows are merged into "Other".
MIN_CATEGORY_COUNT = 100


# --------------------------------------------------------------------------- #
# 1. Load
# --------------------------------------------------------------------------- #
def load_raw(path):
    read_kw = dict(na_values="?", keep_default_na=False, low_memory=False)
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            with z.open("diabetic_data.csv") as f:
                return pd.read_csv(f, **read_kw)
    return pd.read_csv(path, **read_kw)


# --------------------------------------------------------------------------- #
# 2-3. Clean + feature engineering (row-wise only, safe before the split)
# --------------------------------------------------------------------------- #
def icd9_group(code):
    """Map an ICD-9 code to a disease group (Strack et al. 2014)."""
    if pd.isna(code):
        return "Missing"
    code = str(code)
    if code.startswith(("V", "E")):
        return "Other"
    v = float(code)
    if int(v) == 250:
        return "Diabetes"
    if 390 <= v <= 459 or int(v) == 785:
        return "Circulatory"
    if 460 <= v <= 519 or int(v) == 786:
        return "Respiratory"
    if 520 <= v <= 579 or int(v) == 787:
        return "Digestive"
    if 800 <= v <= 999:
        return "Injury"
    if 710 <= v <= 739:
        return "Musculoskeletal"
    if 580 <= v <= 629 or int(v) == 788:
        return "Genitourinary"
    if 140 <= v <= 239:
        return "Neoplasms"
    return "Other"


def clean_and_engineer(df, first_encounter=False, log=print):
    n0 = len(df)

    # Target
    df[TARGET] = (df["readmitted"] == "<30").astype(int)

    # Rows that cannot / should not be used
    df = df[~df["discharge_disposition_id"].isin(EXPIRED_OR_HOSPICE)]
    log(f"  removed expired/hospice discharges: {n0 - len(df)}")
    n1 = len(df)
    df = df[df["gender"] != "Unknown/Invalid"]
    log(f"  removed gender 'Unknown/Invalid':   {n1 - len(df)}")

    if first_encounter:
        n2 = len(df)
        df = df.sort_values("encounter_id").drop_duplicates("patient_nbr", keep="first")
        log(f"  kept first encounter per patient:  removed {n2 - len(df)}")

    df = df.copy()

    # Hidden missing values in ID columns -> explicit "Unknown"
    def id_to_cat(series, unknown_codes):
        s = series.astype(str)
        s[series.isin(unknown_codes)] = "Unknown"
        return s

    df["admission_type_id"] = id_to_cat(df["admission_type_id"], UNKNOWN_ADMISSION_TYPE)
    df["discharge_disposition_id"] = id_to_cat(df["discharge_disposition_id"], UNKNOWN_DISCHARGE)
    df["admission_source_id"] = id_to_cat(df["admission_source_id"], UNKNOWN_ADMISSION_SOURCE)

    # Missing categorical values -> their own "Unknown" category
    # (missingness itself can be informative, e.g. no payer code recorded)
    for col in ["race", "payer_code", "medical_specialty"]:
        df[col] = df[col].fillna("Unknown")

    # "None" in lab results = test not performed: a real category, keep it.

    # Ordinal: age bracket "[70-80)" -> midpoint 75
    df["age_num"] = df["age"].str.extract(r"\[(\d+)-")[0].astype(int) + 5

    # Binary
    df["gender_male"] = (df["gender"] == "Male").astype(int)
    df["change"] = (df["change"] == "Ch").astype(int)
    df["diabetesMed"] = (df["diabetesMed"] == "Yes").astype(int)

    # Diagnoses: ~700 ICD-9 codes -> 9 disease groups
    for c in ["diag_1", "diag_2", "diag_3"]:
        df[f"{c}_group"] = df[c].map(icd9_group)

    # Prior visits: heavily right-skewed -> log1p; plus total
    df["total_visits"] = df[VISIT_COLS].sum(axis=1)
    for c in VISIT_COLS + ["total_visits"]:
        df[f"log_{c}"] = np.log1p(df[c])

    # Medication summaries (computed over all 23 drugs before dropping any)
    meds = df[ALL_MEDS]
    df["num_meds_taken"] = (meds != "No").sum(axis=1)
    df["num_med_changes"] = meds.isin(["Up", "Down"]).sum(axis=1)

    keep = (["encounter_id", "patient_nbr"] + NUMERIC_FEATURES + BINARY_FEATURES
            + CATEGORICAL_FEATURES + [TARGET])
    # Dropped: weight (97% missing), raw diag codes (replaced by groups),
    # raw age/gender, raw visit counts (replaced by log versions),
    # near-constant meds, original 3-class readmitted.
    return df[keep].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 4. Split: 70/15/15, grouped by patient, stratified on target
# --------------------------------------------------------------------------- #
def split_train_val_test(df, seed):
    # 20 folds of ~5% each -> 3 folds test, 3 folds val, 14 folds train
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    fold = np.empty(len(df), dtype=int)
    for i, (_, idx) in enumerate(sgkf.split(df, df[TARGET], groups=df["patient_nbr"])):
        fold[idx] = i
    test = df[fold < 3]
    val = df[(fold >= 3) & (fold < 6)]
    train = df[fold >= 6]
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 5. Encoding + scaling: fit on train only
# --------------------------------------------------------------------------- #
def fit_encoder(train):
    """Learn category levels and scaling stats from the training set."""
    levels = {}
    for col in CATEGORICAL_FEATURES:
        counts = train[col].value_counts()
        levels[col] = sorted(counts[counts >= MIN_CATEGORY_COUNT].index.tolist())
    means = train[NUMERIC_FEATURES].mean()
    stds = train[NUMERIC_FEATURES].std().replace(0, 1)
    return {"levels": levels, "means": means.to_dict(), "stds": stds.to_dict()}


def transform(df, enc):
    parts = [df[["encounter_id", "patient_nbr"]]]

    num = (df[NUMERIC_FEATURES] - pd.Series(enc["means"])) / pd.Series(enc["stds"])
    parts.append(num)
    parts.append(df[BINARY_FEATURES])

    for col in CATEGORICAL_FEATURES:
        lv = enc["levels"][col]
        # Rare levels and levels unseen in train -> "Other"
        s = df[col].where(df[col].isin(lv), "Other")
        cats = lv + (["Other"] if "Other" not in lv else [])
        dummies = pd.get_dummies(pd.Categorical(s, categories=cats), prefix=col, dtype=int)
        dummies.index = df.index
        parts.append(dummies)

    parts.append(df[[TARGET]])
    return pd.concat(parts, axis=1)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="archive.zip",
                    help="archive.zip or diabetic_data.csv (default: archive.zip)")
    ap.add_argument("--out", default="data/processed", help="output directory")
    ap.add_argument("--seed", type=int, default=42, help="random seed for the split")
    ap.add_argument("--first-encounter", action="store_true",
                    help="keep only each patient's first encounter")
    args = ap.parse_args()

    print(f"Loading {args.input}")
    raw = load_raw(args.input)
    print(f"  raw shape: {raw.shape}")

    print("Cleaning + feature engineering")
    df = clean_and_engineer(raw, first_encounter=args.first_encounter)
    print(f"  clean shape: {df.shape}")

    print("Splitting 70/15/15 (grouped by patient, stratified)")
    train, val, test = split_train_val_test(df, args.seed)
    overlap = (set(train.patient_nbr) & set(val.patient_nbr)) | \
              (set(train.patient_nbr) & set(test.patient_nbr)) | \
              (set(val.patient_nbr) & set(test.patient_nbr))
    assert not overlap, "patient leakage between splits"

    print("Encoding + scaling (fit on train only)")
    enc = fit_encoder(train)
    train_m, val_m, test_m = (transform(d, enc) for d in (train, val, test))

    clean_dir = os.path.join(args.out, "clean")
    model_dir = os.path.join(args.out, "model_ready")
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    summary = {"seed": args.seed, "first_encounter": args.first_encounter, "splits": {}}
    for name, c, m in [("train", train, train_m), ("val", val, val_m), ("test", test, test_m)]:
        c.to_csv(os.path.join(clean_dir, f"{name}.csv"), index=False)
        m.to_csv(os.path.join(model_dir, f"{name}.csv"), index=False)
        summary["splits"][name] = {
            "rows": len(c),
            "patients": int(c.patient_nbr.nunique()),
            "share": round(len(c) / len(df), 4),
            "positive_rate": round(float(c[TARGET].mean()), 4),
        }
    summary["n_model_features"] = train_m.shape[1] - 3  # minus 2 ids + target
    summary["feature_names"] = [c for c in train_m.columns
                                if c not in ("encounter_id", "patient_nbr", TARGET)]
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out, "encoder.json"), "w") as f:
        json.dump(enc, f, indent=2)

    print(f"\n{'split':<6} {'rows':>7} {'patients':>9} {'share':>7} {'pos_rate':>9}")
    for name, s in summary["splits"].items():
        print(f"{name:<6} {s['rows']:>7} {s['patients']:>9} {s['share']:>7.2%} {s['positive_rate']:>9.2%}")
    print(f"\nmodel-ready features: {summary['n_model_features']}")
    print(f"saved to {args.out}/ (clean/, model_ready/, summary.json, encoder.json)")


if __name__ == "__main__":
    main()
