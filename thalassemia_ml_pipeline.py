"""
thalassemia_ml_pipeline.py
==========================
Complete reproducible analysis pipeline for:

  Rahman MR, Rose JI, Rahman MR. Machine Learning-Based Multi-Class
  Classification of Beta-Thalassemia Trait and Hemoglobin E Carrier Status
  Using HPLC and Complete Blood Count Parameters. PLOS ONE, 2026. (Under review)

Usage:
    python thalassemia_ml_pipeline.py --data "data/HPLC data.csv"

All outputs (metrics CSV, figures, trained models) are saved to results/.
"""

import argparse
import os
import re
import warnings
import joblib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
TARGET_CLASSES = ["Normal", "Beta Thalassaemia carrier", "HbE carrier"]
FULL_FEATURES = ["HbA0", "HbA2", "HbF", "RBC", "HB", "MCV", "MCH", "MCHC", "RDWcv",
                 "Age_num", "Gender_num"]
CBC_FEATURES  = ["RBC", "HB", "MCV", "MCH", "MCHC", "RDWcv", "Age_num", "Gender_num"]


# ── Data loading and preprocessing ───────────────────────────────────────────

def parse_age(age_str):
    if pd.isna(age_str):
        return np.nan
    match = re.search(r"(\d+)\s*Yr", str(age_str))
    return int(match.group(1)) if match else np.nan


def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)
    df["Diagnosis_clean"] = df["Diagnosis"].str.strip()
    df = df[df["Diagnosis_clean"].isin(TARGET_CLASSES)].copy()
    df["Age_num"] = df["Age"].apply(parse_age)
    df["Gender_num"] = (df["Gender"] == "Male").astype(int)
    df = df.dropna(subset=FULL_FEATURES).copy()
    print(f"Loaded {len(df)} records after class selection and missing-value exclusion.")
    print(df["Diagnosis_clean"].value_counts())
    return df


def prepare_splits(df):
    le = LabelEncoder()
    y = le.fit_transform(df["Diagnosis_clean"].values)

    X_full = df[FULL_FEATURES].values
    X_cbc  = df[CBC_FEATURES].values

    # Stratified 80/20 split
    Xf_tr, Xf_te, y_tr, y_te = train_test_split(
        X_full, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    Xc_tr, Xc_te, _,    _   = train_test_split(
        X_cbc,  y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

    # SMOTE on training set only
    smote = SMOTE(random_state=RANDOM_SEED)
    Xf_tr_sm, y_tr_sm = smote.fit_resample(Xf_tr, y_tr)
    Xc_tr_sm, _       = smote.fit_resample(Xc_tr, y_tr)

    # Scaling for scale-sensitive models
    sc_full = StandardScaler()
    Xf_tr_sc = sc_full.fit_transform(Xf_tr_sm)
    Xf_te_sc = sc_full.transform(Xf_te)

    sc_cbc = StandardScaler()
    Xc_tr_sc = sc_cbc.fit_transform(Xc_tr_sm)
    Xc_te_sc = sc_cbc.transform(Xc_te)

    return {
        "le": le,
        "Xf_tr_sm": Xf_tr_sm, "Xf_te": Xf_te,
        "Xf_tr_sc": Xf_tr_sc, "Xf_te_sc": Xf_te_sc,
        "Xc_tr_sm": Xc_tr_sm, "Xc_te": Xc_te,
        "Xc_tr_sc": Xc_tr_sc, "Xc_te_sc": Xc_te_sc,
        "y_tr_sm": y_tr_sm, "y_te": y_te,
        "sc_full": sc_full, "sc_cbc": sc_cbc,
        "X_full": X_full, "y": y
    }


# ── Model definitions ─────────────────────────────────────────────────────────

MODEL_SPECS = [
    ("Logistic Regression",   LogisticRegression(max_iter=1000, random_state=RANDOM_SEED), True),
    ("K-Nearest Neighbors",   KNeighborsClassifier(n_neighbors=5),                          True),
    ("Support Vector Machine", SVC(probability=True, random_state=RANDOM_SEED),              True),
    ("Random Forest",          RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1), False),
    ("XGBoost",                XGBClassifier(n_estimators=200, random_state=RANDOM_SEED,
                                              eval_metric="mlogloss", verbosity=0),           False),
    ("LightGBM",               LGBMClassifier(n_estimators=200, random_state=RANDOM_SEED, verbose=-1), False),
]


def evaluate_model(model, Xtr, Xte, y_tr, y_te):
    import copy
    m = copy.deepcopy(model)
    m.fit(Xtr, y_tr)
    yp    = m.predict(Xte)
    yprob = m.predict_proba(Xte)
    return m, {
        "Accuracy (%)":           round(accuracy_score(y_te, yp) * 100, 2),
        "Precision-Macro (%)":    round(precision_score(y_te, yp, average="macro") * 100, 2),
        "Recall-Macro (%)":       round(recall_score(y_te, yp, average="macro") * 100, 2),
        "F1-Macro (%)":           round(f1_score(y_te, yp, average="macro") * 100, 2),
        "F1-Weighted (%)":        round(f1_score(y_te, yp, average="weighted") * 100, 2),
        "AUC-ROC (OvR Macro)":   round(roc_auc_score(y_te, yprob, multi_class="ovr", average="macro"), 4),
    }, yp, yprob


def run_all_models(splits, out_dir):
    results_full, results_cbc = {}, {}
    trained_models = {}

    print("\n=== Full Feature Set ===")
    for name, model, scaled in MODEL_SPECS:
        Xtr = splits["Xf_tr_sc"] if scaled else splits["Xf_tr_sm"]
        Xte = splits["Xf_te_sc"] if scaled else splits["Xf_te"]
        m, res, yp, yprob = evaluate_model(model, Xtr, Xte, splits["y_tr_sm"], splits["y_te"])
        results_full[name] = res
        trained_models[name] = (m, scaled)
        print(f"  {name:25s}: Acc={res['Accuracy (%)']:.2f}%  F1={res['F1-Macro (%)']:.2f}%  AUC={res['AUC-ROC (OvR Macro)']:.4f}")

    print("\n=== CBC-Only Feature Set ===")
    for name, model, scaled in MODEL_SPECS:
        Xtr = splits["Xc_tr_sc"] if scaled else splits["Xc_tr_sm"]
        Xte = splits["Xc_te_sc"] if scaled else splits["Xc_te"]
        _, res, _, _ = evaluate_model(model, Xtr, Xte, splits["y_tr_sm"], splits["y_te"])
        results_cbc[name] = res
        print(f"  {name:25s}: Acc={res['Accuracy (%)']:.2f}%  F1={res['F1-Macro (%)']:.2f}%  AUC={res['AUC-ROC (OvR Macro)']:.4f}")

    pd.DataFrame(results_full).T.to_csv(os.path.join(out_dir, "model_results_full.csv"))
    pd.DataFrame(results_cbc).T.to_csv(os.path.join(out_dir, "model_results_cbc.csv"))

    return results_full, results_cbc, trained_models


# ── Cross-validation ──────────────────────────────────────────────────────────

def run_cross_validation(splits):
    print("\n=== 10-Fold Cross-Validation (XGBoost, Full Features) ===")
    xgb_cv = XGBClassifier(n_estimators=200, random_state=RANDOM_SEED,
                            eval_metric="mlogloss", verbosity=0)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    acc = cross_val_score(xgb_cv, splits["X_full"], splits["y"], cv=cv, scoring="accuracy")
    f1  = cross_val_score(xgb_cv, splits["X_full"], splits["y"], cv=cv, scoring="f1_macro")
    print(f"  CV Accuracy : {acc.mean()*100:.2f}% ± {acc.std()*100:.2f}%")
    print(f"  CV F1-Macro : {f1.mean()*100:.2f}% ± {f1.std()*100:.2f}%")
    return acc, f1


# ── XGBoost detailed evaluation ───────────────────────────────────────────────

def detailed_xgb_evaluation(splits, out_dir):
    classes = list(splits["le"].classes_)
    print("\n=== XGBoost Detailed Evaluation (Full Features) ===")

    xgb = XGBClassifier(n_estimators=200, random_state=RANDOM_SEED,
                        eval_metric="mlogloss", verbosity=0)
    xgb.fit(splits["Xf_tr_sm"], splits["y_tr_sm"])
    yp = xgb.predict(splits["Xf_te"])
    print(classification_report(splits["y_te"], yp, target_names=classes))

    cm = confusion_matrix(splits["y_te"], yp)
    print("Confusion matrix (order:", classes, "):")
    print(cm)

    joblib.dump(xgb, os.path.join(out_dir, "..", "models", "xgboost_full_features.pkl"))
    return xgb, yp, cm, classes


# ── SHAP analysis ─────────────────────────────────────────────────────────────

def run_shap(xgb_model, splits, out_dir):
    print("\n=== SHAP Feature Importance (XGBoost, Full Features) ===")
    explainer = shap.TreeExplainer(xgb_model)
    sv = explainer.shap_values(splits["Xf_te"])  # (n_samples, n_features, n_classes)

    classes = list(splits["le"].classes_)
    fn_display = ["HbA0", "HbA2", "HbF", "RBC", "Hemoglobin", "MCV", "MCH",
                  "MCHC", "RDW-CV", "Age", "Sex"]

    shap_summary = {}
    for i, cls in enumerate(classes):
        mean_sv = np.abs(sv[:, :, i]).mean(axis=0)
        df_s = pd.DataFrame({"Feature": FULL_FEATURES, "Feature_Display": fn_display,
                              "Mean_SHAP": mean_sv.round(5)})
        df_s = df_s.sort_values("Mean_SHAP", ascending=False).reset_index(drop=True)
        shap_summary[cls] = df_s

    overall = np.abs(sv).mean(axis=2).mean(axis=0)
    df_overall = pd.DataFrame({"Feature": FULL_FEATURES, "Feature_Display": fn_display,
                                "Mean_SHAP_Overall": overall.round(5)})
    df_overall = df_overall.sort_values("Mean_SHAP_Overall", ascending=False).reset_index(drop=True)
    print(df_overall.to_string(index=False))

    return sv, shap_summary, df_overall, fn_display


# ── Figure generation ─────────────────────────────────────────────────────────

def generate_figures(df, splits, results_full, results_cbc, xgb_model, shap_vals,
                     fn_display, cm, classes, out_dir):
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    COLORS = {"Normal": "#2196F3", "Beta Thalassaemia carrier": "#E53935", "HbE carrier": "#43A047"}

    # Fig 1 — Class distribution
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cats = ["Normal", "BTT Carrier", "HbE Carrier"]
    cnts = [11777, 562, 546]
    pcts = [c / sum(cnts) * 100 for c in cnts]
    cols = ["#2196F3", "#E53935", "#43A047"]
    bars = ax.bar(cats, cnts, color=cols, width=0.5, edgecolor="white", linewidth=0.8)
    for bar, cnt, pct in zip(bars, cnts, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 120,
                f"{cnt:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of Records", fontsize=12)
    ax.set_xlabel("Diagnostic Class", fontsize=12)
    ax.set_title("Fig 1. Dataset Class Distribution (n = 12,885)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 13500)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "Fig1.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Fig 2 — Box plots
    order = ["Normal", "Beta Thalassaemia carrier", "HbE carrier"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, (col, ylabel) in zip(axes, [("MCV", "MCV (fL)"), ("MCH", "MCH (pg)"), ("HbA2", "HbA2 (%)")]):
        data_plot = [df[df["Diagnosis_clean"] == cls][col].dropna().values for cls in order]
        bp = ax.boxplot(data_plot, patch_artist=True, widths=0.5,
                        medianprops=dict(color="white", linewidth=2),
                        flierprops=dict(marker="o", markersize=2, alpha=0.3))
        for patch, fc in zip(bp["boxes"], ["#2196F3", "#E53935", "#43A047"]):
            patch.set_facecolor(fc); patch.set_alpha(0.75)
        ax.set_xticklabels(["Normal", "BTT\nCarrier", "HbE\nCarrier"], fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11); ax.set_title(ylabel, fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.suptitle("Fig 2. Distribution of Key Discriminating Parameters by Diagnostic Class",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "Fig2.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Fig 3 — Model comparison
    model_names = ["LR", "KNN", "SVM", "RF", "XGBoost", "LightGBM"]
    acc_f = [results_full[k]["Accuracy (%)"] for k in results_full]
    f1_f  = [results_full[k]["F1-Macro (%)"] for k in results_full]
    acc_c = [results_cbc[k]["Accuracy (%)"] for k in results_cbc]
    f1_c  = [results_cbc[k]["F1-Macro (%)"] for k in results_cbc]
    x = np.arange(len(model_names)); w = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - 1.5*w, acc_f, w, label="Accuracy - Full", color="#1565C0", alpha=0.85)
    b2 = ax.bar(x - 0.5*w, f1_f,  w, label="F1-Macro - Full", color="#42A5F5", alpha=0.85)
    b3 = ax.bar(x + 0.5*w, acc_c, w, label="Accuracy - CBC Only", color="#B71C1C", alpha=0.85)
    b4 = ax.bar(x + 1.5*w, f1_c,  w, label="F1-Macro - CBC Only", color="#EF9A9A", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylabel("Score (%)", fontsize=12); ax.set_ylim(40, 108)
    ax.set_title("Fig 3. Model Performance Comparison: Full Feature Set vs CBC-Only",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.axhline(y=90, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "Fig3.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Fig 4 — Confusion matrix
    labels = ["BTT\nCarrier", "HbE\nCarrier", "Normal"]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, linecolor="white", annot_kws={"size": 14, "weight": "bold"})
    ax.set_xlabel("Predicted Class", fontsize=12); ax.set_ylabel("Actual Class", fontsize=12)
    ax.set_title("Fig 4. XGBoost Confusion Matrix\n(Full Feature Set, Test Set n = 2,577)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "Fig4.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Fig 5 — ROC curves
    from sklearn.preprocessing import label_binarize
    yprob = xgb_model.predict_proba(splits["Xf_te"])
    y_bin = label_binarize(splits["y_te"], classes=[0, 1, 2])
    from sklearn.metrics import roc_curve, auc as sk_auc
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    cls_names_short = ["BTT Carrier", "HbE Carrier", "Normal"]
    roc_cols = ["#E53935", "#43A047", "#2196F3"]
    for i, (cls_name, col) in enumerate(zip(cls_names_short, roc_cols)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], yprob[:, i])
        ax.plot(fpr, tpr, lw=2, color=col, label=f"{cls_name} (AUC = {sk_auc(fpr, tpr):.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random Classifier")
    ax.set_xlim([-0.01, 1.0]); ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12); ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Fig 5. ROC Curves by Class\n(XGBoost, Full Feature Set, One-vs-Rest)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "Fig5.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Fig 6 — SHAP per-class
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    cls_short = ["BTT Carrier", "HbE Carrier", "Normal"]
    cls_cols  = ["#E53935", "#43A047", "#2196F3"]
    for ax, i, cls_name, col in zip(axes, range(3), cls_short, cls_cols):
        sv = np.abs(shap_vals[:, :, i]).mean(axis=0)
        order_idx = np.argsort(sv)
        ax.barh([fn_display[j] for j in order_idx], sv[order_idx], color=col, alpha=0.82)
        ax.set_title(cls_name, fontsize=12, fontweight="bold", color=col)
        ax.set_xlabel("Mean |SHAP Value|", fontsize=10)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.suptitle("Fig 6. SHAP Feature Importance by Diagnostic Class (XGBoost)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "Fig6.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Fig 7 — SHAP overall
    overall = np.abs(shap_vals).mean(axis=2).mean(axis=0)
    order_idx = np.argsort(overall)[::-1]
    bar_cols = ["#1a5276" if i < 2 else "#5dade2" if i < 5 else "#aed6f1"
                for i in range(len(order_idx))]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([fn_display[j] for j in order_idx], overall[order_idx],
           color=bar_cols, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Mean |SHAP Value| (Overall)", fontsize=12)
    ax.set_xlabel("Feature", fontsize=12)
    ax.set_title("Fig 7. Overall SHAP Feature Importance (XGBoost, Full Feature Set)",
                 fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "Fig7.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nAll figures saved to {fig_dir}/")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Thalassemia ML Classification Pipeline")
    parser.add_argument("--data", required=True, help="Path to 'HPLC data.csv'")
    parser.add_argument("--out", default="results", help="Output directory (default: results)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "..", "models"), exist_ok=True)

    print("Loading and preprocessing data...")
    df = load_and_preprocess(args.data)

    print("\nPreparing train/test splits with SMOTE...")
    splits = prepare_splits(df)

    print("\nTraining and evaluating all models...")
    results_full, results_cbc, trained_models = run_all_models(splits, args.out)

    print("\nRunning 10-fold cross-validation...")
    cv_acc, cv_f1 = run_cross_validation(splits)

    print("\nRunning detailed XGBoost evaluation...")
    xgb_model, yp_xgb, cm, classes = detailed_xgb_evaluation(splits, args.out)

    print("\nRunning SHAP analysis...")
    shap_vals, shap_summary, df_overall, fn_display = run_shap(xgb_model, splits, args.out)

    print("\nGenerating manuscript figures...")
    generate_figures(df, splits, results_full, results_cbc, xgb_model,
                     shap_vals, fn_display, cm, classes, args.out)

    print("\nPipeline complete. All outputs saved to:", args.out)
    print("Trained XGBoost model saved to: models/xgboost_full_features.pkl")


if __name__ == "__main__":
    main()
