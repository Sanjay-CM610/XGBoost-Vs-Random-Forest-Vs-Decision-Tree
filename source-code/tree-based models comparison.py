# %%
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# %%
# ====================================================
# Tunable Parameters
# ====================================================
save_path = 'D:/Projects/study-main/tree_based_models_comparison/' 
save_figure = True
display_figure = False
no_of_folds = 10
sizes_to_test = [0.2, 0.8]

# INTEGRATED PLOTTING SUITE
sns.set_theme(style="whitegrid")
model_names = ["Decision Tree", "Random Forest", "XGBoost"]
plot_size = max(sizes_to_test)

# ====================================================
# Global Structures for Harvesting Pipeline Results
# ====================================================
harvested_cancer_data = {}
harvested_housing_data = {}

# %%
# ====================================================
# Core Cross-Validation Pipeline
# ====================================================

def run_advanced_cv_pipeline(X, y, feature_names, train_size, num_folds, dataset_name):
    """
    Executes a Stratified Cross-Validation loop for a specific training data size,
    collecting aggregated diagnostic metrics.
    """
    if dataset_name not in ["cancer", "housing"]:
        raise ValueError("dataset_name must be either 'cancer' or 'housing' ")

    # Temporary stratification labels used solely to preserve class proportions across CV folds.
    if dataset_name == "housing":
        stratification_median = np.median(y)
        stratify_y = (y > stratification_median).astype(int)
    else:
        stratify_y = y


    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    models_config = {
        "Decision Tree": lambda: DecisionTreeClassifier(random_state=42, max_depth=None),

        "Random Forest": lambda: RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1),

        "XGBoost": lambda: XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, eval_metric="logloss", random_state=42, n_jobs=1),
    }


    # Dictionary to aggregate metrics across all folds
    results = {
    name: {
        "roc_auc_list": [],
        "train_time_list": [],
        "pred_time_list": [],
        "total_tn": 0,
        "total_fp": 0,
        "total_fn": 0,
        "total_tp": 0,
        "feature_importances_list": [],
        "train_sample_sizes": [],
        "test_sample_sizes": [],
        }
    for name in models_config
    }


    # ====================================================
    # Main Cross-Validation Loop
    # ====================================================

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, stratify_y), 1):
        X_train_full, X_test = X[train_idx], X[test_idx]
        y_train_full, y_test = y[train_idx], y[test_idx]

        for name in models_config:
            results[name]["test_sample_sizes"].append(len(X_test))

        # Dataset-specific target handling
        if dataset_name == "housing":
            # Calculate the median using ONLY the current training fold
            median_val = np.median(y_train_full)

            y_train_full = (y_train_full > median_val).astype(int)
            y_test = (y_test > median_val).astype(int)

        elif dataset_name == "cancer":
            # Target was previously transformed to:
            # 0 = benign
            # 1 = malignant
            pass


        # ====================================================
        # Select the requested percentage of the ORIGINAL dataset
        # while keeping the CV test fold completely untouched.
        # ====================================================

        outer_train_fraction = (num_folds - 1) / num_folds

        # The CV test fold occupies 10% with 10-fold CV,
        # so a maximum of 90% of the original dataset can be used for training.
        if train_size > outer_train_fraction:
            raise ValueError(
                f"train_size={train_size:.0%} is impossible with "
                f"{num_folds}-fold CV. The maximum training size is "
                f"{outer_train_fraction:.0%} because "
                f"{1 - outer_train_fraction:.0%} is reserved for the CV test fold."
            )

        if train_size < outer_train_fraction:

            # Convert desired percentage of ORIGINAL dataset
            # into percentage of the available CV training fold.
            desired_ratio = train_size / outer_train_fraction

            X_train, _, y_train, _ = train_test_split(
                X_train_full,
                y_train_full,
                train_size=desired_ratio,
                stratify=y_train_full,
                random_state=42 + fold
            )

        else:
            # train_size == 90% with 10-fold CV: use the entire 90% CV training fold.
            X_train, y_train = X_train_full, y_train_full


        # ====================================================
        # Train and Evaluate Each Model
        # ====================================================

        for name, model_fn in models_config.items():
            results[name]["train_sample_sizes"].append(len(X_train))

            model = model_fn()


            # 1. Isolate Training Time
            start_train = time.perf_counter()
            model.fit(X_train, y_train)
            train_time_ms = (time.perf_counter() - start_train) * 1000

            # 2. Isolate Prediction Time
            start_pred = time.perf_counter()

            preds = model.predict(X_test)
            pred_time_ms = (time.perf_counter() - start_pred) * 1000
            probs = model.predict_proba(X_test)[:, 1]


            # Confusion Matrix
            tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

            # Store metrics for later calculation
            results[name]["roc_auc_list"].append(roc_auc_score(y_test, probs))
            results[name]["train_time_list"].append(train_time_ms)
            results[name]["pred_time_list"].append(pred_time_ms)

            # Sum errors continuously across all folds
            results[name]["total_tn"] += tn
            results[name]["total_fp"] += fp
            results[name]["total_fn"] += fn
            results[name]["total_tp"] += tp

            # Store feature importances from every fold
            results[name]["feature_importances_list"].append(model.feature_importances_)


    # ====================================================
    # Calculate Final Statistical Metrics
    # ====================================================

    final_output = {}

    for name, metrics in results.items():
        importance_matrix = np.array(metrics["feature_importances_list"])

        mean_importances = np.mean(importance_matrix, axis=0)

        feature_imp_dict = dict(zip(feature_names, mean_importances))

        sorted_imp = sorted(feature_imp_dict.items(), key=lambda x: x[1], reverse=True)


        tn = metrics["total_tn"]
        fp = metrics["total_fp"]
        fn = metrics["total_fn"]
        tp = metrics["total_tp"]

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        final_output[name] = {
            "avg_train_time_ms": np.mean(metrics["train_time_list"]),
            "std_train_time_ms": np.std(metrics["train_time_list"]),
            "avg_pred_time_ms": np.mean(metrics["pred_time_list"]),
            "std_pred_time_ms": np.std(metrics["pred_time_list"]),
            "avg_pred_time_per_sample_ms": np.mean(np.array(metrics["pred_time_list"]) / np.array(metrics["test_sample_sizes"])),
            "mean_roc_auc": np.mean(metrics["roc_auc_list"]),
            "std_roc_auc": np.std(metrics["roc_auc_list"]),
            "roc_auc_raw_list": metrics["roc_auc_list"],

            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,

            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "accuracy": accuracy,

            "top_features": sorted_imp[:3],
            "actual_samples": int(np.mean(metrics["train_sample_sizes"])),

        }

    return final_output


# %%
# ====================================================
# 1. RUN AND DISPLAY FOR BREAST CANCER
# ====================================================

print("\n=============================================")
print(f"CLINICAL DATA SCALING ANALYSIS ({no_of_folds}-FOLD CV)")
print("=============================================")

cancer_data = load_breast_cancer()

# 0 = benign
# 1 = malignant
y_cancer = (cancer_data.target == 0).astype(int)

for size in sizes_to_test:
    print(f"\n[TRAINING WITH {size * 100:.0f}% OF TOTAL DATASET]")

    cancer_res = run_advanced_cv_pipeline(
    cancer_data.data,
    y_cancer,
    cancer_data.feature_names,
    train_size=size,
    num_folds=no_of_folds,
    dataset_name="cancer"
    )
    harvested_cancer_data[size] = cancer_res

    for name, metrics in cancer_res.items():
        print(f"\n[{name}] (Avg samples per fold: {metrics['actual_samples']})")
        print(f"   Avg Training Time   : {metrics['avg_train_time_ms']:.2f} ms")
        print(f"   Avg Prediction Time       : {metrics['avg_pred_time_ms']:.2f} ms " f"(±{metrics['std_pred_time_ms']:.2f} ms)")
        print(f"   Prediction Time per Sample: " f"{metrics['avg_pred_time_per_sample_ms']:.6f} ms/sample")
        print(f"   Mean ROC-AUC Score    : {metrics['mean_roc_auc']:.4f} (±{metrics['std_roc_auc']:.4f})")

        print("   Aggregated Confusion Matrix Breakdown:")
        print(f"       • True Negatives  (Correct Benign)       : {metrics['tn']}")
        print(f"       • False Positives (False Cancer Alarm)   : {metrics['fp']}")
        print(f"       • False Negatives (Missed Cancer)        : {metrics['fn']}")
        print(f"       • True Positives  (Cancer Detected)      : {metrics['tp']}")

        print(f"   Sensitivity (Recall) : {metrics['sensitivity']:.4f}")
        print(f"   Specificity          : {metrics['specificity']:.4f}")
        print(f"   Precision            : {metrics['precision']:.4f}")
        print(f"   Accuracy             : {metrics['accuracy']:.4f}")


# %%
# ====================================================
# 2. RUN AND DISPLAY FOR CALIFORNIA HOUSING
# ====================================================

housing_data = fetch_california_housing()

X_housing = housing_data.data
y_housing = housing_data.target


print("\n=============================================")
print(f"CALIFORNIA HOUSING VALUE CLASSIFICATION ({no_of_folds}-FOLD CV)")
print("=============================================")


for size in sizes_to_test:
    print(f"\n[TRAINING WITH {size * 100:.0f}% OF TOTAL DATASET]")

    housing_res = run_advanced_cv_pipeline(
    X_housing,
    y_housing,
    housing_data.feature_names,
    train_size=size,
    num_folds=no_of_folds,
    dataset_name="housing"
    )
    harvested_housing_data[size] = housing_res

    for name, metrics in housing_res.items():
        print(f"\n[{name}] (Avg samples per fold: {metrics['actual_samples']})")
        print(f"   Avg Training Time   : {metrics['avg_train_time_ms']:.2f} ms")
        print(f"   Avg Prediction Time       : {metrics['avg_pred_time_ms']:.2f} ms " f"(±{metrics['std_pred_time_ms']:.2f} ms)")
        print(f"   Prediction Time per Sample: " f"{metrics['avg_pred_time_per_sample_ms']:.6f} ms/sample")
        print(f"   Mean ROC-AUC Score    : {metrics['mean_roc_auc']:.4f} (±{metrics['std_roc_auc']:.4f})")

        print("   Aggregated Confusion Matrix Breakdown:")
        print(f"       • True Negatives  (Correct Below Median): {metrics['tn']}")
        print(f"       • False Positives (False Above Median)  : {metrics['fp']}")
        print(f"       • False Negatives (Missed Above Median)  : {metrics['fn']}")
        print(f"       • True Positives  (Correct Above Median) : {metrics['tp']}")


        print("   Top 3 Market Drivers:")

        for feat, score in metrics["top_features"]:
            print(f"       • {feat:<15}: {score * 100:.2f}% impact")

# %% [markdown]
# ====================================================
#           INTEGRATED PLOTTING SUITE
# ====================================================

# %%
# ====================================================
# PLOT A: ROC-AUC VS TRAINING DATASET SIZE
# ====================================================

fig_a, axes_a = plt.subplots(1, 2, figsize=(14, 5))

title_text_a = "ROC-AUC vs Training Dataset Size"

fig_a.suptitle(title_text_a, fontsize=14, fontweight="bold")

for idx, (data_harvest, title) in enumerate([
    (harvested_cancer_data, "Breast Cancer Detection"),
    (harvested_housing_data, "California Housing Classification"),
]):

    ax = axes_a[idx]

    x_indexes = np.arange(len(model_names))
    bar_width = 0.8 / len(sizes_to_test)

    for size_idx, size in enumerate(sizes_to_test):
        auc_values = [
            data_harvest[size][model]["mean_roc_auc"]
            for model in model_names
        ]

        offset = (size_idx - (len(sizes_to_test) - 1) / 2) * bar_width

        ax.bar(
            x_indexes + offset,
            auc_values,
            bar_width,
            label=f"{size * 100:.0f}% Train Size"
        )

    ax.set_xticks(x_indexes)
    ax.set_xticklabels(model_names)
    ax.set_xlabel("Model")
    ax.set_ylabel("Mean ROC-AUC Score")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)


plt.tight_layout()
if save_figure: plt.savefig(save_path + title_text_a + ".png")
if display_figure: plt.show()


# %%
# ====================================================
# PLOT B: CROSS-VALIDATION STABILITY BOXPLOT
# ====================================================

fig_b, axes_b = plt.subplots(1, 2, figsize=(14, 5))
title_text_b = (f"Cross-Validation Score Variance Profiles " f"(At {plot_size * 100:.0f}% Data Size)")


fig_b.suptitle(title_text_b, fontsize=14, fontweight="bold")

for idx, (data_harvest, title) in enumerate([
    (harvested_cancer_data, "Breast Cancer Variance"),
    (harvested_housing_data, "California Housing Variance"),
    ]):
    ax = axes_b[idx]

    plot_data = [data_harvest[plot_size][m]["roc_auc_raw_list"] for m in model_names]


    # Boxplot showing median and interquartile range
    sns.boxplot(data=plot_data, ax=ax, width=0.5, color="#AFC6E9", showmeans=True,
        meanprops={
            "marker": "*",
            "markerfacecolor": "red",
            "markeredgecolor": "black",
            "markersize": 6
        }
    )

    # Individual fold observations
    sns.stripplot(data=plot_data, ax=ax, color="#333333", size=6, jitter=0.08, alpha=0.8)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names)
    ax.set_ylabel(f"ROC-AUC Score Across {no_of_folds} CV Folds")
    ax.set_ylim(0.75, 1.01)

    # Mean marker explanation
    ax.plot(0.04, 0.02, marker="*", markersize=5, markerfacecolor="red", markeredgecolor="black", transform=ax.transAxes, linestyle="None")
    ax.text(0.04, 0.02, "= Mean", transform=ax.transAxes, fontsize=9, color="red", va="center")

plt.tight_layout()
if save_figure: plt.savefig(save_path + title_text_b + ".png")
if display_figure: plt.show()


# %%
# ====================================================
# PLOT C: CROSS-MODEL FEATURE IMPORTANCE COMPARISON
# ====================================================

fig_c, axes_c = plt.subplots(1, 3, figsize=(16, 5))
title_text_c = (f"Comparative Top 3 Feature Importances " f"(California Housing at {plot_size * 100:.0f}% Size)")

fig_c.suptitle(title_text_c, fontsize=14, fontweight="bold")

for idx, name in enumerate(model_names):
    ax = axes_c[idx]
    top_feats = harvested_housing_data[plot_size][name]["top_features"]

    features = [item[0] for item in top_feats]
    weights = [item[1] * 100 for item in top_feats]

    sns.barplot(x=weights, y=features, ax=ax, palette="viridis", hue=features, legend=False)

    ax.set_title(f"{name} Drivers", fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean Feature Importance (%)")
    ax.set_xlim(0, 50)

plt.tight_layout()
if save_figure: plt.savefig(save_path + title_text_c + ".png")
if display_figure: plt.show()


# %%
# ====================================================
# PLOT D: AGGREGATED CONFUSION MATRIX QUADRANT HEATMAPS
# ====================================================

fig_d, axes_d = plt.subplots(1, 2, figsize=(14, 5.5))
title_text_d = (f"Aggregated Diagnostic Error Footprints " f"(XGBoost at {plot_size * 100:.0f}% Data Size)")

fig_d.suptitle(title_text_d, fontsize=14, fontweight="bold")


# ====================================================
# Breast Cancer Matrix Setup
# ====================================================

m_cn = harvested_cancer_data[plot_size]["XGBoost"]

matrix_cn = np.array([[m_cn["tn"], m_cn["fp"]], [m_cn["fn"], m_cn["tp"]]])

labels_cn = [
    f"True Benign\n{m_cn['tn']}",
    f"False Cancer Alarm\n{m_cn['fp']}",
    f"Missed Cancer\n{m_cn['fn']}",
    f"True Cancer Detected\n{m_cn['tp']}",
]

labels_cn = np.asarray(labels_cn).reshape(2, 2)

sns.heatmap(matrix_cn, annot=labels_cn, fmt="", cmap="Reds", cbar=False, ax=axes_d[0], annot_kws={"size": 11, "weight": "bold"})

axes_d[0].set_title("Breast Cancer XGBoost Matrix", fontsize=12, fontweight="bold")
axes_d[0].set_xticklabels(["Predicted Benign", "Predicted Malignant"])
axes_d[0].set_yticklabels(["Actual Benign", "Actual Malignant"], rotation=0)


# ====================================================
# California Housing Matrix Setup
# ====================================================

m_ho = harvested_housing_data[plot_size]["XGBoost"]

matrix_ho = np.array([[m_ho["tn"], m_ho["fp"]], [m_ho["fn"], m_ho["tp"]]])

labels_ho = [
    f"True Below Median\n{m_ho['tn']}",
    f"False Above Median\n{m_ho['fp']}",
    f"Missed Above Median\n{m_ho['fn']}",
    f"True Above Median\n{m_ho['tp']}",
]

labels_ho = np.asarray(labels_ho).reshape(2, 2)

sns.heatmap(matrix_ho, annot=labels_ho, fmt="", cmap="Blues", cbar=False, ax=axes_d[1], annot_kws={"size": 11, "weight": "bold"})

axes_d[1].set_title("California Housing XGBoost Matrix", fontsize=12, fontweight="bold")
axes_d[1].set_xticklabels(["Predicted Below Median", "Predicted Above Median"])
axes_d[1].set_yticklabels(["Actual Below Median", "Actual Above Median"], rotation=0)


plt.tight_layout()
if save_figure: plt.savefig(save_path + title_text_d + ".png")
if display_figure: plt.show()
