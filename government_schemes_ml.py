# ============================================================
#  Machine Learning on Government Schemes Dataset
#  Models:
#    1. Random Forest Regressor  → Predict Success_Rate_%
#    2. Random Forest Classifier → Predict PM_Flagship (Yes/No)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error, r2_score, mean_squared_error,
    accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\pythonProject\government_schemes.csv")   # ← update path if needed
print(f"Dataset shape: {df.shape}")
print(df.head(3))

# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────

# Drop non-informative / ID columns
drop_cols = ["Scheme_ID", "Scheme_Name", "Nodal_Agency", "Remarks",
             "Target_Beneficiaries", "States_Covered", "Scheme_Duration_Years"]
df.drop(columns=drop_cols, inplace=True)

# Fill numeric nulls with median
num_cols = df.select_dtypes(include=["float64", "int64"]).columns
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Fill categorical nulls with mode
cat_cols = df.select_dtypes(include=["object"]).columns
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Label encode all categorical columns
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

print("\nCleaned dataset shape:", df.shape)
print(df.dtypes)

# ─────────────────────────────────────────────
# 3. FEATURE SELECTION
# ─────────────────────────────────────────────
# Features shared by both models (exclude the two targets)
FEATURES = [c for c in df.columns if c not in ["Success_Rate_%", "PM_Flagship"]]

# ═══════════════════════════════════════════════════════
# MODEL 1 — REGRESSION: Predict Success_Rate_%
# ═══════════════════════════════════════════════════════
print("\n" + "═"*55)
print("  MODEL 1 — Random Forest Regressor (Success_Rate_%)  ")
print("═"*55)

reg_df = df.dropna(subset=["Success_Rate_%"])
X_reg = reg_df[FEATURES]
y_reg = reg_df["Success_Rate_%"]

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

rf_reg = RandomForestRegressor(n_estimators=200, max_depth=10,
                                random_state=42, n_jobs=-1)
rf_reg.fit(X_train_r, y_train_r)
y_pred_r = rf_reg.predict(X_test_r)

mae  = mean_absolute_error(y_test_r, y_pred_r)
rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
r2   = r2_score(y_test_r, y_pred_r)
cv   = cross_val_score(RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
                       X_reg, y_reg, cv=5, scoring="r2").mean()

print(f"  MAE  : {mae:.2f}")
print(f"  RMSE : {rmse:.2f}")
print(f"  R²   : {r2:.4f}")
print(f"  5-Fold CV R²: {cv:.4f}")

# Plot: Actual vs Predicted
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(y_test_r, y_pred_r, alpha=0.5, color="steelblue", edgecolors="k", lw=0.3)
axes[0].plot([y_test_r.min(), y_test_r.max()],
             [y_test_r.min(), y_test_r.max()], "r--", lw=2)
axes[0].set_xlabel("Actual Success Rate (%)")
axes[0].set_ylabel("Predicted Success Rate (%)")
axes[0].set_title(f"Regressor: Actual vs Predicted\nR² = {r2:.4f}  |  MAE = {mae:.2f}")
axes[0].grid(alpha=0.3)

# Feature importance (Regressor)
imp_r = pd.Series(rf_reg.feature_importances_, index=FEATURES).sort_values(ascending=True).tail(10)
imp_r.plot(kind="barh", ax=axes[1], color="steelblue")
axes[1].set_title("Top 10 Feature Importances (Regressor)")
axes[1].set_xlabel("Importance")
axes[1].grid(alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig("regression_results.png", dpi=150)
plt.show()
print("  → Saved: regression_results.png")

# ═══════════════════════════════════════════════════════
# MODEL 2 — CLASSIFICATION: Predict PM_Flagship (Yes/No)
# ═══════════════════════════════════════════════════════
print("\n" + "═"*55)
print("  MODEL 2 — Random Forest Classifier (PM_Flagship)   ")
print("═"*55)

X_clf = df[FEATURES]
y_clf = df["PM_Flagship"]          # already label-encoded (0 = No, 1 = Yes)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

rf_clf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                 class_weight="balanced",
                                 random_state=42, n_jobs=-1)
rf_clf.fit(X_train_c, y_train_c)
y_pred_c = rf_clf.predict(X_test_c)

acc = accuracy_score(y_test_c, y_pred_c)
cv_acc = cross_val_score(rf_clf, X_clf, y_clf, cv=5, scoring="accuracy").mean()

print(f"  Accuracy      : {acc:.4f}")
print(f"  5-Fold CV Acc : {cv_acc:.4f}")
print("\n  Classification Report:")
print(classification_report(y_test_c, y_pred_c, target_names=["Not Flagship", "PM Flagship"]))

# Plot: Confusion Matrix + Feature Importance
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

cm = confusion_matrix(y_test_c, y_pred_c)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=["Not Flagship", "PM Flagship"])
disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title(f"Confusion Matrix\nAccuracy = {acc:.4f}")

imp_c = pd.Series(rf_clf.feature_importances_, index=FEATURES).sort_values(ascending=True).tail(10)
imp_c.plot(kind="barh", ax=axes[1], color="darkorange")
axes[1].set_title("Top 10 Feature Importances (Classifier)")
axes[1].set_xlabel("Importance")
axes[1].grid(alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig("classification_results.png", dpi=150)
plt.show()
print("  → Saved: classification_results.png")

# ─────────────────────────────────────────────
# 4. SUMMARY
# ─────────────────────────────────────────────
print("\n" + "─"*55)
print("  SUMMARY")
print("─"*55)
print(f"  Regressor  → R²: {r2:.4f}  |  MAE: {mae:.2f}  |  RMSE: {rmse:.2f}")
print(f"  Classifier → Accuracy: {acc:.4f}  |  CV Acc: {cv_acc:.4f}")
print("─"*55)
