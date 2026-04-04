"""
fraud_pipeline.py
=================
IS 455 — Machine Learning in Python
CRISP-DM Fraud Detection Pipeline

Connects to a Supabase (PostgreSQL) database using credentials stored in a
.env file, engineers features from the shop database, and runs the full
MLPipeline class through all CRISP-DM phases.

Usage:
    python fraud_pipeline.py

Requirements (.env file in same directory):
    SUPABASE_HOST=db.<your-project-ref>.supabase.co
    SUPABASE_PORT=5432
    SUPABASE_DB=postgres
    SUPABASE_USER=postgres
    SUPABASE_PASS=<your-database-password>
"""

# =============================================================================
# IMPORTS
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import os
import sqlite3
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import psycopg2
import joblib
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt

from dotenv import load_dotenv

from sklearn.model_selection import (
    train_test_split, cross_val_score, RandomizedSearchCV, StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, f1_score, ConfusionMatrixDisplay,
)
from sklearn.feature_selection import SelectKBest, f_classif


# =============================================================================
# PART 1 — DATA LOADING HELPERS
# =============================================================================

def load_credentials() -> Dict[str, str]:
    """
    Load Supabase connection credentials from a .env file.
    Raises EnvironmentError if required variables are missing.
    """
    load_dotenv()

    host = os.getenv("SUPABASE_HOST")
    port = os.getenv("SUPABASE_PORT", "5432")
    db   = os.getenv("SUPABASE_DB",   "postgres")
    user = os.getenv("SUPABASE_USER", "postgres")
    pwd  = os.getenv("SUPABASE_PASS")

    if not host or not pwd:
        raise EnvironmentError(
            "Missing credentials. Make sure .env exists and contains "
            "SUPABASE_HOST and SUPABASE_PASS."
        )

    print(f"Connecting to: {host}:{port}/{db}")
    return {"host": host, "port": port, "dbname": db, "user": user, "password": pwd}


def load_from_supabase(query: str) -> pd.DataFrame:
    """
    Connect to Supabase (PostgreSQL) and return a query result as a DataFrame.
    Credentials are read automatically from the .env file.
    """
    creds = load_credentials()
    conn = psycopg2.connect(**creds, sslmode="require")
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"Loaded {len(df):,} rows x {df.shape[1]} cols from Supabase")
    return df


def load_from_sqlite(db_path: str, query: str) -> pd.DataFrame:
    """Load a SQL query result from a local SQLite database into a DataFrame."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"Loaded {len(df):,} rows x {df.shape[1]} cols from '{db_path}'")
    return df


# =============================================================================
# PART 2 — MLPipeline CLASS
# =============================================================================

class MLPipeline:
    """
    End-to-end CRISP-DM machine learning pipeline for classification tasks.
    Works with any pandas DataFrame.

    Parameters
    ----------
    df            : pd.DataFrame  — raw input data
    target        : str           — name of the target / label column
    models        : list[str]     — models to train; choose from:
                                     "lr"  Logistic Regression
                                     "rf"  Random Forest        (default)
                                     "gb"  Gradient Boosting    (default)
                                     "dt"  Decision Tree
                                     "ada" AdaBoost
                                     "knn" K-Nearest Neighbors
                                     "svm" Support Vector Machine
                                     "all" trains every model above
    tune          : bool          — run RandomizedSearchCV on the best model
    output_path   : str           — file path to save the serialized model
    drop_cols     : list[str]     — columns to drop before training (IDs, etc.)
    cat_strategy  : str           — "onehot" or "ordinal" encoding
    scale         : bool          — apply StandardScaler to numeric features
    test_size     : float         — fraction held out for test set
    random_state  : int           — seed for reproducibility
    n_features    : int | None    — SelectKBest top-k features (None = use all)
    cv_folds      : int           — number of cross-validation folds
    verbose       : bool          — print progress messages
    """

    MODEL_REGISTRY: Dict[str, Any] = {
        "lr":  LogisticRegression(max_iter=1000, random_state=42),
        "rf":  RandomForestClassifier(n_estimators=200, random_state=42),
        "gb":  GradientBoostingClassifier(n_estimators=200, random_state=42),
        "dt":  DecisionTreeClassifier(random_state=42),
        "ada": AdaBoostClassifier(n_estimators=100, random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "svm": SVC(probability=True, random_state=42),
    }

    TUNING_PARAMS: Dict[str, Dict] = {
        "rf": {
            "classifier__n_estimators":      [100, 200, 300],
            "classifier__max_depth":         [None, 5, 10, 20],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__max_features":      ["sqrt", "log2"],
        },
        "gb": {
            "classifier__n_estimators":  [100, 200, 300],
            "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "classifier__max_depth":     [3, 5, 7],
            "classifier__subsample":     [0.7, 0.8, 1.0],
        },
        "lr": {
            "classifier__C":       [0.001, 0.01, 0.1, 1, 10, 100],
            "classifier__penalty": ["l2"],
            "classifier__solver":  ["lbfgs", "saga"],
        },
        "dt": {
            "classifier__max_depth":         [None, 5, 10, 20],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__criterion":         ["gini", "entropy"],
        },
        "knn": {
            "classifier__n_neighbors": [3, 5, 7, 11, 15],
            "classifier__weights":     ["uniform", "distance"],
        },
    }

    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        models: Optional[List[str]] = None,
        tune: bool = False,
        output_path: str = "trained_model.joblib",
        drop_cols: Optional[List[str]] = None,
        cat_strategy: str = "onehot",
        scale: bool = True,
        test_size: float = 0.2,
        random_state: int = 42,
        n_features: Optional[int] = None,
        cv_folds: int = 5,
        verbose: bool = True,
    ):
        self.df           = df.copy()
        self.target       = target
        self.models       = models if models is not None else ["rf", "gb"]
        if "all" in self.models:
            self.models = list(self.MODEL_REGISTRY.keys())
        self.tune         = tune
        self.output_path  = output_path
        self.drop_cols    = drop_cols or []
        self.cat_strategy = cat_strategy
        self.scale        = scale
        self.test_size    = test_size
        self.random_state = random_state
        self.n_features   = n_features
        self.cv_folds     = cv_folds
        self.verbose      = verbose

        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.results: Dict[str, Dict] = {}
        self.best_model_key: Optional[str] = None
        self.final_pipeline = None
        self._numeric_cols: List[str] = []
        self._categorical_cols: List[str] = []

    # ── Phase 1: Data Understanding ───────────────────────────────────
    def data_understanding(self) -> None:
        self._log("\n" + "=" * 60)
        self._log("PHASE 1 — DATA UNDERSTANDING")
        self._log("=" * 60)
        df = self.df
        self._log(f"\nShape         : {df.shape[0]:,} rows x {df.shape[1]} columns")
        self._log(f"Target column : '{self.target}'")
        self._log("\n--- Column dtypes ---")
        self._log(df.dtypes.to_string())
        self._log("\n--- Missing values ---")
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        self._log(
            "  None ✓" if missing.empty else
            pd.concat([missing.rename("count"),
                       (missing / len(df) * 100).round(2).rename("pct%")], axis=1).to_string()
        )
        self._log("\n--- Target class distribution ---")
        counts = df[self.target].value_counts()
        pcts   = df[self.target].value_counts(normalize=True).mul(100).round(2)
        self._log(pd.concat([counts.rename("count"), pcts.rename("pct%")], axis=1).to_string())
        self._log("\n--- Numeric summary ---")
        self._log(df.describe().T.to_string())

    # ── Phase 2: Data Preparation ─────────────────────────────────────
    def data_preparation(self) -> None:
        self._log("\n" + "=" * 60)
        self._log("PHASE 2 — DATA PREPARATION")
        self._log("=" * 60)
        df = self.df.copy()
        cols_to_drop = [c for c in self.drop_cols if c in df.columns]
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)
            self._log(f"Dropped : {cols_to_drop}")
        if self.target not in df.columns:
            raise ValueError(f"Target column '{self.target}' not found.")
        X = df.drop(columns=[self.target])
        y = df[self.target]
        if y.dtype == object or str(y.dtype) == "category":
            le = LabelEncoder()
            y  = pd.Series(le.fit_transform(y), name=self.target)
            self._log(f"Target encoded: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        numeric_cols     = X.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        self._log(f"Numeric features ({len(numeric_cols)}):     "
                  f"{numeric_cols[:8]}{'...' if len(numeric_cols) > 8 else ''}")
        self._log(f"Categorical features ({len(categorical_cols)}): "
                  f"{categorical_cols[:8]}{'...' if len(categorical_cols) > 8 else ''}")
        num_t = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler() if self.scale else "passthrough"),
        ])
        cat_enc = (
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            if self.cat_strategy == "ordinal"
            else OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        )
        cat_t = Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("enc", cat_enc),
        ])
        transformers = []
        if numeric_cols:     transformers.append(("num", num_t, numeric_cols))
        if categorical_cols: transformers.append(("cat", cat_t, categorical_cols))
        self.preprocessor = ColumnTransformer(transformers, remainder="drop")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y,
        )
        self._log(f"Train: {self.X_train.shape[0]:,} rows | Test: {self.X_test.shape[0]:,} rows")
        self._numeric_cols     = numeric_cols
        self._categorical_cols = categorical_cols

    # ── Phase 3: Modeling ─────────────────────────────────────────────
    def modeling(self) -> None:
        self._log("\n" + "=" * 60)
        self._log("PHASE 3 — MODELING")
        self._log("=" * 60)
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                              random_state=self.random_state)
        for key in self.models:
            if key not in self.MODEL_REGISTRY:
                self._log(f"  [WARN] Unknown model '{key}' — skipping.")
                continue
            clf = self.MODEL_REGISTRY[key]
            self._log(f"\nTraining: {clf.__class__.__name__} ...")
            steps = [("preprocessor", self.preprocessor)]
            if self.n_features:
                steps.append(("feature_select", SelectKBest(f_classif, k=self.n_features)))
            steps.append(("classifier", clf))
            pipeline = Pipeline(steps)
            pipeline.fit(self.X_train, self.y_train)
            y_pred  = pipeline.predict(self.X_test)
            y_proba = (pipeline.predict_proba(self.X_test)[:, 1]
                       if hasattr(clf, "predict_proba") else None)
            cv = cross_val_score(pipeline, self.X_train, self.y_train,
                                 cv=skf, scoring="roc_auc")
            self.results[key] = {
                "pipeline": pipeline,
                "acc":      accuracy_score(self.y_test, y_pred),
                "f1":       f1_score(self.y_test, y_pred, average="weighted"),
                "roc":      roc_auc_score(self.y_test, y_proba) if y_proba is not None else None,
                "cv_mean":  cv.mean(),
                "cv_std":   cv.std(),
                "y_pred":   y_pred,
                "y_proba":  y_proba,
                "report":   classification_report(self.y_test, y_pred),
                "cm":       confusion_matrix(self.y_test, y_pred),
            }
            self._log(f"  Accuracy : {self.results[key]['acc']:.4f}")
            self._log(f"  F1 (wtd) : {self.results[key]['f1']:.4f}")
            if y_proba is not None:
                self._log(f"  ROC-AUC  : {self.results[key]['roc']:.4f}")
            self._log(f"  CV ROC   : {cv.mean():.4f} +/- {cv.std():.4f}")

    # ── Phase 4: Evaluation ───────────────────────────────────────────
    def evaluation(self) -> None:
        self._log("\n" + "=" * 60)
        self._log("PHASE 4 — EVALUATION")
        self._log("=" * 60)
        if not self.results:
            self._log("No models trained yet.")
            return
        rows = [{
            "Model":    self.results[k]["pipeline"].named_steps["classifier"].__class__.__name__,
            "Accuracy": round(self.results[k]["acc"], 4),
            "F1 (wtd)": round(self.results[k]["f1"],  4),
            "ROC-AUC":  round(self.results[k]["roc"],  4) if self.results[k]["roc"] else "N/A",
            "CV ROC":   f"{self.results[k]['cv_mean']:.4f} +/- {self.results[k]['cv_std']:.4f}",
        } for k in self.results]
        self._log("\n--- Model Comparison ---")
        self._log(pd.DataFrame(rows).to_string(index=False))
        self.best_model_key = max(self.results, key=lambda k: self.results[k]["cv_mean"])
        best     = self.results[self.best_model_key]
        clf_name = best["pipeline"].named_steps["classifier"].__class__.__name__
        self._log(f"\nBest model: {clf_name} (CV ROC = {best['cv_mean']:.4f})")
        self._log("\n--- Classification Report (Best Model) ---")
        self._log(best["report"])
        # Confusion matrix
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(best["cm"]).plot(ax=ax, colorbar=False)
        ax.set_title(f"Confusion Matrix — {clf_name}")
        plt.tight_layout()
        cm_path = os.path.join(os.path.dirname(self.output_path) or ".", "confusion_matrix.png")
        plt.savefig(cm_path, dpi=120)
        plt.close()
        self._log(f"Saved: {cm_path}")
        # Feature importance
        clf = best["pipeline"].named_steps["classifier"]
        if hasattr(clf, "feature_importances_"):
            self._plot_feature_importance(best["pipeline"], clf)

    def _plot_feature_importance(self, pipeline: Pipeline, clf) -> None:
        try:
            pre       = pipeline.named_steps["preprocessor"]
            num_names = self._numeric_cols.copy()
            cat_names: List[str] = []
            for name, trans, cols in pre.transformers_:
                if name == "cat":
                    enc = trans.named_steps["enc"]
                    cat_names = (list(enc.get_feature_names_out(cols))
                                 if hasattr(enc, "get_feature_names_out") else list(cols))
            all_names   = num_names + cat_names
            importances = clf.feature_importances_
            if len(importances) != len(all_names):
                all_names = [f"feature_{i}" for i in range(len(importances))]
            imp_df = (pd.DataFrame({"feature": all_names, "importance": importances})
                        .sort_values("importance", ascending=False).head(20))
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1], color="steelblue")
            ax.set_xlabel("Importance")
            ax.set_title(f"Top-20 Feature Importances — {clf.__class__.__name__}")
            plt.tight_layout()
            fi_path = os.path.join(os.path.dirname(self.output_path) or ".", "feature_importance.png")
            plt.savefig(fi_path, dpi=120)
            plt.close()
            self._log(f"Saved: {fi_path}")
        except Exception as e:
            self._log(f"[WARN] Could not plot feature importances: {e}")

    # ── Phase 5: Hyperparameter Tuning ───────────────────────────────
    def hyperparameter_tuning(self) -> None:
        self._log("\n" + "=" * 60)
        self._log("PHASE 5 — HYPERPARAMETER TUNING")
        self._log("=" * 60)
        if self.best_model_key is None:
            self._log("Run evaluation() first.")
            return
        key = self.best_model_key
        if key not in self.TUNING_PARAMS:
            self._log(f"No tuning grid for '{key}'. Skipping.")
            self.final_pipeline = self.results[key]["pipeline"]
            return
        clf_name = self.results[key]["pipeline"].named_steps["classifier"].__class__.__name__
        self._log(f"Tuning: {clf_name} ...")
        search = RandomizedSearchCV(
            self.results[key]["pipeline"],
            self.TUNING_PARAMS[key],
            n_iter=20,
            scoring="roc_auc",
            cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                               random_state=self.random_state),
            random_state=self.random_state,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(self.X_train, self.y_train)
        self.final_pipeline = search.best_estimator_
        y_pred  = self.final_pipeline.predict(self.X_test)
        y_proba = self.final_pipeline.predict_proba(self.X_test)[:, 1]
        self._log(f"Best params   : {search.best_params_}")
        self._log(f"CV ROC-AUC    : {search.best_score_:.4f}")
        self._log(f"Test ROC-AUC  : {roc_auc_score(self.y_test, y_proba):.4f}")
        self._log(f"Test Accuracy : {accuracy_score(self.y_test, y_pred):.4f}")
        self._log("\n--- Tuned Classification Report ---")
        self._log(classification_report(self.y_test, y_pred))

    # ── Phase 6: Deployment ───────────────────────────────────────────
    def deployment(self) -> None:
        self._log("\n" + "=" * 60)
        self._log("PHASE 6 — DEPLOYMENT")
        self._log("=" * 60)
        p = self.final_pipeline or (self.results.get(self.best_model_key) or {}).get("pipeline")
        if p is None:
            self._log("No trained pipeline found.")
            return
        joblib.dump(p, self.output_path)
        self._log(f"Model saved to : {self.output_path}")
        self._log("To reload later:")
        self._log(f"  model = joblib.load('{self.output_path}')")
        self._log("  preds = model.predict(new_df)")

    # ── Run all phases ────────────────────────────────────────────────
    def run(self) -> Dict[str, Dict]:
        """Execute all CRISP-DM phases in sequence and return results dict."""
        self.data_understanding()
        self.data_preparation()
        self.modeling()
        self.evaluation()
        if self.tune:
            self.hyperparameter_tuning()
        self.deployment()
        self._log("\n" + "=" * 60)
        self._log("PIPELINE COMPLETE ✓")
        self._log("=" * 60)
        return self.results

    def predict(self, new_df: pd.DataFrame) -> np.ndarray:
        """Run inference on new data using the final trained pipeline."""
        p = self.final_pipeline or (self.results.get(self.best_model_key) or {}).get("pipeline")
        if p is None:
            raise RuntimeError("Call run() first.")
        return p.predict(new_df)

    def predict_proba(self, new_df: pd.DataFrame) -> np.ndarray:
        """Return class probabilities for new data."""
        p = self.final_pipeline or (self.results.get(self.best_model_key) or {}).get("pipeline")
        if p is None:
            raise RuntimeError("Call run() first.")
        return p.predict_proba(new_df)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)


# =============================================================================
# PART 3 — FRAUD DETECTION ON SUPABASE
# =============================================================================

# Feature engineering query — PostgreSQL syntax
# (uses EXTRACT / DATE_PART instead of SQLite's strftime / julianday)
QUERY = """
SELECT
    o.order_subtotal,
    o.shipping_fee,
    o.tax_amount,
    o.order_total,
    o.risk_score,
    o.promo_used,
    o.payment_method,
    o.device_type,
    o.ip_country,
    o.shipping_state,
    CASE WHEN o.billing_zip != o.shipping_zip THEN 1 ELSE 0 END  AS zip_mismatch,
    EXTRACT(HOUR FROM o.order_datetime::timestamp)               AS order_hour,
    EXTRACT(DOW  FROM o.order_datetime::timestamp)               AS order_dow,
    c.gender,
    c.customer_segment,
    c.loyalty_tier,
    c.is_active                                                  AS customer_is_active,
    DATE_PART('year', AGE(NOW(), c.birthdate::timestamp))::INT   AS customer_age,
    DATE_PART('day',  o.order_datetime::timestamp
                    - c.created_at::timestamp) / 365.25          AS customer_tenure_years,
    s.carrier,
    s.shipping_method,
    s.distance_band,
    s.late_delivery,
    COUNT(oi.order_item_id)                                      AS num_items,
    SUM(oi.quantity)                                             AS total_qty,
    o.is_fraud
FROM orders o
JOIN customers   c  ON o.customer_id = c.customer_id
JOIN shipments   s  ON o.order_id    = s.order_id
JOIN order_items oi ON o.order_id    = oi.order_id
GROUP BY
    o.order_id, o.order_subtotal, o.shipping_fee, o.tax_amount, o.order_total,
    o.risk_score, o.promo_used, o.payment_method, o.device_type, o.ip_country,
    o.shipping_state, o.billing_zip, o.shipping_zip, o.order_datetime, o.is_fraud,
    c.gender, c.customer_segment, c.loyalty_tier, c.is_active,
    c.birthdate, c.created_at,
    s.carrier, s.shipping_method, s.distance_band, s.late_delivery
"""


def main():
    # ── Load data from Supabase ───────────────────────────────────────
    df = load_from_supabase(QUERY)
    print(f"Fraud rate: {df.is_fraud.mean():.2%}")

    # ── Exploratory plots ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    df["is_fraud"].value_counts().rename({0: "Legit", 1: "Fraud"}).plot(
        kind="bar", ax=axes[0], color=["steelblue", "tomato"], edgecolor="black")
    axes[0].set_title("Class Distribution")
    axes[0].tick_params(axis="x", rotation=0)

    df.groupby("payment_method")["is_fraud"].mean().sort_values().plot(
        kind="barh", ax=axes[1], color="steelblue")
    axes[1].set_title("Fraud Rate by Payment Method")
    axes[1].set_xlabel("Rate")

    df[df.is_fraud == 0]["risk_score"].plot(
        kind="hist", bins=30, alpha=0.6, ax=axes[2], label="Legit", color="steelblue")
    df[df.is_fraud == 1]["risk_score"].plot(
        kind="hist", bins=30, alpha=0.6, ax=axes[2], label="Fraud", color="tomato")
    axes[2].set_title("Risk Score by Class")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("eda_plots.png", dpi=120)
    plt.close()
    print("Saved: eda_plots.png")

    # ── Run the full CRISP-DM pipeline ────────────────────────────────
    pipe = MLPipeline(
        df           = df,
        target       = "is_fraud",
        models       = ["lr", "rf", "gb"],
        tune         = True,
        output_path  = "fraud_model.joblib",
        cat_strategy = "onehot",
        scale        = True,
        test_size    = 0.2,
        random_state = 42,
        cv_folds     = 5,
    )
    results = pipe.run()

    # ── Inference example ─────────────────────────────────────────────
    # Reload the saved model and score new data:
    #
    #   model = joblib.load("fraud_model.joblib")
    #   preds = model.predict(new_df)
    #   proba = model.predict_proba(new_df)[:, 1]
    #
    # Or use the live pipeline object:
    #
    #   preds = pipe.predict(new_df)
    #   proba = pipe.predict_proba(new_df)

    return results


if __name__ == "__main__":
    main()