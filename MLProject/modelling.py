import mlflow
import mlflow.sklearn
import joblib
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# ================== SETUP MLFLOW ==================
# mlruns disimpan di root repo (satu level di atas folder MLProject)
script_dir = os.path.dirname(os.path.abspath(__file__))
mlruns_path = os.path.join(script_dir, "..", "mlruns")
mlruns_path = os.path.abspath(mlruns_path)

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
mlflow.set_tracking_uri("file:" + mlruns_path)

print("🚀 Mulai training...")
print(f"📁 MLflow tracking URI: file:{mlruns_path}")

# ================== MULAI RUN ==================
with mlflow.start_run() as run:
    print(f"✅ Run ID: {run.info.run_id}")

    # ================== LOAD DATA ==================
    data_path = os.path.join(script_dir, "dataset_preprocessing")

    print(f"Current dir: {os.getcwd()}")
    print(f"Load data dari: {data_path}")

    if not os.path.exists(data_path):
        print("❌ Folder dataset_preprocessing tidak ditemukan!")
        sys.exit(1)

    X_train = joblib.load(os.path.join(data_path, "X_train.pkl"))
    X_test  = joblib.load(os.path.join(data_path, "X_test.pkl"))
    y_train = joblib.load(os.path.join(data_path, "y_train.pkl"))
    y_test  = joblib.load(os.path.join(data_path, "y_test.pkl"))
    print("✅ Data berhasil dimuat!")

    # ================== TRAINING ==================
    param_grid = {
        "n_estimators":    [100, 200],
        "max_depth":       [10, 15, None],
        "min_samples_split": [2, 5],
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid,
        cv=3,
        scoring="f1_weighted",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred     = best_model.predict(X_test)
    print(f"✅ Best params: {grid_search.best_params_}")

    # ================== METRICS ==================
    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "f1_score":  f1_score(y_test, y_pred, average="weighted"),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall":    recall_score(y_test, y_pred, average="weighted"),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(
            y_test, best_model.predict_proba(X_test)[:, 1]
        )
    except Exception:
        metrics["roc_auc"] = 0.0

    # ================== LOGGING ==================
    mlflow.log_params(grid_search.best_params_)
    for k, v in metrics.items():
        mlflow.log_metric(k, v)

    # Log model — disimpan di artifacts/model/
    mlflow.sklearn.log_model(best_model, "model")

    # ================== ARTIFACT ==================
    cm_path = os.path.join(script_dir, "confusion_matrix.png")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path)

    # ================== PRINT HASIL ==================
    print("🎉 Training selesai!")
    print(f"Best F1 Score : {metrics['f1_score']:.4f}")
    print(f"Accuracy      : {metrics['accuracy']:.4f}")
    print(f"Precision     : {metrics['precision']:.4f}")
    print(f"Recall        : {metrics['recall']:.4f}")
    print(f"ROC AUC       : {metrics['roc_auc']:.4f}")

    # Path model untuk referensi Docker build
    artifact_uri = mlflow.get_artifact_uri("model")
    # Ubah file:/// menjadi path biasa
    model_local_path = artifact_uri.replace("file://", "").replace("file:", "")
    print(f"✅ Model artifact URI  : {artifact_uri}")
    print(f"✅ Model local path    : {model_local_path}")