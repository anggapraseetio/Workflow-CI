import mlflow
import mlflow.sklearn
import joblib
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ================== SETUP MLFLOW ==================
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
mlflow.set_tracking_uri("file:./mlruns")

# Ambil Run ID dari environment variable yang diset oleh MLflow Project
run_id = os.environ.get("MLFLOW_RUN_ID")
print(f"✅ MLFLOW_RUN_ID dari env: {run_id}")

if not run_id:
    print("⚠️ Tidak menemukan MLFLOW_RUN_ID, membuat run baru...")
    mlflow.start_run()
    run_id = mlflow.active_run().info.run_id

# ================== LOAD DATA ==================
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "dataset_preprocessing")

print(f"Current working directory: {os.getcwd()}")
print(f"Script directory: {script_dir}")
print(f"Load data dari: {data_path}")

if not os.path.exists(data_path):
    print(f"❌ Folder dataset_preprocessing tidak ditemukan!")
    print("Isi folder:", os.listdir(script_dir))
    sys.exit(1)

try:
    X_train = joblib.load(os.path.join(data_path, "X_train.pkl"))
    X_test = joblib.load(os.path.join(data_path, "X_test.pkl"))
    y_train = joblib.load(os.path.join(data_path, "y_train.pkl"))
    y_test = joblib.load(os.path.join(data_path, "y_test.pkl"))
    print("✅ Data berhasil dimuat!")
except Exception as e:
    print(f"❌ Gagal load data: {e}")
    sys.exit(1)

# ================== TRAINING ==================
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15, None],
    'min_samples_split': [2, 5]
}

rf = RandomForestClassifier(random_state=42, n_jobs=-1)
print("Training model dengan GridSearchCV...")
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# ================== METRICS ==================
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred, average='weighted'),
    "precision": precision_score(y_test, y_pred, average='weighted'),
    "recall": recall_score(y_test, y_pred, average='weighted'),
}

try:
    y_proba = best_model.predict_proba(X_test)
    metrics["roc_auc"] = roc_auc_score(y_test, y_proba[:, 1])
except:
    metrics["roc_auc"] = 0.0

# ================== LOGGING ==================
with mlflow.start_run(run_id=run_id, nested=True):
    mlflow.log_params(grid_search.best_params_)
    
    for k, v in metrics.items():
        mlflow.log_metric(k, v)

    mlflow.sklearn.log_model(best_model, "model")

    # Confusion Matrix Artifact
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    plt.close()

    mlflow.log_artifact("confusion_matrix.png")

print("🎉 Training CI berhasil!")
print(f"Best Params : {grid_search.best_params_}")
print(f"Best F1 Score: {metrics['f1_score']:.4f}")