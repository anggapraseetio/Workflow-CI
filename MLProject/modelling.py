import mlflow
import mlflow.sklearn
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

data_path = os.getenv("DATA_PATH", "dataset_preprocessing")

X_train = joblib.load(f"{data_path}/X_train.pkl")
X_test = joblib.load(f"{data_path}/X_test.pkl")
y_train = joblib.load(f"{data_path}/y_train.pkl")
y_test = joblib.load(f"{data_path}/y_test.pkl")

with mlflow.start_run(run_name="CI_RandomForest_Tuning"):
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, None],
        'min_samples_split': [2, 5]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
    }
    
    mlflow.log_params(grid_search.best_params_)
    for k, v in metrics.items():
        mlflow.log_metric(k, v)
    
    mlflow.sklearn.log_model(best_model, "model")
    
    print("Training CI berhasil!")
    print(f"Best F1: {metrics['f1_score']:.4f}")