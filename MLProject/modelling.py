import os
import joblib
import sys

# ================== LOAD DATA ==================
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "dataset_preprocessing")

print(f"Current working directory: {os.getcwd()}")
print(f"Script directory: {script_dir}")
print(f"Looking for data in: {data_path}")

if not os.path.exists(data_path):
    print(f"ERROR: Folder dataset_preprocessing tidak ditemukan!")
    print("Isi folder saat ini:", os.listdir(script_dir))
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