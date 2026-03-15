# Script ini melatih model machine learning dan mencatat hasilnya ke MLflow.

from pathlib import Path # cari direktori
import mlflow # mencatat eksperimen
import pandas as pd # mencatat eksperimen
from lightgbm import LGBMRegressor # model machine learning
import os # membuat path file, membaca folder, mengatur file path
import numpy as np # dipakai untuk perhitungan matematika dan array
import warnings # menyembunyikan warning dari Python
import argparse # menerima parameter dari command line
import joblib # menyimpan atau membaca model / object machine learning
from sklearn.metrics import mean_squared_error, r2_score # metric evaluation

# Filtering warnings
warnings.filterwarnings(action='ignore')

# Fungsi ini akan melatih model dengan parameter:
def train_model(n_estimators, max_depth):
    # Mencari folder utama project.
    base_path = Path(__file__).resolve().parent.parent
    data_path = base_path / "preprocessing" / "diamond_preprocessing"
    
    # Membaca file preprocessing yang sudah disimpan.
    transformer = joblib.load(os.path.join(data_path, 'power_transformers.joblib'))

    # Mengambil transformer khusus untuk kolom harga.
    price_transformer = transformer['price']

    # Membaca data training dan testing
    X_train = pd.read_csv(os.path.join(data_path, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(data_path, 'y_train.csv'))
    X_test = pd.read_csv(os.path.join(data_path, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(data_path, 'y_test.csv'))

    # Karena harga sebelumnya ditransformasi, di sini dikembalikan ke skala asli.
    y_test = price_transformer.inverse_transform(y_test.to_numpy().reshape(-1,1))

    # Melakukan Experimen pelatihan model
    with mlflow.start_run():
        model = LGBMRegressor(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)

    # Mencatat performa model
        y_pred_transform = model.predict(X_test)
        y_pred = price_transformer.inverse_transform(y_pred_transform.reshape(-1,1))

        # R2 = mengukur seberapa baik model menjelaskan data sebenarnya
        r2_skor = r2_score(y_test, y_pred)
        # RMSE = mengukur rata-rata kesalahan prediksi model
        rmse_skor = np.sqrt(mean_squared_error(y_test, y_pred))

        # Supaya MLFlow bisa tracking
        mlflow.log_params(model.get_params())
        mlflow.log_metric("RMSE", rmse_skor)
        mlflow.log_metric("R2", r2_skor)

        # Bagian ini dipakai untuk menyimpan model yang sudah dilatih ke MLflow
        # lengkap dengan environment yang dibutuhkan agar model bisa dijalankan lagi
        conda_path = Path(__file__).resolve().parent / "conda.yaml"
        mlflow.sklearn.log_model(model, artifact_path="model", conda_env=str(conda_path))

        print(f'R2 Score : {r2_skor}')
        print(f'RMSE : {rmse_skor}')

# kode ini hanya dijalankan jika file dijalankan langsung
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', type=int, default=100, help='Jumlah pohon dalam Model')
    parser.add_argument('--max_depth', type=int, default=-1, help='Kedalaman pohon dalam Model')
    args = parser.parse_args()

    train_model(args.n_estimators, args.max_depth)