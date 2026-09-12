import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# Config
ROOT = r"C:\Users\dayinta agustina\Downloads\Documents\PENELITIAN FMIPA FORECAST FINAL"
DATA_CSV = os.path.join(ROOT, "data/harga_saham_yfinance.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODELS_DIR = os.path.normpath(MODELS_DIR)

WINDOW = 30
ticker_list = ["ADRO", "PTBA", "INDY", "ITMG"]
CUTOFF = "2022-12-31"

# helper functions

def buat_sequences(data: np.ndarray, window: int):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i])
        y.append(data[i])
    return np.array(X)[..., np.newaxis], np.array(y)


def siapkan_data_lstm(series_train: pd.Series, series_test: pd.Series, window: int):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_sc = scaler.fit_transform(series_train.values.reshape(-1, 1)).flatten()
    test_sc  = scaler.transform(series_test.values.reshape(-1, 1)).flatten()
    X_tr, y_tr = buat_sequences(train_sc, window)
    X_te, y_te = buat_sequences(test_sc,  window)
    return X_tr, y_tr, X_te, y_te, scaler

# Load data
print("Loading data...")
df = pd.read_csv(DATA_CSV, index_col=0, parse_dates=True)
df_train = df[df.index <= CUTOFF].copy()
df_test  = df[df.index > CUTOFF].copy()

# Evaluate
evaluasi_lstm_standalone = []

for ticker in ticker_list:
    print(f"Evaluating LSTM standalone — {ticker}")
    series_train = df_train[ticker].dropna()
    series_test  = df_test[ticker].dropna()

    # prepare sequences and scaler
    X_tr, y_tr, X_te, y_te, scaler = siapkan_data_lstm(series_train, series_test, WINDOW)

    if X_te.shape[0] == 0:
        print(f"  Skipping {ticker}: not enough test data after window={WINDOW}")
        continue

    # load model
    model_path = os.path.join(MODELS_DIR, f"lstm_pure_{ticker}.keras")
    if not os.path.exists(model_path):
        model_path = os.path.join(MODELS_DIR, f"lstm_pure_{ticker}.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found for {ticker}: {model_path}")

    mdl = tf.keras.models.load_model(model_path)

    # predict (scaled -> inverse)
    pred_sc = mdl.predict(X_te, verbose=0).flatten()
    pred    = scaler.inverse_transform(pred_sc.reshape(-1,1)).flatten()

    # actual aligned with sequences: series_test.values[window:]
    aktual = series_test.values[WINDOW:WINDOW+len(pred)]

    mae  = mean_absolute_error(aktual, pred)
    rmse = np.sqrt(mean_squared_error(aktual, pred))
    mape = np.mean(np.abs((aktual - pred) / aktual)) * 100
    r2   = r2_score(aktual, pred)

    evaluasi_lstm_standalone.append({
        "Emiten": ticker,
        "Model": "LSTM",
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE (%)": round(mape, 4),
        "R²": round(r2, 6)
    })

# Save results
out_df = pd.DataFrame(evaluasi_lstm_standalone)
out_csv = os.path.join(os.path.dirname(__file__), "..", "data", "tabel_lstm_standalone_evaluasi.csv")
out_csv = os.path.normpath(out_csv)
out_df.to_csv(out_csv, index=False)

print("Done. Results:")
print(out_df.to_string(index=False))
print(f"Saved to: {out_csv}")
