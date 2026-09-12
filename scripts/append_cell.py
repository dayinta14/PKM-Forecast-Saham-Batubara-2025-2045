import nbformat
from nbformat.v4 import new_code_cell

nb_path = r"model_ann_lstm.ipynb"
nb = nbformat.read(nb_path, as_version=4)

cell_source = '''from sklearn.metrics import r2_score

# Evaluasi LSTM standalone (aligned with notebook variables)
evaluasi_lstm_standalone = []
for ticker in ticker_list:
    pred = lstm_pred_pure[ticker]
    aktual = df_test[ticker].values
    n_fc = min(len(aktual), len(pred))
    aktual, pred = aktual[:n_fc], pred[:n_fc]

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

# Save to CSV
import pandas as pd
out_df = pd.DataFrame(evaluasi_lstm_standalone)
out_path = f"{DIR_DATA}/tabel_lstm_standalone_evaluasi.csv"
out_df.to_csv(out_path, index=False)
print("✓  tabel_lstm_standalone_evaluasi.csv saved")
print(out_df)
'''

nb['cells'].append(new_code_cell(cell_source))
nbformat.write(nb, nb_path)
print('Appended new cell to', nb_path)
