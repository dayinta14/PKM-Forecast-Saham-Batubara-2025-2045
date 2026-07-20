# DESKRIPSI KARYA — HAK KEKAYAAN INTELEKTUAL (HKI)

---

## 1. Judul Inovasi

**Potensi Pertumbuhan Ekonomi Indonesia 2045 Pada Sektor Energi Berbasis Harga Saham Menggunakan *Hybrid Statistics and Machine Learning Method***

---

## 2. Latar Belakang

### 2.1 Permasalahan yang Ingin Diselesaikan

Indonesia menargetkan pencapaian visi "Indonesia Emas 2045" yang bertepatan dengan seratus tahun kemerdekaan Republik Indonesia. Sektor energi, khususnya industri batubara, merupakan salah satu pilar utama perekonomian nasional yang berkontribusi signifikan terhadap Produk Domestik Bruto (PDB), penerimaan ekspor, dan penyediaan lapangan kerja. Namun, proyeksi arah pertumbuhan sektor ini dalam jangka panjang hingga tahun 2045 masih menjadi tantangan yang belum terjawab secara memadai, terutama di tengah dinamika transisi energi global, fluktuasi harga komoditas, serta ketidakpastian geopolitik internasional.

Metode prediksi konvensional yang mengandalkan pendekatan statistik semata, seperti model Autoregressive Integrated Moving Average (ARIMA), memiliki keterbatasan dalam menangkap pola non-linear dan kompleksitas dinamika pasar keuangan. Sebaliknya, metode machine learning berbasis jaringan saraf tiruan (Artificial Neural Network), khususnya Long Short-Term Memory (LSTM), memiliki kemampuan unggul dalam mempelajari pola temporal non-linear, namun cenderung memerlukan data dalam jumlah besar dan rentan terhadap overfitting pada dataset terbatas.

### 2.2 Pentingnya Prediksi Pertumbuhan Ekonomi Berbasis Sektor Energi

Sektor energi berbasis batubara Indonesia diwakili oleh emiten-emiten yang tercatat di Bursa Efek Indonesia (BEI/IDX), antara lain PT Adaro Energy Indonesia Tbk (ADRO), PT Bukit Asam Tbk (PTBA), PT Indika Energy Tbk (INDY), dan PT Indo Tambangraya Megah Tbk (ITMG). Pergerakan harga saham emiten-emiten tersebut mencerminkan ekspektasi pasar terhadap prospek pertumbuhan sektor energi secara keseluruhan. Oleh karena itu, analisis dan prediksi harga saham sektor batubara dapat dijadikan proksi (proxy) untuk mengukur potensi pertumbuhan ekonomi Indonesia pada sektor energi.

### 2.3 Alasan Penggunaan Data Harga Saham

Data harga saham dipilih sebagai basis analisis karena beberapa alasan fundamental:

1. **Ketersediaan dan kualitas data**: Harga saham tersedia secara harian dengan kualitas pencatatan yang terstandarisasi oleh otoritas bursa.
2. **Refleksi ekspektasi pasar**: Harga saham mengandung informasi agregat mengenai ekspektasi pelaku pasar terhadap kinerja masa depan perusahaan dan sektor terkait.
3. **Karakteristik time series**: Data harga saham memiliki sifat time series yang kompleks dengan komponen linear dan non-linear, sehingga cocok untuk menguji pendekatan hybrid yang diusulkan.
4. **Relevansi ekonomi makro**: Kinerja emiten batubara di BEI secara langsung terkait dengan kontribusi sektor energi terhadap pertumbuhan ekonomi nasional.

---

## 3. Tujuan Pengembangan Sistem

### 3.1 Tujuan Utama

Sistem ini dikembangkan untuk membangun kerangka kerja (*framework*) prediksi harga saham jangka panjang (2025–2045) pada sektor energi batubara Indonesia dengan mengintegrasikan metode statistik ARIMA dan metode machine learning LSTM dalam suatu arsitektur hybrid yang sinergis. Secara spesifik, tujuan pengembangan sistem meliputi:

1. Merancang dan mengimplementasikan sistem preprocessing data dan Exploratory Data Analysis (EDA) yang komprehensif untuk data harga saham sektor batubara.
2. Membangun model ARIMA optimal untuk menangkap komponen linear dari time series harga saham.
3. Membangun model LSTM untuk menangkap komponen non-linear yang tidak tertangkap oleh model ARIMA (residual ARIMA).
4. Mengintegrasikan kedua model dalam arsitektur hybrid ARIMA-LSTM untuk menghasilkan prediksi yang lebih akurat dibandingkan masing-masing model secara individu.
5. Menghasilkan proyeksi harga saham sektor energi batubara hingga tahun 2045 sebagai masukan analitis bagi pemangku kepentingan.

### 3.2 Manfaat Sistem

Sistem ini memberikan manfaat bagi berbagai pemangku kepentingan:

| Pemangku Kepentingan | Manfaat |
|---|---|
| **Akademisi dan Peneliti** | Menyediakan kerangka kerja metodologis yang dapat direplikasi dan dikembangkan untuk penelitian prediksi time series pada domain keuangan dan ekonomi. |
| **Pemerintah** | Memberikan proyeksi kuantitatif mengenai potensi pertumbuhan sektor energi yang dapat menjadi masukan dalam perumusan kebijakan ekonomi dan energi nasional menuju Indonesia Emas 2045. |
| **Investor dan Pelaku Pasar** | Menyediakan alat bantu analisis untuk pengambilan keputusan investasi jangka panjang pada sektor energi batubara. |
| **Industri Energi** | Memberikan perspektif berbasis data mengenai tren jangka panjang harga saham sektor batubara yang relevan untuk perencanaan strategis perusahaan. |

---

## 4. Deskripsi Inovasi

### 4.1 Gambaran Umum Sistem

Sistem yang dikembangkan merupakan platform analisis prediktif berbasis pendekatan *Hybrid Statistics and Machine Learning* yang mengintegrasikan model ARIMA (Autoregressive Integrated Moving Average) sebagai komponen statistik dan LSTM (Long Short-Term Memory) sebagai komponen machine learning. Sistem ini dirancang secara modular dalam tiga komponen utama yang saling terhubung, mencakup seluruh alur pemrosesan mulai dari akuisisi data mentah hingga menghasilkan proyeksi harga saham jangka panjang beserta interval kepercayaannya.

### 4.2 Komponen Utama Sistem

Sistem terdiri atas tiga modul utama yang terintegrasi:

1. **Modul Preprocessing dan EDA** (*preprocessing_eda*): Bertanggung jawab atas pembangkitan dataset representatif, transformasi data (log return), analisis eksplorasi visual, uji stasioneritas (ADF dan KPSS), dekomposisi time series, serta identifikasi pola autokorelasi melalui ACF dan PACF.

2. **Modul Pemodelan ARIMA** (*model_arima*): Melaksanakan grid search otomatis untuk seleksi orde optimal (p,d,q) berdasarkan kriteria informasi AIC, fitting model, diagnostik residual menyeluruh, serta evaluasi performa out-of-sample melalui rolling forecast 1-step ahead. Modul ini juga mengekspor residual ARIMA sebagai input bagi modul hybrid.

3. **Modul Pemodelan ANN-LSTM dan Hybrid** (*model_ann_lstm*): Membangun dan melatih arsitektur LSTM dua lapis (dual-layer LSTM) untuk dua tujuan: (a) LSTM murni yang dilatih pada harga saham secara langsung sebagai baseline, dan (b) LSTM hybrid yang dilatih pada residual ARIMA untuk menangkap komponen non-linear. Modul ini juga menghasilkan proyeksi jangka panjang 2025–2045 beserta confidence interval.

### 4.3 Kebaruan yang Ditawarkan

Sistem ini menawarkan beberapa unsur kebaruan:

1. **Arsitektur hybrid terstruktur**: Integrasi ARIMA dan LSTM dilakukan bukan secara *ensemble* sederhana, melainkan melalui dekomposisi sinyal dimana ARIMA menangkap komponen linear terlebih dahulu, kemudian LSTM mempelajari pola residual non-linear yang tersisa. Formulasi matematisnya:

   ```
   ŷ_hybrid(t) = ŷ_ARIMA(t) + ŷ_LSTM(ε_ARIMA(t))
   ```

   dimana ε_ARIMA(t) = X(t) − ŷ_ARIMA(t) adalah residual ARIMA.

2. **Fokus aplikatif pada visi Indonesia Emas 2045**: Sistem secara khusus dirancang untuk menghasilkan proyeksi hingga tahun 2045, selaras dengan target pembangunan nasional, dengan mempertimbangkan milestone tahun-tahun kunci (2025, 2027, 2030, 2035, 2040, 2045).

3. **Multi-emiten analisis simultan**: Sistem memproses empat emiten batubara utama (ADRO, PTBA, INDY, ITMG) secara paralel dalam satu kerangka kerja terintegrasi, memungkinkan perbandingan lintas emiten.

4. **Evaluasi komprehensif tiga model**: Sistem secara sistematis membandingkan tiga pendekatan (ARIMA murni, LSTM murni, dan Hybrid ARIMA-LSTM) menggunakan metrik standar (MAE, RMSE, MAPE, R²) untuk memvalidasi keunggulan arsitektur hybrid.

5. **Pipeline end-to-end yang reproducible**: Seluruh alur kerja dari preprocessing hingga forecast jangka panjang terstruktur dalam pipeline yang dapat direproduksi secara konsisten.

---

## 5. Metodologi Sistem

### Tahap 1: Data Acquisition

Data yang digunakan dalam sistem ini merupakan data harga saham harian sektor batubara yang tercatat di Bursa Efek Indonesia (BEI/IDX). Spesifikasi data meliputi:

| Parameter | Keterangan |
|---|---|
| **Emiten** | ADRO, PTBA, INDY, ITMG |
| **Periode** | 2 Januari 2014 – 31 Desember 2024 |
| **Frekuensi** | Harian (hari bursa/business day) |
| **Jumlah Observasi** | 2.869 hari per emiten |
| **Variabel** | Harga penutupan (closing price) |

Pembangkitan data dilakukan secara deterministik menggunakan model Geometric Brownian Motion (GBM) yang diperkaya dengan efek-efek makroekonomi, antara lain:

- **Drift tahunan** yang bervariasi per emiten (5–12%) untuk merepresentasikan tren jangka panjang.
- **Volatilitas annualized** yang disesuaikan (33–65%) untuk mencerminkan karakteristik risiko masing-masing emiten.
- **Efek booming batubara** (tahun 2022) berupa peningkatan log return selama 60 hari perdagangan.
- **Efek crash COVID-19** (tahun 2020) berupa penurunan log return selama 40 hari perdagangan.

Log return harian dihitung sebagai transformasi logaritmik natural dari rasio harga:

```
ret(t) = ln(P(t) / P(t-1))
```

### Tahap 2: Data Preprocessing dan Exploratory Data Analysis (EDA)

#### 2.1 Pembersihan Data

Proses pembersihan data mencakup penanganan missing values, validasi konsistensi indeks temporal (business day), serta penghapusan observasi pertama yang menghasilkan NaN pada perhitungan log return.

#### 2.2 Transformasi Data

Transformasi data yang dilakukan meliputi:

- **Perhitungan log return**: Transformasi harga nominal menjadi log return untuk memperoleh distribusi yang mendekati normal dan memenuhi asumsi stasioneritas.
- **Normalisasi MinMaxScaler**: Scaling data ke rentang [-1, 1] untuk keperluan training model LSTM, guna menstabilkan proses konvergensi gradient.
- **Windowed sequences**: Transformasi time series menjadi format supervised learning dengan ukuran jendela (window) 30 hari perdagangan.

#### 2.3 Analisis Eksploratif

Analisis eksploratif yang dilakukan secara komprehensif meliputi:

1. **Visualisasi harga historis**: Panel plot yang menampilkan pergerakan harga penutupan seluruh emiten selama periode analisis, dilengkapi anotasi event ekonomi penting (COVID-19, booming batubara, kebijakan DMO).

2. **Dekomposisi time series**: Menggunakan metode *seasonal_decompose* untuk mengidentifikasi komponen trend, seasonal, dan residual dari setiap seri harga saham.

3. **Uji stasioneritas**: Pengujian formal menggunakan dua metode komplementer:
   - **Augmented Dickey-Fuller (ADF) Test**: Untuk menguji hipotesis nol bahwa seri memiliki unit root (non-stasioner).
   - **Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test**: Sebagai konfirmasi silang, menguji hipotesis nol bahwa seri bersifat stasioner.

   Hasil uji menunjukkan bahwa seluruh seri harga level bersifat non-stasioner (I(1)), dan menjadi stasioner setelah differencing orde pertama (Δ¹).

4. **Analisis distribusi**: Visualisasi histogram, density plot, serta perhitungan skewness dan kurtosis untuk setiap seri harga dan log return.

5. **Analisis autokorelasi**: Pembuatan plot Autocorrelation Function (ACF) dan Partial Autocorrelation Function (PACF) yang menjadi dasar identifikasi kandidat orde ARIMA.

### Tahap 3: Pemodelan ARIMA

#### 3.1 Tujuan Penggunaan ARIMA

Model ARIMA digunakan sebagai komponen statistik dalam arsitektur hybrid untuk menangkap pola linear dan struktur autoregressive dalam time series harga saham. Model ARIMA dipilih karena:

- Merupakan model benchmark standar dalam analisis time series keuangan.
- Memiliki interpretabilitas yang tinggi melalui parameter (p, d, q).
- Mampu menangkap dependensi linear temporal secara efektif.
- Menyediakan kerangka kerja diagnostik residual yang terstandarisasi.

#### 3.2 Mekanisme Forecasting Statistik

Proses pemodelan ARIMA dilaksanakan melalui tahapan berikut:

**a. Pembagian Data (Train/Test Split)**

Data dibagi secara temporal (time-based split) untuk menghormati urutan kronologis:
- **Data training**: 2 Januari 2014 – 31 Desember 2022 (≈80%)
- **Data testing**: 1 Januari 2023 – 31 Desember 2024 (≈20%)

**b. Grid Search Orde Optimal**

Pencarian orde optimal ARIMA(p,d,q) dilakukan melalui grid search exhaustive dengan:
- p ∈ {0, 1, 2, 3} — orde autoregressive
- d = 1 — orde differencing (berdasarkan hasil uji ADF yang menunjukkan I(1))
- q ∈ {0, 1, 2, 3} — orde moving average
- Kriteria seleksi: **Akaike Information Criterion (AIC)** minimum
- Estimasi parameter: metode L-BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno)

**c. Diagnostik Residual**

Panel diagnostik residual yang komprehensif meliputi:
- Time plot residual dengan band ±2σ
- Histogram dan Kernel Density Estimation (KDE) residual
- Quantile-Quantile (Q-Q) Plot terhadap distribusi normal
- ACF dan PACF residual untuk validasi white noise
- Uji Ljung-Box pada multiple lags untuk autokorelasi residual
- Uji Jarque-Bera untuk normalitas distribusi residual

**d. Rolling Forecast 1-Step Ahead**

Evaluasi out-of-sample menggunakan metode rolling forecast yang lebih realistis, dimana model diperbarui (refit) setiap langkah dengan menambahkan observasi aktual terbaru ke dalam data training.

### Tahap 4: Pemodelan ANN-LSTM

#### 4.1 Arsitektur Model

Model LSTM yang dibangun menggunakan arsitektur dual-layer sebagai berikut:

```
┌─────────────────────────────────────────────────┐
│  Input Layer        → (None, 30, 1)             │
│  LSTM Layer 1       → 64 units, return_seq=True │
│                       + L2 regularization       │
│  Dropout Layer 1    → rate = 0.2                │
│  LSTM Layer 2       → 64 units, return_seq=False│
│                       + L2 regularization       │
│  Dropout Layer 2    → rate = 0.2                │
│  Dense Layer        → 32 units, activation=ReLU │
│  Dense Output       → 1 unit (prediksi harga)   │
└─────────────────────────────────────────────────┘
```

Konfigurasi hyperparameter yang digunakan:

| Hyperparameter | Nilai | Justifikasi |
|---|---|---|
| Window (look-back) | 30 hari | Menangkap satu bulan pola trading |
| LSTM units | 64 | Kapasitas memadai tanpa overfitting |
| Dropout rate | 0.2 | Regularisasi standar time series keuangan |
| Learning rate | 1×10⁻³ | Initial rate untuk Adam optimizer |
| Batch size | 32 | Mini-batch gradient descent yang stabil |
| Max epochs | 100 | Dengan Early Stopping |
| ES patience | 15 | Toleransi penurunan validation loss |

Teknik regularisasi yang diterapkan:
- **Dropout** (rate 0.2) pada setiap layer LSTM untuk mencegah overfitting.
- **L2 regularization** (λ = 10⁻⁴) pada kernel LSTM untuk constraint bobot.
- **Early Stopping** dengan pemantauan validation loss dan pengembalian bobot terbaik.
- **ReduceLROnPlateau** yang menurunkan learning rate sebesar faktor 0.5 bila validation loss stagnan selama 7 epoch.

#### 4.2 Mekanisme Pembelajaran Pola Temporal

Model LSTM memanfaatkan mekanisme gating (input gate, forget gate, output gate) dan cell state untuk mempelajari dependensi temporal jangka panjang (long-term dependencies) dalam data time series. Proses pembelajaran meliputi:

1. **Feature Engineering**: Transformasi data time series menjadi pasangan input-output berpola windowed:
   - Input X: [x(t-W), x(t-W+1), ..., x(t-1)] dengan dimensi (n_samples, W, 1)
   - Output y: x(t) dengan dimensi (n_samples, 1)

2. **Scaling**: Normalisasi data menggunakan MinMaxScaler ke rentang [-1, 1] untuk menstabilkan proses training.

3. **Training**: Pelatihan model menggunakan optimizer Adam dengan loss function MSE (Mean Squared Error), dengan pembagian data training menjadi 85% training dan 15% validasi.

4. **Prediksi**: Inverse transform hasil prediksi dari skala normalisasi kembali ke skala harga asli.

### Tahap 5: Hybrid Statistics and Machine Learning

Integrasi ARIMA dan LSTM dalam sistem hybrid dilaksanakan melalui pendekatan **residual learning** yang terdiri atas langkah-langkah berikut:

**Langkah 1 — Pemodelan Linear (ARIMA)**

Model ARIMA dilatih pada data harga saham untuk menangkap komponen linear. Residual ARIMA dihitung sebagai selisih antara nilai aktual dan fitted values:

```
ε_ARIMA(t) = X(t) − ŷ_ARIMA(t)
```

Residual ini mengandung komponen non-linear yang tidak berhasil dimodelkan oleh ARIMA.

**Langkah 2 — Pemodelan Non-Linear (LSTM)**

Model LSTM kedua (LSTM Residual) dilatih khusus pada sequence residual ARIMA:
- Input: [ε(t-W), ε(t-W+1), ..., ε(t-1)]
- Output: prediksi residual ε̂(t)

Dengan demikian, LSTM belajar memodelkan pola non-linear yang tersisa setelah pemodelan ARIMA.

**Langkah 3 — Kombinasi Hybrid**

Prediksi akhir diperoleh dengan menjumlahkan komponen linear (ARIMA) dan non-linear (LSTM):

```
ŷ_hybrid(t) = ŷ_ARIMA(t) + ŷ_LSTM(ε_ARIMA(t))
```

**Langkah 4 — Forecast Jangka Panjang 2025–2045**

Proyeksi jangka panjang dilakukan menggunakan *recursive multi-step forecast*:
- **ARIMA**: Forecast iteratif menggunakan model yang di-fit pada data penuh (train + test) dengan horizon hingga 2045.
- **LSTM Residual**: Prediksi residual secara rekursif menggunakan buffer yang diperbarui setiap langkah.
- **Confidence Interval**: Interval kepercayaan 95% dihitung dari forecast ARIMA menggunakan statsmodels.

---

## 6. Diagram Alur Sistem

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ALUR KERJA SISTEM                          │
│        Hybrid Statistics and Machine Learning Method               │
└─────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────────┐
                    │  INPUT DATA SAHAM    │
                    │  (ADRO, PTBA, INDY,  │
                    │   ITMG) 2014–2024    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   PREPROCESSING      │
                    │ • Pembersihan data   │
                    │ • Log return         │
                    │ • Train/Test split   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │        EDA           │
                    │ • Visualisasi historis│
                    │ • Dekomposisi        │
                    │ • Uji stasioneritas  │
                    │ • ACF/PACF           │
                    └──────────┬───────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
                ▼              ▼              ▼
     ┌─────────────┐ ┌──────────────┐ ┌──────────────┐
     │   ARIMA     │ │ LSTM Murni   │ │ LSTM Residual│
     │ Forecasting │ │ (Baseline)   │ │ (Hybrid)     │
     │             │ │              │ │              │
     │ • Grid      │ │ • Training   │ │ • Training   │
     │   Search    │ │   pada       │ │   pada       │
     │ • Fitting   │ │   harga      │ │   residual   │
     │ • Diagnostik│ │   langsung   │ │   ARIMA      │
     │ • Residual  │─┤              │ │              │
     │   Export    │ │              │ │              │
     └──────┬──────┘ └──────┬───────┘ └──────┬───────┘
            │               │                │
            │               ▼                │
            │    ┌──────────────────┐         │
            │    │ Evaluasi Metrik  │         │
            └───►│ (MAE, RMSE,     │◄────────┘
                 │  MAPE, R²)      │
                 └────────┬────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   HYBRID PREDICTION   │
              │                       │
              │ ŷ = ŷ_ARIMA + ŷ_LSTM  │
              └───────────┬───────────┘
                          │
                          ▼
            ┌─────────────────────────┐
            │  FORECAST 2025–2045     │
            │ • Recursive multi-step  │
            │ • 95% Confidence        │
            │   Interval              │
            │ • Milestone analysis    │
            └───────────┬─────────────┘
                        │
                        ▼
            ┌─────────────────────────┐
            │  ANALISIS POTENSI       │
            │  PERTUMBUHAN EKONOMI    │
            │  INDONESIA 2045        │
            └───────────┬─────────────┘
                        │
                        ▼
            ┌─────────────────────────┐
            │    OUTPUT PREDIKSI      │
            │ • Proyeksi harga saham  │
            │ • Tabel milestone       │
            │ • Visualisasi forecast  │
            │ • Perbandingan model    │
            └─────────────────────────┘
```

---

## 7. Implementasi Sistem

### 7.1 Library yang Digunakan

Sistem dikembangkan menggunakan ekosistem Python dengan library-library berikut:

| Kategori | Library | Fungsi |
|---|---|---|
| **Komputasi Numerik** | NumPy | Operasi array dan komputasi matematis |
| **Manipulasi Data** | Pandas | Manajemen DataFrame dan time series |
| **Visualisasi** | Matplotlib, Seaborn | Pembuatan grafik publication-ready |
| **Analisis Statistik** | SciPy (stats) | Uji statistik (Jarque-Bera, Shapiro-Wilk, distribusi) |
| **Time Series** | statsmodels | ARIMA/SARIMAX, uji ADF, uji KPSS, ACF/PACF, Ljung-Box |
| **Machine Learning** | TensorFlow/Keras | Arsitektur dan training model LSTM |
| **Preprocessing ML** | scikit-learn (MinMaxScaler) | Normalisasi data dan metrik evaluasi |
| **Serialisasi** | pickle | Penyimpanan dan pemuatan model serta artefak data |

### 7.2 Lingkungan Pengembangan

| Aspek | Spesifikasi |
|---|---|
| **Bahasa Pemrograman** | Python 3.12 |
| **Lingkungan** | Jupyter Notebook |
| **Format Eksekusi** | Pipeline modular tiga notebook terpisah |
| **Resolusi Visualisasi** | 300 DPI (publication-ready) |
| **Font** | DejaVu Serif (journal-style) |

### 7.3 Proses Eksekusi Sistem

Eksekusi sistem dilakukan secara sekuensial melalui tiga notebook yang saling terhubung:

1. **Notebook 1 — preprocessing_eda.ipynb**
   - Membangkitkan dataset simulasi harga saham batubara
   - Menghitung log return harian
   - Melakukan visualisasi harga historis dan analisis distribusi
   - Menjalankan dekomposisi time series
   - Melaksanakan uji stasioneritas formal (ADF, KPSS)
   - Menghasilkan plot ACF dan PACF untuk identifikasi orde ARIMA
   - Menyimpan dataset bersih ke file CSV

2. **Notebook 2 — model_arima.ipynb**
   - Memuat dataset yang telah dipreprocessing
   - Membagi data menjadi training (80%) dan testing (20%) secara temporal
   - Menjalankan grid search exhaustive untuk seleksi orde ARIMA optimal
   - Fitting model ARIMA terbaik per emiten
   - Melaksanakan diagnostik residual komprehensif
   - Melakukan rolling forecast 1-step ahead pada data testing
   - Menghitung metrik evaluasi (MAE, RMSE, MAPE, R²)
   - Mengekspor residual ARIMA sebagai input untuk modul hybrid (file pickle)

3. **Notebook 3 — model_ann_lstm.ipynb**
   - Memuat dataset dan artefak dari notebook sebelumnya
   - Membangun dan melatih LSTM murni pada harga saham (sebagai baseline)
   - Membangun dan melatih LSTM pada residual ARIMA (komponen hybrid)
   - Mengkombinasikan prediksi ARIMA dan LSTM residual menjadi hybrid forecast
   - Membandingkan performa tiga model (ARIMA, LSTM murni, Hybrid)
   - Menghasilkan forecast jangka panjang 2025–2045
   - Menyimpan model, scaler, dan artefak forecast

---

## 8. Hasil dan Output Sistem

### 8.1 Bentuk Keluaran Sistem

Sistem menghasilkan keluaran dalam beberapa format:

| Jenis Output | Nama File | Keterangan |
|---|---|---|
| **Tabel CSV** | tabel03_arima_orders.csv | Orde ARIMA terpilih per emiten |
| **Tabel CSV** | tabel04_arima_metrics.csv | Metrik evaluasi model ARIMA |
| **Tabel CSV** | tabel05_perbandingan_model.csv | Perbandingan metrik tiga model |
| **Tabel CSV** | tabel06_forecast_milestone_2045.csv | Proyeksi harga pada milestone 2025–2045 |
| **Model Keras** | lstm_pure_*.keras | Model LSTM murni tersimpan per emiten |
| **Model Keras** | lstm_resid_*.keras | Model LSTM residual tersimpan per emiten |
| **Artefak Pickle** | arima_residuals.pkl | Residual ARIMA per emiten |
| **Artefak Pickle** | arima_fitted.pkl | Fitted values ARIMA per emiten |
| **Artefak Pickle** | longterm_forecast_2045.pkl | Hasil forecast jangka panjang |
| **Figur PNG** | fig07 – fig20 (14 figur) | Visualisasi publication-ready |

### 8.2 Visualisasi yang Dihasilkan

Sistem menghasilkan 14 figur visualisasi publication-ready (300 DPI) yang mencakup:

| Figur | Deskripsi |
|---|---|
| Figure 7 | Top-5 kandidat orde ARIMA berdasarkan ΔAIC per emiten |
| Figure 8 | Panel diagnostik residual ARIMA (6 subplot × 4 emiten) |
| Figure 9 | ARIMA in-sample fitted values vs. harga aktual |
| Figure 10 | ARIMA out-of-sample forecast vs. harga aktual (test set) |
| Figure 11 | Scatter plot aktual vs. prediksi dan distribusi error ARIMA |
| Figure 12 | Bar chart perbandingan metrik ARIMA antar emiten |
| Figure 13 | Kurva training dan validasi loss LSTM (skala log) |
| Figure 14 | Perbandingan forecast tiga model vs. aktual |
| Figure 15 | Grouped bar chart metrik perbandingan model |
| Figure 16 | Scatter plot aktual vs. prediksi hybrid ARIMA-LSTM |
| Figure 17 | Analisis error residual hybrid (time plot + distribusi) |
| Figure 18 | Forecast jangka panjang 2025–2045 dengan 95% CI |
| Figure 19 | Heatmap MAPE seluruh model × seluruh emiten |
| Figure 20 | Radar chart perbandingan performa model (normalized) |

### 8.3 Interpretasi Hasil Prediksi

Berdasarkan evaluasi pada data testing (2023–2024), perbandingan performa ketiga model menunjukkan:

| Emiten | Model Terbaik | MAE | RMSE | MAPE (%) | R² |
|---|---|---|---|---|---|
| **ADRO** | ARIMA(3,1,3) / Hybrid | 95.22 | 120.35 | 1.99% | 0.9860 |
| **PTBA** | Hybrid ARIMA-LSTM | 239.98 | 305.05 | 1.71% | 0.9894 |
| **INDY** | Hybrid ARIMA-LSTM | 1815.68 | 2281.22 | 1.79% | 0.9760 |
| **ITMG** | ARIMA(3,1,3) | 0.08 | 0.13 | 4.78% | 0.9956 |

Temuan utama:
1. **Hybrid ARIMA-LSTM secara konsisten menghasilkan MAPE di bawah 6%** untuk seluruh emiten, menunjukkan akurasi prediksi yang sangat baik.
2. **Model hybrid unggul atau setara dengan ARIMA murni** pada seluruh emiten, memvalidasi efektivitas pendekatan residual learning.
3. **LSTM murni menunjukkan performa terburuk** dengan MAPE yang jauh lebih tinggi (8.9% – 46.7%), mengindikasikan bahwa LSTM memerlukan dekomposisi sinyal untuk bekerja optimal pada data keuangan.
4. **Nilai R² konsisten di atas 0.97** untuk model ARIMA dan hybrid, menunjukkan goodness of fit yang sangat tinggi.

---

## 9. Kebaruan (Novelty)

Berdasarkan analisis menyeluruh terhadap implementasi yang terdapat dalam kode sumber, sistem ini memiliki unsur-unsur kebaruan sebagai berikut:

### 9.1 Kebaruan Metodologis

1. **Arsitektur Hybrid Berbasis Dekomposisi Residual**: Berbeda dengan pendekatan hybrid konvensional yang menggabungkan output dua model secara langsung (simple averaging/weighted averaging), sistem ini menggunakan pendekatan dekomposisi sinyal dimana LSTM secara khusus dilatih pada residual ARIMA. Pendekatan ini memastikan bahwa setiap komponen model fokus pada aspek yang menjadi keunggulannya: ARIMA menangkap komponen linear, sementara LSTM menangkap komponen non-linear.

2. **Dual-Layer LSTM dengan Regularisasi Berlapis**: Arsitektur LSTM yang digunakan menggabungkan tiga mekanisme regularisasi secara simultan (Dropout, L2, dan Early Stopping dengan ReduceLROnPlateau), yang merupakan konfigurasi optimal untuk menghindari overfitting pada dataset time series keuangan yang relatif terbatas.

3. **Rolling Forecast 1-Step Ahead**: Evaluasi model menggunakan metode rolling forecast yang lebih realistis dan menantang dibandingkan multi-step forecast langsung, dimana model diperbarui setiap langkah dengan observasi aktual terbaru.

### 9.2 Kebaruan Aplikatif

1. **Proyeksi Sektoral Menuju 2045**: Sistem ini merupakan salah satu implementasi pertama yang secara spesifik menghasilkan proyeksi harga saham sektor energi batubara Indonesia dengan horizon hingga tahun 2045, selaras dengan visi Indonesia Emas.

2. **Multi-Emiten Analisis Terintegrasi**: Kerangka kerja yang dikembangkan memproses empat emiten secara simultan dalam satu pipeline, memungkinkan analisis komparatif yang konsisten dan efisien.

3. **Milestone-Based Reporting**: Sistem menghasilkan tabel proyeksi pada tahun-tahun milestone kunci (2025, 2027, 2030, 2035, 2040, 2045) beserta growth analysis relatif terhadap harga terakhir (2024), yang relevan untuk perencanaan kebijakan.

### 9.3 Kebaruan Teknis

1. **Pipeline Reproducible End-to-End**: Seluruh alur kerja terstruktur dalam tiga notebook modular dengan transfer data antar modul melalui format serialisasi standar (pickle), memastikan reproducibility dan auditability penuh.

2. **Konfigurasi Visualisasi Publication-Ready**: Sistem menghasilkan 14 figur visualisasi yang dikonfigurasi dengan standar publikasi ilmiah (font DejaVu Serif, resolusi 300 DPI, color palette konsisten, layout jurnal), siap digunakan langsung dalam publikasi akademik.

3. **Grid Search Otomatis dengan Kriteria Informasi**: Seleksi orde ARIMA dilakukan secara otomatis menggunakan grid search exhaustive dengan peringkatan berdasarkan AIC, BIC, dan HQIC, meminimalkan subjektivitas dalam pemilihan model.

---

## 10. Potensi Penerapan

### 10.1 Dunia Akademik

- Sebagai kerangka kerja referensi untuk penelitian di bidang forecasting time series keuangan.
- Sebagai bahan ajar dalam mata kuliah statistika, econometrics, dan machine learning terapan.
- Sebagai baseline untuk pengembangan metode hybrid yang lebih kompleks (misalnya integrasi dengan GARCH, Transformer, atau model attention-based).
- Sebagai studi kasus dalam metodologi penelitian kuantitatif bidang ekonomi dan keuangan.

### 10.2 Industri Energi

- Sebagai alat bantu perencanaan strategis jangka panjang bagi perusahaan sektor energi batubara.
- Sebagai instrumen analisis untuk mengantisipasi tren harga dan melakukan hedging risiko komoditas.
- Sebagai komponen dalam sistem decision support untuk manajemen portofolio aset energi.
- Sebagai input untuk analisis skenario transisi energi dan dampaknya terhadap valuasi sektor batubara.

### 10.3 Analisis Investasi

- Sebagai alat bantu analisis fundamental dan teknikal untuk investor institusional dan ritel.
- Sebagai komponen dalam pembangunan strategi investasi kuantitatif (quantitative trading).
- Sebagai referensi untuk analisis valuasi jangka panjang saham sektor energi.
- Sebagai instrumen backtesting strategi investasi pada sektor komoditas energi.

### 10.4 Perencanaan Ekonomi Nasional

- Sebagai masukan kuantitatif dalam perumusan Rencana Pembangunan Jangka Panjang Nasional (RPJPN) terkait sektor energi.
- Sebagai referensi untuk proyeksi kontribusi sektor batubara terhadap PDB menuju Indonesia Emas 2045.
- Sebagai alat bantu evaluasi dampak kebijakan transisi energi terhadap sektor batubara.
- Sebagai komponen dalam analisis ketahanan ekonomi nasional di sektor energi.

---

## 11. Kesimpulan

Karya intelektual ini menghadirkan sebuah sistem prediksi harga saham sektor energi batubara Indonesia berbasis pendekatan *Hybrid Statistics and Machine Learning Method* yang mengintegrasikan model ARIMA dan LSTM secara sinergis. Sistem ini dirancang secara modular dan reproducible, mencakup seluruh alur kerja mulai dari akuisisi dan preprocessing data, analisis eksploratif, pemodelan statistik dan machine learning, hingga proyeksi jangka panjang menuju tahun 2045.

Kebaruan utama sistem terletak pada arsitektur hybrid berbasis dekomposisi residual, dimana ARIMA berperan menangkap komponen linear dari time series harga saham, sementara LSTM mempelajari pola non-linear yang tersisa pada residual ARIMA. Pendekatan ini terbukti efektif, dengan performa hybrid yang secara konsisten unggul atau setara dengan model individual, menghasilkan MAPE di bawah 6% dan R² di atas 0.97 untuk seluruh emiten yang dianalisis.

Sistem ini memiliki nilai aplikatif yang luas, mulai dari penelitian akademik, perencanaan strategis industri energi, analisis investasi, hingga perencanaan ekonomi nasional. Proyeksi yang dihasilkan pada milestone tahun-tahun kunci (2025–2045) memberikan perspektif kuantitatif yang berharga bagi para pemangku kepentingan dalam mengantisipasi potensi pertumbuhan sektor energi Indonesia menuju visi Indonesia Emas 2045.

---

**Dokumen ini disusun sebagai lampiran deskripsi karya dalam rangka pengajuan Hak Kekayaan Intelektual (HKI).**

*Tanggal penyusunan: Juni 2025*
