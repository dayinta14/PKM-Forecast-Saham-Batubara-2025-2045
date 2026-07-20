# BAB 2 METODE PENELITIAN

---

## 2.1 Kerangka Kerja Penelitian

Penelitian ini menerapkan pendekatan prediktif hibrida (*hybrid predictive modeling*) yang mengintegrasikan metode statistik klasik *Autoregressive Integrated Moving Average* (ARIMA) dengan metode *machine learning* berbasis *Artificial Neural Network* (ANN), yaitu *Long Short-Term Memory* (LSTM). Tujuan utamanya adalah melakukan proyeksi tren harga saham sektor energi batubara di Bursa Efek Indonesia (BEI/IDX) untuk rentang tahun 2025 hingga 2045 sebagai indikator proksi pertumbuhan ekonomi nasional.

Kerangka kerja penelitian ini dirancang secara sistematis melalui 6 tahapan utama yang ditunjukkan pada alur kerja berikut:

```
[1. Akuisisi Data] ➔ [2. Pra-pemrosesan & EDA] ➔ [3. Pemodelan Linear (ARIMA)]
                                                        │
[6. Dashboard Interaktif] ◄── [5. Evaluasi Model] ◄── [4. Pemodelan Non-Linear (LSTM)]
                                                        │
                                               (Hybrid ARIMA-LSTM)
```

1. **Akuisisi Data (*Data Acquisition*)**: Mengumpulkan data historis harga saham harian (*Daily Closing Price*) emiten batubara utama melalui API `yfinance`.
2. **Pra-pemrosesan Data & EDA (*Preprocessing & Exploratory Data Analysis*)**: Pembersihan data, pengisian *missing values*, uji stasioneritas (*Augmented Dickey-Fuller*), transformasi skala, dan ekstraksi fitur deret waktu.
3. **Pemodelan Linear ARIMA**: Mengidentifikasi orde $(p, d, q)$ optimal menggunakan kriteria informasi AIC/BIC, fitting model ARIMA, dan mengekstrak residual linear ($\epsilon_t$).
4. **Pemodelan Non-Linear LSTM (Residual Modeling)**: Membangun arsitektur jaringan LSTM untuk mempelajari pola non-linear dan ketidakpastian temporal dari residual ARIMA.
5. **Evaluasi dan Komparasi Model**: Menguji performa model ARIMA murni, LSTM murni, dan Hybrid ARIMA-LSTM menggunakan metrik evaluasi standar (MAE, RMSE, MAPE, $R^2$).
6. **Implementasi Dashboard Interaktif**: Mengintegrasikan hasil prediksi jangka panjang 2025–2045 ke dalam aplikasi web berbasis JavaScript dan Chart.js untuk visualisasi publik dan pemangku kepentingan.

---

## 2.2 Dataset Penelitian

Dataset yang digunakan dalam penelitian ini berupa data harga saham harian penutupan (*Adjusted Closing Price*) dari 4 emiten sektor energi batubara terkemuka di Bursa Efek Indonesia (BEI) yang terdaftar dalam indeks komoditas utama:

| Ticker Saham | Nama Perusahaan | Rentang Data | Frekuensi | Sumber Data |
| :--- | :--- | :--- | :--- | :--- |
| **ADRO** | PT Adaro Energy Indonesia Tbk | 2010 – 2024 | Harian (*Daily*) | Yahoo Finance API (`yfinance`) |
| **PTBA** | PT Bukit Asam Tbk | 2010 – 2024 | Harian (*Daily*) | Yahoo Finance API (`yfinance`) |
| **INDY** | PT Indika Energy Tbk | 2010 – 2024 | Harian (*Daily*) | Yahoo Finance API (`yfinance`) |
| **ITMG** | PT Indo Tambangraya Megah Tbk | 2010 – 2024 | Harian (*Daily*) | Yahoo Finance API (`yfinance`) |

### Karakteristik Data
Data time-series historis ini mencakup periode dinamika ekonomi yang krusial, seperti krisis komoditas global, siklus *supercycle* harga energi, serta periode pasca-pandemi. Struktur data harian terdiri dari atribut: *Date, Open, High, Low, Close, Adj Close,* dan *Volume*. Atribut utama yang digunakan untuk pemodelan adalah **`Adj Close`**.

---

## 2.3 Pra-pemrosesan Data

Pra-pemrosesan data dilakukan untuk memastikan data memenuhi asumsi pemodelan deret waktu dan mengoptimalkan proses pemelajaran pada jaringan saraf tiruan.

### 2.3.1 Penanganan Data Hilang dan Pembersihan
* **Deteksi Hari Libur Bursa**: Mengatasi *gap* tanggal akibat akhir pekan dan hari libur nasional menggunakan metode *forward fill* (`ffill`) dilanjutkan dengan *backward fill* (`bfill`).
* **Pembersihan Anomali**: Memeriksa nilai nol atau negatif pada harga saham.

### 2.3.2 Uji Stasioneritas Data (Augmented Dickey-Fuller Test)
Model statistik ARIMA mensyaratkan data deret waktu yang stasioner (rata-rata dan varians konstan sepanjang waktu). Stasioneritas diuji menggunakan uji **Augmented Dickey-Fuller (ADF)** dengan hipotesis:
* $H_0$: Data memiliki unit root (tidak stasioner).
* $H_1$: Data stasioner ($p \text{-value} < 0.05$).

Jika data awal tidak stasioner pada tingkat level ($d=0$), dilakukan proses deferensiasi orde pertama (*first-order differencing*, $d=1$):

$$\Delta Y_t = Y_t - Y_{t-1}$$

### 2.3.3 Transformasi Skala Data (Min-Max Scaling)
Untuk pemodelan LSTM, data harga dan residual ditransformasikan ke dalam rentang $[0, 1]$ menggunakan `MinMaxScaler` untuk mencegah *exploding/vanishing gradient* serta mempercepat konvergensi optimasi Gradient Descent:

$$X_{\text{scaled}} = \frac{X - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}}}$$

### 2.3.4 Pembentukan Struktur Windowing Sequence
Data deret waktu ditransformasikan menjadi struktur *sliding window* berpasangan $(X, y)$ untuk pembelajaran terawasi (*supervised learning*) pada LSTM:
* **Input Window ($X_t$)**: Jendela observasi historis sebanyak $k$ langkah waktu sebelumnya ($t-k, t-k+1, \dots, t-1$).
* **Target ($y_t$)**: Nilai pada langkah waktu ke-$t$.

---

## 2.4 Pemodelan Hybrid ARIMA-LSTM

Pendekatan hibrida didasarkan pada asumsi bahwa deret waktu harga saham $Y_t$ terdiri dari komponen linear $L_t$ dan komponen non-linear $N_t$:

$$Y_t = L_t + N_t$$

### 2.4.1 Komponen Linear: Model ARIMA
Model **ARIMA$(p, d, q)$** digunakan untuk memodelkan komponen linear $L_t$. Persamaan umum model ARIMA$(p, d, q)$ adalah:

$$\Phi_p(B) (1 - B)^d Y_t = \Theta_q(B) \epsilon_t$$

Di mana:
* $B$ adalah operator backshift ($B^k Y_t = Y_{t-k}$).
* $\Phi_p(B) = 1 - \sum_{i=1}^p \phi_i B^i$ adalah komponen Autoregressive (AR).
* $\Theta_q(B) = 1 + \sum_{j=1}^q \theta_j B^j$ adalah komponen Moving Average (MA).
* $d$ adalah orde diferensiasi.

Orde optimal $(p, d, q)$ dipilih berdasarkan nilai terkecil dari *Akaike Information Criterion* (AIC) dan *Bayesian Information Criterion* (BIC):

$$\text{AIC} = 2k - 2\ln(\hat{L}), \quad \text{BIC} = k\ln(n) - 2\ln(\hat{L})$$

Hasil estimasi ARIMA menghasilkan nilai prediksi linear $\hat{L}_t$ dan residual linear $e_t$:

$$e_t = Y_t - \hat{L}_t$$

### 2.4.2 Komponen Non-Linear: Model LSTM (Residual Modeling)
Residual $e_t$ yang mengandung pola ketergantungan non-linear diprediksi menggunakan jaringan **Long Short-Term Memory (LSTM)**. Arsitektur LSTM terdiri dari sel memori yang dikendalikan oleh tiga gerbang utama:

1. **Forget Gate ($f_t$)**: Menentukan informasi yang akan dibuang dari cell state:
   $$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
2. **Input Gate ($i_t$)**: Menentukan informasi baru yang disimpan ke cell state:
   $$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
   $$\tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$
3. **Cell State Update ($C_t$)**:
   $$C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$$
4. **Output Gate ($o_t$) & Hidden State ($h_t$)**:
   $$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
   $$h_t = o_t * \tanh(C_t)$$

**Spesifikasi Arsitektur Jaringan LSTM:**
* **Input Layer**: Shape $(N_{\text{samples}}, \text{look\_back}, 1)$
* **Hidden Layer 1**: LSTM Layer (50 units) + Batch Normalization + Dropout ($r = 0.2$)
* **Hidden Layer 2**: LSTM Layer (50 units) + Dropout ($r = 0.2$)
* **Output Layer**: Dense Layer (1 unit)
* **Optimizer**: Adam ($\eta = 0.001$) dengan Callback `ReduceLROnPlateau` dan `EarlyStopping` (patience = 15).

### 2.4.3 Penggabungan Prediksi (Hybrid Forecast)
Hasil prediksi non-linear residual dari LSTM ($\hat{N}_t = \hat{e}_t$) digabungkan secara aditif dengan hasil prediksi linear ARIMA ($\hat{L}_t$) untuk menghasilkan nilai prediksi final $\hat{Y}_t$:

$$\hat{Y}_{\text{Hybrid}, t} = \hat{L}_t + \hat{N}_t$$

---

## 2.5 Evaluasi Model

Performa prediksi diuji menggunakan 4 metrik evaluasi standar untuk mengukur tingkat akurasi dan kesesuaian model:

### 2.5.1 Mean Absolute Error (MAE)
Mengukur rata-rata besarnya kesalahan absolut antara nilai aktual $y_i$ dan nilai prediksi $\hat{y}_i$:

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$

### 2.5.2 Root Mean Squared Error (RMSE)
Mengukur akar dari rata-rata kuadrat kesalahan, memberikan bobot lebih besar pada kesalahan yang bernilai besar (*outliers*):

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$

### 2.5.3 Mean Absolute Percentage Error (MAPE)
Mengukur rata-rata persentase kesalahan relatif terhadap nilai aktual:

$$\text{MAPE} = \frac{100\%}{n} \sum_{i=1}^n \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$

### 2.5.4 Koefisien Determinasi ($R^2$)
Mengukur proporsi varians dari variabel dependen yang dapat dijelaskan oleh model:

$$R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$$

---

## 2.6 Implementasi Dashboard Interaktif

Untuk menyajikan hasil proyeksi jangka panjang (2025–2045) secara transparan dan mudah diakses oleh pengambil keputusan, dikembangkan **Dashboard Interaktif Web-based**.

### 2.6.1 Arsitektur Teknologi Dashboard
* **Frontend**: HTML5, Vanilla CSS3 (Custom Responsive Layout), dan JavaScript (ES6+).
* **Visualisasi Grafik**: `Chart.js` (v4.4.2) terintegrasi dengan adaptor tanggal `chartjs-adapter-date-fns` untuk rendering grafik time-series interaktif.
* **Format Data**: JSON (`data_saham.json`) yang memuat data historis, hasil fitting model, dan proyeksi *milestone* tahunan (2025, 2030, 2035, 2040, 2045).

### 2.6.2 Fitur Utama Dashboard
1. **Navigasi Tab Emiten**: Memungkinkan pengguna beralih antara 4 emiten batubara (ADRO, PTBA, INDY, ITMG).
2. **KPI Scorecard / Accuracy Cards**: Menampilkan metrik evaluasi model secara real-time (MAE, RMSE, MAPE, $R^2$) beserta status kualitas prediksi (*Exemplary / Good / Acceptable*).
3. **Interactive Forecast Plot**: Visualisasi garis interaktif dengan fitur *hover tooltip*, zoom, toggling seri data (Historis vs Prediksi Model), dan batas rentang waktu proyeksi hingga tahun 2045.
4. **Ringkasan Proyeksi Milestone**: Tabel proyeksi tren harga saham pada tahun-tahun kunci menuju Indonesia Emas 2045.
