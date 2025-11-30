# 📈 Prediksi Harga Saham Harian Menggunakan LSTM & GRU

**Mini Project – Bagus Darmawan**
**Universitas Bhayangkara Jakarta Raya**

---

## 🔍 Deskripsi Singkat

Project ini bertujuan untuk memprediksi harga saham harian menggunakan model Deep Learning seperti **LSTM**, **GRU**, dan **Hybrid LSTM+GRU** yang telah dilatih menggunakan data historis saham dari Yahoo Finance.

Selain regresi untuk memprediksi harga, project ini juga menyediakan model **klasifikasi arah 5 hari ke depan (Naik / Tidak Naik)** menggunakan pendekatan sequence.

Project ini di-deploy menggunakan **Streamlit Cloud**, sehingga dapat digunakan secara interaktif melalui browser.

---

## 🚀 Link Aplikasi & Notebook

### 🔗 Aplikasi Streamlit

👉 [https://prediksi-saham-menggunakan-lstm-dan-gru.streamlit.app/](https://prediksi-saham-menggunakan-lstm-dan-gru.streamlit.app/)

### 🔗 Notebook Google Colab

👉 [https://colab.research.google.com/drive/1S9ttFNq6L2kbGOu4l5viN-UG2eXnQLqJ?usp=sharing](https://colab.research.google.com/drive/1S9ttFNq6L2kbGOu4l5viN-UG2eXnQLqJ?usp=sharing)

---

## 📦 Fitur Utama Aplikasi

### ✔ **1. Visualisasi Data Historis**

* OHLC + Volume
* Indikator teknikal:

  * SMA10, SMA30
  * EMA10
  * RSI14
  * MACD & Signal
  * Bollinger Bands

### ✔ **2. Prediksi Harga 1 Hari ke Depan**

* Model LSTM
* Model GRU
* Baseline Naive (harga besok = harga hari ini)
* Perbandingan dalam bentuk grafik bar

### ✔ **3. Prediksi Multi-step 5 Hari (Hybrid LSTM+GRU)**

* Model hybrid memprediksi 5 titik harga sekaligus
* Visualisasi kurva prediksi vs baseline

### ✔ **4. Prediksi Arah (Naik/Turun) 5 Hari**

* Model klasifikasi LSTM/GRU untuk arah
* Probabilitas naik
* Simulasi sederhana (Buy/Hold based on model)

### ✔ **5. Catatan & Analisis Backtest**

* Penjelasan mengapa baseline naive sulit dikalahkan
* Link ke notebook untuk evaluasi lengkap

---

## 🧠 Model yang Digunakan

### 🔹 **LSTM (Long Short-Term Memory)**

Model RNN yang mampu menangkap pola jangka panjang.

### 🔹 **GRU (Gated Recurrent Unit)**

Lebih ringan dari LSTM dan sering performa mirip.

### 🔹 **Hybrid LSTM+GRU**

Model kombinasi untuk multi-step forecasting (t+1 s/d t+5).

### 🔹 **Klasifikasi Arah (Binary Classification)**

Memastikan apakah harga t+5 lebih tinggi dari t.

---

## 📉 Insight Utama dari Hasil Evaluasi Model

* Baseline *naive* memiliki RMSE sekitar **146** → ini sangat rendah.
* Model LSTM, GRU, Hybrid menghasilkan RMSE **lebih besar**, artinya:
  👉 **Harga saham harian cenderung random walk, sangat sulit dikalahkan.**
* Model masih berguna untuk edukasi dan eksplorasi pola historis.
* Model klasifikasi memberikan probabilitas arah, namun tetap noisy.

---

## 🏗 Struktur Project

```
/
├── app.py                     # Aplikasi Streamlit
├── lstm_stock_model.h5        # Model regresi 1 hari
├── gru_stock_model.h5         # Model regresi 1 hari
├── hybrid_5day_model.h5       # Model hybrid multi-step
├── direction_5day_model.h5    # Model klasifikasi arah
├── feature_scaler_stock.save
├── target_scaler_stock.save
├── cls_feature_scaler.save
├── logo_ubhara.png            # Logo kampus (untuk sidebar)
└── README.md
```

---

## 🧩 Cara Menjalankan Secara Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## ⚠️ Disclaimer

**Aplikasi ini dibuat untuk keperluan edukasi.
Tidak diperuntukkan sebagai rekomendasi investasi atau trading.**


