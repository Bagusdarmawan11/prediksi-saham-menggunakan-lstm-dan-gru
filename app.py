import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import tensorflow as tf

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ==========================================================
# 1. Konfigurasi Halaman
# ==========================================================

st.set_page_config(
    page_title="Prediksi Harga Saham Harian - LSTM & GRU",
    page_icon="📈",
    layout="wide",
)


# ==========================================================
# 2. Fungsi Utilitas: Load Model & Scaler (di-cache)
# ==========================================================

@st.cache_resource
def load_models_and_scalers():
    lstm_model = tf.keras.models.load_model("lstm_stock_model.h5")
    gru_model = tf.keras.models.load_model("gru_stock_model.h5")

    feature_scaler = joblib.load("feature_scaler_stock.save")
    target_scaler = joblib.load("target_scaler_stock.save")

    return lstm_model, gru_model, feature_scaler, target_scaler


# ==========================================================
# 3. Fungsi: Ambil Data Harga & Buat Fitur Teknikal
# ==========================================================

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghasilkan dataframe fitur yang sama seperti di notebook:
    Open, High, Low, Close, Volume + indikator teknikal.
    """
    df = df.copy()

    # Moving Averages
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_30"] = df["Close"].rolling(window=30).mean()
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()

    # RSI 14
    window_rsi = 14
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window_rsi).mean()
    avg_loss = loss.rolling(window=window_rsi).mean()
    rs = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands (20, 2 std)
    window_bb = 20
    bb_middle = df["Close"].rolling(window=window_bb).mean()
    bb_std = df["Close"].rolling(window=window_bb).std()
    df["BB_middle"] = bb_middle
    df["BB_upper"] = bb_middle + 2 * bb_std
    df["BB_lower"] = bb_middle - 2 * bb_std

    return df


@st.cache_data
def load_price_data(ticker: str, start_date: dt.date):
    data = yf.download(ticker, start=start_date, end=dt.date.today())
    if data.empty:
        return None
    data = data.reset_index()
    data.rename(columns={"Date": "Date"}, inplace=True)
    return data


# ==========================================================
# 4. Fungsi: Siapkan Window Terakhir untuk Prediksi
# ==========================================================

FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA_10", "SMA_30", "EMA_10",
    "RSI_14",
    "MACD", "MACD_signal",
    "BB_middle", "BB_upper", "BB_lower",
]

LOOK_BACK = 60  # sama seperti di notebook


def prepare_last_window(df_features: pd.DataFrame, feature_scaler):
    """
    Mengambil 60 baris terakhir dari df_features,
    men-scale dengan feature_scaler, dan mengubah menjadi shape (1, 60, n_features).
    """
    if len(df_features) < LOOK_BACK:
        return None

    latest_features = df_features[FEATURE_COLS].values[-LOOK_BACK:]
    latest_scaled = feature_scaler.transform(latest_features)
    X_last = latest_scaled.reshape(1, LOOK_BACK, len(FEATURE_COLS))

    last_close = df_features["Close"].iloc[-1]

    return X_last, float(last_close)


# ==========================================================
# 5. Sidebar: Pengaturan Aplikasi
# ==========================================================

st.sidebar.title("⚙️ Pengaturan")

default_ticker = "BBCA.JK"
ticker = st.sidebar.text_input("Ticker saham (format Yahoo Finance)", default_ticker)

years_back = st.sidebar.slider("Ambil data historis (tahun ke belakang)", 1, 10, 5)

start_date = dt.date.today() - dt.timedelta(days=365 * years_back)

st.sidebar.markdown("---")
st.sidebar.write("Model yang digunakan:")
st.sidebar.write("- LSTM (1-day forecast)")
st.sidebar.write("- GRU (1-day forecast)")
st.sidebar.write("- Naive baseline (harga kemarin)")

st.sidebar.markdown("---")
st.sidebar.caption("Catatan: ini adalah demo edukasi.\n"
                   "Prediksi **bukan** rekomendasi investasi.")


# ==========================================================
# 6. Load Model & Data
# ==========================================================

with st.spinner("Memuat model & scaler..."):
    lstm_model, gru_model, feature_scaler, target_scaler = load_models_and_scalers()

with st.spinner(f"Mengambil data {ticker} dari Yahoo Finance..."):
    price_data = load_price_data(ticker, start_date)

if price_data is None or price_data.empty:
    st.error("Tidak dapat mengambil data harga. Coba ganti ticker atau rentang waktu.")
    st.stop()


# ==========================================================
# 7. Buat Fitur & Window untuk Prediksi
# ==========================================================

price_data_feat = price_data.set_index("Date")
price_data_feat = add_technical_features(price_data_feat)

# Drop baris yang masih mengandung NaN dari indikator teknikal
price_data_feat = price_data_feat.dropna().copy()

X_last_and_close = prepare_last_window(price_data_feat, feature_scaler)

if X_last_and_close is None:
    st.error(f"Data terlalu pendek (< {LOOK_BACK} hari) untuk membuat window prediksi.")
    st.stop()

X_last, last_close = X_last_and_close


# ==========================================================
# 8. Layout Utama: Judul & Deskripsi
# ==========================================================

st.title("📈 Prediksi Harga Saham Harian dengan LSTM & GRU")

st.markdown(f"""
Aplikasi ini menggunakan model **LSTM** dan **GRU** yang telah dilatih
untuk memprediksi **harga penutupan (Close)** 1 hari ke depan.

- Ticker: **{ticker}**
- Data historis: {start_date.strftime("%d-%m-%Y")} s/d hari ini
- Window input: **{LOOK_BACK} hari terakhir**
- Fitur: harga (Open, High, Low, Close, Volume) + indikator teknikal (SMA, EMA, RSI, MACD, Bollinger Bands)

> ⚠️ Prediksi bersifat edukatif dan **bukan** saran investasi.
""")


# ==========================================================
# 9. Tab Navigasi
# ==========================================================

tab1, tab2, tab3 = st.tabs(["📊 Data Historis", "🔮 Prediksi 1 Hari ke Depan", "📈 Backtest Sederhana"])


# ==========================================================
# 9a. Tab 1 – Data Historis
# ==========================================================

with tab1:
    st.subheader(f"Data Historis {ticker}")

    fig_price = px.line(
        price_data,
        x="Date",
        y="Close",
        title=f"Harga Penutupan (Close) - {ticker}",
        labels={"Date": "Tanggal", "Close": "Harga"}
    )
    st.plotly_chart(fig_price, use_container_width=True)

    st.markdown("### Contoh 5 baris terakhir")
    st.dataframe(price_data.tail(), use_container_width=True)


# ==========================================================
# 9b. Tab 2 – Prediksi 1 Hari ke Depan
# ==========================================================

with tab2:
    st.subheader("Prediksi Harga 1 Hari ke Depan")

    # Prediksi dalam skala ter-normalisasi
    y_pred_lstm_scaled = lstm_model.predict(X_last)
    y_pred_gru_scaled = gru_model.predict(X_last)

    # Inverse transform ke skala harga asli
    y_pred_lstm = target_scaler.inverse_transform(y_pred_lstm_scaled.reshape(-1, 1))[0, 0]
    y_pred_gru = target_scaler.inverse_transform(y_pred_gru_scaled.reshape(-1, 1))[0, 0]

    # Baseline naive
    naive_pred = last_close

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Harga Terakhir (Close)", f"{last_close:,.2f}")
    col_b.metric("Prediksi LSTM (besok)", f"{y_pred_lstm:,.2f}",
                 f"{y_pred_lstm - last_close:,.2f}")
    col_c.metric("Prediksi GRU (besok)", f"{y_pred_gru:,.2f}",
                 f"{y_pred_gru - last_close:,.2f}")

    st.markdown("#### Perbandingan Prediksi")
    df_pred = pd.DataFrame({
        "Model": ["Naive (harga hari ini)", "LSTM", "GRU"],
        "Predicted_Close": [naive_pred, y_pred_lstm, y_pred_gru]
    })

    fig_bar = px.bar(
        df_pred,
        x="Model",
        y="Predicted_Close",
        text="Predicted_Close",
        labels={"Predicted_Close": "Perkiraan Harga Penutupan"},
        title="Perbandingan Prediksi 1 Hari ke Depan"
    )
    fig_bar.update_traces(texttemplate='%{text:.2f}', textposition="outside")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("""
**Interpretasi ringkas:**

- Baseline *naive* memprediksi harga besok = harga hari ini.
- LSTM & GRU mencoba mengoreksi berdasarkan pola 60 hari terakhir dan indikator teknikal.
- Karena harga saham harian sangat noisy, sering kali baseline naive masih sangat sulit dikalahkan.
""")


# ==========================================================
# 9c. Tab 3 – Backtest Sederhana di Data Historis
# ==========================================================

with tab3:
    st.subheader("Backtest Sederhana (Hanya Visual)")

    st.markdown("""
Backtest ini **tidak** melakukan retraining model di Streamlit.
Sebagai gantinya, kita hanya menampilkan lagi **grafik hasil evaluasi**
dari model yang sudah dilatih di notebook:

- Garis biru: harga aktual
- Garis hijau: GRU (biasanya sedikit lebih baik dari LSTM)
- Garis oranye: LSTM
- Garis merah putus: baseline naive

Untuk backtest yang lebih mendalam (RMSE/MAE dsb.), analisis sudah dilakukan di notebook.
Di aplikasi ini fokusnya adalah eksplorasi visual & prediksi harian.
""")

    st.info(
        "Untuk menampilkan grafik backtest yang sama seperti di notebook, "
        "kamu bisa mengekspor data prediksi ke CSV dari notebook, lalu memuatnya di sini "
        "sebagai tambahan. Saat ini tab ini hanya berfungsi sebagai placeholder penjelasan."
    )


# ==========================================================
# 10. Footer
# ==========================================================

st.markdown("---")
st.caption(
    "Dikembangkan sebagai mini project prediksi harga saham harian menggunakan LSTM & GRU. "
    "Prediksi hanya untuk keperluan edukasi dan eksplorasi, bukan rekomendasi investasi."
)
