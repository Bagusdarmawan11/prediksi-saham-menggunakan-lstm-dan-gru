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
    lstm_model = tf.keras.models.load_model(
        "lstm_stock_model.h5",
        compile=False
    )
    gru_model = tf.keras.models.load_model(
        "gru_stock_model.h5",
        compile=False
    )
    hybrid_model = tf.keras.models.load_model(
        "hybrid_5day_model.h5",
        compile=False
    )
    direction_model = tf.keras.models.load_model(
        "direction_5day_model.h5",
        compile=False
    )

    feature_scaler = joblib.load("feature_scaler_stock.save")
    target_scaler = joblib.load("target_scaler_stock.save")
    cls_feature_scaler = joblib.load("cls_feature_scaler.save")

    return (
        lstm_model,
        gru_model,
        hybrid_model,
        direction_model,
        feature_scaler,
        target_scaler,
        cls_feature_scaler,
    )




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

    data = data.reset_index()  # index → kolom pertama
    first_col = data.columns[0]  # bisa "Date", "Datetime", atau "index"
    data = data.rename(columns={first_col: "Date"})  # paksa jadi "Date"

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


def prepare_last_window_reg(df_features: pd.DataFrame, feature_scaler):
    """
    Untuk regresi (LSTM, GRU, Hybrid 5-day):
    Mengambil 60 baris terakhir fitur, scaling, dan reshape ke (1, 60, n_features).
    """
    if len(df_features) < LOOK_BACK:
        return None

    latest_features = df_features[FEATURE_COLS].values[-LOOK_BACK:]
    latest_scaled = feature_scaler.transform(latest_features)
    X_last = latest_scaled.reshape(1, LOOK_BACK, len(FEATURE_COLS))

    last_close = df_features["Close"].iloc[-1]

    return X_last, float(last_close)


def prepare_last_window_cls(df_features: pd.DataFrame, cls_feature_scaler):
    """
    Untuk klasifikasi arah 5 hari:
    Sama seperti regresi, tapi pakai scaler klasifikasi.
    """
    if len(df_features) < LOOK_BACK:
        return None

    latest_features = df_features[FEATURE_COLS].values[-LOOK_BACK:]
    latest_scaled = cls_feature_scaler.transform(latest_features)
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
st.sidebar.write("- LSTM (regresi 1 hari)")
st.sidebar.write("- GRU (regresi 1 hari)")
st.sidebar.write("- Hybrid LSTM+GRU (regresi 5 hari)")
st.sidebar.write("- LSTM+GRU (klasifikasi arah 5 hari)")
st.sidebar.write("- Naive baseline (harga kemarin)")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Catatan: ini adalah demo edukasi.\n"
    "Prediksi **bukan** rekomendasi investasi."
)


# ==========================================================
# 6. Load Model & Data
# ==========================================================

with st.spinner("Memuat model & scaler..."):
    (
        lstm_model,
        gru_model,
        hybrid_model,
        direction_model,
        feature_scaler,
        target_scaler,
        cls_feature_scaler,
    ) = load_models_and_scalers()

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

reg_last = prepare_last_window_reg(price_data_feat, feature_scaler)
cls_last = prepare_last_window_cls(price_data_feat, cls_feature_scaler)

if reg_last is None or cls_last is None:
    st.error(f"Data terlalu pendek (< {LOOK_BACK} hari) untuk membuat window prediksi.")
    st.stop()

X_last_reg, last_close_reg = reg_last
X_last_cls, last_close_cls = cls_last  # last_close_cls sama dengan last_close_reg, tapi tidak masalah


# ==========================================================
# 8. Layout Utama: Judul & Deskripsi
# ==========================================================

st.title("📈 Prediksi Harga Saham Harian dengan LSTM & GRU")

st.markdown(f"""
Aplikasi ini menggunakan beberapa model berbasis **LSTM** dan **GRU** yang telah dilatih untuk:

1. Memprediksi **harga penutupan (Close)** 1 hari ke depan.
2. Memprediksi **harga penutupan 5 hari ke depan** (multi-step).
3. Memprediksi **arah pergerakan harga dalam 5 hari ke depan** (Naik / Tidak Naik).

- Ticker: **{ticker}**
- Data historis: {start_date.strftime("%d-%m-%Y")} s/d hari ini
- Window input: **{LOOK_BACK} hari terakhir**
- Fitur: harga (Open, High, Low, Close, Volume) + indikator teknikal (SMA, EMA, RSI, MACD, Bollinger Bands)

> ⚠️ Prediksi bersifat edukatif dan **bukan** saran investasi.
""")


# ==========================================================
# 9. Tab Navigasi
# ==========================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Historis",
    "🔮 Prediksi 1 Hari",
    "📅 Prediksi 5 Hari (Hybrid)",
    "📈 Arah 5 Hari (Klasifikasi)",
    "ℹ️ Catatan & Backtest"
])


# ==========================================================
# 9a. Tab 1 – Data Historis
# ==========================================================

with tab1:
    st.subheader(f"Data Historis {ticker}")

    # Debug: tampilkan nama kolom yang tersedia
    st.caption(f"Kolom yang terdeteksi di data: {list(price_data.columns)}")

    # Coba buat grafik harga
    try:
        fig_price = px.line(
            price_data,
            x="Date",
            y="Close",
            title=f"Harga Penutupan (Close) - {ticker}",
            labels={"Date": "Tanggal", "Close": "Harga"}
        )
        st.plotly_chart(fig_price, use_container_width=True)

    except Exception as e:
        st.error(f"Gagal membuat grafik harga. Error: {e}")
        st.markdown("### Berikut cuplikan data untuk pengecekan:")
        st.dataframe(price_data.head(), use_container_width=True)

    # Tampilkan data tail di bagian bawah
    st.markdown("### 5 Baris Terakhir Data")
    st.dataframe(price_data.tail(), use_container_width=True)

# ==========================================================
# 9b. Tab 2 – Prediksi 1 Hari ke Depan (Regresi)
# ==========================================================

with tab2:
    st.subheader("Prediksi Harga 1 Hari ke Depan")

    # Prediksi dalam skala ter-normalisasi
    y_pred_lstm_scaled = lstm_model.predict(X_last_reg)
    y_pred_gru_scaled = gru_model.predict(X_last_reg)

    # Inverse transform ke skala harga asli
    y_pred_lstm = target_scaler.inverse_transform(
        y_pred_lstm_scaled.reshape(-1, 1)
    )[0, 0]
    y_pred_gru = target_scaler.inverse_transform(
        y_pred_gru_scaled.reshape(-1, 1)
    )[0, 0]

    # Baseline naive
    naive_pred = last_close_reg

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Harga Terakhir (Close)", f"{last_close_reg:,.2f}")
    col_b.metric("Prediksi LSTM (besok)", f"{y_pred_lstm:,.2f}",
                 f"{y_pred_lstm - last_close_reg:,.2f}")
    col_c.metric("Prediksi GRU (besok)", f"{y_pred_gru:,.2f}",
                 f"{y_pred_gru - last_close_reg:,.2f}")

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
# 9c. Tab 3 – Prediksi 5 Hari ke Depan (Hybrid)
# ==========================================================

with tab3:
    st.subheader("Prediksi 5 Hari ke Depan (Hybrid LSTM+GRU)")

    horizon = st.slider("Pilih horizon prediksi (hari ke-)", 1, 5, 5)

    # Prediksi 5 hari dalam skala ter-normalisasi
    y_pred_5_scaled = hybrid_model.predict(X_last_reg)  # shape (1, 5)
    y_pred_5_flat = target_scaler.inverse_transform(
        y_pred_5_scaled.reshape(-1, 1)
    ).flatten()  # shape (5,)

    days = np.arange(1, 6)
    naive_path = np.full_like(days, fill_value=last_close_reg, dtype=float)

    # Tampilkan angka untuk horizon terpilih
    selected_price = y_pred_5_flat[horizon - 1]

    col1, col2 = st.columns(2)
    col1.metric(f"Prediksi Hybrid untuk hari ke-{horizon}",
                f"{selected_price:,.2f}",
                f"{selected_price - last_close_reg:,.2f}")
    col2.metric("Harga terakhir (acuan Naive)",
                f"{last_close_reg:,.2f}")

    # Plot jalur 5 hari ke depan
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(
        x=days,
        y=y_pred_5_flat,
        mode="lines+markers",
        name="Prediksi Hybrid"
    ))
    fig_forecast.add_trace(go.Scatter(
        x=days,
        y=naive_path,
        mode="lines",
        name="Naive (harga sekarang)",
        line=dict(dash="dash")
    ))
    fig_forecast.update_layout(
        title=f"Prediksi 5 Hari ke Depan untuk {ticker}",
        xaxis_title="Hari ke-",
        yaxis_title="Perkiraan Harga Penutupan"
    )
    st.plotly_chart(fig_forecast, use_container_width=True)

    st.markdown("""
**Catatan:**

- Model hybrid memprediksi **5 titik harga ke depan sekaligus** (t+1 ... t+5).
- Garis putus-putus menunjukkan baseline *naive* (harga selalu sama dengan hari ini).
- Performa nyata hybrid vs naive sudah dianalisis di notebook, dan untuk data harian biasanya naive masih sangat kuat.
""")


# ==========================================================
# 9d. Tab 4 – Prediksi Arah 5 Hari ke Depan (Klasifikasi)
# ==========================================================

with tab4:
    st.subheader("Prediksi Arah 5 Hari ke Depan (Naik / Tidak Naik)")

    # Probabilitas naik (label 1)
    proba_up = direction_model.predict(X_last_cls).flatten()[0]  # antara 0..1

    threshold = st.slider(
        "Ambang probabilitas untuk sinyal beli (%)",
        min_value=50,
        max_value=90,
        value=60,
        step=1
    )
    th = threshold / 100.0

    # Keputusan sederhana
    if proba_up >= th:
        signal = "BUY (sinyal beli)"
        color = "green"
    elif proba_up <= 1 - th:
        signal = "SELL / AVOID"
        color = "red"
    else:
        signal = "NEUTRAL / NO ACTION"
        color = "orange"

    col1, col2 = st.columns(2)
    col1.metric("Probabilitas harga 5 hari lagi LEBIH TINGGI daripada hari ini",
                f"{proba_up * 100:.2f}%")
    col2.markdown(f"### Sinyal sederhana: :{color}[{signal}]")

    st.markdown("""
**Penjelasan:**

- Model klasifikasi dilatih untuk memprediksi apakah harga **5 hari ke depan** akan
  **lebih tinggi** daripada harga hari ini.
- Probabilitas di atas hanyalah estimasi statistika dari model, **bukan kepastian**.
- Aturan sinyal di atas (mis. beli jika probabilitas > 60%) hanyalah contoh strategi
  yang sangat sederhana dan *tidak* mempertimbangkan risiko, biaya transaksi, dsb.
""")

    # Visualisasikan probabilitas sebagai gauge / bar
    fig_prob = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba_up * 100,
        title={"text": "Probabilitas Naik (5 hari)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "royalblue"},
            "steps": [
                {"range": [0, 50], "color": "#ffcccc"},
                {"range": [50, 70], "color": "#fff3cd"},
                {"range": [70, 100], "color": "#d4edda"},
            ],
        }
    ))
    st.plotly_chart(fig_prob, use_container_width=True)


# ==========================================================
# 9e. Tab 5 – Catatan & Backtest
# ==========================================================

with tab5:
    st.subheader("Catatan Penting & Backtest (Dari Notebook)")

    st.markdown("""
### Kenapa model sering kalah dari baseline *naive*?

Dari eksperimen di notebook:

- Baseline *naive* (harga besok = harga hari ini) memiliki RMSE sekitar **146**.
- Model LSTM, GRU, dan Hybrid memiliki RMSE jauh lebih tinggi.
- Hal ini konsisten dengan teori bahwa **harga saham harian cenderung random walk**,
  sehingga sulit diprediksi jauh lebih baik daripada strategi naive.

Di aplikasi ini, fokusnya adalah pada **eksplorasi interaktif**:

- Melihat bagaimana model bereaksi terhadap 60 hari terakhir.
- Membandingkan prediksi model vs baseline.
- Melihat probabilitas arah pergerakan 5 hari.

### Backtest Lengkap

Backtest numerik (RMSE, MAE, ekspanding-window CV, dsb.)
telah dilakukan di notebook pengembangan model.

Jika ingin, kamu bisa:
1. Mengekspor hasil prediksi backtest ke CSV dari notebook.
2. Memuat CSV tersebut di sini dan menambahkan grafik perbandingan aktual vs prediksi.
""")

    st.info(
        "Ingat: semua output di aplikasi ini bersifat edukatif, "
        "bukan rekomendasi beli/jual saham."
    )


# ==========================================================
# 10. Footer
# ==========================================================

st.markdown("---")
st.caption(
    "Dikembangkan sebagai mini project prediksi harga saham harian menggunakan LSTM, GRU, "
    "model hybrid multi-step, dan klasifikasi arah. "
    "Prediksi hanya untuk keperluan edukasi dan eksplorasi, bukan rekomendasi investasi."
)
