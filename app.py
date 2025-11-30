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
    lstm_model = tf.keras.models.load_model("lstm_stock_model.h5", compile=False)
    gru_model = tf.keras.models.load_model("gru_stock_model.h5", compile=False)
    hybrid_model = tf.keras.models.load_model("hybrid_5day_model.h5", compile=False)
    direction_model = tf.keras.models.load_model("direction_5day_model.h5", compile=False)

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
    df = df.copy()

    # Moving Averages
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_30"] = df["Close"].rolling(30).mean()
    df["EMA_10"] = df["Close"].ewm(span=10).mean()

    # RSI 14
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI_14"] = 100 - 100 / (1 + rs)

    # MACD
    ema_12 = df["Close"].ewm(span=12).mean()
    ema_26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_signal"] = df["MACD"].ewm(span=9).mean()

    # Bollinger Bands
    bb_mid = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["BB_middle"] = bb_mid
    df["BB_upper"] = bb_mid + 2 * bb_std
    df["BB_lower"] = bb_mid - 2 * bb_std

    return df


@st.cache_data
def load_price_data(ticker: str, start_date: dt.date):
    """
    Ambil data dari yfinance dan rapikan kolom:
    - Flatten MultiIndex kalau ada
    - Pastikan ada Date, Open, High, Low, Close, Volume
    """
    data = yf.download(ticker, start=start_date, end=dt.date.today())
    if data.empty:
        return None

    data = data.reset_index()

    # Flatten MultiIndex → single level
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [
            "_".join([str(c) for c in col if c not in ("", None)])
            for col in data.columns
        ]

    # Rename Open_<TICKER> → Open, dst.
    for base in ["Open", "High", "Low", "Close", "Volume"]:
        col_name = f"{base}_{ticker}"
        if col_name in data.columns:
            data = data.rename(columns={col_name: base})

    # Pastikan kolom pertama bernama Date
    if data.columns[0] != "Date":
        data = data.rename(columns={data.columns[0]: "Date"})

    return data


# ==========================================================
# 4. Window untuk Prediksi
# ==========================================================

FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA_10", "SMA_30", "EMA_10",
    "RSI_14",
    "MACD", "MACD_signal",
    "BB_middle", "BB_upper", "BB_lower",
]
LOOK_BACK = 60


def prepare_last_window(df_features: pd.DataFrame, scaler):
    """
    Ambil 60 baris terakhir, scaling, reshape ke (1, 60, n_features)
    """
    if len(df_features) < LOOK_BACK:
        return None

    latest = df_features[FEATURE_COLS].values[-LOOK_BACK:]
    latest_scaled = scaler.transform(latest)
    X_last = latest_scaled.reshape(1, LOOK_BACK, len(FEATURE_COLS))

    last_close = float(df_features["Close"].iloc[-1])
    return X_last, last_close


# ==========================================================
# 5. Sidebar – Pengaturan
# ==========================================================

st.sidebar.title("⚙️ Pengaturan")

popular_tickers = [
    "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "ASII.JK",
    "TLKM.JK", "ICBP.JK", "UNVR.JK", "ANTM.JK", "MDKA.JK",
    "Custom (manual)",
]

ticker_choice = st.sidebar.selectbox(
    "Pilih Ticker Saham",
    popular_tickers,
    help="Ketik untuk mencari. Pilih 'Custom (manual)' untuk memasukkan ticker lain."
)

if ticker_choice == "Custom (manual)":
    ticker = st.sidebar.text_input(
        "Ticker (format Yahoo Finance)",
        value="BBCA.JK",
        placeholder="Contoh: BBCA.JK"
    )
else:
    ticker = ticker_choice

years_back = st.sidebar.selectbox(
    "Ambil data historis (tahun ke belakang)",
    options=[1, 2, 3, 4, 5, 7, 10],
    index=3,
    help="Rentang data yang digunakan untuk visualisasi dan fitur model."
)

start_date = dt.date.today() - dt.timedelta(days=365 * years_back)

st.sidebar.markdown("---")
st.sidebar.caption("Model digunakan: LSTM, GRU, Hybrid 5-hari, Klasifikasi arah 5-hari.")
st.sidebar.caption("⚠️ Prediksi hanya untuk edukasi – bukan rekomendasi investasi.")


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

price_data_feat = price_data.set_index("Date")
price_data_feat = add_technical_features(price_data_feat)
price_data_feat = price_data_feat.dropna().copy()

reg_last = prepare_last_window(price_data_feat, feature_scaler)
cls_last = prepare_last_window(price_data_feat, cls_feature_scaler)

if reg_last is None or cls_last is None:
    st.error(f"Data terlalu pendek (< {LOOK_BACK} hari) untuk membuat window prediksi.")
    st.stop()

X_last_reg, last_close_reg = reg_last
X_last_cls, last_close_cls = cls_last  # sama saja nilainya


# ==========================================================
# 7. Layout Utama – Judul
# ==========================================================

st.title("📈 Mini Project – Prediksi Harga Saham Harian (LSTM & GRU)")

st.markdown(f"""
Aplikasi ini dibuat sebagai mini project untuk:
- Membandingkan model **LSTM**, **GRU**, dan baseline **Naive (harga kemarin)**.
- Melihat prediksi **harga 1 hari** dan **5 hari ke depan**.
- Memprediksi **arah pergerakan 5 hari ke depan (Naik / Tidak Naik)**.

**Ticker:** `{ticker}`  
**Rentang data historis:** {start_date.strftime("%d-%m-%Y")} s/d hari ini  
**Window input model:** {LOOK_BACK} hari terakhir  
""")


# ==========================================================
# 8. Tab Navigasi
# ==========================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Historis",
    "🔮 Prediksi 1 Hari",
    "📅 Prediksi 5 Hari (Hybrid)",
    "📈 Arah 5 Hari (Klasifikasi)",
    "ℹ️ Catatan & Backtest"
])


# ==========================================================
# 9a. TAB 1 – Data Historis
# ==========================================================

with tab1:
    st.subheader("📊 Data Historis & Indikator")

    st.markdown("""
Tab ini menampilkan data historis harga saham yang digunakan model, beserta
gambaran pergerakan harga. Kamu bisa memilih rentang waktu di sidebar.
""")

    df_show = price_data.copy()
    df_show.insert(0, "No", range(1, len(df_show) + 1))

    st.caption(f"Kolom data: {list(df_show.columns)}")

    # Pilih window untuk grafik
    view_option = st.selectbox(
        "Tampilkan grafik untuk:",
        ["Seluruh data", "6 bulan terakhir", "3 bulan terakhir"],
        index=0
    )

    df_plot = df_show.copy()
    if view_option == "6 bulan terakhir":
        cutoff = df_plot["Date"].max() - pd.Timedelta(days=180)
        df_plot = df_plot[df_plot["Date"] >= cutoff]
    elif view_option == "3 bulan terakhir":
        cutoff = df_plot["Date"].max() - pd.Timedelta(days=90)
        df_plot = df_plot[df_plot["Date"] >= cutoff]

    try:
        fig_price = px.line(
            df_plot,
            x="Date",
            y="Close",
            title=f"Harga Penutupan (Close) - {ticker}",
            labels={"Date": "Tanggal", "Close": "Harga"}
        )
        st.plotly_chart(fig_price, use_container_width=True)
    except Exception as e:
        st.error(f"Gagal membuat grafik harga. Error: {e}")
        st.dataframe(df_plot.head(), use_container_width=True)

    st.markdown("### Seluruh Data Historis")
    st.dataframe(df_show, use_container_width=True)

    st.markdown("### 5 Baris Terakhir Data")
    st.dataframe(df_show.tail(), use_container_width=True)


# ==========================================================
# 9b. TAB 2 – Prediksi 1 Hari (LSTM vs GRU vs Naive)
# ==========================================================

with tab2:
    st.subheader("🔮 Prediksi Harga 1 Hari ke Depan")

    st.markdown("""
Tab ini membandingkan prediksi **harga penutupan besok** dari:
- Baseline **Naive** (harga besok = harga hari ini),
- Model **LSTM**,
- Model **GRU**.

Model dilatih pada data historis BBCA.JK dengan fitur harga + indikator teknikal.
""")

    # Prediksi dalam skala ter-normalisasi
    y_pred_lstm_scaled = lstm_model.predict(X_last_reg)
    y_pred_gru_scaled = gru_model.predict(X_last_reg)

    # Inverse ke skala harga asli
    y_pred_lstm = target_scaler.inverse_transform(
        y_pred_lstm_scaled.reshape(-1, 1)
    )[0, 0]
    y_pred_gru = target_scaler.inverse_transform(
        y_pred_gru_scaled.reshape(-1, 1)
    )[0, 0]

    naive_pred = last_close_reg

    delta_lstm = y_pred_lstm - last_close_reg
    delta_gru = y_pred_gru - last_close_reg

    col1, col2, col3 = st.columns(3)
    col1.metric("Harga Terakhir (Close)", f"{last_close_reg:,.2f}")

    col2.metric(
        "Prediksi LSTM (besok)",
        f"{y_pred_lstm:,.2f}",
        f"{delta_lstm:,.2f}"
    )

    col3.metric(
        "Prediksi GRU (besok)",
        f"{y_pred_gru:,.2f}",
        f"{delta_gru:,.2f}"
    )

    st.markdown("#### Perbandingan Prediksi")

    df_pred = pd.DataFrame({
        "Model": ["Naive (hari ini)", "LSTM", "GRU"],
        "Predicted_Close": [naive_pred, y_pred_lstm, y_pred_gru]
    })

    fig_bar = px.bar(
        df_pred,
        x="Model",
        y="Predicted_Close",
        color="Model",
        color_discrete_map={
            "Naive (hari ini)": "#ff7f0e",
            "LSTM": "#1f77b4",
            "GRU": "#2ca02c",
        },
        text="Predicted_Close",
        labels={"Predicted_Close": "Perkiraan Harga Penutupan"},
        title="Perbandingan Prediksi 1 Hari ke Depan"
    )
    fig_bar.update_traces(texttemplate='%{text:.2f}', textposition="outside")
    ymin = df_pred["Predicted_Close"].min() * 0.98
    ymax = df_pred["Predicted_Close"].max() * 1.02
    fig_bar.update_layout(yaxis_range=[ymin, ymax])
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("""
**Insight singkat:**

- Kalau bar LSTM/GRU hampir sama dengan Naive, artinya model sulit mengalahkan strategi
  "harga besok = harga hari ini".
- Hal ini umum pada data harga saham harian yang sangat noisy (mendekati random walk).
""")


# ==========================================================
# 9c. TAB 3 – Prediksi 5 Hari ke Depan (Hybrid)
# ==========================================================

with tab3:
    st.subheader("📅 Prediksi 5 Hari ke Depan (Hybrid LSTM+GRU)")

    st.markdown("""
Model **Hybrid** ini memprediksi **5 titik harga ke depan sekaligus** (t+1 s.d. t+5).
Dropdown di bawah menentukan **hari ke berapa** yang ingin kamu fokuskan.
""")

    horizon = st.selectbox(
        "Pilih horizon prediksi (hari ke-)",
        options=[1, 2, 3, 4, 5],
        index=4,
        help="Hari ke-1 = besok, Hari ke-5 = 5 hari kerja berikutnya."
    )

    y_pred_5_scaled = hybrid_model.predict(X_last_reg)
    y_pred_5_flat = target_scaler.inverse_transform(
        y_pred_5_scaled.reshape(-1, 1)
    ).flatten()

    days = np.arange(1, 6)
    naive_path = np.full_like(days, last_close_reg, dtype=float)

    selected_price = y_pred_5_flat[horizon - 1]
    delta_sel = selected_price - last_close_reg

    col1, col2 = st.columns(2)
    col1.metric(
        f"Prediksi Hybrid untuk hari ke-{horizon}",
        f"{selected_price:,.2f}",
        f"{delta_sel:,.2f}"
    )
    col2.metric("Harga terakhir (acuan Naive)", f"{last_close_reg:,.2f}")

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
**Insight singkat:**

- Garis putus-putus adalah baseline *naive* (harga konstan = hari ini).
- Kalau garis Hybrid tidak jauh dari garis Naive, berarti model belum mampu
  menangkap pola lanjutan yang kuat di luar pergerakan random jangka pendek.
""")


# ==========================================================
# 9d. TAB 4 – Klasifikasi Arah 5 Hari + Simulasi
# ==========================================================

with tab4:
    st.subheader("📈 Prediksi Arah 5 Hari ke Depan (Naik / Tidak Naik)")

    st.markdown("""
Model klasifikasi ini memprediksi probabilitas bahwa **harga 5 hari ke depan** akan
**lebih tinggi** dibanding harga hari ini.

Slider threshold di bawah hanya mengubah **aturan sinyal** (kapan BUY/SELL/NEUTRAL),
bukan mengubah nilai probabilitas model.
""")

    proba_up = direction_model.predict(X_last_cls).flatten()[0]

    threshold = st.slider(
        "Ambang probabilitas untuk sinyal BUY (%)",
        min_value=50,
        max_value=90,
        value=60,
        step=1,
        help="Misal 60% artinya model harus cukup yakin (>=0.60) baru memberi sinyal BUY."
    )
    th = threshold / 100.0

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
    col1.metric(
        "Probabilitas harga 5 hari lagi LEBIH TINGGI",
        f"{proba_up * 100:.2f}%"
    )
    col2.markdown(
        f"### Sinyal sederhana (threshold {threshold}%): "
        f":{color}[{signal}]"
    )

    # Gauge probabilitas
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

    st.markdown("### Simulasi Strategi Sederhana (Teoritis)")

    st.markdown("""
Simulasi ini **bukan backtest sungguhan**, hanya ilustrasi:

- Jika sinyal = BUY, anggap beli 1 unit di harga sekarang,
- lalu **gunakan prediksi Hybrid hari ke-5** sebagai perkiraan harga jual.
""")

    # Gunakan prediksi hybrid hari ke-5 untuk simulasi
    y_pred_5_scaled_sim = hybrid_model.predict(X_last_reg)
    y_pred_5_flat_sim = target_scaler.inverse_transform(
        y_pred_5_scaled_sim.reshape(-1, 1)
    ).flatten()
    price_5d = y_pred_5_flat_sim[4]  # hari ke-5
    gain = price_5d - last_close_reg
    gain_pct = gain / last_close_reg * 100

    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("Harga sekarang (entry)", f"{last_close_reg:,.2f}")
    col_s2.metric("Perkiraan harga 5 hari lagi (Hybrid)", f"{price_5d:,.2f}")
    col_s3.metric("Perkiraan P/L 5 hari", f"{gain:,.2f}", f"{gain_pct:,.2f}%")

    st.info("""
Ini hanya simulasi teoritis dengan **prediksi model**, bukan hasil backtest historis.
Untuk trading nyata, perlu backtest ketat, manajemen risiko, dan biaya transaksi.
""")


# ==========================================================
# 9e. TAB 5 – Catatan & Backtest
# ==========================================================

with tab5:
    st.subheader("ℹ️ Catatan Penting & Backtest (Dari Notebook)")

    st.markdown("""
### Kenapa model sulit mengalahkan baseline *naive*?

Dari eksperimen di notebook:

- Baseline *naive* (harga besok = harga hari ini) punya RMSE sekitar **146**.
- Model LSTM, GRU, dan Hybrid punya RMSE jauh lebih besar.
- Ini konsisten dengan teori bahwa **harga saham harian cenderung random walk**,
  sehingga sulit diprediksi jauh lebih baik daripada strategi naive.

Aplikasi ini lebih berfungsi sebagai **alat eksplorasi & edukasi**:
- Melihat reaksi model terhadap 60 hari terakhir,
- Membandingkan prediksi model vs baseline,
- Melihat probabilitas arah pergerakan 5 hari.

### Notebook Pengembangan Model

📘 Notebook lengkap (preprocessing, training, evaluasi, cross-validation):

👉 [Buka di Google Colab](https://colab.research.google.com/drive/1S9ttFNq6L2kbGOu4l5viN-UG2eXnQLqJ?usp=sharing)

Di sana kamu bisa melihat:
- Detail arsitektur LSTM, GRU, dan Hybrid,
- Eksperimen window size & hyperparameter,
- Expanding-window cross-validation,
- Analisis mengapa baseline naive sangat kuat.

### Ide Pengembangan Lanjutan

- Tambah pilihan model (misal: Linear Regression / Random Forest) sebagai pembanding non-deep-learning.
- Tambah upload CSV untuk user yang ingin pakai data sendiri.
- Tambah backtest strategi yang lebih realistis (dengan biaya transaksi, stop loss, dll).
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
    "Mini Project Prediksi Harga Saham Harian menggunakan LSTM, GRU, "
    "model hybrid multi-step, dan klasifikasi arah. "
    "Prediksi hanya untuk edukasi & eksplorasi, bukan saran investasi."
)
