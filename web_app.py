# web_app.py
import os
import pickle
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Stock Price Predictor")

# --- Config: change filenames here if yours differ ---
MODEL_PATH = "stock_lstm.h5"   # change to your model filename if needed
SCALER_PATH = "scaler.pkl"     # change to your scaler filename if needed
MODEL_WINDOW = 60              # model input window (days) - adjust if your model uses different

# --- Load model & scaler (cached resource) ---
try:
    from tensorflow.keras.models import load_model # type: ignore
    TENSORFLOW_AVAILABLE = True
except Exception:
    TENSORFLOW_AVAILABLE = False

@st.cache_resource
def load_resources():
    model = None
    scaler = None
    # load model if TF available and file exists
    if TENSORFLOW_AVAILABLE and os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH)
        except Exception as e:
            st.warning(f"Could not load model {MODEL_PATH}: {e}")
            model = None
    # load scaler if present
    if os.path.exists(SCALER_PATH):
        try:
            with open(SCALER_PATH, "rb") as f:
                scaler = pickle.load(f)
        except Exception as e:
            st.warning(f"Could not load scaler {SCALER_PATH}: {e}")
            scaler = None
    return model, scaler

model, scaler = load_resources()

# --- Sidebar UI ---
with st.sidebar:
    st.header("Stock Input")
    symbols_input = st.text_input("Symbol(s), comma-separated", "AAPL")
    symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
    start_date = st.date_input("Start date", value=(datetime.today() - timedelta(days=180)).date())
    end_date = st.date_input("End date", value=datetime.today().date())
    show_ma = st.checkbox("Show moving averages (MA7, MA21)", value=True)
    predict_btn = st.button("Predict")
    # optionally choose history days by slider (keeps compatibility)
    # days = st.slider("History (days)", min_value=30, max_value=720, value=180, step=30)

# --- caching helper for yfinance data ---
@st.cache_data(ttl=600)
def fetch_yf(ticker: str, start: datetime.date, end: datetime.date) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end + timedelta(days=1), progress=False, threads=False)
    if not df.empty:
        df.index = pd.to_datetime(df.index)
    return df

# --- plotting helper ---
def plot_history(df: pd.DataFrame, symbol: str, show_ma: bool = True, predicted_value: float | None = None):
    df2 = df.copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df2.index, y=df2["Close"], mode="lines+markers", name="Close", marker=dict(size=6)))
    if show_ma:
        df2["MA7"] = df2["Close"].rolling(7).mean()
        df2["MA21"] = df2["Close"].rolling(21).mean()
        fig.add_trace(go.Scatter(x=df2.index, y=df2["MA7"], mode="lines", name="MA7"))
        fig.add_trace(go.Scatter(x=df2.index, y=df2["MA21"], mode="lines", name="MA21"))
    if predicted_value is not None:
        next_date = df2.index[-1] + pd.Timedelta(days=1)
        fig.add_trace(go.Scatter(
            x=[next_date],
            y=[predicted_value],
            mode="markers+text",
            name="Predicted",
            marker=dict(size=14, symbol="star"),
            text=[f"${predicted_value:.2f}"],
            textposition="top center"
        ))
    fig.update_layout(
        title=f"{symbol} Closing Prices",
        xaxis_title="Date",
        yaxis_title="Close Price (USD)",
        template="plotly_white",
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# --- prediction helper ---
def predict_next_day(model, scaler, df: pd.DataFrame, window: int = MODEL_WINDOW) -> float:
    closes = df["Close"].values
    if len(closes) < window:
        raise ValueError(f"Not enough data for prediction (need {window} days, have {len(closes)})")
    last_window = closes[-window:].reshape(-1, 1)
    scaled = scaler.transform(last_window)            # (window, 1)
    X = np.array([scaled]).reshape(1, window, 1)      # (1, window, 1)
    pred_scaled = model.predict(X)
    pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
    return float(pred)

# --- small CSS tweak ---
st.markdown("""
<style>
.main .block-container{padding-top:1.5rem;}
.stButton>button {background-color: #6c8cff; color: white;}
</style>
""", unsafe_allow_html=True)

# --- session history setup ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- Main layout ---
st.title("📈 Stock Price Predictor Dashboard")
st.write("Enter one or more stock symbols (comma-separated) in the sidebar and click **Predict**.")

if predict_btn:
    # update recent searches
    for s in symbols:
        if s not in st.session_state.history:
            st.session_state.history.insert(0, s)
    st.session_state.history = st.session_state.history[:10]

    for symbol in symbols:
        st.subheader(f"🔎 {symbol}")
        with st.spinner(f"Fetching data for {symbol}..."):
            try:
                df = fetch_yf(symbol, start_date, end_date)
            except Exception as e:
                st.error(f"Failed to fetch data for {symbol}: {e}")
                continue

        if df is None or df.empty:
            st.error(f"No data found for {symbol} in the selected date range.")
            continue

        # metrics row
        col1, col2, col3 = st.columns([1, 1, 1])
        last_close = float(df["Close"].iloc[-1])
        pct_change = float(df["Close"].pct_change().iloc[-1] * 100) if len(df) > 1 else 0.0
        col1.metric("Last Close", f"${last_close:.2f}")
        col2.metric("Change (1d)", f"{pct_change:.2f}%")
        col3.metric("Records", len(df))

        # show base chart
        fig = plot_history(df, symbol, show_ma=show_ma, predicted_value=None)
        st.plotly_chart(fig, use_container_width=True)

        # Attempt model prediction if available
        if model is not None and scaler is not None:
            with st.spinner("Running model prediction..."):
                try:
                    pred = predict_next_day(model, scaler, df)
                    st.success(f"💰 Predicted next-day closing price for {symbol}: ${pred:.2f}")
                    # show chart with predicted point included
                    fig_pred = plot_history(df, symbol, show_ma=show_ma, predicted_value=pred)
                    st.plotly_chart(fig_pred, use_container_width=True)
                except Exception as e:
                    st.warning(f"Model prediction unavailable for {symbol}: {e}")
        else:
            st.info("Model or scaler not found — showing historical chart and latest price only.")

        # table + download
        with st.expander("Show table / Download CSV"):
            st.dataframe(df[["Open", "High", "Low", "Close", "Volume"]].tail(300))
            csv_data = df.to_csv().encode("utf-8")
            st.download_button("Download CSV", data=csv_data, file_name=f"{symbol}_history.csv", mime="text/csv")

# show recent searches in sidebar
with st.sidebar:
    st.write("---")
    st.write("Recent searches:")
    for s in st.session_state.history:
        st.write(f"- {s}")
