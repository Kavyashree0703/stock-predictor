# web_app.py
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import os
import pickle

st.set_page_config(layout="wide", page_title="Stock Price Predictor")

st.title("📈 Stock Price Predictor Dashboard")

# Try to load model + scaler (optional)
model = None
scaler = None
MODEL_PATH = "stock_lstm.h5"   # change if your model filename is different
SCALER_PATH = "scaler.pkl"     # change if your scaler filename is different

try:
    from tensorflow.keras.models import load_model
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
except Exception:
    model = None

try:
    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
except Exception:
    scaler = None

# Sidebar input
with st.sidebar:
    st.header("Stock Input")
    symbol = st.text_input("Enter stock symbol (e.g., AAPL, TSLA, MSFT):", "AAPL").upper()
    days = st.slider("History (days)", min_value=30, max_value=365, value=90, step=30)
    predict_btn = st.button("Predict")

# Fetch data helper
@st.cache_data(ttl=300)
def fetch_data(ticker: str, period_days: int) -> pd.DataFrame:
    # use yfinance period in months if > 90 else days; keep it simple
    period = f"{max(30, period_days)}d"
    df = yf.download(ticker, period=period, progress=False, threads=False)
    # ensure datetime index
    if isinstance(df.index, pd.DatetimeIndex):
        return df
    else:
        df.index = pd.to_datetime(df.index)
        return df

# Show historical chart
try:
    data = fetch_data(symbol, days)
    if data.empty:
        st.error(f"No data found for symbol: {symbol}")
    else:
        st.subheader(f"📊 Historical Prices for {symbol} (last {days} days)")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["Close"],
                mode="lines+markers",
                name="Close",
                marker=dict(size=6),
            )
        )
        fig.update_layout(
            title=f"{symbol} Closing Prices",
            xaxis_title="Date",
            yaxis_title="Close Price (USD)",
            template="plotly_white",
            height=600,
        )
        st.plotly_chart(fig, use_container_width=True)

        # If user pressed Predict, run model (if available)
        if predict_btn:
            with st.spinner("Running prediction..."):
                # If model+scaler are available, use them. Model typically expects last 60 values.
                if model is not None and scaler is not None:
                    try:
                        # require at least 60 days; adjust if your model uses different window
                        window = 60
                        closes = data["Close"].values
                        if len(closes) < window:
                            st.warning(f"Not enough historical data for model (need {window} days).")
                        else:
                            last_window = closes[-window:].reshape(-1, 1)
                            scaled = scaler.transform(last_window)            # (window, 1)
                            X = np.array([scaled])                           # (1, window, 1)
                            X = X.reshape((X.shape[0], X.shape[1], 1))
                            pred_scaled = model.predict(X)
                            pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
                            st.success(f"💰 {symbol} predicted next-day closing price: ${pred:.2f}")
                    except Exception as e:
                        st.error(f"Prediction error: {e}")
                else:
                    # graceful fallback: show latest close and a simple naive next-day estimate (e.g., last close)
                    last_close = float(data["Close"].iloc[-1])
                    st.info("Model or scaler not found — skipping ML prediction.")
                    st.write(f"Latest close for {symbol}: **${last_close:.2f}**")
                    # optional naive estimate: last_close (or a small percent diff)
                    naive = last_close
                    st.write(f"Naive next-day estimate (last close): **${naive:.2f}**")

except Exception as e:
    st.error(f"An error occurred while loading data: {e}")
