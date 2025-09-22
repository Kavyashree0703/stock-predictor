# web_app.py
import os
import pickle
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Stock Price Predictor Dashboard")

# --- Config ---
MODEL_PATH = "stock_lstm.h5"
SCALER_PATH = "scaler.pkl"
MODEL_WINDOW = 60
FORECAST_DAYS = 5

# --- Load model & scaler ---
try:
    from tensorflow.keras.models import load_model # type: ignore
    TENSORFLOW_AVAILABLE = True
except Exception:
    TENSORFLOW_AVAILABLE = False

@st.cache_resource
def load_resources():
    model = None
    scaler = None
    if TENSORFLOW_AVAILABLE and os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH)
        except Exception as e:
            st.warning(f"Could not load model {MODEL_PATH}: {e}")
            model = None
    if os.path.exists(SCALER_PATH):
        try:
            with open(SCALER_PATH, "rb") as f:
                scaler = pickle.load(f)
        except Exception as e:
            st.warning(f"Could not load scaler {SCALER_PATH}: {e}")
            scaler = None
    return model, scaler

model, scaler = load_resources()

# --- Sidebar navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Stock Predictor"])

# Style for sidebar
st.markdown(
    """
    <style>
    /* Sidebar background and text */
    section[data-testid="stSidebar"] {
        background-color: #0E1117;
    }
    .css-1v3fvcr, .css-16idsys, .css-qrbaxs, .css-1d391kg { 
        color: #00C0F0 !important; 
        font-weight: 700 !important;
        font-size: 1.1rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ======================== Home Page ========================
if page == "Home":
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://cdn.pixabay.com/photo/2018/01/15/07/51/chart-3081197_1280.png"); 
            background-size: cover; 
            background-position: center; 
            background-repeat: no-repeat; 
            background-attachment: fixed;
        }
        .home-container {
            background-color: rgba(255, 255, 255, 0.85); 
            padding: 2.5rem; 
            border-radius: 15px;
        }
        h1 { font-size: 3rem !important; font-weight: 800 !important; color: #222 !important; }
        h2 { font-size: 2.2rem !important; font-weight: 700 !important; color: #222 !important; }
        p { font-size: 1.2rem !important; color: #222 !important; }
        ul { font-size: 1.1rem !important; }
        </style>
        """, unsafe_allow_html=True
    )

    # 🚀 Banner instead of white input box
    st.markdown(
        """
        <div style="background-color:#00c0f0; color:white; padding:14px; border-radius:12px; 
        text-align:center; font-size:1.2rem; font-weight:600; margin-bottom:20px;">
            🚀 Start by choosing <b>Stock Predictor</b> from the sidebar to begin!
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="home-container">', unsafe_allow_html=True)
    st.title("📊 Welcome to Stock Price Predictor")
    st.markdown("""
    ### Predict and Visualize Stock Prices Easily
    This app helps you **predict next-day stock prices** and visualize historical trends with beginner-friendly charts.

    **How to use:**
    1. Go to the **Stock Predictor** page using the sidebar.
    2. Enter one or more stock symbols (like `AAPL`, `TSLA`) in the input box.
    3. Click **Predict** to see historical charts, trends, and predicted prices.

    **Features:**
    - Trend arrows 🔼 🔽 ➡️ show **uptrend, downtrend, or sideways movement**
    - Color-coded predicted price for instant market insight
    - Zoomed-in last 60 days and 5-day forecast
    - Download historical data as CSV
    """)
    st.info("No prior stock knowledge needed! Just select your stock and see the data 📈")
    st.markdown('</div>', unsafe_allow_html=True)

# ======================== Stock Predictor Page ========================
elif page == "Stock Predictor":
    # --- Sidebar UI ---
    with st.sidebar:
        st.header("Stock Input")
        symbols_input = st.text_input("Symbol(s), comma-separated", "AAPL")
        symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
        start_date = st.date_input("Start date", value=(datetime.today() - timedelta(days=180)).date())
        end_date = st.date_input("End date", value=datetime.today().date())
        show_ma = st.checkbox("Show moving average (7-day)", value=True)
        predict_btn = st.button("Predict")

    # --- fetch data from Yahoo Finance ---
    @st.cache_data(ttl=600)
    def fetch_yf(ticker: str, start: datetime.date, end: datetime.date) -> pd.DataFrame:
        df = yf.download(ticker, start=start, end=end + timedelta(days=1), progress=False, threads=False)
        if not df.empty:
            df.index = pd.to_datetime(df.index)
        return df

    # --- Prediction helpers ---
    def predict_next_day(model, scaler, df: pd.DataFrame, window: int = MODEL_WINDOW) -> float:
        closes = df["Close"].values
        last_window = closes[-window:].reshape(-1, 1)
        scaled = scaler.transform(last_window)
        X = np.array([scaled]).reshape(1, window, 1)
        pred_scaled = model.predict(X)
        pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
        return float(pred)

    def forecast_days(model, scaler, df: pd.DataFrame, days: int = FORECAST_DAYS, window: int = MODEL_WINDOW):
        last_window = df["Close"].values[-window:].reshape(-1, 1)
        scaled_window = scaler.transform(last_window)
        preds = []
        current_window = scaled_window.copy()
        for _ in range(days):
            X = current_window.reshape(1, window, 1)
            pred_scaled = model.predict(X)
            pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
            preds.append(pred)
            next_scaled = pred_scaled[0][0]
            current_window = np.roll(current_window, -1)
            current_window[-1] = next_scaled
        return preds

    # --- Plot helper ---
    def plot_history(df: pd.DataFrame, symbol: str, predicted_value=None, forecast=None, last_n_days=None):
        df2 = df.copy()
        df2.index = pd.to_datetime(df2.index)
        if last_n_days:
            df2 = df2.tail(last_n_days)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df2.index, y=df2["Close"], mode="lines+markers", name="Close", line=dict(color="blue", width=3)))
        if show_ma:
            df2["MA7"] = df2["Close"].rolling(7).mean()
            fig.add_trace(go.Scatter(x=df2.index, y=df2["MA7"], mode="lines", name="MA7", line=dict(color="orange", width=2, dash="dash")))

        # Trend arrows
        trend_color = "gray"; trend_text = "Sideways ➡️"
        if show_ma and len(df2["MA7"].dropna()) >= 2:
            recent_ma = df2["MA7"].iloc[-1]
            prev_ma = df2["MA7"].iloc[-2]
            if recent_ma > prev_ma: trend_text, trend_color = "Uptrend 🔼", "green"
            elif recent_ma < prev_ma: trend_text, trend_color = "Downtrend 🔽", "red"
        y_pos = df2["Close"].dropna().max() * 1.02 if not df2["Close"].dropna().empty else 0
        fig.add_annotation(x=df2.index[int(len(df2)/10)], y=y_pos, text=f"<b>{trend_text}</b>", showarrow=False, font=dict(color=trend_color, size=16))

        # Next-day prediction
        if predicted_value is not None:
            next_date = df2.index[-1] + pd.Timedelta(days=1)
            fig.add_trace(go.Scatter(x=[next_date], y=[predicted_value], mode="markers+text", name="Predicted Price", marker=dict(size=14, color=trend_color, symbol="diamond"), text=[f"${predicted_value:.2f}"], textposition="top center"))

        # 5-day forecast
        if forecast is not None:
            forecast_dates = [df2.index[-1] + pd.Timedelta(days=i+1) for i in range(len(forecast))]
            fig.add_trace(go.Scatter(x=forecast_dates, y=forecast, mode="lines+markers", name="5-Day Forecast", line=dict(color="purple", width=2, dash="dot")))
        fig.update_layout(title=f"{symbol} Stock Price Overview", xaxis_title="Date", yaxis_title="Price (USD)", template="plotly_white", height=520, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), hovermode="x unified")
        return fig

    # --- Session history ---
    if "history" not in st.session_state:
        st.session_state.history = []

    # --- Main layout ---
    st.title("📈 Stock Price Predictor Dashboard")
    st.write("Enter stock symbols in the sidebar and click **Predict**.")

    if predict_btn:
        for s in symbols:
            if s not in st.session_state.history:
                st.session_state.history.insert(0, s)
        st.session_state.history = st.session_state.history[:10]

        for symbol in symbols:
            st.subheader(f"🔎 {symbol}")
            with st.spinner(f"Fetching data for {symbol}..."):
                try: df = fetch_yf(symbol, start_date, end_date)
                except Exception as e: st.error(f"Failed to fetch data for {symbol}: {e}"); continue
            if df is None or df.empty: st.error(f"No data found for {symbol}"); continue

            # Metrics
            col1, col2, col3 = st.columns([1,1,1])
            last_close = float(df["Close"].iloc[-1])
            pct_change = float(df["Close"].pct_change().iloc[-1]*100) if len(df)>1 else 0
            col1.metric("Last Close", f"${last_close:.2f}")
            col2.metric("Change (1d)", f"{pct_change:.2f}%")
            col3.metric("Records", len(df))

            # Prediction
            pred = None; forecast = None
            if model is not None and scaler is not None:
                with st.spinner("Running model prediction..."):
                    try:
                        pred = predict_next_day(model, scaler, df)
                        forecast = forecast_days(model, scaler, df)
                    except Exception as e:
                        st.warning(f"Model unavailable: {e}")

            # Plot charts
            fig = plot_history(df, symbol, predicted_value=pred, forecast=forecast, last_n_days=60)
            st.plotly_chart(fig, use_container_width=True)

            # Table + download
            with st.expander("Show table / Download CSV"):
                st.dataframe(df[["Open","High","Low","Close","Volume"]].tail(300))
                st.download_button("Download CSV", data=df.to_csv().encode("utf-8"), file_name=f"{symbol}_history.csv", mime="text/csv")

    # Recent searches
    with st.sidebar:
        st.write("---")
        st.write("Recent searches:")
        for s in st.session_state.history: st.write(f"- {s}")
