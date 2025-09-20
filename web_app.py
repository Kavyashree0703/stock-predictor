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

# --- Load model & scaler ---
try:
    from tensorflow.keras.models import load_model  # type: ignore
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

# --- Home Page ---
if page == "Home":
    st.title("📊 Welcome to Stock Price Predictor")
    st.write("""
    This app helps you **predict next-day stock prices** and visualize historical trends.
    
    **How to use:**
    1. Go to the **Stock Predictor** page using the sidebar.
    2. Enter one or more stock symbols (like `AAPL`, `TSLA`) in the input box.
    3. Click **Predict** to see historical charts, trends, and predicted next-day price.
    
    Even if you have **no knowledge of stocks**, the trend arrows 🔼 🔽 ➡️ and color-coded predicted price will help you understand the market.
    """)
    st.image("https://images.unsplash.com/photo-1565372919472-24a9dcb3f819?ixlib=rb-4.0.3&auto=format&fit=crop&w=1470&q=80",
             caption="Track your favorite stocks easily!", use_column_width=True)

# --- Stock Predictor Page ---
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

    # --- Enhanced beginner-friendly plot ---
    def plot_history(df: pd.DataFrame, symbol: str, predicted_value: float | None = None, last_n_days: int | None = None):
        df2 = df.copy()
        df2.index = pd.to_datetime(df2.index)
        if last_n_days:
            df2 = df2.tail(last_n_days)

        fig = go.Figure()

        # Closing price
        fig.add_trace(go.Scatter(
            x=df2.index,
            y=df2["Close"],
            mode="lines+markers",
            name="Closing Price",
            line=dict(color="blue", width=3),
            marker=dict(size=6),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Price: $%{y:.2f}<extra></extra>"
        ))

        # 7-day moving average
        df2["MA7"] = df2["Close"].rolling(7).mean()
        fig.add_trace(go.Scatter(
            x=df2.index,
            y=df2["MA7"],
            mode="lines",
            name="7-Day Avg",
            line=dict(color="orange", width=2, dash="dash"),
            hovertemplate="7-Day Avg: $%{y:.2f}<extra></extra>"
        ))

        # Determine trend
        trend_color = "gray"
        trend_text = "Stock Trend: Sideways ➡️"
        if len(df2["MA7"].dropna()) >= 2:
            recent_ma = df2["MA7"].iloc[-1]
            prev_ma = df2["MA7"].iloc[-2]
            if recent_ma > prev_ma:
                trend_text = "Stock Trend: Uptrend 🔼"
                trend_color = "green"
            elif recent_ma < prev_ma:
                trend_text = "Stock Trend: Downtrend 🔽"
                trend_color = "red"
        else:
            trend_text = "Stock Trend: Unknown ❓"
            trend_color = "gray"

        # Safely get max close price ignoring NaNs
        if df2["Close"].dropna().empty:
            y_pos = 0
        else:
            y_pos = df2["Close"].dropna().max() * 1.02

        # Trend annotation
        fig.add_annotation(
            x=df2.index[int(len(df2)/10)],
            y=y_pos,
            text=f"<b>{trend_text}</b>",
            showarrow=False,
            font=dict(color=trend_color, size=16)
        )

        # Predicted next-day price
        if predicted_value is not None:
            next_date = df2.index[-1] + pd.Timedelta(days=1)
            fig.add_trace(go.Scatter(
                x=[next_date],
                y=[predicted_value],
                mode="markers+text",
                name="Predicted Price",
                marker=dict(size=14, color=trend_color, symbol="diamond"),
                text=[f"${predicted_value:.2f}"],
                textposition="top center",
                hovertemplate="Predicted Price: $%{y:.2f}<extra></extra>"
            ))
            fig.update_xaxes(range=[df2.index[0], next_date + pd.Timedelta(days=2)])

        # Layout
        fig.update_layout(
            title=f"{symbol} Stock Price Overview",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_white",
            height=520,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified"
        )

        return fig

    # --- Prediction helper ---
    def predict_next_day(model, scaler, df: pd.DataFrame, window: int = MODEL_WINDOW) -> float:
        closes = df["Close"].values
        if len(closes) < window:
            raise ValueError(f"Not enough data for prediction (need {window} days, have {len(closes)})")
        last_window = closes[-window:].reshape(-1, 1)
        scaled = scaler.transform(last_window)
        X = np.array([scaled]).reshape(1, window, 1)
        pred_scaled = model.predict(X)
        pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
        return float(pred)

    # --- CSS tweaks ---
    st.markdown("""
    <style>
    .main .block-container{padding-top:1.5rem;}
    .stButton>button {background-color: #6c8cff; color: white;}
    </style>
    """, unsafe_allow_html=True)

    # --- Session history ---
    if "history" not in st.session_state:
        st.session_state.history = []

    # --- Main layout ---
    st.title("📈 Stock Price Predictor Dashboard")
    st.write("Enter one or more stock symbols (comma-separated) in the sidebar and click **Predict**.")

    if predict_btn:
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

            # Metrics row
            col1, col2, col3 = st.columns([1, 1, 1])
            last_close = float(df["Close"].iloc[-1])
            pct_change = float(df["Close"].pct_change().iloc[-1] * 100) if len(df) > 1 else 0.0
            col1.metric("Last Close", f"${last_close:.2f}")
            col2.metric("Change (1d)", f"{pct_change:.2f}%")
            col3.metric("Records", len(df))

            # Full historical chart
            fig = plot_history(df, symbol, predicted_value=None)
            st.plotly_chart(fig, use_container_width=True)

            # Prediction
            if model is not None and scaler is not None:
                with st.spinner("Running model prediction..."):
                    try:
                        pred = predict_next_day(model, scaler, df)
                        st.success(f"💰 Predicted next-day closing price for {symbol}: ${pred:.2f}")

                        # Full chart with prediction
                        fig_pred = plot_history(df, symbol, predicted_value=pred)
                        st.plotly_chart(fig_pred, use_container_width=True)

                        # Zoomed-in last 60 days
                        if len(df) > 60:
                            fig_zoom = plot_history(df, symbol, predicted_value=pred, last_n_days=60)
                            st.subheader("📊 Last 60 Days (Zoomed-in)")
                            st.plotly_chart(fig_zoom, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Model prediction unavailable for {symbol}: {e}")
            else:
                st.info("Model or scaler not found — showing historical chart only.")

            # Table + download
            with st.expander("Show table / Download CSV"):
                st.dataframe(df[["Open", "High", "Low", "Close", "Volume"]].tail(300))
                csv_data = df.to_csv().encode("utf-8")
                st.download_button("Download CSV", data=csv_data, file_name=f"{symbol}_history.csv", mime="text/csv")

    # Recent searches
    with st.sidebar:
        st.write("---")
        st.write("Recent searches:")
        for s in st.session_state.history:
            st.write(f"- {s}")
