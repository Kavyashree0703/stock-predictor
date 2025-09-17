import streamlit as st
import requests
import yfinance as yf
import plotly.graph_objects as go
import joblib

# Load your trained model
model = joblib.load("scaler.pkl")

# ---------------- Page Config ----------------
st.set_page_config(page_title="📈 Stock Price Predictor", layout="wide")
st.title("📈 Stock Price Predictor Dashboard")

# ---------------- Load CSS ----------------
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
local_css("style.css")

# ---------------- Sidebar Input ----------------
st.sidebar.header("Stock Input")
symbols_input = st.sidebar.text_input("Enter stock symbols (comma-separated, e.g., AAPL, TSLA, GOOGL):", "")
symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

# Optional symbol mapping for Yahoo Finance quirks
symbol_map = {
    "BRK.B": "BRK-B",
    "BF.B": "BF-B",
}

# ---------------- Predict Button ----------------
if st.sidebar.button("Predict"):
    if symbols:
        st.subheader("Predictions")
        for symbol in symbols:
            yf_symbol = symbol_map.get(symbol, symbol)
            try:
                # --- Call Flask API ---
                res = requests.get(f"http://127.0.0.1:5000/predict?symbol={symbol}")
                data = res.json()

                if "prediction" in data:
                    predicted_price = round(data['prediction'], 2)

                    # --- Fetch historical data ---
                    try:
                        df = yf.download(yf_symbol, period="3mo", auto_adjust=True)
                        last_close = round(df['Close'][-1], 2) if not df.empty else None
                    except:
                        df = None
                        last_close = None

                    # --- Determine trend ---
                    if last_close:
                        if predicted_price > last_close:
                            trend_class = "trend-up"
                            trend_arrow = "⬆️ Up"
                        elif predicted_price < last_close:
                            trend_class = "trend-down"
                            trend_arrow = "⬇️ Down"
                        else:
                            trend_class = "trend-stable"
                            trend_arrow = "➡️ Stable"
                    else:
                        trend_class = ""
                        trend_arrow = ""

                    # --- Display prediction card ---
                    st.markdown(f"""
                    <div class='prediction-card'>
                        <h3>{symbol} Prediction: ${predicted_price} <span class='{trend_class}'>{trend_arrow}</span></h3>
                    </div>
                    """, unsafe_allow_html=True)

                    # --- Plot interactive chart ---
                    if df is not None and not df.empty:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=df.index, y=df['Close'],
                            mode='lines+markers', name='Close Price'
                        ))
                        fig.update_layout(
                            title=f"{symbol} Closing Prices (Last 3 Months)",
                            xaxis_title="Date",
                            yaxis_title="Price ($)",
                            template="plotly_white"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                else:
                    st.error(f"API error for {symbol}: {data.get('error', 'Unknown error')}")

            except Exception as e:
                st.error(f"Failed to fetch data for {symbol}: {e}")
    else:
        st.warning("Please enter at least one stock symbol.")
