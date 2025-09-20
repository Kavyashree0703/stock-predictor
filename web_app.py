# web_app.py
import streamlit as st

# --- Page configuration ---
st.set_page_config(page_title="Home - Stock Dashboard", layout="centered")

# --- Home Page Content ---
st.title("🏠 Welcome to the Stock Price Predictor Dashboard")
st.write("""
This app allows you to:
- View historical stock prices for any stock symbol
- Display moving averages (MA7, MA21)
- Predict next-day closing prices using an LSTM model
- Download stock data as CSV
""")
st.write("Use the sidebar on the left to navigate to the **Stock Predictor** page and start analyzing stocks.")

# Optional: Add tips or instructions
st.info("Tip: Enter one or more stock symbols (comma-separated) in the Stock Predictor page, select the date range, and click Predict to see the forecast.")
