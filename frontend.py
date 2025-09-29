# frontend.py
import streamlit as st
import openai # type: ignore
import yfinance as yf
import pandas as pd

st.title("Stock Predictor")

st.write("Enter a stock symbol to fetch recent data:")

symbol = st.text_input("Stock Symbol", "AAPL")

if symbol:
    data = yf.Ticker(symbol).history(period="5d")
    st.write(data)

st.write("This app can also call OpenAI API (example):")

openai_api_key = st.text_input("OpenAI API Key", type="password")

if openai_api_key:
    openai.api_key = openai_api_key
    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=f"Give a one line advice for investing in {symbol}",
            max_tokens=20
        )
        st.write("OpenAI says:", response.choices[0].text.strip())
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
