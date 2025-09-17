from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import pandas as pd
import yfinance as yf
import pickle
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# Load model and scaler
model = tf.keras.models.load_model('stock_lstm.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

app = Flask(__name__)
CORS(app)

# Function to get last 60 days of stock data and scale it
def get_last_60_days(symbol):
    end = datetime.now()
    start = end - timedelta(days=100)
    data = yf.download(symbol, start=start, end=end)

    # Ensure we have at least 60 days of data
    if len(data) < 60:
        raise ValueError("Not enough data available to make prediction")

    close_prices = data['Close'].values[-60:].reshape(-1, 1)
    scaled = scaler.transform(close_prices)
    return np.array([scaled])

# Prediction API route
@app.route('/predict', methods=['GET'])
def predict():
    symbol = request.args.get('symbol', 'AAPL')  # Default is Apple
    try:
        X_input = get_last_60_days(symbol)
        pred_scaled = model.predict(X_input)
        pred_price = scaler.inverse_transform(pred_scaled)[0][0]
        return jsonify({
            "symbol": symbol,
            "prediction": round(float(pred_price), 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
