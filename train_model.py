# 🚀 LSTM Stock Price Predictor Training Script

# ✅ Environment setup
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU usage (optional)

# ✅ Imports
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
import pickle
from datetime import datetime

# ✅ Parameters
SYMBOL = "AAPL"
LOOKBACK = 60
EPOCHS = 20
BATCH_SIZE = 8

# ✅ Step 1: Download stock data
print(f"📥 Downloading {SYMBOL} data...")
try:
    data = yf.download(SYMBOL, start="2020-01-01", end="2023-12-31", progress=False)
    print(f"✅ Data downloaded: {data.shape}")
    close_prices = data["Close"].dropna().values.reshape(-1, 1)
except Exception as e:
    print(f"❌ Failed to download data: {e}")
    exit(1)

# ✅ Step 2: Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# ✅ Step 3: Prepare training sequences
X_train, y_train = [], []
for i in range(LOOKBACK, len(scaled_data)):
    X_train.append(scaled_data[i - LOOKBACK:i, 0])
    y_train.append(scaled_data[i, 0])

X_train = np.array(X_train).reshape(-1, LOOKBACK, 1)
y_train = np.array(y_train)

# ✅ Step 4: Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(LOOKBACK, 1)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])
model.compile(optimizer="adam", loss="mean_squared_error")

# ✅ Step 5: Train model
print("🧠 Training model...")
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

# ✅ Step 6: Save model and scaler
model.save("stock_lstm.h5")
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Training complete. Model and scaler saved.")
