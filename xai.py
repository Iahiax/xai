#pip install requests numpy pandas scikit-learn tensorflow ta
# ===============================
# CAPITAL.COM TREND AI BOT
# SINGLE FILE - NO EXTERNAL FILES
# ===============================

import time
import requests
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from ta.trend import ADXIndicator
from ta.volatility import AverageTrueRange

# ===============================
# CONFIG
# ===============================

API_KEY = "PUT_API_KEY_HERE"
IDENTIFIER = "PUT_EMAIL_HERE"
PASSWORD = "PUT_PASSWORD_HERE"

BASE_URL = "https://demo-api-capital.backend-capital.com/api/v1"
EPIC = "BTCUSD"
TIMEFRAME = "MINUTE"
SEQ_LEN = 30

THRESHOLD_BUY = 0.6
THRESHOLD_SELL = 0.4

# ===============================
# MEMORY (IN-RAM ONLY)
# ===============================

TRAINING_MEMORY = {
    "strong": [],
    "medium": [],
    "max_size": 5000
}

stats = {
    "equity": [1.0],
    "returns": [],
    "wins": 0,
    "losses": 0
}

model_stats = {
    "strong": {"wins": 0, "losses": 0},
    "medium": {"wins": 0, "losses": 0}
}

# ===============================
# AUTH
# ===============================

def login():
    r = requests.post(
        f"{BASE_URL}/session",
        headers={"X-CAP-API-KEY": API_KEY},
        json={"identifier": IDENTIFIER, "password": PASSWORD}
    )
    return {
        "CST": r.headers["CST"],
        "X-SECURITY-TOKEN": r.headers["X-SECURITY-TOKEN"]
    }

# ===============================
# MARKET DATA
# ===============================

def get_prices(tokens, n=200):
    r = requests.get(
        f"{BASE_URL}/prices/{EPIC}",
        headers={
            "X-CAP-API-KEY": API_KEY,
            "CST": tokens["CST"],
            "X-SECURITY-TOKEN": tokens["X-SECURITY-TOKEN"]
        },
        params={"resolution": TIMEFRAME, "max": n}
    )

    prices = r.json()["prices"]
    df = pd.DataFrame([{
        "open": float(p["openPrice"]["bid"]),
        "high": float(p["highPrice"]["bid"]),
        "low": float(p["lowPrice"]["bid"]),
        "close": float(p["closePrice"]["bid"])
    } for p in prices])

    return df

# ===============================
# MEMORY UPDATE (SMART FILTER)
# ===============================

def update_memory(df):
    adx = ADXIndicator(df.high, df.low, df.close).adx()

    for price, a in zip(df.close, adx):
        if a >= 30:
            TRAINING_MEMORY["strong"].append(price)
        elif a >= 20:
            TRAINING_MEMORY["medium"].append(price)

    for k in ["strong", "medium"]:
        if len(TRAINING_MEMORY[k]) > TRAINING_MEMORY["max_size"]:
            TRAINING_MEMORY[k] = TRAINING_MEMORY[k][-TRAINING_MEMORY["max_size"]:]

# ===============================
# MODEL
# ===============================

def build_model(shape):
    m = Sequential([
        LSTM(64, return_sequences=True, input_shape=shape),
        Dropout(0.2),
        LSTM(32),
        Dense(1, activation="sigmoid")
    ])
    m.compile("adam", "binary_crossentropy")
    return m

def prepare(bucket):
    data = TRAINING_MEMORY[bucket]
    if len(data) < SEQ_LEN + 50:
        return None, None, None

    scaler = MinMaxScaler()
    data = scaler.fit_transform(np.array(data).reshape(-1,1))

    X, y = [], []
    for i in range(SEQ_LEN, len(data)):
        X.append(data[i-SEQ_LEN:i])
        y.append(1 if data[i] > data[i-1] else 0)

    return np.array(X), np.array(y), scaler

def train_models():
    models = {}
    for bucket in ["strong", "medium"]:
        X, y, scaler = prepare(bucket)
        if X is None:
            models[bucket] = (None, None)
            continue
        model = build_model((X.shape[1], X.shape[2]))
        model.fit(X, y, epochs=5, batch_size=32, verbose=0)
        models[bucket] = (model, scaler)
        print(f"🧠 trained {bucket} model ({len(X)})")
    return models

# ===============================
# SL / TP
# ===============================

def calc_sl_tp(df, side):
    atr = AverageTrueRange(df.high, df.low, df.close).average_true_range().iloc[-1]
    adx = ADXIndicator(df.high, df.low, df.close).adx().iloc[-1]
    price = df.close.iloc[-1]

    sl_m, tp_m = (1.5, 3) if adx >= 30 else (1, 2)

    if side == "BUY":
        return price - atr*sl_m, price + atr*tp_m
    else:
        return price + atr*sl_m, price - atr*tp_m

# ===============================
# TRADING
# ===============================

def open_trade(tokens, side, df):
    sl, tp = calc_sl_tp(df, side)

    requests.post(
        f"{BASE_URL}/positions",
        headers={
            "X-CAP-API-KEY": API_KEY,
            "CST": tokens["CST"],
            "X-SECURITY-TOKEN": tokens["X-SECURITY-TOKEN"],
            "Content-Type": "application/json"
        },
        json={
            "epic": EPIC,
            "direction": side,
            "size": 1,
            "orderType": "MARKET",
            "stopLevel": round(sl, 2),
            "limitLevel": round(tp, 2)
        }
    )

    print(f"📈 {side} | SL {sl:.2f} | TP {tp:.2f}")

# ===============================
# MAIN LOOP
# ===============================

def run():
    tokens = login()
    models = train_models()

    while True:
        df = get_prices(tokens)
        update_memory(df)

        adx = ADXIndicator(df.high, df.low, df.close).adx().iloc[-1]

        if adx >= 30:
            regime = "strong"
        elif adx >= 20:
            regime = "medium"
        else:
            time.sleep(30)
            continue

        model, scaler = models[regime]
        if not model:
            time.sleep(30)
            continue

        x = scaler.transform(df.close.values[-SEQ_LEN:].reshape(-1,1))
        pred = model.predict(x.reshape(1, SEQ_LEN, 1), verbose=0)[0][0]

        if pred > THRESHOLD_BUY:
            open_trade(tokens, "BUY", df)
        elif pred < THRESHOLD_SELL:
            open_trade(tokens, "SELL", df)

        time.sleep(60)

# ===============================
# START
# ===============================

if __name__ == "__main__":
    run()
