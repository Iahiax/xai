# =========================================================
# CAPITAL.COM EURUSD AI BOT - FINAL FIXED VERSION
# LSTM + RL + NEWS + REAL WIN RATE + AUTO SESSION
# SINGLE FILE - DEMO READY
# =========================================================

import time
import requests
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from ta.trend import ADXIndicator
from ta.volatility import AverageTrueRange

# ================== CONFIG ==================

API_KEY = "4YVCSQJ8FGNXZQ7K"
IDENTIFIER = "yahia.x@outlook.sa"
PASSWORD = "Yahia-1411"

NEWS_API_KEY = "ba0a4d6fcb234b57a7a05fbfe80ae701"
NEWS_QUERY = "EURUSD OR euro OR dollar OR ECB OR FED"
NEWS_NEGATIVE_THRESHOLD = -3

BASE_URL = "https://demo-api-capital.backend-capital.com/api/v1"
EPIC = "EURUSD"
TIMEFRAME = "MINUTE"

HISTORY_BARS = 1200
SEQ_LEN = 30

THRESHOLD_BUY = 0.57
THRESHOLD_SELL = 0.43

POLL_INTERVAL = 15
COOLDOWN_SECONDS = 600
RETRAIN_EVERY = 25

# ================== MEMORY ==================

TRAINING_MEMORY = {
    "strong": [],
    "medium": [],
    "max": 9000
}

STATS = {
    "trades": 0,
    "wins": 0,
    "losses": 0,
    "last_trade_time": 0
}

# Reinforcement Learning Q-table
Q = {
    "strong_BUY": 1.0,
    "strong_SELL": 1.0,
    "medium_BUY": 1.0,
    "medium_SELL": 1.0
}

ALPHA = 0.1
GAMMA = 0.9

# ================== AUTH (FIXED) ==================

def login():
    r = requests.post(
        f"{BASE_URL}/session",
        headers={
            "X-CAP-API-KEY": API_KEY,
            "Content-Type": "application/json"
        },
        json={
            "identifier": IDENTIFIER,
            "password": PASSWORD
        },
        timeout=10
    )

    if r.status_code != 200:
        print("❌ LOGIN FAILED")
        print("Status:", r.status_code)
        print("Response:", r.text)
        raise SystemExit("Fix API credentials or enable API access")

    if "CST" not in r.headers or "X-SECURITY-TOKEN" not in r.headers:
        print("❌ SESSION TOKENS NOT RETURNED")
        print("Headers:", dict(r.headers))
        raise SystemExit("Capital.com did not return CST")

    print("✅ Login successful")
    return {
        "CST": r.headers.get("CST"),
        "X-SECURITY-TOKEN": r.headers.get("X-SECURITY-TOKEN")
    }

def safe_request(method, url, tokens, **kw):
    headers = kw.pop("headers", {})
    headers.update({
        "X-CAP-API-KEY": API_KEY,
        "CST": tokens["CST"],
        "X-SECURITY-TOKEN": tokens["X-SECURITY-TOKEN"]
    })

    r = requests.request(
        method,
        url,
        headers=headers,
        timeout=10,
        **kw
    )

    if r.status_code in [401, 403]:
        print("🔄 Session expired — relogin")
        tokens.update(login())
        headers.update({
            "CST": tokens["CST"],
            "X-SECURITY-TOKEN": tokens["X-SECURITY-TOKEN"]
        })
        r = requests.request(method, url, headers=headers, timeout=10, **kw)

    return r

# ================== NEWS ==================

def get_news_score():
    try:
        r = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": NEWS_QUERY,
                "language": "en",
                "pageSize": 10,
                "sortBy": "publishedAt",
                "apiKey": NEWS_API_KEY
            },
            timeout=5
        )
        articles = r.json().get("articles", [])
        negative = ["crisis", "recession", "panic", "collapse", "risk", "inflation"]
        score = 0
        for a in articles:
            title = (a["title"] or "").lower()
            for w in negative:
                if w in title:
                    score -= 1
        return score
    except:
        return 0

# ================== DATA ==================

def get_prices(tokens):
    r = safe_request(
        "GET",
        f"{BASE_URL}/prices/{EPIC}",
        tokens,
        params={
            "resolution": TIMEFRAME,
            "max": HISTORY_BARS
        }
    )

    prices = r.json()["prices"]
    return pd.DataFrame([{
        "open": float(p["openPrice"]["bid"]),
        "high": float(p["highPrice"]["bid"]),
        "low": float(p["lowPrice"]["bid"]),
        "close": float(p["closePrice"]["bid"])
    } for p in prices])

def data_quality_ok(df):
    if df.isna().sum().sum() > 0:
        return False
    if df.close.pct_change().abs().max() > 0.03:
        return False
    return True

# ================== ML ==================

def build_model(shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=shape),
        Dropout(0.2),
        LSTM(32),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy")
    return model

def update_memory(df):
    adx = ADXIndicator(df.high, df.low, df.close).adx()
    for price, a in zip(df.close, adx):
        if a >= 30:
            TRAINING_MEMORY["strong"].append(price)
        elif a >= 20:
            TRAINING_MEMORY["medium"].append(price)

    for k in ["strong", "medium"]:
        TRAINING_MEMORY[k] = TRAINING_MEMORY[k][-TRAINING_MEMORY["max"]:]

def prepare(bucket):
    data = TRAINING_MEMORY[bucket]
    if len(data) < SEQ_LEN + 50:
        return None, None, None

    scaler = MinMaxScaler()
    data = scaler.fit_transform(np.array(data).reshape(-1, 1))

    X, y = [], []
    for i in range(SEQ_LEN, len(data)):
        X.append(data[i - SEQ_LEN:i])
        y.append(1 if data[i] > data[i - 1] else 0)

    return np.array(X), np.array(y), scaler

def train_models():
    models = {}
    for bucket in ["strong", "medium"]:
        X, y, scaler = prepare(bucket)
        if X is None:
            models[bucket] = (None, None)
            continue

        model = build_model((X.shape[1], 1))
        model.fit(X, y, epochs=5, batch_size=32, verbose=0)
        models[bucket] = (model, scaler)
        print(f"🧠 Trained {bucket} model | samples {len(X)}")

    return models

# ================== RL ==================

def update_q(state, reward):
    Q[state] = Q[state] + ALPHA * (reward + GAMMA * Q[state] - Q[state])

# ================== TRADING ==================

def can_trade():
    return time.time() - STATS["last_trade_time"] > COOLDOWN_SECONDS

def close_opposite(tokens, side):
    r = safe_request("GET", f"{BASE_URL}/positions", tokens)
    for p in r.json().get("positions", []):
        pos = p["position"]
        if pos["epic"] == EPIC and pos["direction"] != side:
            safe_request(
                "DELETE",
                f"{BASE_URL}/positions/{pos['dealId']}",
                tokens
            )
            print("❌ Closed opposite position")

def open_trade(tokens, side, df, regime, news_score):
    if not can_trade():
        return

    close_opposite(tokens, side)

    atr = AverageTrueRange(df.high, df.low, df.close)\
        .average_true_range().iloc[-1]

    price = df.close.iloc[-1]

    if side == "BUY":
        sl = price - atr * 1.2
        tp = price + atr * 2.4
    else:
        sl = price + atr * 1.2
        tp = price - atr * 2.4

    safe_request(
        "POST",
        f"{BASE_URL}/positions",
        tokens,
        json={
            "epic": EPIC,
            "direction": side,
            "size": 1,
            "orderType": "MARKET",
            "stopLevel": round(sl, 5),
            "limitLevel": round(tp, 5)
        }
    )

    STATS["trades"] += 1
    STATS["last_trade_time"] = time.time()

    print(f"📈 {side} | Regime:{regime} | News:{news_score}")

# ================== WIN RATE ==================

def update_winrate(tokens):
    r = safe_request("GET", f"{BASE_URL}/history/activity", tokens)
    activities = r.json().get("activities", [])

    wins = losses = 0
    for a in activities:
        pnl = a.get("profitAndLoss")
        if pnl is None:
            continue
        if pnl > 0:
            wins += 1
        elif pnl < 0:
            losses += 1

    STATS["wins"] = wins
    STATS["losses"] = losses

    if wins + losses > 0:
        winrate = wins / (wins + losses) * 100
        print(f"📊 Win Rate: {winrate:.2f}%")

# ================== MAIN LOOP ==================

def run():
    tokens = login()
    models = train_models()
    step = 0

    while True:
        df = get_prices(tokens)

        if not data_quality_ok(df):
            time.sleep(POLL_INTERVAL)
            continue

        update_memory(df)

        if step % RETRAIN_EVERY == 0:
            models = train_models()
            update_winrate(tokens)

        news_score = get_news_score()
        if news_score <= NEWS_NEGATIVE_THRESHOLD:
            print("📰 Negative news — trading paused")
            time.sleep(POLL_INTERVAL)
            continue

        adx = ADXIndicator(df.high, df.low, df.close).adx().iloc[-1]
        if adx >= 30:
            regime = "strong"
        elif adx >= 20:
            regime = "medium"
        else:
            time.sleep(POLL_INTERVAL)
            continue

        model, scaler = models[regime]
        if not model:
            time.sleep(POLL_INTERVAL)
            continue

        x = scaler.transform(
            df.close.values[-SEQ_LEN:].reshape(-1, 1)
        )

        prob = model.predict(
            x.reshape(1, SEQ_LEN, 1),
            verbose=0
        )[0][0]

        buy_score = prob * Q[f"{regime}_BUY"] * (1 + news_score * 0.05)
        sell_score = (1 - prob) * Q[f"{regime}_SELL"] * (1 + news_score * 0.05)

        if buy_score > THRESHOLD_BUY:
            open_trade(tokens, "BUY", df, regime, news_score)
            update_q(f"{regime}_BUY", 1)
        elif sell_score > (1 - THRESHOLD_SELL):
            open_trade(tokens, "SELL", df, regime, news_score)
            update_q(f"{regime}_SELL", 1)

        step += 1
        time.sleep(POLL_INTERVAL)

# ================== START ==================

if __name__ == "__main__":
    run()
