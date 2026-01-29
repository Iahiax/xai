# =========================================================
# CAPITAL.COM EURUSD AI BOT - FULL FINAL VERSION
# LSTM + RL + NEWS + REAL WIN RATE + AUTO SESSION
# SINGLE FILE - EVERYTHING INCLUDED
# =========================================================

import time, requests, numpy as np, pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from ta.trend import ADXIndicator
from ta.volatility import AverageTrueRange

# ================== CONFIG ==================

API_KEY = "PUT_API_KEY"
IDENTIFIER = "PUT_EMAIL"
PASSWORD = "PUT_PASSWORD"

NEWS_API_KEY = "PUT_NEWS_API_KEY"
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

TRAINING_MEMORY = {"strong": [], "medium": [], "max": 9000}

STATS = {
    "trades": 0,
    "wins": 0,
    "losses": 0,
    "last_trade_time": 0
}

Q = {
    "strong_BUY": 1.0,
    "strong_SELL": 1.0,
    "medium_BUY": 1.0,
    "medium_SELL": 1.0
}

ALPHA = 0.1
GAMMA = 0.9

# ================== AUTH ==================

def login():
    r = requests.post(
        f"{BASE_URL}/session",
        headers={"X-CAP-API-KEY": API_KEY},
        json={"identifier": IDENTIFIER, "password": PASSWORD}
    )
    return {"CST": r.headers["CST"], "X-SECURITY-TOKEN": r.headers["X-SECURITY-TOKEN"]}

def safe_request(method, url, tokens, **kw):
    h = kw.pop("headers", {})
    h.update({
        "X-CAP-API-KEY": API_KEY,
        "CST": tokens["CST"],
        "X-SECURITY-TOKEN": tokens["X-SECURITY-TOKEN"]
    })
    r = requests.request(method, url, headers=h, timeout=10, **kw)
    if r.status_code in [401, 403]:
        tokens.update(login())
        h.update(tokens)
        r = requests.request(method, url, headers=h, timeout=10, **kw)
    return r

# ================== NEWS FILTER ==================

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
        negative = ["crisis", "inflation", "recession", "panic", "collapse", "risk"]
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
        params={"resolution": TIMEFRAME, "max": HISTORY_BARS}
    )
    p = r.json()["prices"]
    return pd.DataFrame([{
        "open": float(x["openPrice"]["bid"]),
        "high": float(x["highPrice"]["bid"]),
        "low": float(x["lowPrice"]["bid"]),
        "close": float(x["closePrice"]["bid"])
    } for x in p])

def data_quality_ok(df):
    if df.isna().sum().sum() > 0:
        return False
    if df.close.pct_change().abs().max() > 0.03:
        return False
    return True

# ================== ML ==================

def build_model(shape):
    m = Sequential([
        LSTM(64, return_sequences=True, input_shape=shape),
        Dropout(0.2),
        LSTM(32),
        Dense(1, activation="sigmoid")
    ])
    m.compile("adam", "binary_crossentropy")
    return m

def update_memory(df):
    adx = ADXIndicator(df.high, df.low, df.close).adx()
    for c, a in zip(df.close, adx):
        if a >= 30:
            TRAINING_MEMORY["strong"].append(c)
        elif a >= 20:
            TRAINING_MEMORY["medium"].append(c)
    for k in ["strong", "medium"]:
        TRAINING_MEMORY[k] = TRAINING_MEMORY[k][-TRAINING_MEMORY["max"]:]

def prepare(bucket):
    d = TRAINING_MEMORY[bucket]
    if len(d) < SEQ_LEN + 50:
        return None, None, None
    sc = MinMaxScaler()
    d = sc.fit_transform(np.array(d).reshape(-1, 1))
    X, y = [], []
    for i in range(SEQ_LEN, len(d)):
        X.append(d[i-SEQ_LEN:i])
        y.append(1 if d[i] > d[i-1] else 0)
    return np.array(X), np.array(y), sc

def train_models():
    out = {}
    for b in ["strong", "medium"]:
        X, y, sc = prepare(b)
        if X is None:
            out[b] = (None, None)
            continue
        m = build_model((X.shape[1], 1))
        m.fit(X, y, epochs=5, batch_size=32, verbose=0)
        out[b] = (m, sc)
    return out

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
            safe_request("DELETE", f"{BASE_URL}/positions/{pos['dealId']}", tokens)

def open_trade(tokens, side, df, regime, news_score):
    if not can_trade():
        return

    close_opposite(tokens, side)

    atr = AverageTrueRange(df.high, df.low, df.close).average_true_range().iloc[-1]
    price = df.close.iloc[-1]

    sl, tp = (
        (price - atr * 1.2, price + atr * 2.4)
        if side == "BUY"
        else (price + atr * 1.2, price - atr * 2.4)
    )

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
    acts = r.json().get("activities", [])
    wins = losses = 0
    for a in acts:
        if a.get("profitAndLoss") is not None:
            if a["profitAndLoss"] > 0:
                wins += 1
            elif a["profitAndLoss"] < 0:
                losses += 1
    STATS["wins"], STATS["losses"] = wins, losses
    if wins + losses > 0:
        print(f"📊 Win Rate: {wins/(wins+losses)*100:.2f}%")

# ================== RUN ==================

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
        regime = "strong" if adx >= 30 else "medium" if adx >= 20 else None
        if not regime:
            time.sleep(POLL_INTERVAL)
            continue

        model, sc = models[regime]
        if not model:
            time.sleep(POLL_INTERVAL)
            continue

        x = sc.transform(df.close.values[-SEQ_LEN:].reshape(-1, 1))
        prob = model.predict(x.reshape(1, SEQ_LEN, 1), verbose=0)[0][0]

        score_buy = prob * Q[f"{regime}_BUY"] * (1 + news_score * 0.05)
        score_sell = (1 - prob) * Q[f"{regime}_SELL"] * (1 + news_score * 0.05)

        if score_buy > THRESHOLD_BUY:
            open_trade(tokens, "BUY", df, regime, news_score)
            update_q(f"{regime}_BUY", 1)
        elif score_sell > (1 - THRESHOLD_SELL):
            open_trade(tokens, "SELL", df, regime, news_score)
            update_q(f"{regime}_SELL", 1)

        step += 1
        time.sleep(POLL_INTERVAL)

# ================== START ==================

if __name__ == "__main__":
    run()
