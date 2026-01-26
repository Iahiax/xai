#pip install requests websocket-client tensorflow scikit-learn ta polars rich matplotlib numpy
import requests
import websocket
import json
import numpy as np
import datetime
import ta
import polars as pl
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Attention
from sklearn.preprocessing import StandardScaler
from rich.console import Console
from rich.table import Table

# ============================
# إعدادات الحساب
# ============================
API_KEY = "4YVCSQJ8FGNXZQ7K"
BASE_URL = "https://demo-api-capital.com"

headers = {
    "X-CAP-API-KEY": API_KEY,
    "Content-Type": "application/json"
}

# جلب رصيد الحساب التجريبي مباشرة
response = requests.get(BASE_URL + "/api/v1/accounts", headers=headers)
if response.status_code == 200:
    account_data = response.json()
    balance = account_data[0]["balance"]["available"]  # الرصيد المتاح من الحساب
    print("💰 رصيد الحساب التجريبي:", balance)
else:
    print("خطأ في الاتصال:", response.text)
    balance = 1000  # fallback إذا فشل الاتصال

# ============================
# بناء نموذج LSTM + Attention
# ============================
def build_model(window, features):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(window, features)))
    model.add(Dropout(0.2))
    model.add(Attention())
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="tanh"))
    model.compile(optimizer="adam", loss="mse")
    return model

window = 60
features = 10
model = build_model(window, features)
scaler = StandardScaler()

# ============================
# منطق التداول التراكمي مع رافعة 1:200
# ============================
position = 0
entry_price = 0
leverage = 200
profits = []
wins = 0
losses = 0
trade_count = 0
equity_curve = []  # منحنى الرصيد

console = Console()

def show_stats():
    table = Table(title="📊 إحصائيات التداول المباشرة", show_lines=True)
    table.add_column("المؤشر", justify="center")
    table.add_column("القيمة", justify="center")

    win_rate = (wins / trade_count) * 100 if trade_count > 0 else 0
    avg_profit = sum([p for p in profits if p > 0]) / wins if wins > 0 else 0
    avg_loss = sum([p for p in profits if p < 0]) / losses if losses > 0 else 0

    table.add_row("عدد الصفقات", str(trade_count))
    table.add_row("الصفقات الرابحة", str(wins))
    table.add_row("الصفقات الخاسرة", str(losses))
    table.add_row("نسبة النجاح %", f"{win_rate:.2f}")
    table.add_row("متوسط الربح", f"{avg_profit:.2f}")
    table.add_row("متوسط الخسارة", f"{avg_loss:.2f}")
    table.add_row("الرصيد التراكمي", f"{balance:.2f}")

    console.clear()
    console.print(table)

    # رسم منحنى الرصيد
    if len(equity_curve) > 1:
        plt.figure(figsize=(10,5))
        plt.plot(equity_curve, color="blue", label="منحنى الرصيد")
        plt.axhline(equity_curve[0], color="gray", linestyle="--", label="الرصيد الابتدائي")
        plt.title("📈 منحنى الرصيد التراكمي")
        plt.xlabel("عدد الصفقات")
        plt.ylabel("الرصيد ($)")
        plt.legend()
        plt.show()

def execute_trade(signal, price):
    global balance, position, entry_price, profits, wins, losses, trade_count
    
    trade_size = (balance * leverage) / price
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if signal == 1:  # شراء
        if position == -1:
            profit = (entry_price - price) * trade_size
            balance += profit
            profits.append(profit)
            trade_count += 1
            if profit > 0: wins += 1
            else: losses += 1
            console.print(f"[{now}] ✦ إغلاق بيع | ربح/خسارة: {profit:.2f} | رصيد: {balance:.2f}")
        position = 1
        entry_price = price
        console.print(f"[{now}] ✦ فتح شراء عند {price:.5f}")

    elif signal == -1:  # بيع
        if position == 1:
            profit = (price - entry_price) * trade_size
            balance += profit
            profits.append(profit)
            trade_count += 1
            if profit > 0: wins += 1
            else: losses += 1
            console.print(f"[{now}] ✦ إغلاق شراء | ربح/خسارة: {profit:.2f} | رصيد: {balance:.2f}")
        position = -1
        entry_price = price
        console.print(f"[{now}] ✦ فتح بيع عند {price:.5f}")
    
    equity_curve.append(balance)
    show_stats()

# ============================
# WebSocket لجلب بيانات EUR/USD
# ============================
prices = []

def run_model(prices, current_price):
    df = pl.DataFrame({"Close": prices})
    df = df.with_columns([
        pl.Series("RSI", ta.momentum.RSIIndicator(df["Close"].to_numpy(), window=14).rsi()),
        pl.Series("MACD", ta.trend.MACD(df["Close"].to_numpy()).macd()),
        pl.Series("EMA20", ta.trend.EMAIndicator(df["Close"].to_numpy(), window=20).ema_indicator()),
        pl.Series("EMA50", ta.trend.EMAIndicator(df["Close"].to_numpy(), window=50).ema_indicator()),
        pl.Series("EMA200", ta.trend.EMAIndicator(df["Close"].to_numpy(), window=200).ema_indicator()),
        pl.Series("ATR", ta.volatility.AverageTrueRange(df["Close"].to_numpy(), df["Close"].to_numpy(), df["Close"].to_numpy()).average_true_range()),
        pl.Series("STOCH", ta.momentum.StochasticOscillator(df["Close"].to_numpy(), df["Close"].to_numpy(), df["Close"].to_numpy()).stoch()),
        pl.Series("BB_High", ta.volatility.BollingerBands(df["Close"].to_numpy(), window=20).bollinger_hband()),
        pl.Series("BB_Low", ta.volatility.BollingerBands(df["Close"].to_numpy(), window=20).bollinger_lband()),
        pl.Series("ADX", ta.trend.ADXIndicator(df["Close"].to_numpy(), df["Close"].to_numpy(), df["Close"].to_numpy(), window=14).adx())
    ]).drop_nulls()

    if df.shape[0] < window:
        return

    features_data = df.select(["Close","RSI","MACD","EMA20","EMA50","EMA200","ATR","STOCH","BB_High","BB_Low","ADX"]).to_numpy()[-window:]
    scaled = scaler.fit_transform(features_data)
    X = np.array([scaled])
    pred = model.predict(X, verbose=0)
    signal = 1 if pred > 0 else -1

    execute_trade(signal, current_price)

def on_message(ws, message):
    global prices
    data = json.loads(message)
    if "tick" in data:
        price = float(data["tick"]["bid"])
        prices.append(price)
        if len(prices) > window:
            run_model(prices, price)

def on_open(ws):
    sub_msg = {
        "destination": "marketData.subscribe",
        "payload": {
            "instrument": "EUR/USD",
            "interval": "MINUTE"
        }
    }
    ws.send(json.dumps(sub_msg))

ws = websocket.WebSocketApp(
    "wss://demo-streaming-capital.com/connect",
    header={"X-CAP-API-KEY": API_KEY},
    on_message=on_message,
    on_open=on_open
)

ws.run_forever()
