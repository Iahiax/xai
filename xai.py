#pip install requests websocket-client numpy polars ta scikit-learn tensorflow pycryptodome
import os
import time
import json
import datetime
import requests
import websocket
import numpy as np
import polars as pl
import threading
import io
import pickle
import traceback
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import ta
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_v1_5
import base64

# ============================
# إعدادات الحساب والـ API
# ============================
API_KEY = os.getenv("CAP_API_KEY", "4YVCSQJ8FGNXZQ7K")
EMAIL = os.getenv("CAP_EMAIL", "yahia.x@outlook.sa")
PASSWORD = os.getenv("CAP_PASSWORD", "Yahia-1411")

BASE_URL = "https://demo-api-capital.backend-capital.com/api/v1"
STREAM_URL = "wss://demo-streaming-capital.com/connect"

INSTRUMENT = {"instrument": "EUR/USD", "interval": "MINUTE"}

# ============================
# إعدادات النموذج والتدريب
# ============================
WINDOW = 60
FEATURES = 11
TRAIN_EPOCHS = 10
BATCH_SIZE = 32
RETRAIN_INTERVAL_HOURS = 6  # قابل للتعديل

# ============================
# تخزين داخل الذاكرة
# ============================
_model_weights_bytes = None
_scaler_bytes = None
_training_data_bytes = None
retrain_metrics = []

# ============================
# دوال تشفير وكائنات الجلسة
# ============================
def encrypt_password(encryption_key_b64: str, timestamp: int, password: str) -> str:
    input_str = f"{password}|{timestamp}"
    input_bytes = base64.b64encode(input_str.encode("utf-8"))
    pub_key_bytes = base64.b64decode(encryption_key_b64.encode("utf-8"))
    pub_key = RSA.importKey(pub_key_bytes)
    cipher = PKCS1_v1_5.new(pub_key)
    encrypted_bytes = cipher.encrypt(input_bytes)
    return base64.b64encode(encrypted_bytes).decode("utf-8")

def create_session():
    headers = {"X-CAP-API-KEY": API_KEY}
    resp = requests.get(f"{BASE_URL}/session/encryptionKey", headers=headers, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    encryption_key = data["encryptionKey"]
    timestamp = data["time"]
    encrypted_password = encrypt_password(encryption_key, timestamp, PASSWORD)
    payload = {
        "identifier": EMAIL,
        "encryptedPassword": encrypted_password,
        "time": timestamp
    }
    resp_login = requests.post(f"{BASE_URL}/session", headers=headers, json=payload, timeout=10)
    resp_login.raise_for_status()
    login_data = resp_login.json()
    token = login_data.get("session", {}).get("token")
    if token:
        print("✅ Session token obtained.")
    return token

def ensure_session():
    """يحاول إنشاء جلسة جديدة إذا انتهت صلاحية التوكن أو فشل الاتصال"""
    while True:
        try:
            token = create_session()
            if token:
                return token
        except Exception as e:
            print("❌ خطأ في إنشاء الجلسة:", e)
        print("🔄 إعادة المحاولة بعد 10 ثواني...")
        time.sleep(10)

# ============================
# بناء نموذج GRU أخف
# ============================
def build_model():
    m = Sequential()
    m.add(GRU(64, return_sequences=True, input_shape=(WINDOW, FEATURES)))
    m.add(Dropout(0.2))
    m.add(GRU(32))
    m.add(Dropout(0.2))
    m.add(Dense(32, activation='relu'))
    m.add(Dropout(0.2))
    m.add(Dense(1, activation='tanh'))
    m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse')
    return m

scaler = StandardScaler()
model = build_model()

# ============================
# دوال الحفظ والاستعادة داخل الذاكرة
# ============================
def save_model_in_memory(keras_model):
    global _model_weights_bytes
    try:
        buf = io.BytesIO()
        pickle.dump(keras_model.get_weights(), buf)
        _model_weights_bytes = buf.getvalue()
        print("💾 تم حفظ أوزان النموذج داخل الذاكرة.")
    except Exception as e:
        print("❌ خطأ أثناء حفظ النموذج في الذاكرة:", e)

def load_model_from_memory(keras_model):
    global _model_weights_bytes
    if not _model_weights_bytes:
        return False
    try:
        buf = io.BytesIO(_model_weights_bytes)
        weights = pickle.load(buf)
        keras_model.set_weights(weights)
        print("🔁 تم استعادة أوزان النموذج من الذاكرة.")
        return True
    except Exception as e:
        print("❌ خطأ أثناء استعادة النموذج من الذاكرة:", e)
        return False

def save_scaler_in_memory(scaler_obj):
    global _scaler_bytes
    try:
        _scaler_bytes = pickle.dumps(scaler_obj)
        print("💾 تم حفظ الـ scaler داخل الذاكرة.")
    except Exception as e:
        print("❌ خطأ أثناء حفظ الـ scaler:", e)

def load_scaler_from_memory():
    global _scaler_bytes
    if not _scaler_bytes:
        return None
    try:
        obj = pickle.loads(_scaler_bytes)
        print("🔁 تم استعادة الـ scaler من الذاكرة.")
        return obj
    except Exception as e:
        print("❌ خطأ أثناء استعادة الـ scaler:", e)
        return None

def save_training_data_in_memory(X, y):
    global _training_data_bytes
    try:
        _training_data_bytes = pickle.dumps({"X": X, "y": y})
        print("💾 تم حفظ بيانات التدريب داخل الذاكرة.")
    except Exception as e:
        print("❌ خطأ أثناء حفظ بيانات التدريب:", e)

def load_training_data_from_memory():
    global _training_data_bytes
    if not _training_data_bytes:
        return None, None
    try:
        d = pickle.loads(_training_data_bytes)
        return d.get("X"), d.get("y")
    except Exception as e:
        print("❌ خطأ أثناء استعادة بيانات التدريب:", e)
        return None, None

# ============================
# متغيرات التداول وإحصاءات
# ============================
prices = []
trade_log = []
profits = []
wins = 0
losses = 0
trade_count = 0
position = 0
entry_price = 0.0
leverage = 200

def get_balance():
    return 1000.0

def execute_trade(signal, price):
    global position, entry_price, wins, losses, trade_count
    balance = get_balance()
    trade_size = (balance * leverage) / price
    now = datetime.datetime.utcnow().isoformat()

    record = {
        "time": now,
        "instrument": INSTRUMENT["instrument"],
        "signal": int(signal),
        "price": price,
        "trade_size": trade_size,
        "position_before": position
    }

    if signal == 1:
        if position == -1:
            profit = (entry_price - price) * trade_size
            profits.append(profit)
            trade_count += 1
            if profit > 0:
                wins += 1
            else:
                losses += 1
            record.update({"action": "close_short", "profit": profit})
        position = 1
        entry_price = price
        record.update({"action": "open_long"})
    elif signal == -1:
        if position == 1:
            profit = (price - entry_price) * trade_size
            profits.append(profit)
            trade_count += 1
            if profit > 0:
                wins += 1
            else:
                losses += 1
            record.update({"action": "close_long", "profit": profit})
        position = -1
        entry_price = price
        record.update({"action": "open_short"})

    trade_log.append(record)
    print("🚀 صفقة جديدة:", record)
    show_stats()

def show_stats():
    win_rate = (wins / trade_count) * 100 if trade_count > 0 else 0
    avg_profit = sum([p for p in profits if p > 0]) / wins if wins > 0 else 0
    avg_loss = sum([p for p in profits if p < 0]) / losses if losses > 0 else 0
    print("📊 إحصاءات التداول")
    print("عدد الصفقات:", trade_count, "الرابحة:", wins, "الخاسرة:", losses, f"نسبة النجاح: {win_rate:.2f}%")
    print("متوسط الربح:", f"{avg_profit:.2f}", "متوسط الخسارة:", f"{avg_loss:.2f}", "الرصيد الحالي:", f"{get_balance():.2f}")

# ============================
# جلب بيانات تاريخية من Capital
# ============================
def get_historical_data(instrument="EUR/USD", interval="MINUTE", limit=2000):
    url = f"{BASE_URL}/prices/{instrument}/{interval}"
    headers = {"X-CAP-API-KEY": API_KEY}
    resp = requests.get(url, headers=headers, params={"limit": limit}, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    closes = []
    if isinstance(data, dict) and "prices" in data and isinstance(data["prices"], list):
        for item in data["prices"]:
            if isinstance(item, dict) and ("close" in item or "c" in item):
                closes.append(float(item.get("close", item.get("c"))))
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and ("close" in item or "c" in item):
                closes.append(float(item.get("close", item.get("c"))))
    if not closes:
        raise RuntimeError("❌ لم أتمكن من استخراج أسعار الإغلاق من استجابة الـ API.")
    print(f"📥 جلبت {len(closes)} نقطة سعرية تاريخية.")
    return closes

# ============================
# تدريب النموذج على البيانات التاريخية
# ============================
def train_model_on_history():
    global scaler, model, wins, losses, trade_count
    try:
        prices_hist = get_historical_data(INSTRUMENT["instrument"], INSTRUMENT["interval"], limit=2000)
        df = pl.DataFrame({"Close": prices_hist}).drop_nulls()

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

        features = df.select([
            "Close","RSI","MACD","EMA20","EMA50","EMA200",
            "ATR","STOCH","BB_High","BB_Low","ADX"
        ]).to_numpy()

        X, y = [], []
        for i in range(WINDOW, len(features)):
            X.append(features[i-WINDOW:i])
            y.append(1 if features[i][0] > features[i-1][0] else -1)

        X = np.array(X)
        y = np.array(y)
        if X.shape[0] == 0:
            print("⚠️ لا توجد بيانات كافية للتدريب.")
            return False

        n_samples = X.shape[0]
        X_flat = X.reshape((n_samples * WINDOW, FEATURES))
        scaler = StandardScaler().fit(X_flat)
        X_scaled = np.array([scaler.transform(x) for x in X])

        model = build_model()
        es = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True, verbose=1)
        model.fit(X_scaled, y, epochs=TRAIN_EPOCHS, batch_size=BATCH_SIZE, callbacks=[es], verbose=1)

        save_model_in_memory(model)
        save_scaler_in_memory(scaler)
        save_training_data_in_memory(X_scaled, y)

        # سجل المقاييس داخل الذاكرة (يعتمد على سجل التداول الحالي)
        win_rate = (wins / trade_count) * 100 if trade_count > 0 else 0
        retrain_metrics.append({
            "time": datetime.datetime.utcnow().isoformat(),
            "win_rate": win_rate,
            "trades": trade_count,
            "wins": wins,
            "losses": losses
        })
        print("✅ تم تدريب النموذج وحفظه داخل الذاكرة.")
        return True
    except Exception as e:
        print("❌ خطأ أثناء تدريب النموذج على البيانات التاريخية:", e)
        traceback.print_exc()
        return False

# ============================
# التنبؤ من الأسعار اللحظية
# ============================
def compute_indicators_and_predict(prices_list):
    try:
        df = pl.DataFrame({"Close": prices_list})
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
    except Exception as e:
        print("❌ خطأ في حساب المؤشرات:", e)
        return None

    if df.shape[0] < WINDOW:
        return None

    features_data = df.select([
        "Close","RSI","MACD","EMA20","EMA50","EMA200",
        "ATR","STOCH","BB_High","BB_Low","ADX"
    ]).to_numpy()[-WINDOW:]

    loaded_scaler = load_scaler_from_memory()
    if loaded_scaler is not None:
        try:
            scaled = loaded_scaler.transform(features_data)
        except Exception:
            scaled = StandardScaler().fit_transform(features_data)
    else:
        scaled = StandardScaler().fit_transform(features_data)

    X = np.array([scaled])
    load_model_from_memory(model)

    try:
        pred = model.predict(X, verbose=0)
        signal = 1 if pred > 0 else -1
        return float(pred), int(signal)
    except Exception as e:
        print("❌ خطأ أثناء التنبؤ:", e)
        return None

# ============================
# WebSocket للتشغيل الحي مع إعادة اتصال تلقائي
# ============================
def start_websocket(token):
    headers = [f"X-CAP-API-KEY: {API_KEY}", f"X-SECURITY-TOKEN: {token}"]

    def on_message(ws, message):
        try:
            data = json.loads(message)
            if "tick" in data and isinstance(data["tick"], dict) and "bid" in data["tick"]:
                price = float(data["tick"]["bid"])
            elif "price" in data:
                price = float(data["price"])
            else:
                return
            prices.append(price)
            if len(prices) > 5000:
                del prices[:len(prices)-5000]
            if len(prices) >= WINDOW:
                result = compute_indicators_and_predict(prices)
                if result:
                    _, signal = result
                    execute_trade(signal, price)
        except Exception as e:
            print("❌ خطأ في on_message:", e)
            traceback.print_exc()

    def on_open(ws):
        print("✅ WebSocket متصل")
        try:
            sub_msg = {"destination": "marketData.subscribe", "payload": INSTRUMENT}
            ws.send(json.dumps(sub_msg))
        except Exception as e:
            print("❌ خطأ أثناء إرسال رسالة الاشتراك:", e)

    def on_error(ws, error):
        print("❌ WebSocket error:", error)

    def on_close(ws, close_status_code, close_msg):
        print("🔌 WebSocket مغلق", close_status_code, close_msg)
        # إعادة الاتصال تلقائيًا بعد تأخير قصير
        time.sleep(5)
        new_token = ensure_session()
        start_websocket(new_token)

    ws = websocket.WebSocketApp(
        STREAM_URL,
        header=headers,
        on_message=on_message,
        on_open=on_open,
        on_error=on_error,
        on_close=on_close
    )
    ws.run_forever()

# ============================
# إعادة التدريب الدورية مع طباعة سجل المقاييس
# ============================
def periodic_retrain(interval_hours=RETRAIN_INTERVAL_HOURS):
    def job():
        print(f"⏱️ بدء إعادة تدريب دورية: {datetime.datetime.utcnow().isoformat()}")
        ok = train_model_on_history()
        if ok:
            print("🔁 إعادة التدريب الدورية اكتملت بنجاح.")
        else:
            print("⚠️ إعادة التدريب الدورية فشلت.")
        # طباعة سجل المقاييس داخل البوت
        if retrain_metrics:
            print("📊 سجل المقاييس حتى الآن:")
            for m in retrain_metrics:
                print(f"- {m['time']} | WinRate: {m['win_rate']:.2f}% | صفقات: {m['trades']} | رابحة: {m['wins']} | خاسرة: {m['losses']}")
        else:
            print("📊 لا توجد مقاييس مسجلة بعد.")
        # جدولة التشغيل التالي
        threading.Timer(interval_hours * 3600, job).start()
    # بدء أول تشغيل بعد 5 ثواني لتجنب حجب الإقلاع
    threading.Timer(5, job).start()
    print(f"🕒 تم جدولة إعادة التدريب كل {interval_hours} ساعة.")

# ============================
# تشغيل البوت
# ============================
if __name__ == "__main__":
    print("🚀 بدء تشغيل بوت EUR/USD — تدريب أولي + حفظ داخل الذاكرة + بث لحظي + إعادة تدريب دورية")
    try:
        trained = train_model_on_history()
        if not trained:
            print("⚠️ التدريب الأولي فشل. سيتابع البوت محاولة التشغيل لكن التنبؤات قد تكون غير دقيقة.")
        # استعادة النموذج والـ scaler من الذاكرة إن وُجدت
        load_model_from_memory(model)
        loaded_scaler = load_scaler_from_memory()
        if loaded_scaler:
            scaler = loaded_scaler
        # بدء إعادة التدريب الدورية (تطبع السجل بعد كل تشغيل)
        periodic_retrain(RETRAIN_INTERVAL_HOURS)
        # إنشاء الجلسة والتشغيل الحي
        token = ensure_session()
        start_websocket(token)
    except KeyboardInterrupt:
        print("🛑 تم إيقاف البوت بواسطة المستخدم.")
    except Exception as e:
        print("❌ خطأ أثناء تشغيل البوت:", e)
        traceback.print_exc()
