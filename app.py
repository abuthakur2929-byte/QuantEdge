import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="QuantEdge",
    page_icon="📈",
    layout="wide",
)

st.title("📈 QuantEdge")
st.caption("Research & Backtesting Platform • Paper Trading Only")


# =========================================================
# INDICATORS
# =========================================================

def calculate_indicators(df):
    df = df.copy()

    # Trend
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # ATR
    prev_close = df["Close"].shift(1)

    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    df["ATR"] = true_range.rolling(14).mean()

    # Relative Volume
    df["Volume_MA20"] = df["Volume"].rolling(20).mean()
    df["RelativeVolume"] = (
        df["Volume"] /
        df["Volume_MA20"].replace(0, np.nan)
    )

    # VWAP
    typical_price = (
        df["High"] +
        df["Low"] +
        df["Close"]
    ) / 3

    cumulative_volume = df["Volume"].cumsum()

    df["VWAP"] = (
        (typical_price * df["Volume"]).cumsum()
        / cumulative_volume.replace(0, np.nan)
    )

    # Price-action breakout
    df["PreviousHigh20"] = (
        df["High"].rolling(20).max().shift(1)
    )

    df["PreviousLow20"] = (
        df["Low"].rolling(20).min().shift(1)
    )

    return df


# =========================================================
# STRATEGY BY YOGI
# =========================================================

def generate_signals(df):
    df = df.copy()

    df["BuyScore"] = 0
    df["SellScore"] = 0

    # -------------------------
    # BUY CONDITIONS
    # -------------------------

    trend_buy = (
        (df["Close"] > df["EMA20"]) &
        (df["EMA20"] > df["EMA50"]) &
        (df["EMA50"] > df["EMA200"])
    )

    vwap_buy = df["Close"] > df["VWAP"]

    rsi_buy = (
        (df["RSI"] >= 50) &
        (df["RSI"] <= 70)
    )

    volume_buy = df["RelativeVolume"] >= 1.20

    breakout_buy = (
        df["Close"] > df["PreviousHigh20"]
    )

    # -------------------------
    # SELL CONDITIONS
    # -------------------------

    trend_sell = (
        (df["Close"] < df["EMA20"]) &
        (df["EMA20"] < df["EMA50"]) &
        (df["EMA50"] < df["EMA200"])
    )

    vwap_sell = df["Close"] < df["VWAP"]

    rsi_sell = (
        (df["RSI"] >= 30) &
        (df["RSI"] <= 50)
    )

    volume_sell = df["RelativeVolume"] >= 1.20

    breakout_sell = (
        df["Close"] < df["PreviousLow20"]
    )

    # Score each confirmation
    df["BuyScore"] = (
        trend_buy.astype(int)
        + vwap_buy.astype(int)
        + rsi_buy.astype(int)
        + volume_buy.astype(int)
        + breakout_buy.astype(int)
    )

    df["SellScore"] = (
        trend_sell.astype(int)
        + vwap_sell.astype(int)
        + rsi_sell.astype(int)
        + volume_sell.astype(int)
        + breakout_sell.astype(int)
    )

    # Market regime
    trending_market = (
        (df["EMA20"] > df["EMA50"]) |
        (df["EMA20"] < df["EMA50"])
    )

    df["MarketRegime"] = np.where(
        trending_market,
        "TRENDING",
        "SIDEWAYS"
    )

    # Minimum 4/5 confirmations
    df["Signal"] = "HOLD"

    df.loc[
        (df["BuyScore"] >= 4) &
        (df["MarketRegime"] == "TRENDING"),
        "Signal"
    ] = "BUY"

    df.loc[
        (df["SellScore"] >= 4) &
        (df["MarketRegime"] == "TRENDING"),
        "Signal"
    ] = "SELL"

    # Dynamic ATR risk levels
    df["Entry"] = np.nan
    df["StopLoss"] = np.nan
    df["Target"] = np.nan

    buy_mask = df["Signal"] == "BUY"
    sell_mask = df["Signal"] == "SELL"

    # 1.5 ATR stop, 3 ATR target = 1:2 risk/reward
    df.loc[buy_mask, "Entry"] = df.loc[buy_mask, "Close"]

    df.loc[buy_mask, "StopLoss"] = (
        df.loc[buy_mask, "Close"]
        - 1.5 * df.loc[buy_mask, "ATR"]
    )

    df.loc[buy_mask, "Target"] = (
        df.loc[buy_mask, "Close"]
        + 3.0 * df.loc[buy_mask, "ATR"]
    )

    df.loc[sell_mask, "Entry"] = df.loc[sell_mask, "Close"]

    df.loc[sell_mask, "StopLoss"] = (
        df.loc[sell_mask, "Close"]
        + 1.5 * df.loc[sell_mask, "ATR"]
    )

    df.loc[sell_mask, "Target"] = (
        df.loc[sell_mask, "Close"]
        - 3.0 * df.loc[sell_mask, "ATR"]
    )

    return df


# =========================================================
# UPLOAD DATA
# =========================================================

st.subheader("📂 Upload Market Data")

uploaded_file = st.file_uploader(
    "Upload OHLCV CSV",
    type=["csv"]
)

if uploaded_file is None:
    st.info(
        "Upload OHLCV market data to run the QuantEdge strategy."
    )
    st.stop()


# =========================================================
# LOAD CSV
# =========================================================

try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()


df.columns = [
    str(col).strip()
    for col in df.columns
]


required_columns = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]


missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]


if missing_columns:
    st.error(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )

    st.write("Required columns:")
    st.write(required_columns)

    st.write("Columns found:")
    st.write(list(df.columns))

    st.stop()


# Convert numeric columns
for column in required_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# Remove invalid rows
df = df.dropna(
    subset=required_columns
).reset_index(drop=True)


if len(df) < 250:
    st.warning(
        "At least 250 OHLCV rows are recommended "
        "for EMA200 and reliable strategy calculations."
    )


# =========================================================
# RUN STRATEGY
# =========================================================

df = calculate_indicators(df)

df = generate_signals(df)


# =========================================================
# DASHBOARD
# =========================================================

st.success(
    f"QuantEdge strategy engine loaded — {len(df):,} rows"
)


buy_count = int(
    (df["Signal"] == "BUY").sum()
)

sell_count = int(
    (df["Signal"] == "SELL").sum()
)

hold_count = int(
    (df["Signal"] == "HOLD").sum()
)


col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Rows",
    f"{len(df):,}"
)

col2.metric(
    "BUY Signals",
    f"{buy_count:,}"
)

col3.metric(
    "SELL Signals",
    f"{sell_count:,}"
)

col4.metric(
    "HOLD",
    f"{hold_count:,}"
)


# =========================================================
# LATEST SIGNAL
# =========================================================

st.subheader("🎯 Latest QuantEdge Signal")

latest = df.iloc[-1]

signal = latest["Signal"]

if signal == "BUY":
    st.success("🟢 BUY")

elif signal == "SELL":
    st.error("🔴 SELL")

else:
    st.info("⚪ HOLD")


signal_col1, signal_col2, signal_col3 = st.columns(3)

signal_col1.metric(
    "Setup Score",
    f"{max(latest['BuyScore'], latest['SellScore'])}/5"
)

signal_col2.metric(
    "Market Regime",
    latest["MarketRegime"]
)

signal_col3.metric(
    "RSI",
    f"{latest['RSI']:.2f}"
    if pd.notna(latest["RSI"])
    else "N/A"
)


# =========================================================
# ENTRY / SL / TARGET
# =========================================================

if signal in ["BUY", "SELL"]:

    st.subheader("🛡️ Trade Plan")

    p1, p2, p3 = st.columns(3)

    p1.metric(
        "Entry",
        f"{latest['Entry']:.2f}"
    )

    p2.metric(
        "Stop Loss",
        f"{latest['StopLoss']:.2f}"
    )

    p3.metric(
        "Target",
        f"{latest['Target']:.2f}"
    )

    st.caption(
        "Dynamic levels use ATR. "
        "Current model uses 1.5× ATR stop and 3× ATR target "
        "(approximately 1:2 risk/reward)."
    )


# =========================================================
# INDICATORS
# =========================================================

st.subheader("📊 Indicators")

indicator_columns = [
    "Close",
    "EMA20",
    "EMA50",
    "EMA200",
    "VWAP",
    "RSI",
    "ATR",
    "RelativeVolume",
    "BuyScore",
    "SellScore",
    "MarketRegime",
    "Signal",
    "Entry",
    "StopLoss",
    "Target",
]

st.dataframe(
    df[indicator_columns].tail(50),
    use_container_width=True
)


# =========================================================
# DATA PREVIEW
# =========================================================

st.subheader("📋 Recent Signals")

signal_columns = [
    "Close",
    "BuyScore",
    "SellScore",
    "MarketRegime",
    "Signal",
    "Entry",
    "StopLoss",
    "Target",
]

st.dataframe(
    df[signal_columns].tail(25),
    use_container_width=True
)


st.divider()

st.caption(
    "QuantEdge • Strategy by Yogi • Research & Backtesting • "
    "Paper Trading Only"
)
