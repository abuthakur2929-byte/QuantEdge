import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="QuantEdge",
    page_icon="📈",
    layout="wide",
)

st.title("📈 QuantEdge")
st.caption("Research & Backtesting Platform • Paper Trading Only")

st.subheader("📂 Upload Market Data")

uploaded_file = st.file_uploader(
    "Upload OHLCV CSV",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload a CSV file to begin.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

df.columns = [str(col).strip() for col in df.columns]

required_columns = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )

    st.write("Required:")
    st.write(required_columns)

    st.write("Found:")
    st.write(list(df.columns))

    st.stop()

for column in required_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

invalid_rows = int(
    df[required_columns]
    .isna()
    .any(axis=1)
    .sum()
)

st.success(
    f"CSV loaded successfully — {len(df):,} rows"
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Rows",
    f"{len(df):,}"
)

col2.metric(
    "Columns",
    f"{len(df.columns):,}"
)

col3.metric(
    "Invalid OHLCV Rows",
    f"{invalid_rows:,}"
)

if invalid_rows > 0:
    st.warning(
        f"{invalid_rows:,} rows contain missing or invalid OHLCV values."
    )

st.subheader("📊 Data Preview")

st.dataframe(
    df.head(25),
    use_container_width=True
)

st.divider()

st.success(
    "QuantEdge foundation is working. "
    "Strategy and backtesting modules will be added next."
)
