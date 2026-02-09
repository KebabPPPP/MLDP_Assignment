import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="COE Premium Predictor", layout="centered")

st.title("COE Next Premium Predictor")
st.write("Select a vehicle class. The app uses the latest record + engineered history features to predict the next premium.")

# ---- Load model ----
@st.cache_resource
def load_model():
    return joblib.load("best_model.joblib")

# ---- Load dataset ----
@st.cache_data
def load_data():
    return pd.read_csv("COEBiddingResultsPrices.csv")

model = load_model()
df = load_data()

# ---- Required raw columns ----
needed_raw = {"month", "bidding_no", "vehicle_class", "quota", "bids_received", "bids_success", "premium"}
missing_raw = needed_raw - set(df.columns)
if missing_raw:
    st.error(f"CSV missing required columns: {sorted(list(missing_raw))}")
    st.stop()

# ---- Parse month ----
df["month"] = pd.to_datetime(df["month"], errors="coerce")
df = df.dropna(subset=["month"])

# ---- Force numeric columns (fix strings / commas / blanks) ----
num_cols = ["quota", "bids_received", "bids_success", "premium", "bidding_no"]
for c in num_cols:
    df[c] = (
        df[c].astype(str)
             .str.replace(",", "", regex=False)
             .str.strip()
    )
    df[c] = pd.to_numeric(df[c], errors="coerce")
df[num_cols] = df[num_cols].fillna(0)

# ---- Sort properly ----
df = df.sort_values(["vehicle_class", "month", "bidding_no"]).reset_index(drop=True)

# ---- Feature engineering (match engineered columns your model expects) ----
df["demand_supply_ratio"] = df["bids_received"] / df["quota"].replace(0, pd.NA)
df["demand_supply_ratio"] = df["demand_supply_ratio"].fillna(0)

df["success_rate"] = df["bids_success"] / df["bids_received"].replace(0, pd.NA)
df["success_rate"] = df["success_rate"].fillna(0)

df["year"] = df["month"].dt.year
df["month_num"] = df["month"].dt.month
df["quarter"] = df["month"].dt.quarter

df["premium_lag1"] = df.groupby("vehicle_class")["premium"].shift(1)
df["premium_lag2"] = df.groupby("vehicle_class")["premium"].shift(2)
df["premium_lag3"] = df.groupby("vehicle_class")["premium"].shift(3)

# Rolling stats using lag1 (safer, avoids peeking at current premium)
df["premium_roll_mean_3"] = (
    df.groupby("vehicle_class")["premium_lag1"]
      .transform(lambda s: s.rolling(3, min_periods=1).mean())
)
df["premium_roll_std_3"] = (
    df.groupby("vehicle_class")["premium_lag1"]
      .transform(lambda s: s.rolling(3, min_periods=1).std())
)

for c in ["premium_lag1", "premium_lag2", "premium_lag3", "premium_roll_std_3"]:
    df[c] = df[c].fillna(0)

# ---- UI ----
vehicle_classes = sorted(df["vehicle_class"].dropna().unique().tolist())
vc = st.selectbox("Vehicle Class", vehicle_classes)

latest = df[df["vehicle_class"] == vc].sort_values(["month", "bidding_no"]).tail(1)

st.subheader("Latest Record Used")
st.dataframe(latest)

# ---- Scenario Inputs (ONE version only) ----
st.subheader("Adjust Scenario Inputs (Optional)")
st.caption("You can tweak these 3 inputs to see how the prediction changes. History-based features (lags/rolling) stay the same.")

base_quota = int(latest["quota"].iloc[0])
base_received = int(latest["bids_received"].iloc[0])
base_success = int(latest["bids_success"].iloc[0])

# Per-category keys so switching Category A/B/C doesn't keep old values
k_quota = f"quota_in_{vc}"
k_recv = f"received_in_{vc}"
k_succ = f"success_in_{vc}"

# Init defaults once per category
if k_quota not in st.session_state:
    st.session_state[k_quota] = base_quota
if k_recv not in st.session_state:
    st.session_state[k_recv] = base_received
if k_succ not in st.session_state:
    st.session_state[k_succ] = base_success

# Reset button
if st.button("Reset inputs to latest record"):
    st.session_state[k_quota] = base_quota
    st.session_state[k_recv] = base_received
    st.session_state[k_succ] = base_success

quota_in = st.number_input("Quota", min_value=0, value=st.session_state[k_quota], step=1, key=k_quota)
received_in = st.number_input("Bids Received", min_value=0, value=st.session_state[k_recv], step=1, key=k_recv)
success_in = st.number_input("Bids Successful", min_value=0, value=st.session_state[k_succ], step=1, key=k_succ)

# Clamp: bids_success cannot exceed bids_received
if success_in > received_in:
    st.warning("Bids Successful cannot exceed Bids Received. Adjusting it to match.")
    success_in = received_in


# ---- Build input to model ----
expected_cols = [
    "quota", "bids_success", "bids_received",
    "demand_supply_ratio", "success_rate",
    "year", "month_num", "quarter",
    "premium_lag1", "premium_lag2", "premium_lag3",
    "premium_roll_mean_3", "premium_roll_std_3",
    "bidding_no", "vehicle_class"
]

missing = set(expected_cols) - set(latest.columns)
if missing:
    st.error(f"Still missing columns: {sorted(list(missing))}")
    st.stop()

X_latest = latest[expected_cols].copy()

# Apply overrides
X_latest.loc[:, "quota"] = quota_in
X_latest.loc[:, "bids_received"] = received_in
X_latest.loc[:, "bids_success"] = success_in

# Recompute dependent ratio features
X_latest.loc[:, "demand_supply_ratio"] = (received_in / quota_in) if quota_in != 0 else 0
X_latest.loc[:, "success_rate"] = (success_in / received_in) if received_in != 0 else 0

with st.expander("Show final model input row"):
    st.dataframe(X_latest)

if st.button("Predict Next Premium"):
    try:
        pred = model.predict(X_latest)[0]
        st.success(f"Predicted next premium for {vc}: {pred:,.2f}")
    except Exception as e:
        st.error("Prediction failed:")
        st.code(str(e))
