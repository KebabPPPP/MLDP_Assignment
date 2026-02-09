import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="COE Premium Predictor", layout="centered")

st.title("COE Next Premium Predictor")
st.write("Predict the **next COE premium** using the latest record + engineered history features.")

# -----------------------------
# Load model & data
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("best_model.joblib")

@st.cache_data
def load_data():
    return pd.read_csv("COEBiddingResultsPrices.csv")

model = load_model()
df = load_data()

# -----------------------------
# Validate schema
# -----------------------------
needed_raw = {"month", "bidding_no", "vehicle_class", "quota", "bids_received", "bids_success", "premium"}
missing_raw = needed_raw - set(df.columns)
if missing_raw:
    st.error(f"CSV missing required columns: {sorted(list(missing_raw))}")
    st.stop()

# -----------------------------
# Parse & clean
# -----------------------------
df["month"] = pd.to_datetime(df["month"], errors="coerce")
df = df.dropna(subset=["month"])

num_cols = ["quota", "bids_received", "bids_success", "premium", "bidding_no"]
for c in num_cols:
    df[c] = (
        df[c].astype(str)
             .str.replace(",", "", regex=False)
             .str.strip()
    )
    df[c] = pd.to_numeric(df[c], errors="coerce")
df[num_cols] = df[num_cols].fillna(0)

df = df.sort_values(["vehicle_class", "month", "bidding_no"]).reset_index(drop=True)

# -----------------------------
# Feature engineering
# -----------------------------
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

expected_cols = [
    "quota", "bids_success", "bids_received",
    "demand_supply_ratio", "success_rate",
    "year", "month_num", "quarter",
    "premium_lag1", "premium_lag2", "premium_lag3",
    "premium_roll_mean_3", "premium_roll_std_3",
    "bidding_no", "vehicle_class"
]

missing_engineered = set(expected_cols) - set(df.columns)
if missing_engineered:
    st.error(f"Engineered columns missing: {sorted(list(missing_engineered))}")
    st.stop()

# -----------------------------
# UI Tabs (polish)
# -----------------------------
tab1, tab2 = st.tabs(["📌 Predictor", "ℹ️ Model Info"])

with tab1:
    vehicle_classes = sorted(df["vehicle_class"].dropna().unique().tolist())

    # Requirement-friendly: "before any option is selected"
    vc_choice = st.selectbox(
        "Vehicle Class",
        ["-- Select a category --"] + vehicle_classes,
        index=0
    )

    if vc_choice == "-- Select a category --":
        st.info("Select a vehicle class to view the latest record and run a prediction.")
        st.stop()

    vc = vc_choice

    latest = df[df["vehicle_class"] == vc].sort_values(["month", "bidding_no"]).tail(1)
    if latest.empty:
        st.error("No records found for this vehicle class.")
        st.stop()

    st.subheader("Latest Record Used (default context)")
    st.dataframe(latest)

    base_quota = int(latest["quota"].iloc[0])
    base_received = int(latest["bids_received"].iloc[0])
    base_success = int(latest["bids_success"].iloc[0])
    base_bidding_no = int(latest["bidding_no"].iloc[0])

    # Per-category keys
    k_quota = f"quota_in_{vc}"
    k_recv = f"received_in_{vc}"
    k_succ = f"success_in_{vc}"
    k_bidno = f"bidno_in_{vc}"

    # Init defaults once per category
    if k_quota not in st.session_state:
        st.session_state[k_quota] = base_quota
    if k_recv not in st.session_state:
        st.session_state[k_recv] = base_received
    if k_succ not in st.session_state:
        st.session_state[k_succ] = base_success
    if k_bidno not in st.session_state:
        st.session_state[k_bidno] = base_bidding_no

    st.subheader("Adjust Scenario Inputs (Optional)")
    st.caption("Scenario testing: you can tweak bidding stats. History features (lags/rolling) remain from the latest record.")

    col1, col2 = st.columns(2)
    with col1:
        quota_in = st.number_input("Quota", min_value=0, value=st.session_state[k_quota], step=1, key=k_quota)
        received_in = st.number_input("Bids Received", min_value=0, value=st.session_state[k_recv], step=1, key=k_recv)
    with col2:
        success_in = st.number_input("Bids Successful", min_value=0, value=st.session_state[k_succ], step=1, key=k_succ)
        bidding_no_in = st.number_input("Bidding No (optional)", min_value=0, value=st.session_state[k_bidno], step=1, key=k_bidno)

    # Validation
    if success_in > received_in:
        st.warning("Bids Successful cannot exceed Bids Received. Please adjust.")
    if quota_in == 0:
        st.warning("Quota is 0 → demand/supply ratio becomes 0. This is an unrealistic scenario (but allowed).")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Reset inputs to latest record"):
            st.session_state[k_quota] = base_quota
            st.session_state[k_recv] = base_received
            st.session_state[k_succ] = base_success
            st.session_state[k_bidno] = base_bidding_no
            st.rerun()
    with c2:
        st.caption("Tip: Use Reset after switching categories or testing extreme values.")

    # Build model row
    X_latest = latest[expected_cols].copy()

    # Apply overrides
    X_latest.loc[:, "quota"] = quota_in
    X_latest.loc[:, "bids_received"] = received_in
    X_latest.loc[:, "bids_success"] = success_in
    X_latest.loc[:, "bidding_no"] = bidding_no_in

    # Recompute dependent features
    X_latest.loc[:, "demand_supply_ratio"] = (received_in / quota_in) if quota_in != 0 else 0
    X_latest.loc[:, "success_rate"] = (success_in / received_in) if received_in != 0 else 0

    with st.expander("Show final model input row (what the model actually sees)"):
        st.dataframe(X_latest)

    # Predict
    if st.button("Predict Next Premium"):
        if success_in > received_in:
            st.error("Fix the scenario inputs first: Bids Successful must be ≤ Bids Received.")
            st.stop()

        try:
            pred = model.predict(X_latest)[0]
            st.success(f"Predicted next premium for {vc}: {pred:,.2f}")
        except Exception as e:
            st.error("Prediction failed:")
            st.code(str(e))

with tab2:
    st.subheader("What this model predicts")
    st.write("**Target:** next bidding premium for the selected vehicle class (a regression problem).")

    st.subheader("Features used")
    st.write(", ".join(expected_cols))

    st.subheader("How scenario inputs affect prediction")
    st.write(
        "- Scenario inputs change **quota / bids_received / bids_success (and optional bidding_no)**.\n"
        "- The app recomputes **demand_supply_ratio** and **success_rate**.\n"
        "- History features (**premium_lag1/2/3** and rolling stats) stay based on the latest record."
    )
