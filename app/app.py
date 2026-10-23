from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODELS = Path(__file__).resolve().parent.parent / "models"


@st.cache_resource
def load():
    model = joblib.load(MODELS / "rf_model.joblib")
    cols = joblib.load(MODELS / "feature_columns.joblib")
    opts = joblib.load(MODELS / "options.joblib")
    return model, cols, opts


model, feature_cols, opts = load()

st.title("India Used Car Price Advisor")
st.caption("Random Forest trained on about 15,000 CarDekho listings. An estimate, not a valuation.")

c1, c2 = st.columns(2)
with c1:
    brand = st.selectbox("Brand", opts["brand"], index=opts["brand"].index("Maruti") if "Maruti" in opts["brand"] else 0)
    fuel = st.selectbox("Fuel type", opts["fuel_type"], index=opts["fuel_type"].index("Petrol") if "Petrol" in opts["fuel_type"] else 0)
    transmission = st.selectbox("Transmission", opts["transmission_type"])
    seller = st.selectbox("Seller type", opts["seller_type"])
with c2:
    age = st.number_input("Vehicle age (years)", 0, 30, 5)
    km = st.number_input("Kilometers driven", 0, 500000, 50000, step=5000)
    engine = st.number_input("Engine (cc)", 600, 6600, 1200, step=50)
    power = st.number_input("Max power (bhp)", 30.0, 650.0, 85.0, step=1.0)

mileage = st.slider("Mileage (km/l)", 4.0, 34.0, 19.0)
seats = st.selectbox("Seats", [2, 4, 5, 6, 7, 8, 9], index=2)

if st.button("Estimate price"):
    row = pd.DataFrame([{
        "vehicle_age": age, "km_driven": km, "mileage": mileage,
        "engine": engine, "max_power": power, "seats": seats,
        "brand": brand, "seller_type": seller,
        "fuel_type": fuel, "transmission_type": transmission,
    }])
    row = pd.get_dummies(row, columns=["brand", "seller_type", "fuel_type", "transmission_type"])
    row = row.reindex(columns=feature_cols, fill_value=0)
    x = row.values.astype(float)

    estimate = float(np.exp(model.predict(x)[0]))
    per_tree = np.exp(np.array([t.predict(x)[0] for t in model.estimators_]))
    low, high = np.percentile(per_tree, [10, 90])

    st.subheader(f"Estimated price: Rs {estimate:,.0f}")
    st.write(f"Most trees in the model fall between Rs {low:,.0f} and Rs {high:,.0f}.")
    st.caption("On test data, about 80% of predictions were within 20% of the actual price. Less common cars (very old, very cheap, or rare brands) are predicted less reliably.")
