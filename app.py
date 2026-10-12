from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "house_proce_prediction4.pickle"
TRAIN_PATH = BASE_DIR / "train.csv"
FEATURES_PATH = BASE_DIR / "model_features.pkl"

st.set_page_config(
    page_title="House Price Prediction", page_icon="🏠", layout="wide"
)
st.title("🏠 House Price Prediction")
st.write("Enter the property details to estimate its sale price.")


@st.cache_resource
def load_artifacts():
  with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
  with open(FEATURES_PATH, "rb") as f:
    features = pickle.load(f)
  return model, features


@st.cache_data
def load_training_data():
  return pd.read_csv(TRAIN_PATH)


# Validation checks
if not MODEL_PATH.exists():
  st.error(f"Model file not found: {MODEL_PATH.name}")
  st.stop()
if not TRAIN_PATH.exists():
  st.error(f"Dataset file not found: {TRAIN_PATH.name}")
  st.stop()
if not FEATURES_PATH.exists():
  st.error(
      "model_features.pkl nahi mila! Pehle Step 1 wala notebook cell run karo."
  )
  st.stop()

try:
  model, model_features = load_artifacts()
  train_df = load_training_data()
except Exception as e:
  st.error(f"Could not load artifacts: {e}")
  st.stop()

# Detect Target Column
target_candidates = [
    "SalePrice",
    "saleprice",
    "sale_price",
    "price",
    "Price",
    "house_price",
    "HousePrice",
    "selling_price",
    "target",
]
target_col = next((c for c in target_candidates if c in train_df.columns), None)

if target_col is None:
  st.error("Could not identify the target column in train.csv.")
  st.stop()

raw_feature_cols = [c for c in train_df.columns if c != target_col]

st.subheader("Property details")
form_values = {}

with st.form("house_details"):
  left, right = st.columns(2)
  for i, col_name in enumerate(raw_feature_cols):
    series = train_df[col_name].dropna()
    container = left if i % 2 == 0 else right
    label = col_name.replace("_", " ").replace(".", " ").title()

    with container:
      if pd.api.types.is_numeric_dtype(train_df[col_name]):
        if series.empty:
          min_val, max_val, default_val = 0.0, 100.0, 0.0
        else:
          min_val = float(series.min())
          max_val = float(series.max())
          default_val = float(series.median())
          if min_val == max_val:
            min_val -= 1.0
            max_val += 1.0

        is_integer = pd.api.types.is_integer_dtype(train_df[col_name])
        chosen = st.number_input(
            label,
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(min(max(default_val, min_val), max_val)),
            step=1.0 if is_integer else 0.1,
            key=f"input_{col_name}",
        )
        form_values[col_name] = int(chosen) if is_integer else float(chosen)
      else:
        options = series.astype(str).unique().tolist() or ["Unknown"]
        form_values[col_name] = st.selectbox(
            label, options, key=f"input_{col_name}"
        )

  submitted = st.form_submit_button("Predict House Price", type="primary")

if submitted:
  try:
    # 1. User form values ko DataFrame me load karo
    input_raw = pd.DataFrame([form_values], columns=raw_feature_cols)

    # 2. Numeric columns ko strictly numeric convert karo
    for col_name in raw_feature_cols:
      if pd.api.types.is_numeric_dtype(train_df[col_name]):
        input_raw[col_name] = pd.to_numeric(
            input_raw[col_name], errors="coerce"
        )

    # 3. Categorical one-hot encoding
    input_encoded = pd.get_dummies(input_raw)

    # 4. EXACT 479 COLUMNS me match karo (Jo column nahi mila use 0 set karega)
    input_ready = input_encoded.reindex(columns=model_features, fill_value=0)
    input_ready = input_ready.fillna(0)

    # 5. Model prediction run karo
    prediction = model.predict(input_ready)
    price = float(prediction[0])

    st.success("Prediction completed!")
    st.metric("Estimated House Price", f"${price:,.2f}")
    st.caption("This is a model estimate, not a guaranteed market price.")

  except Exception as e:
    st.error("Prediction failed. Details:")
    st.code(str(e))

with st.expander("Preview training data"):
  st.dataframe(train_df.head(10), use_container_width=True)
  st.write(f"Rows: {train_df.shape[0]} · Columns: {train_df.shape[1]}")
  st.write("Target column:", target_col)
  st.write("Encoded feature count:", len(model_features))