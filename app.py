from pathlib import Path
import pickle
import textwrap

import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# 1. PATHS AND PAGE CONFIG
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "house_proce_prediction4.pickle"
TRAIN_PATH = BASE_DIR / "train.csv"
FEATURES_PATH = BASE_DIR / "model_features.pkl"

st.set_page_config(
    page_title="EstateIQ | House Price AI",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 2. CUSTOM CSS
# =========================================================

st.markdown(
    textwrap.dedent("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #f4f7fb;
        color: #172033;
    }

    .block-container {
        max-width: 1500px;
        padding: 1.5rem 2rem 3rem;
    }

    [data-testid="stSidebar"] {
        background: #101b2d;
    }

    [data-testid="stSidebar"] * {
        color: #e5edf8 !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: #26364d;
    }

    .hero {
        background:
            linear-gradient(105deg, rgba(13,27,46,.97), rgba(13,27,46,.78)),
            url("https://images.unsplash.com/photo-1600596542815-ffad4c1539a9?auto=format&fit=crop&w=1800&q=85");
        background-size: cover;
        background-position: center;
        border-radius: 24px;
        padding: 42px;
        min-height: 260px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        margin-bottom: 28px;
        box-shadow: 0 18px 40px rgba(15,23,42,.12);
    }

    .hero-tag {
        color: #70e1d1;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .hero-title {
        color: #ffffff;
        font-size: 38px;
        font-weight: 800;
        letter-spacing: -1.5px;
        margin: 12px 0;
    }

    .hero-desc {
        color: #d5e1ef;
        font-size: 15px;
        max-width: 650px;
        line-height: 1.7;
    }

    .hero-pill {
        display: inline-block;
        margin-top: 18px;
        background: rgba(255,255,255,.12);
        color: #d6fff7;
        border: 1px solid rgba(255,255,255,.2);
        border-radius: 30px;
        padding: 9px 16px;
        font-size: 12px;
        font-weight: 600;
        width: fit-content;
    }

    .page-heading {
        color: #14253e;
        font-size: 27px;
        font-weight: 800;
        letter-spacing: -.8px;
        margin-bottom: 5px;
    }

    .page-subheading {
        color: #77869b;
        font-size: 13px;
        margin-bottom: 22px;
    }

    .section-heading {
        font-size: 19px;
        font-weight: 800;
        color: #172b46;
        margin: 25px 0 14px;
    }

    .stat-card {
        background: #ffffff;
        border: 1px solid #e8edf5;
        border-radius: 18px;
        padding: 20px;
        min-height: 145px;
        box-shadow: 0 4px 16px rgba(23,43,70,.035);
        transition: transform .2s ease, box-shadow .2s ease;
    }

    .stat-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 28px rgba(23,43,70,.08);
    }

    .stat-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 40px;
        height: 40px;
        border-radius: 12px;
        font-size: 20px;
        margin-bottom: 14px;
    }

    .stat-label {
        color: #8190a5;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: .7px;
    }

    .stat-value {
        color: #172b46;
        font-size: 25px;
        font-weight: 800;
        letter-spacing: -.7px;
        margin-top: 5px;
    }

    .stat-note {
        color: #93a0b2;
        font-size: 11px;
        margin-top: 5px;
    }

    .panel {
        background: #ffffff;
        border: 1px solid #e8edf5;
        border-radius: 20px;
        padding: 22px;
        box-shadow: 0 4px 16px rgba(23,43,70,.035);
    }

    .panel-title {
        color: #172b46;
        font-size: 16px;
        font-weight: 800;
    }

    .panel-subtitle {
        color: #93a0b2;
        font-size: 12px;
        margin-top: 5px;
    }

    .result-card {
        background: linear-gradient(125deg, #10243c 0%, #163d5b 55%, #087f8c 100%);
        border-radius: 24px;
        padding: 34px;
        color: white;
        box-shadow: 0 16px 40px rgba(8,80,105,.16);
    }

    .result-label {
        color: #b5e9e7;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 1.5px;
    }

    .result-price {
        color: #ffffff;
        font-size: 43px;
        font-weight: 800;
        margin: 12px 0;
        letter-spacing: -1.5px;
    }

    .result-caption {
        color: #c5dce8;
        font-size: 13px;
    }

    [data-testid="stForm"] {
        background: #ffffff;
        border: 1px solid #e7edf5;
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 5px 20px rgba(23,43,70,.04);
    }

    .stButton button,
    .stFormSubmitButton button,
    .stDownloadButton button {
        background: #087f8c;
        color: white;
        border: none;
        border-radius: 11px;
        font-weight: 700;
        min-height: 45px;
    }

    .stButton button:hover,
    .stFormSubmitButton button:hover,
    .stDownloadButton button:hover {
        background: #066c77;
        color: white;
    }

    .info-note {
        background: #e8f7f6;
        color: #17636b;
        padding: 15px 18px;
        border-radius: 13px;
        border-left: 4px solid #0891a2;
        font-size: 13px;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #e8edf5;
        border-radius: 15px;
        overflow: hidden;
    }

    .footer {
        text-align: center;
        color: #98a5b5;
        font-size: 11px;
        padding: 28px 0 8px;
    }
    </style>
    """),
    unsafe_allow_html=True,
)


# =========================================================
# 3. LOAD MODEL AND DATA
# =========================================================

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


# =========================================================
# 4. VALIDATE FILES
# =========================================================

for required_file in [MODEL_PATH, TRAIN_PATH, FEATURES_PATH]:
    if not required_file.exists():
        st.error(f"Required file not found: {required_file.name}")
        st.stop()

try:
    model, model_features = load_artifacts()
    train_df = load_training_data()
except Exception as e:
    st.error(f"Could not load model/data files: {e}")
    st.stop()


# =========================================================
# 5. FIND TARGET COLUMN
# =========================================================

target_candidates = [
    "SalePrice", "saleprice", "sale_price", "price", "Price",
    "house_price", "HousePrice", "selling_price", "target"
]

target_col = next(
    (col for col in target_candidates if col in train_df.columns),
    None
)

if target_col is None:
    st.error("Target column not found in train.csv.")
    st.stop()

raw_feature_cols = [col for col in train_df.columns if col != target_col]


# =========================================================
# 6. DATASET STATISTICS
# =========================================================

prices = pd.to_numeric(train_df[target_col], errors="coerce").dropna()

total_properties = len(train_df)
average_price = float(prices.mean()) if not prices.empty else 0
median_price = float(prices.median()) if not prices.empty else 0
minimum_price = float(prices.min()) if not prices.empty else 0
maximum_price = float(prices.max()) if not prices.empty else 0
missing_values = int(train_df.isnull().sum().sum())


# =========================================================
# 7. SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown(
        '<div style="font-size:25px;font-weight:800;">◈ EstateIQ</div>',
        unsafe_allow_html=True
    )
    st.caption("PROPERTY INTELLIGENCE PLATFORM")
    st.divider()

    page = st.radio(
        "WORKSPACE",
        [
            "Overview",
            "Price Estimator",
            "Data Explorer",
            "Project Details"
        ]
    )

    st.divider()
    st.caption("MODEL STATUS")
    st.markdown("🟢 Model loaded")
    st.caption(f"{len(model_features)} model features")
    st.caption(f"{total_properties:,} dataset records")
    st.divider()
    st.caption("POWERED BY")
    st.markdown("Python · Machine Learning · Streamlit")


# =========================================================
# 8. REUSABLE UI FUNCTIONS
# =========================================================

def show_hero():
    st.markdown(
        textwrap.dedent("""
        <div class="hero">
            <div class="hero-tag">PROPERTY INTELLIGENCE · AI POWERED</div>
            <div class="hero-title">Find the value behind every home.</div>
            <div class="hero-desc">
                Explore housing market insights and estimate property values
                using a machine learning model trained on historical data.
            </div>
            <div class="hero-pill">✦ Smart valuation · Data-driven insights</div>
        </div>
        """),
        unsafe_allow_html=True,
    )


def show_stat(icon, label, value, note, icon_bg="#e0f2fe"):
    st.markdown(
        textwrap.dedent(f"""
        <div class="stat-card">
            <div class="stat-icon" style="background:{icon_bg};">{icon}</div>
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
            <div class="stat-note">{note}</div>
        </div>
        """),
        unsafe_allow_html=True,
    )


# =========================================================
# 9. OVERVIEW DASHBOARD
# =========================================================

if page == "Overview":

    show_hero()

    st.markdown(
        '<div class="page-heading">Market Overview</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="page-subheading">A quick snapshot of your housing dataset.</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        show_stat("🏘️", "PROPERTIES", f"{total_properties:,}",
                  "Records in dataset", "#e0f2fe")

    with c2:
        show_stat("💰", "AVERAGE VALUE", f"${average_price:,.0f}",
                  "Mean sale price", "#dcfce7")

    with c3:
        show_stat("📈", "MEDIAN VALUE", f"${median_price:,.0f}",
                  "Middle sale price", "#fef3c7")

    with c4:
        show_stat("🧠", "MODEL FEATURES", f"{len(model_features):,}",
                  "Encoded input features", "#ede9fe")

    st.write("")

    left, right = st.columns([1.4, 1])

    with left:
        st.markdown(
            textwrap.dedent("""
            <div class="panel">
                <div class="panel-title">House Price Distribution</div>
                <div class="panel-subtitle">
                    Distribution of sale prices in the training data
                </div>
            </div>
            """),
            unsafe_allow_html=True,
        )

        if not prices.empty:
            counts, bins = np.histogram(prices, bins=20)
            labels = [
                f"${bins[i]/1000:.0f}k–${bins[i+1]/1000:.0f}k"
                for i in range(len(bins) - 1)
            ]
            hist_df = pd.DataFrame({
                "Price Range": labels,
                "Properties": counts
            })
            st.bar_chart(
                hist_df.set_index("Price Range"),
                color="#0891a2"
            )
        else:
            st.info("No valid price data available.")

    with right:
        st.markdown(
            textwrap.dedent("""
            <div class="panel">
                <div class="panel-title">Price Highlights</div>
                <div class="panel-subtitle">
                    Key values from the training dataset
                </div>
            </div>
            """),
            unsafe_allow_html=True,
        )

        st.write("")
        st.metric("Lowest Sale Price", f"${minimum_price:,.0f}")
        st.metric("Highest Sale Price", f"${maximum_price:,.0f}")
        st.metric("Missing Dataset Values", f"{missing_values:,}")

    # Optional location-based chart, if the dataset has such a column
    area_col = next(
        (
            col for col in train_df.columns
            if col.lower() in ["neighborhood", "location", "area", "city"]
        ),
        None
    )

    if area_col is not None:
        st.markdown(
            '<div class="section-heading">🏙️ Property Values by Area</div>',
            unsafe_allow_html=True
        )

        area_df = train_df[[area_col, target_col]].copy()
        area_df[target_col] = pd.to_numeric(area_df[target_col], errors="coerce")

        area_summary = (
            area_df.groupby(area_col)[target_col]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )

        st.bar_chart(area_summary, color="#6366f1")

    st.markdown(
        '<div class="section-heading">📋 Dataset Preview</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        train_df.head(8),
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# 10. PRICE ESTIMATOR
# =========================================================

elif page == "Price Estimator":

    st.markdown(
        '<div class="page-heading">Property Price Estimator</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="page-subheading">'
        'Enter property details to estimate its value.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="info-note">'
        '✦ Fill in the property information below. The trained '
        'machine learning model will estimate the sale price.'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("")
    form_values = {}

    with st.form("property_prediction_form"):

        st.markdown(
            '<div class="section-heading">🏡 Property Information</div>',
            unsafe_allow_html=True
        )

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

                    is_integer = pd.api.types.is_integer_dtype(
                        train_df[col_name]
                    )

                    chosen = st.number_input(
                        label,
                        min_value=min_val,
                        max_value=max_val,
                        value=float(min(max(default_val, min_val), max_val)),
                        step=1.0 if is_integer else 0.1,
                        key=f"field_{col_name}"
                    )

                    form_values[col_name] = (
                        int(chosen) if is_integer else float(chosen)
                    )

                else:

                    options = series.astype(str).unique().tolist()

                    if not options:
                        options = ["Unknown"]

                    form_values[col_name] = st.selectbox(
                        label,
                        options,
                        key=f"field_{col_name}"
                    )

        st.write("")
        submitted = st.form_submit_button(
            "✨ Estimate Property Value",
            type="primary",
            use_container_width=True
        )

    if submitted:

        try:

            input_raw = pd.DataFrame(
                [form_values],
                columns=raw_feature_cols
            )

            for col_name in raw_feature_cols:
                if pd.api.types.is_numeric_dtype(train_df[col_name]):
                    input_raw[col_name] = pd.to_numeric(
                        input_raw[col_name],
                        errors="coerce"
                    )

            input_encoded = pd.get_dummies(input_raw)

            input_ready = input_encoded.reindex(
                columns=model_features,
                fill_value=0
            ).fillna(0)

            prediction = model.predict(input_ready)
            price = float(prediction[0])

            st.session_state["prediction"] = price
            st.session_state["property_details"] = form_values.copy()

        except Exception as e:
            st.error("Prediction failed. Check the model and feature files.")
            st.code(str(e))

    if "prediction" in st.session_state:

        price = st.session_state["prediction"]
        details = st.session_state.get("property_details", {})

        st.write("")
        st.success("Prediction completed successfully!")

        st.markdown(
            textwrap.dedent(f"""
            <div class="result-card">
                <div class="result-label">YOUR ESTIMATED PROPERTY VALUE</div>
                <div class="result-price">${price:,.2f}</div>
                <div class="result-caption">
                    Estimated by your trained machine learning model
                </div>
            </div>
            """),
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-heading">Prediction Details</div>',
            unsafe_allow_html=True
        )

        details_df = pd.DataFrame(
            list(details.items()),
            columns=["Property Feature", "Selected Value"]
        )

        st.dataframe(
            details_df,
            use_container_width=True,
            hide_index=True
        )

        report_df = pd.DataFrame([
            {**details, "Estimated Price": price}
        ])

        st.download_button(
            "⬇️ Download Prediction Report",
            data=report_df.to_csv(index=False).encode("utf-8"),
            file_name="property_prediction_report.csv",
            mime="text/csv"
        )

        st.caption(
            "This prediction is an estimate, not a guaranteed market valuation."
        )


# =========================================================
# 11. DATA EXPLORER
# =========================================================

elif page == "Data Explorer":

    st.markdown(
        '<div class="page-heading">Dataset Explorer</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="page-subheading">'
        'Explore, filter and download the housing dataset.'
        '</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Dataset Records", f"{train_df.shape[0]:,}")
    c2.metric("Dataset Columns", f"{train_df.shape[1]:,}")
    c3.metric("Missing Values", f"{missing_values:,}")

    st.write("")

    search = st.text_input(
        "🔍 Search dataset",
        placeholder="Enter a feature value or category..."
    )

    display_df = train_df.copy()

    if search:
        mask = display_df.astype(str).apply(
            lambda col: col.str.contains(
                search,
                case=False,
                na=False
            )
        ).any(axis=1)

        display_df = display_df[mask]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Download Full Dataset",
        data=train_df.to_csv(index=False).encode("utf-8"),
        file_name="house_price_dataset.csv",
        mime="text/csv"
    )


# =========================================================
# 12. PROJECT DETAILS
# =========================================================

elif page == "Project Details":

    st.markdown(
        '<div class="page-heading">About EstateIQ</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="page-subheading">'
        'Machine learning based property valuation platform.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        textwrap.dedent("""
        <div class="panel">
            <div class="panel-title">🏡 House Price Prediction</div>
            <br>
            EstateIQ is a machine learning application designed to estimate
            property sale prices using housing data.
            <br><br>
            Users can enter property information, process features and
            receive an estimated value from a trained machine learning model.
        </div>
        """),
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-heading">Technology Stack</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Language", "Python")
    c2.metric("ML Framework", "Scikit-learn")
    c3.metric("Dashboard", "Streamlit")

    st.markdown(
        '<div class="section-heading">Project Workflow</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    1. Load historical housing data.
    2. Preprocess property features.
    3. Train a machine learning model.
    4. Save the trained model and feature list.
    5. Collect user property details.
    6. Generate a house price estimate.
    """)

    st.markdown(
        '<div class="section-heading">Project Files</div>',
        unsafe_allow_html=True
    )

    st.code("""
app.py
train.csv
house_proce_prediction4.pickle
model_features.pkl
requirements.txt
README.md
""")

    st.info(
        "This project is intended for educational and demonstration purposes."
    )


# =========================================================
# 13. FOOTER
# =========================================================

st.markdown(
    textwrap.dedent("""
    <div class="footer">
        ESTATEIQ · PROPERTY INTELLIGENCE
        <br>
        Machine Learning · Python · Streamlit
    </div>
    """),
    unsafe_allow_html=True,
)
=======
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

