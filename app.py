import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="SmartCar AI 🚗",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM CSS - PROFESSIONAL DARK THEME
# =====================================================

st.markdown(
    """
    <style>

    /* Main Background */
    .stApp {
        background: linear-gradient(
            135deg,
            #0f172a 0%,
            #111827 50%,
            #1e293b 100%
        );
        color: #f8fafc;
    }

    /* Main Container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* Header */
    .main-title {
        text-align: center;
        font-size: 45px;
        font-weight: 800;
        color: #38bdf8;
        margin-bottom: 5px;
    }

    .sub-title {
        text-align: center;
        font-size: 18px;
        color: #cbd5e1;
        margin-bottom: 25px;
    }

    /* Cards */
    .info-card {
        background: rgba(30, 41, 59, 0.90);
        border: 1px solid #334155;
        border-radius: 18px;
        padding: 22px;
        margin-bottom: 18px;
        box-shadow: 0px 8px 25px rgba(0, 0, 0, 0.25);
    }

    .card-title {
        color: #38bdf8;
        font-size: 23px;
        font-weight: 700;
        margin-bottom: 12px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #020617 0%,
            #0f172a 100%
        );
        border-right: 1px solid #334155;
    }

    /* Input Labels */
    label {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
    }

    /* Inputs */
    div[data-baseweb="select"] > div,
    input,
    textarea {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border-radius: 10px !important;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        background: linear-gradient(
            90deg,
            #0284c7,
            #2563eb
        );
        color: white;
        font-size: 19px;
        font-weight: 700;
        border: none;
        border-radius: 12px;
        padding: 13px;
        transition: 0.3s;
    }

    .stButton > button:hover {
        background: linear-gradient(
            90deg,
            #0369a1,
            #1d4ed8
        );
        transform: scale(1.02);
    }

    /* Metric */
    div[data-testid="stMetric"] {
        background: linear-gradient(
            135deg,
            #164e63,
            #1e3a8a
        );
        padding: 20px;
        border-radius: 18px;
        border: 1px solid #38bdf8;
    }

    div[data-testid="stMetricLabel"] {
        color: #bae6fd !important;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-size: 32px !important;
    }

    /* Success Message */
    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 14px;
        margin-top: 30px;
        padding: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =====================================================
# LOAD MODEL AND ENCODER
# =====================================================

@st.cache_resource
def load_files():

    encoder = joblib.load(
        "usedcar_price_prediction_encoder.pkl"
    )

    model = joblib.load(
        "usedcar_price_prediction(random_forest).pkl"
    )

    return encoder, model


try:
    encoder, model = load_files()

except Exception as error:
    st.error("❌ Model किंवा Encoder file load होत नाही.")
    st.exception(error)
    st.stop()

# =====================================================
# HEADER
# =====================================================

st.markdown(
    '<div class="main-title">🚗 SmartCar AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    '🤖 Used Car Resale Price Prediction System '
    '| 📊 Machine Learning Powered'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown("## 🚘 SmartCar AI")

    st.write(
        "Welcome! 👋 Enter your car details "
        "and get an estimated resale price."
    )

    st.divider()

    st.markdown("### 📌 Project Information")

    st.write("🔹 Problem Type: Regression")
    st.write("🔹 Algorithm: Random Forest")
    st.write("🔹 Output: Resale Price")
    st.write("🔹 Technology: Python + ML")

    st.divider()

    st.info(
        "💡 Tip: Accurate details like manufacturing year, "
        "kilometers and original price improve prediction quality."
    )

# =====================================================
# INPUT SECTION
# =====================================================

st.markdown(
    '<div class="info-card">',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="card-title">📝 Enter Car Details</div>',
    unsafe_allow_html=True
)

st.write(
    "Please fill in the details below to estimate the resale value. 🚘"
)

st.markdown("</div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

# =====================================================
# COLUMN 1
# =====================================================

with col1:

    st.markdown("### 🏷️ Basic Information")

    brand = st.selectbox(
        "🚘 Brand",
        [
            "maruti",
            "hyundai",
            "tata",
            "honda",
            "toyota",
            "mahindra",
            "kia",
            "ford",
            "renault",
            "volkswagen",
            "skoda",
            "nissan"
        ]
    )

    model_name = st.text_input(
        "🚗 Model Name",
        value="swift"
    ).lower().strip()

    fuel_type = st.selectbox(
        "⛽ Fuel Type",
        [
            "petrol",
            "diesel",
            "cng",
            "electric",
            "hybrid"
        ]
    )

    transmission = st.selectbox(
        "⚙️ Transmission",
        [
            "manual",
            "automatic",
            "amt",
            "cvt",
            "dct"
        ]
    )

# =====================================================
# COLUMN 2
# =====================================================

with col2:

    st.markdown("### 👤 Ownership & Condition")

    owner_type = st.selectbox(
        "👥 Owner Type",
        [
            "first",
            "second",
            "third",
            "fourth"
        ]
    )

    insurance_status = st.selectbox(
        "🛡️ Insurance Status",
        [
            "yes",
            "no",
            "active",
            "expired"
        ]
    )

    service_history = st.selectbox(
        "🔧 Service History",
        [
            "yes",
            "no",
            "available",
            "not available"
        ]
    )

    accident_history = st.selectbox(
        "⚠️ Accident History",
        [
            "no",
            "yes"
        ]
    )

    condition = st.selectbox(
        "✨ Car Condition",
        [
            "excellent",
            "good",
            "average",
            "poor"
        ]
    )

# =====================================================
# COLUMN 3
# =====================================================

with col3:

    st.markdown("### 📊 Car Specifications")

    year = st.number_input(
        "📅 Manufacturing Year",
        min_value=1990,
        max_value=2026,
        value=2020,
        step=1
    )

    kilometers_driven = st.number_input(
        "🛣️ Kilometers Driven",
        min_value=0,
        max_value=500000,
        value=50000,
        step=1000
    )

    engine_cc = st.number_input(
        "🔩 Engine CC",
        min_value=500,
        max_value=6000,
        value=1200,
        step=100
    )

    original_price = st.number_input(
        "💰 Original Price (Lakh)",
        min_value=0.1,
        max_value=500.0,
        value=8.0,
        step=0.1
    )

    city = st.selectbox(
        "📍 City",
        [
            "mumbai",
            "pune",
            "nagpur",
            "delhi",
            "bangalore",
            "hyderabad",
            "chennai",
            "kolkata",
            "ahmedabad",
            "nashik"
        ]
    )

    color = st.selectbox(
        "🎨 Color",
        [
            "white",
            "black",
            "red",
            "blue",
            "silver",
            "grey",
            "green"
        ]
    )

# =====================================================
# FEATURE ENGINEERING
# =====================================================

car_age = 2026 - year

accident_history_value = (
    1 if accident_history == "yes" else 0
)

# =====================================================
# PREDICTION BUTTON
# =====================================================

st.divider()

st.markdown("### 🔮 Predict Your Car's Resale Value")

predict_button = st.button(
    "🚀 Predict Resale Price",
    use_container_width=True
)

if predict_button:

    if model_name == "":
        st.warning("⚠️ Please enter the car model name.")
        st.stop()

    # -------------------------------------------------
    # Input Data
    # Price_Diff intentionally removed
    # -------------------------------------------------

    input_data = pd.DataFrame([{

        "Brand": brand,
        "Model": model_name,
        "Fuel_Type": fuel_type,
        "Transmission": transmission,
        "Owner_Type": owner_type,
        "Insurance_Status": insurance_status,
        "Service_History": service_history,
        "City": city,
        "Color": color,
        "Condition": condition,

        "Year": year,
        "Kilometers_Driven": kilometers_driven,
        "Engine_CC": engine_cc,
        "Accident_History": accident_history_value,
        "Original_Price_Lakh": original_price,
        "Car_Age": car_age

    }])

    categorical_columns = [
        "Brand",
        "Model",
        "Fuel_Type",
        "Transmission",
        "Owner_Type",
        "Insurance_Status",
        "Service_History",
        "City",
        "Color",
        "Condition"
    ]

    numerical_columns = [
        "Year",
        "Kilometers_Driven",
        "Engine_CC",
        "Accident_History",
        "Original_Price_Lakh",
        "Car_Age"
    ]

    try:

        # -------------------------------------------------
        # Encoding
        # -------------------------------------------------

        encoded_data = encoder.transform(
            input_data[categorical_columns]
        )

        if hasattr(encoded_data, "toarray"):
            encoded_data = encoded_data.toarray()

        # -------------------------------------------------
        # Numerical Data
        # -------------------------------------------------

        numerical_data = input_data[
            numerical_columns
        ].values

        # -------------------------------------------------
        # Final Input
        # -------------------------------------------------

        final_input = np.hstack([
            numerical_data,
            encoded_data
        ])

        expected_features = model.n_features_in_
        actual_features = final_input.shape[1]

        # -------------------------------------------------
        # Feature Mismatch Check
        # -------------------------------------------------

        if actual_features != expected_features:

            st.error(
                f"❌ Feature mismatch: Model expects "
                f"{expected_features} features, but input has "
                f"{actual_features} features."
            )

            st.write(
                "🔍 Expected Features:",
                expected_features
            )

            st.write(
                "📊 Actual Features:",
                actual_features
            )

            st.info(
                "ℹ️ Training dataset आणि app मधील columns "
                "same असणे आवश्यक आहे."
            )

            st.stop()

        # -------------------------------------------------
        # Prediction
        # -------------------------------------------------

        prediction = model.predict(final_input)

        predicted_price = float(prediction[0])

        # -------------------------------------------------
        # Result Display
        # -------------------------------------------------

        st.success(
            "🎉 Prediction completed successfully!"
        )

        st.markdown("## 💰 Estimated Resale Price")

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:

            st.metric(
                label="🚗 Resale Price",
                value=f"₹ {predicted_price:.2f} Lakh"
            )

        with result_col2:

            st.metric(
                label="📅 Car Age",
                value=f"{car_age} Years"
            )

        with result_col3:

            st.metric(
                label="🛣️ Kilometers",
                value=f"{kilometers_driven:,} KM"
            )

        
        

        st.info(
            "⚠️ Note: This is an estimated price generated by "
            "the machine learning model. Actual market price may vary."
        )

    except Exception as error:

        st.error("❌ Prediction failed.")

        st.exception(error)

# =====================================================
# FOOTER
# =====================================================

st.markdown(
    """
    <div class="footer">
        🚗 SmartCar AI | Built with Python, Pandas,
        Scikit-learn & Streamlit 🤖
        <br>
        💡 Used Car Resale Price Prediction Project
    </div>
    """,
    unsafe_allow_html=True
)