import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Set page configuration for a premium look
st.set_page_config(
    page_title="Occupancy Sensing System",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design & Visual Excellence (Harmonious colors, clean card design, transitions)
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background: radial-gradient(circle at top left, #1a1e29, #0e1117);
        color: #e2e8f0;
    }
    
    /* Title styling */
    .title-container {
        padding: 2rem;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    .title-main {
        font-family: 'Outfit', 'Inter', sans-serif;
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .subtitle-main {
        color: #94a3b8;
        font-size: 1.1rem;
        font-weight: 400;
    }

    /* Cards styling */
    .card {
        background: rgba(255, 255, 255, 0.03);
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.2);
        margin-bottom: 1.5rem;
    }
    
    /* Result Card Styling */
    .result-card-occupied {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(5, 150, 105, 0.2));
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 2.5rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(16, 185, 129, 0.1);
        margin-top: 1rem;
    }
    .result-card-vacant {
        background: linear-gradient(135deg, rgba(100, 116, 139, 0.1), rgba(71, 85, 105, 0.2));
        border: 1px solid rgba(100, 116, 139, 0.3);
        padding: 2.5rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(100, 116, 139, 0.1);
        margin-top: 1rem;
    }
    .status-text {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    /* Sidebar customization */
    .css-1d391tw {
        background-color: #0e1117;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained models
@st.cache_resource
def load_models():
    dt_model = pickle.load(open("dt_model.pkl", "rb"))
    knn_model = pickle.load(open("knn_model.pkl", "rb"))
    rf_model = pickle.load(open("rf_model.pkl", "rb"))
    label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
    return dt_model, knn_model, rf_model, label_encoder

# Try to load models
try:
    dt_model, knn_model, rf_model, label_encoder = load_models()
    models_loaded = True
except Exception as e:
    st.error(f"Error loading models: {e}. Please ensure that the models are trained and saved in the current directory.")
    models_loaded = False

# Title Banner
st.markdown("""
<div class="title-container">
    <div class="title-main">OCCUPANCY SENSING SYSTEM</div>
    <div class="subtitle-main">Real-time room occupancy prediction using environmental parameters</div>
</div>
""", unsafe_allow_html=True)

if models_loaded:
    # Create sidebar for inputs
    st.sidebar.markdown("### 🎛️ Input Parameters")
    
    # Input Sliders & Number inputs with realistic ranges based on dataset
    temp = st.sidebar.slider("Temperature (°C)", min_value=15.0, max_value=30.0, value=22.0, step=0.1)
    humidity = st.sidebar.slider("Humidity (%)", min_value=15.0, max_value=50.0, value=26.0, step=0.1)
    light = st.sidebar.slider("Light (Lux)", min_value=0.0, max_value=1500.0, value=450.0, step=1.0)
    co2 = st.sidebar.slider("CO2 (ppm)", min_value=300.0, max_value=2500.0, value=800.0, step=1.0)
    humidity_ratio = st.sidebar.number_input("Humidity Ratio", min_value=0.001, max_value=0.010, value=0.004, format="%.6f", step=0.0001)

    # Model selection box
    selected_model_name = st.sidebar.selectbox(
        "🧠 Classification Model",
        ["Random Forest (Recommended)", "Decision Tree", "K-Nearest Neighbors (KNN)", "Ensemble (Majority Vote)"]
    )

    # Create the input dataframe
    input_data = pd.DataFrame(
        [[temp, humidity, light, co2, humidity_ratio]],
        columns=["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
    )

    # Main Layout columns
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.markdown("<div class='card'><h3>📊 Sensor Readings Summary</h3>", unsafe_allow_html=True)
        # Display nicely formatted metric cards for the inputs
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Temperature", f"{temp} °C")
        m_col2.metric("Humidity", f"{humidity} %")
        m_col3.metric("Light Level", f"{light} Lux")
        
        m_col4, m_col5 = st.columns(2)
        m_col4.metric("CO2 Level", f"{co2} ppm")
        m_col5.metric("Humidity Ratio", f"{humidity_ratio:.5f}")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='card'><h3>🔮 Prediction Result</h3>", unsafe_allow_html=True)
        
        # Predict using selected model
        if "Random Forest" in selected_model_name:
            pred = rf_model.predict(input_data)
            confidence = rf_model.predict_proba(input_data)[0][pred[0]] * 100
        elif "Decision Tree" in selected_model_name:
            pred = dt_model.predict(input_data)
            confidence = dt_model.predict_proba(input_data)[0][pred[0]] * 100
        elif "KNN" in selected_model_name:
            pred = knn_model.predict(input_data)
            confidence = knn_model.predict_proba(input_data)[0][pred[0]] * 100
        else:
            # Ensemble majority vote
            pred_rf = rf_model.predict(input_data)[0]
            pred_dt = dt_model.predict(input_data)[0]
            pred_knn = knn_model.predict(input_data)[0]
            votes = [pred_rf, pred_dt, pred_knn]
            pred = [max(set(votes), key=votes.count)]
            confidence = (votes.count(pred[0]) / 3.0) * 100

        # Decode target label
        decoded_label = label_encoder.inverse_transform(pred)[0]
        
        if decoded_label == 1:
            st.markdown(f"""
            <div class="result-card-occupied">
                <div class="status-text" style="color: #10b981;">👤 OCCUPIED</div>
                <p style="color: #a7f3d0; margin-bottom: 0;">The sensors detect occupant presence.</p>
                <p style="font-size: 0.9rem; color: #6ee7b7; margin-top: 5px;">Model Confidence: {confidence:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card-vacant">
                <div class="status-text" style="color: #94a3b8;">🚪 VACANT</div>
                <p style="color: #cbd5e1; margin-bottom: 0;">No occupant presence detected.</p>
                <p style="font-size: 0.9rem; color: #94a3b8; margin-top: 5px;">Model Confidence: {confidence:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)

    # Comparison block at the bottom
    st.markdown("<div class='card'><h3>🔄 Model Comparison</h3>", unsafe_allow_html=True)
    comp_col1, comp_col2, comp_col3 = st.columns(3)
    
    pred_dt_all = label_encoder.inverse_transform(dt_model.predict(input_data))[0]
    pred_knn_all = label_encoder.inverse_transform(knn_model.predict(input_data))[0]
    pred_rf_all = label_encoder.inverse_transform(rf_model.predict(input_data))[0]
    
    status_map = {1: "Occupied 🟢", 0: "Vacant 🔴"}
    
    comp_col1.markdown(f"**Decision Tree Prediction:**")
    comp_col1.markdown(f"### {status_map[pred_dt_all]}")
    comp_col2.markdown(f"**K-Nearest Neighbors Prediction:**")
    comp_col2.markdown(f"### {status_map[pred_knn_all]}")
    comp_col3.markdown(f"**Random Forest Prediction:**")
    comp_col3.markdown(f"### {status_map[pred_rf_all]}")
    st.markdown("</div>", unsafe_allow_html=True)
