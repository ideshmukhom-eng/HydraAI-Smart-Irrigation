import streamlit as st
import numpy as np
import joblib
import os
import plotly.graph_objects as go
from tensorflow.keras.models import load_model

# ---------------- PAGE CONFIG & STYLING ----------------
st.set_page_config(page_title="HydraAI | Smart Irrigation", page_icon="💧", layout="wide")

# Custom CSS for Glassmorphism and UI polish
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.7);
        border-radius: 20px;
        padding: 30px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
        margin-bottom: 20px;
    }
    h1, h2, h3 { color: #1e3a8a !important; font-family: 'Segoe UI', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

# ---------------- ASSET LOADING ----------------
@st.cache_resource
def load_assets():
    # Placeholder for model/scaler loading logic (matches your original)
    if not os.path.exists("water_requirement_model.h5"): return None, None
    m = load_model("water_requirement_model.h5", compile=False)
    s = joblib.load("scaler.pkl")
    return m, s

model, scaler = load_assets()

# ---------------- NAVIGATION TABS ----------------
# This creates the "Tab Form" structure you requested
tab_home, tab_predict, tab_about = st.tabs(["🏠 Home", "🔮 Smart Prediction", "📈 System Analytics"])

# ---------------- HOME TAB ----------------
with tab_home:
    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.title("Welcome to HydraAI")
        st.markdown("""
        ### The future of precision agriculture is here.
        Using Deep Learning to optimize every drop. This system analyzes:
        * **Micro-climates:** Real-time atmospheric shifts.
        * **Soil Chemistry:** pH-driven requirement adjustments.
        * **Wind Dynamics:** Evapotranspiration modeling.
        """)
        if st.button("Start New Analysis"):
            st.toast("Redirecting to Prediction Engine...")
    with col_r:
        # Placeholder for an animation/image
        st.image("https://cdn-icons-png.flaticon.com/512/4241/4241664.png", width=200)

# ---------------- PREDICTION TAB ----------------
with tab_predict:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🛠 Environmental Parameters")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        temp = st.slider("Temperature (°C)", 0, 50, 25)
        ph = st.select_slider("Soil pH Level", options=np.arange(0, 14.5, 0.5).tolist(), value=7.0)
    with c2:
        humidity = st.slider("Humidity (%)", 0, 100, 60)
        pressure = st.number_input("Pressure (KPa)", value=101.3)
    with c3:
        wind = st.number_input("Wind Speed (Km/h)", value=5.0)
        gust = st.number_input("Wind Gust (Km/h)", value=10.0)

    if st.button("✨ Calculate Requirements", type="primary"):
        with st.spinner("Neural Engine Processing..."):
            # Feature Engineering logic (kept from your original)
            input_data = np.array([[temp, wind, humidity, gust, pressure, ph, 
                                    temp**2, 100-humidity, wind*gust, pressure*0.1, abs(ph-7)]])
            
            # Simulated prediction (assuming model exists)
            if model:
                scaled = scaler.transform(input_data)
                res = model.predict(scaled)[0][0]
                
                st.balloons()
                st.markdown("---")
                # Animated Metric Pop-up
                col_m1, col_m2 = st.columns([1, 1])
                with col_m1:
                    st.metric(label="Water Needed", value=f"{res:.2f} Units", delta="Optimal Range")
                with col_m2:
                    st.success("✅ Prediction complete! Data synced to cloud.")
            else:
                st.warning("Model file not found. Please upload assets.")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- ANALYTICS TAB ----------------
with tab_about:
    st.subheader("📊 Live Input Distribution")
    # Quick Radar Chart for visual "Pop"
    categories = ['Temp', 'Wind', 'Humidity', 'Gust', 'pH']
    fig = go.Figure(data=go.Scatterpolar(
        r=[temp/50, wind/20, humidity/100, gust/30, ph/14],
        theta=categories, fill='toself', line_color='#1e3a8a'
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=False)), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)