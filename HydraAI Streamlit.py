import streamlit as st
import numpy as np
import joblib
import os
import time
import plotly.graph_objects as go
from tensorflow.keras.models import load_model

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="HydraAI | Smart Irrigation",
    page_icon="💧",
    layout="wide"
)

# ---------------- LIGHT UI CSS ----------------
st.markdown("""
<style>
html, body, .stApp {
    background-color: #ffffff !important;
    color: #000000 !important;
}

* {
    color: #000000 !important;
}

h1, h2, h3 {
    color: #1e3a8a !important;
}

.stButton > button {
    background-color: #1e3a8a !important;
    color: white !important;
    border-radius: 8px;
    padding: 0.6em 1.2em;
    border: none;
}

[data-testid="stMetricValue"] {
    color: #1e3a8a !important;
    font-weight: bold;
}

.glass-card {
    background: #f9f9f9 !important;
    border-radius: 15px;
    padding: 25px;
    border: 1px solid #e0e0e0;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_assets():
    if not os.path.exists("water_requirement_model.h5"):
        return None, None
    model = load_model("water_requirement_model.h5", compile=False)
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_assets()

# ---------------- TABS ----------------
tab_home, tab_predict, tab_about = st.tabs(
    ["🏠 Home", "🔮 Smart Prediction", "📈 System Analytics"]
)

# ---------------- HOME ----------------
with tab_home:
    col_l, col_r = st.columns([2, 1])

    with col_l:
        st.title("Welcome to HydraAI")
        st.markdown("""
        ### The future of precision agriculture is here.
        Using Deep Learning to optimize every drop.

        **System analyzes:**
        - Micro-climates  
        - Soil Chemistry  
        - Wind Dynamics  
        """)

        st.button("Start New Analysis")

    with col_r:
        st.image(
            "https://cdn-icons-png.flaticon.com/512/4241/4241664.png",
            width=220
        )

# ---------------- PREDICTION ----------------
with tab_predict:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Environmental Parameters")

    c1, c2, c3 = st.columns(3)

    with c1:
        temp = st.slider("Temperature (°C)", 0, 50, 25)
        ph = st.select_slider(
            "Soil pH Level",
            options=np.arange(0, 14.5, 0.5).tolist(),
            value=7.0
        )

    with c2:
        humidity = st.slider("Humidity (%)", 0, 100, 60)
        pressure = st.slider("Pressure (KPa)", 80.0, 120.0, 101.3, 0.1)

    with c3:
        wind = st.slider("Wind Speed (Km/h)", 0.0, 50.0, 5.0, 0.5)
        gust = st.slider("Wind Gust (Km/h)", 0.0, 80.0, 10.0, 0.5)

    if st.button("✨ Calculate Requirements"):

        with st.spinner("🤖 AI Engine Processing..."):
            time.sleep(1)

            input_data = np.array([[
                temp, wind, humidity, gust, pressure, ph,
                temp**2,
                100 - humidity,
                wind * gust,
                pressure * 0.1,
                abs(ph - 7)
            ]])

            if model is not None:
                scaled = scaler.transform(input_data)
                res = model.predict(scaled)[0][0]

                # 🎉 Effects
                st.balloons()
                st.toast("✅ Prediction Completed Successfully!", icon="💧")

                # 📊 Progress Animation
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress.progress(i + 1)

                st.markdown("---")

                # 💎 Premium Result Card
                st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg,#1e3a8a,#2563eb);
                        padding:30px;
                        border-radius:15px;
                        color:white;
                        text-align:center;
                        font-size:22px;
                        font-weight:bold;
                        box-shadow:0 8px 25px rgba(0,0,0,0.25);
                    ">
                        💧 Recommended Water Requirement <br><br>
                        <span style="font-size:45px;">{res:.2f} Units</span>
                    </div>
                """, unsafe_allow_html=True)

            else:
                st.error("⚠ Model file not found.")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- ANALYTICS ----------------
with tab_about:
    st.subheader("Live Input Distribution")

    categories = ['Temp', 'Wind', 'Humidity', 'Gust', 'pH']

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[
            temp / 50,
            wind / 50,
            humidity / 100,
            gust / 80,
            ph / 14
        ],
        theta=categories,
        fill='toself'
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=False)),
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)
