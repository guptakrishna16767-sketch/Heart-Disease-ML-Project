import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Heart Disease AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# DARK PROFESSIONAL CSS
# =========================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background-color: #0b1120;
        color: #f8fafc;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }

    /* Sidebar text */
    section[data-testid="stSidebar"] * {
        color: #f8fafc !important;
    }

    /* Main title */
    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #94a3b8;
        font-size: 18px;
        margin-bottom: 30px;
    }

    /* Cards */
    .card {
        background: linear-gradient(
            145deg,
            #111827,
            #172033
        );
        padding: 25px;
        border-radius: 18px;
        border: 1px solid #263244;
        margin-bottom: 20px;
    }

    .card-title {
        font-size: 17px;
        font-weight: 700;
        color: #cbd5e1;
        margin-bottom: 8px;
    }

    .card-value {
        font-size: 32px;
        font-weight: 800;
        color: #f8fafc;
    }

    .risk-high {
        background: linear-gradient(
            135deg,
            #450a0a,
            #7f1d1d
        );
        padding: 25px;
        border-radius: 18px;
        border: 1px solid #ef4444;
        text-align: center;
    }

    .risk-low {
        background: linear-gradient(
            135deg,
            #052e16,
            #14532d
        );
        padding: 25px;
        border-radius: 18px;
        border: 1px solid #22c55e;
        text-align: center;
    }

    .risk-title {
        font-size: 27px;
        font-weight: 800;
        color: white;
    }

    .risk-text {
        color: #d1d5db;
        font-size: 16px;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        height: 48px;
        font-weight: 700;
        background: #2563eb;
        color: white;
        border: none;
    }

    .stButton > button:hover {
        background: #1d4ed8;
        color: white;
    }

    /* Metric */
    [data-testid="stMetric"] {
        background-color: #111827;
        padding: 18px;
        border-radius: 15px;
        border: 1px solid #263244;
    }

    /* Hide default menu/footer */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = joblib.load("LogisticRegression_heart.pkl")
    scaler = joblib.load("scaler.pkl")
    expected_columns = joblib.load("columns.pkl")

    return model, scaler, expected_columns


model, scaler, expected_columns = load_model()


# =========================================================
# SESSION STATE
# =========================================================

if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "probability" not in st.session_state:
    st.session_state.probability = 0.0

if "patient_data" not in st.session_state:
    st.session_state.patient_data = {}


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

st.sidebar.markdown(
    """
    <div style="text-align:center; padding:20px 0;">
        <div style="font-size:45px;">❤️</div>
        <h2>Heart Disease AI</h2>
        <p style="color:#94a3b8;">
            Machine Learning Healthcare Dashboard
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "❤️ Prediction",
        "📊 Analytics"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    **Model**

    Logistic Regression

    **Dataset**

    Heart Disease Dataset

    **Purpose**

    Educational / demonstration use only.
    """
)


# =========================================================
# HOME PAGE
# =========================================================

if page == "🏠 Home":

    st.markdown(
        '<div class="main-title">Heart Disease AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Intelligent cardiovascular risk prediction dashboard'
        '</div>',
        unsafe_allow_html=True
    )

    # Hero section
    col1, col2 = st.columns([2, 1])

    with col1:

        st.markdown("""
        <div class="card">

        <h2>AI-Powered Heart Disease Prediction</h2>

        <p style="color:#cbd5e1; font-size:17px; line-height:1.7;">

        This application uses a trained Machine Learning model
        to estimate the probability of heart disease based on
        patient health parameters.

        Enter patient information in the Prediction section
        and receive an instant risk assessment.

        </p>

        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="card" style="text-align:center;">

        <div style="font-size:70px;">🫀</div>

        <h2>ML Healthcare</h2>

        <p style="color:#94a3b8;">
        Logistic Regression
        </p>

        </div>
        """, unsafe_allow_html=True)


    # Features
    st.markdown("## Dashboard Features")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("""
        <div class="card">
        <div style="font-size:35px;">❤️</div>
        <h3>Risk Prediction</h3>
        <p style="color:#94a3b8;">
        Predict potential heart disease risk.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="card">
        <div style="font-size:35px;">📊</div>
        <h3>Analytics</h3>
        <p style="color:#94a3b8;">
        Visualize prediction probability.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="card">
        <div style="font-size:35px;">👤</div>
        <h3>Patient Summary</h3>
        <p style="color:#94a3b8;">
        View patient information clearly.
        </p>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="card">
        <div style="font-size:35px;">🤖</div>
        <h3>Machine Learning</h3>
        <p style="color:#94a3b8;">
        Powered by Logistic Regression.
        </p>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# PREDICTION PAGE
# =========================================================

elif page == "❤️ Prediction":

    st.markdown(
        '<div class="main-title">Patient Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Enter patient health information'
        '</div>',
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # INPUT SECTION
    # -----------------------------------------------------

    with st.container():

        st.markdown("### 👤 Patient Information")

        col1, col2, col3 = st.columns(3)

        with col1:

            age = st.slider(
                "Age",
                18,
                100,
                40
            )

            Sex = st.selectbox(
                "Sex",
                ["M", "F"]
            )

            Chest_pain = st.selectbox(
                "Chest Pain Type",
                ["ATA", "NAP", "TA", "ASY"]
            )

        with col2:

            resting_bp = st.number_input(
                "Resting Blood Pressure",
                min_value=80,
                max_value=200,
                value=120
            )

            Cholesterol = st.number_input(
                "Cholesterol",
                min_value=100,
                max_value=600,
                value=200
            )

            FastingBS = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dL",
                [0, 1]
            )

        with col3:

            RestingECG = st.selectbox(
                "Resting ECG",
                ["Normal", "ST", "LVH"]
            )

            Max_HR = st.slider(
                "Maximum Heart Rate",
                60,
                220,
                150
            )

            Exercise_Angina = st.selectbox(
                "Exercise-Induced Angina",
                ["Y", "N"]
            )

    st.markdown("### 🩺 Additional Parameters")

    col1, col2 = st.columns(2)

    with col1:

        oldpeak = st.slider(
            "Oldpeak (ST Depression)",
            0.0,
            6.0,
            1.0
        )

    with col2:

        st_slope = st.selectbox(
            "ST Slope",
            ["Up", "Flat", "Down"]
        )


    # -----------------------------------------------------
    # PREDICT BUTTON
    # -----------------------------------------------------

    if st.button("🔍 Predict Heart Disease"):

        raw_input = {

            "Age": age,
            "RestingBP": resting_bp,
            "Cholesterol": Cholesterol,
            "FastingBS": FastingBS,
            "MaxHR": Max_HR,
            "Oldpeak": oldpeak,

            "Sex_" + Sex: 1,
            "ChestPainType_" + Chest_pain: 1,
            "RestingECG_" + RestingECG: 1,
            "ExerciseAngina_" + Exercise_Angina: 1,
            "ST_Slope_" + st_slope: 1
        }

        input_df = pd.DataFrame([raw_input])

        # Add missing columns
        for col in expected_columns:

            if col not in input_df.columns:
                input_df[col] = 0

        # Exact training order
        input_df = input_df[expected_columns]

        # Scale
        scaled_input = scaler.transform(input_df)

        # Prediction
        prediction = model.predict(scaled_input)[0]

        probability = model.predict_proba(
            scaled_input
        )[0]

        risk_probability = probability[1] * 100

        # Save session
        st.session_state.prediction_done = True
        st.session_state.prediction = prediction
        st.session_state.probability = risk_probability

        st.session_state.patient_data = {
            "Age": age,
            "Sex": Sex,
            "Chest Pain": Chest_pain,
            "Resting BP": resting_bp,
            "Cholesterol": Cholesterol,
            "Fasting BS": FastingBS,
            "Resting ECG": RestingECG,
            "Max HR": Max_HR,
            "Exercise Angina": Exercise_Angina,
            "Oldpeak": oldpeak,
            "ST Slope": st_slope
        }


    # -----------------------------------------------------
    # RESULTS
    # -----------------------------------------------------

    if st.session_state.prediction_done:

        prediction = st.session_state.prediction
        risk_probability = st.session_state.probability

        st.markdown("---")

        st.markdown("## 📋 Prediction Result")

        if prediction == 1:

            st.markdown(
                """
                <div class="risk-high">

                <div class="risk-title">
                ⚠️ High Risk of Heart Disease
                </div>

                <div class="risk-text">
                The model predicts a higher probability of
                heart disease based on the provided information.
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="risk-low">

                <div class="risk-title">
                ✅ Low Risk of Heart Disease
                </div>

                <div class="risk-text">
                The model predicts a lower probability of
                heart disease based on the provided information.
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        # -------------------------------------------------
        # METRICS
        # -------------------------------------------------

        st.markdown("### 📊 Risk Overview")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Risk Probability",
                f"{risk_probability:.2f}%"
            )

        with c2:
            st.metric(
                "Prediction",
                "High Risk" if prediction == 1
                else "Low Risk"
            )

        with c3:
            st.metric(
                "Model",
                "Logistic Regression"
            )


        # -------------------------------------------------
        # GAUGE
        # -------------------------------------------------

        st.markdown("### 🎯 Probability Gauge")

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=risk_probability,
                number={
                    "suffix": "%",
                    "font": {
                        "size": 35,
                        "color": "white"
                    }
                },
                title={
                    "text": "Heart Disease Risk",
                    "font": {
                        "color": "white"
                    }
                },
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickcolor": "white"
                    },

                    "bar": {
                        "color": "#ef4444"
                    },

                    "bgcolor": "#111827",

                    "borderwidth": 2,

                    "bordercolor": "#374151",

                    "steps": [
                        {
                            "range": [0, 30],
                            "color": "#14532d"
                        },
                        {
                            "range": [30, 70],
                            "color": "#713f12"
                        },
                        {
                            "range": [70, 100],
                            "color": "#7f1d1d"
                        }
                    ]
                }
            )
        )

        gauge.update_layout(
            height=350,
            paper_bgcolor="#0b1120",
            font_color="white"
        )

        st.plotly_chart(
            gauge,
            use_container_width=True
        )


        # -------------------------------------------------
        # PATIENT SUMMARY
        # -------------------------------------------------

        st.markdown("### 👤 Patient Summary")

        patient_df = pd.DataFrame(
            list(
                st.session_state.patient_data.items()
            ),
            columns=["Parameter", "Value"]
        )

        st.dataframe(
            patient_df,
            use_container_width=True,
            hide_index=True
        )


        # -------------------------------------------------
        # PROBABILITY BAR CHART
        # -------------------------------------------------

        st.markdown("### 📈 Prediction Probability")

        probability_df = pd.DataFrame({
            "Outcome": [
                "No Heart Disease",
                "Heart Disease"
            ],

            "Probability": [
                100 - risk_probability,
                risk_probability
            ]
        })

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=probability_df["Outcome"],
                y=probability_df["Probability"],
                text=[
                    f"{x:.2f}%"
                    for x in probability_df["Probability"]
                ],
                textposition="auto"
            )
        )

        fig.update_layout(
            yaxis_title="Probability (%)",
            yaxis_range=[0, 100],
            paper_bgcolor="#0b1120",
            plot_bgcolor="#111827",
            font_color="white",
            height=400
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# =========================================================
# ANALYTICS PAGE
# =========================================================

elif page == "📊 Analytics":

    st.markdown(
        '<div class="main-title">Analytics Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Prediction analytics and patient risk visualization'
        '</div>',
        unsafe_allow_html=True
    )


    if not st.session_state.prediction_done:

        st.info(
            "Run a prediction first to view patient analytics."
        )

    else:

        risk_probability = st.session_state.probability

        prediction = st.session_state.prediction

        # -----------------------------------------------
        # TOP METRICS
        # -----------------------------------------------

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Risk",
                f"{risk_probability:.1f}%"
            )

        with c2:
            st.metric(
                "Age",
                st.session_state.patient_data["Age"]
            )

        with c3:
            st.metric(
                "Cholesterol",
                st.session_state.patient_data["Cholesterol"]
            )

        with c4:
            st.metric(
                "Max HR",
                st.session_state.patient_data["Max HR"]
            )


        # -----------------------------------------------
        # RISK CHART
        # -----------------------------------------------

        st.markdown("### 📊 Risk Distribution")

        analytics_df = pd.DataFrame({
            "Category": [
                "Low Risk",
                "High Risk"
            ],

            "Probability": [
                100 - risk_probability,
                risk_probability
            ]
        })

        fig = go.Figure(
            data=[
                go.Bar(
                    x=analytics_df["Category"],
                    y=analytics_df["Probability"],
                    text=[
                        f"{v:.1f}%"
                        for v in analytics_df["Probability"]
                    ],
                    textposition="auto"
                )
            ]
        )

        fig.update_layout(
            yaxis_title="Probability (%)",
            yaxis_range=[0, 100],
            paper_bgcolor="#0b1120",
            plot_bgcolor="#111827",
            font_color="white",
            height=400
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # -----------------------------------------------
        # PATIENT FACTORS
        # -----------------------------------------------

        st.markdown("### 🩺 Patient Health Factors")

        factors = st.session_state.patient_data

        factor_df = pd.DataFrame({

            "Factor": [
                "Age",
                "Resting BP",
                "Cholesterol",
                "Max HR",
                "Oldpeak"
            ],

            "Value": [
                factors["Age"],
                factors["Resting BP"],
                factors["Cholesterol"],
                factors["Max HR"],
                factors["Oldpeak"]
            ]
        })


        fig2 = go.Figure(
            data=[
                go.Bar(
                    x=factor_df["Factor"],
                    y=factor_df["Value"],
                    text=factor_df["Value"],
                    textposition="auto"
                )
            ]
        )

        fig2.update_layout(
            paper_bgcolor="#0b1120",
            plot_bgcolor="#111827",
            font_color="white",
            height=400,
            yaxis_title="Value"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )


        # -----------------------------------------------
        # PATIENT INFORMATION
        # -----------------------------------------------

        st.markdown("### 👤 Patient Information")

        summary = st.session_state.patient_data

        col1, col2 = st.columns(2)

        with col1:

            st.write(
                f"**Sex:** {summary['Sex']}"
            )

            st.write(
                f"**Chest Pain:** {summary['Chest Pain']}"
            )

            st.write(
                f"**Resting ECG:** {summary['Resting ECG']}"
            )

            st.write(
                f"**Exercise Angina:** "
                f"{summary['Exercise Angina']}"
            )

        with col2:

            st.write(
                f"**Fasting Blood Sugar:** "
                f"{summary['Fasting BS']}"
            )

            st.write(
                f"**ST Slope:** {summary['ST Slope']}"
            )

            st.write(
                f"**Oldpeak:** {summary['Oldpeak']}"
            )

            st.write(
                f"**Resting BP:** "
                f"{summary['Resting BP']} mm Hg"
            )


        # -----------------------------------------------
        # DISCLAIMER
        # -----------------------------------------------

        st.warning(
            "⚠️ This application is an educational Machine "
            "Learning project and should not be used as a "
            "substitute for professional medical diagnosis."
        )