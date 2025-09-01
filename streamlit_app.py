# streamlit_app.py
import argparse
import joblib
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from fpdf import FPDF
import base64

DEFAULT_MODEL = "artifacts/model.joblib"
DEFAULT_METADATA = "artifacts/metadata.json"

# -------------------------------
# Helper Functions
# -------------------------------
def load_model(path):
    return joblib.load(path)

def get_user_input():
    """Collect patient info from sidebar"""
    st.sidebar.header("🧍 Patient Information")

    age = st.sidebar.number_input("Age", 20, 100, 50)
    sex = st.sidebar.selectbox("Sex", ["M", "F"])
    cp = st.sidebar.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
    rbp = st.sidebar.number_input("RestingBP", 80, 200, 120)
    chol = st.sidebar.number_input("Cholesterol", 100, 600, 200)
    fbs = st.sidebar.selectbox("FastingBS (Fasting Blood Sugar)", [0, 1])
    recg = st.sidebar.selectbox("RestingECG", ["Normal", "ST", "LVH"])
    maxhr = st.sidebar.number_input("MaxHR", 60, 220, 150)
    exang = st.sidebar.selectbox("ExerciseAngina", ["N", "Y"])
    oldpeak = st.sidebar.number_input("Oldpeak", 0.0, 10.0, 1.0, step=0.1)
    slope = st.sidebar.selectbox("ST_Slope", ["Up", "Flat", "Down"])

    data = {
        "Age": age,
        "Sex": sex,
        "ChestPainType": cp,
        "RestingBP": rbp,
        "Cholesterol": chol,
        "FastingBS": fbs,
        "RestingECG": recg,
        "MaxHR": maxhr,
        "ExerciseAngina": exang,
        "Oldpeak": oldpeak,
        "ST_Slope": slope
    }
    return pd.DataFrame([data])

def get_risk(prob, row):
    """Risk level + explanation"""
    if prob < 0.3:
        risk = "Low"
    elif prob < 0.7:
        risk = "Medium"
    else:
        risk = "High"

    explanation = []
    if row["Cholesterol"] > 240:
        explanation.append("High Cholesterol")
    if row["RestingBP"] > 140:
        explanation.append("High Blood Pressure")
    if row["MaxHR"] < 120:
        explanation.append("Low Max HR")
    if row["ExerciseAngina"] == "Y":
        explanation.append("Exercise Angina detected")

    message = " & ".join(explanation) if explanation else "No major risk factors detected"
    return risk, message

def show_result(pred, prob, risk, explanation):
    """Styled card output"""
    colors = {"Low": "#4CAF50", "Medium": "#FFC107", "High": "#F44336"}
    st.markdown(
        f"""
        <div style="padding:15px; border-radius:10px; background-color:{colors[risk]}; color:white;">
            <h3>🩺 Prediction: {"Heart Disease" if pred == 1 else "No Heart Disease"}</h3>
            <p><b>Probability:</b> {prob:.2f}</p>
            <p><b>Risk Level:</b> {risk}</p>
            <p><b>Explanation:</b> {explanation}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def plot_gauge(prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob*100,
        title={'text': "Heart Disease Probability (%)"},
        gauge={'axis': {'range': [0,100]},
               'bar': {'color': "red"},
               'steps': [
                   {'range': [0, 30], 'color': "green"},
                   {'range': [30, 70], 'color': "yellow"},
                   {'range': [70, 100], 'color': "red"}]}
    ))
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# PDF Report Generator
# -------------------------------
def generate_pdf(patient_data, pred, prob, risk, explanation):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Heart Disease Prediction Report", ln=True, align="C")

    pdf.set_font("Arial", "", 12)
    pdf.ln(10)
    pdf.cell(200, 10, "Patient Information:", ln=True)
    for k, v in patient_data.items():
        pdf.cell(200, 10, f"{k}: {v}", ln=True)

    pdf.ln(5)
    pdf.cell(200, 10, f"Prediction: {'Heart Disease' if pred==1 else 'No Heart Disease'}", ln=True)
    pdf.cell(200, 10, f"Probability: {prob:.2f}", ln=True)
    pdf.cell(200, 10, f"Risk Level: {risk}", ln=True)
    pdf.multi_cell(200, 10, f"Explanation: {explanation}")

    return pdf.output(dest="S").encode("latin-1")

# -------------------------------
# Main App
# -------------------------------
def main(model_path):
    # Branding
    st.image("https://cdn-icons-png.flaticon.com/512/1483/1483336.png", width=80)
    st.markdown("<h1 style='color:#1976D2;'>❤️ Heart Disease Risk Predictor</h1>", unsafe_allow_html=True)
    st.caption("AI-powered screening tool for early detection")

    # Load model
    model = load_model(model_path)

    # Sidebar Navigation
    menu = st.sidebar.radio("Navigation", ["🏠 Home", "🔍 Single Prediction", "📂 Batch Prediction", "📑 Reports"])

    if menu == "🏠 Home":
        st.subheader("📊 About this project")
        st.write("This app predicts **Heart Disease Risk** using a machine learning model trained on patient data.")
        st.write("- Single & Batch Predictions")
        st.write("- Risk Explanations")
        st.write("- PDF Reports for Doctors")
        st.success("✅ Built with Streamlit + Flask + ML")

    elif menu == "🔍 Single Prediction":
        input_df = get_user_input()
        st.subheader("📋 Patient Data")
        st.write(input_df)

        if st.button("Predict", key="single"):
            pred = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0][1]
            risk, message = get_risk(proba, input_df.iloc[0])
            show_result(pred, proba, risk, message)
            plot_gauge(proba)

            # PDF Download
            pdf_bytes = generate_pdf(input_df.iloc[0].to_dict(), pred, proba, risk, message)
            b64 = base64.b64encode(pdf_bytes).decode()
            href = f'<a href="data:application/pdf;base64,{b64}" download="report.pdf">📑 Download Report (PDF)</a>'
            st.markdown(href, unsafe_allow_html=True)

    elif menu == "📂 Batch Prediction":
        uploaded_file = st.file_uploader("Upload patient CSV file", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.subheader("📋 Uploaded Data")
            st.write(df.head())

            if st.button("Predict Batch", key="batch"):
                preds = model.predict(df)
                probs = model.predict_proba(df)[:, 1]

                risks, explanations = [], []
                for i in range(len(df)):
                    risk, message = get_risk(probs[i], df.iloc[i])
                    risks.append(risk)
                    explanations.append(message)

                df["Prediction"] = ["Heart Disease" if p==1 else "No Heart Disease" for p in preds]
                df["Probability"] = probs
                df["Risk Level"] = risks
                df["Explanation"] = explanations

                st.subheader("📝 Batch Prediction Results")
                st.write(df)

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Results as CSV", data=csv, file_name="predictions.csv", mime="text/csv")

    elif menu == "📑 Reports":
        st.subheader("📑 Generate Doctor Reports")
        st.write("You can generate **individual PDF reports** after predictions.")
        st.info("➡️ Go to **Single Prediction** and click 'Download Report (PDF)'.")
        st.write("Batch PDF Export can also be added if needed.")

# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args, _ = parser.parse_known_args()
    main(args.model)






