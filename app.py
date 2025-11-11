import streamlit as st
import pandas as pd
import joblib
import json
import random
import matplotlib.pyplot as plt
from datetime import datetime
from io import BytesIO
from openai import OpenAI

# -------------------------------
# 🎛 INITIAL SETUP
# -------------------------------
st.set_page_config(page_title="Battery Health Chatbot", page_icon="🔋", layout="centered")

# Load model & metrics
model = joblib.load("models/linreg_soh.pkl")
with open("models/test_metrics.json") as f:
    metrics = json.load(f)
threshold = metrics["threshold"]

# Initialize session variables
if "predictions" not in st.session_state:
    st.session_state.predictions = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------------
# 🎨 HEADER / STYLING
# -------------------------------
st.markdown(
    """
    <style>
    .main-title {font-size: 32px; text-align: center; color: #0E1117; font-weight: bold;}
    .battery-healthy {color: #1DB954; font-weight: bold;}
    .battery-problem {color: #E63946; font-weight: bold;}
    .chatbox {background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 10px;}
    .chat-user {color: #0E76A8; font-weight: bold;}
    .chat-bot {color: #333333; font-weight: bold;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1 class='main-title'>🔋 Battery Pack SOH Predictor + Chatbot</h1>", unsafe_allow_html=True)

st.sidebar.header("Settings")
threshold = st.sidebar.slider("SOH Threshold", 0.1, 5.0, float(threshold), 0.1)

# -------------------------------
# ⚙️ PREDICTION SECTION
# -------------------------------
st.subheader("📈 Predict Battery Pack Health")

if st.button("🔍 Check Battery SOH"):
    df = pd.read_csv("final_project_preprocessed_data.csv")
    X = df.drop(columns=["Pack_SOH"], errors="ignore")

    for col in [f"U{i}" for i in range(1, 22)]:
        if col in X.columns:
            X = X.drop(columns=[col])

    row = X.sample(1, random_state=random.randint(0, 9999))
    predicted_soh = model.predict(row)[0]
    status = "Healthy" if predicted_soh >= threshold else "Problem"

    st.session_state.predictions.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "soh": predicted_soh,
        "status": status
    })

    if status == "Healthy":
        st.markdown(f"### 🟢 **Predicted SOH: {predicted_soh:.2f} — The Battery is Healthy!**")
        st.progress(min(predicted_soh / 5, 1.0))
    else:
        st.markdown(f"### 🔴 **Predicted SOH: {predicted_soh:.2f} — The Battery has a Problem!**")
        st.progress(min(predicted_soh / 5, 1.0))

# -------------------------------
# 📊 RECENT PREDICTIONS GRAPH
# -------------------------------
if st.session_state.predictions:
    st.subheader("📉 Recent Predictions")
    df_pred = pd.DataFrame(st.session_state.predictions)
    fig, ax = plt.subplots()
    ax.plot(df_pred["time"], df_pred["soh"], marker='o')
    ax.set_xlabel("Time")
    ax.set_ylabel("Predicted SOH")
    ax.set_title("SOH Trend Over Time")
    st.pyplot(fig)

# -------------------------------
# 💾 DOWNLOADABLE REPORT
# -------------------------------
if st.session_state.predictions:
    st.subheader("📄 Export Prediction Report")

    buffer = BytesIO()
    df_pred.to_csv(buffer, index=False)
    st.download_button(
        label="📥 Download CSV Report",
        data=buffer.getvalue(),
        file_name="battery_predictions.csv",
        mime="text/csv",
    )

# -------------------------------
# 🤖 CHATBOT SECTION
# -------------------------------
st.subheader("💬 Ask the Chatbot")

user_input = st.text_input("Type your question about batteries:")

if user_input:
    try:
        # Initialize client (API key via .env or environment variable)
        client = OpenAI(api_key="YOUR_API_KEY_HERE")
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a friendly assistant knowledgeable about batteries and sustainability."},
                {"role": "user", "content": user_input}
            ]
        )
        answer = completion.choices[0].message.content
    except Exception:
        answer = "⚠️ ChatGPT API unavailable — please configure a valid API key."

    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Chatbot", answer))

for role, text in st.session_state.chat_history:
    if role == "You":
        st.markdown(f"<div class='chatbox'><span class='chat-user'>🧑 You:</span> {text}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chatbox'><span class='chat-bot'>🤖 Chatbot:</span> {text}</div>", unsafe_allow_html=True)
