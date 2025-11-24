import streamlit as st
import pandas as pd
import joblib
import json
import random
import matplotlib.pyplot as plt
from datetime import datetime
from io import BytesIO
import os
from dotenv import load_dotenv
from google import genai  # Gemini SDK

# -------------------------------
# 🎛 PAGE & GEMINI SETUP
# -------------------------------
st.set_page_config(
    page_title="Battery SOH Dashboard & Chatbot",
    page_icon="🔋",
    layout="wide",
)

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.0-flash"

client = None
if GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        st.sidebar.error(f"Gemini client error: {e}")
else:
    st.sidebar.error("GEMINI_API_KEY not set in .env")

# -------------------------------
# 🔋 LOAD ML MODEL & METRICS
# -------------------------------
model = joblib.load("models/linreg_soh.pkl")
with open("models/test_metrics.json") as f:
    metrics = json.load(f)

threshold_default = metrics.get("threshold", 0.6)
r2 = metrics.get("r2", None)
mae = metrics.get("mae", None)
rmse = metrics.get("rmse", None)

# -------------------------------
# 🧠 SESSION STATE
# -------------------------------
if "predictions" not in st.session_state:
    st.session_state.predictions = []  # list of dicts: time, soh, status
if "messages" not in st.session_state:
    st.session_state.messages = []     # list of {"role": "user"/"assistant", "content": str}

# -------------------------------
# 🎨 SIDEBAR
# -------------------------------
st.sidebar.title("Settings")
threshold = st.sidebar.slider(
    "SOH Threshold (healthy if ≥ threshold)",
    min_value=0.1,
    max_value=5.0,
    value=float(threshold_default),
    step=0.1,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Performance (test set)**")
if r2 is not None:
    st.sidebar.metric("R²", f"{r2:.3f}")
if mae is not None:
    st.sidebar.metric("MAE", f"{mae:.4f}")
if rmse is not None:
    st.sidebar.metric("RMSE", f"{rmse:.4f}")

# -------------------------------
# 🧩 MAIN LAYOUT – TABS
# -------------------------------
st.markdown(
    "<h1 style='text-align:center;'>🔋 Battery Pack SOH Dashboard & Chatbot</h1>",
    unsafe_allow_html=True,
)
st.caption(
    "Use the **Dashboard** to predict battery State of Health (SOH) and visualize trends, "
    "and the **Chatbot** to ask battery-related questions using Google Gemini."
)

tab_dash, tab_chat = st.tabs(["📊 Dashboard", "🤖 Chatbot"])

# --------------------------------------------------
# TAB 1 – DASHBOARD
# --------------------------------------------------
with tab_dash:
    st.subheader("📈 Predict Battery Pack Health")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        if st.button("🔍 Check Battery SOH", use_container_width=True):
            df = pd.read_csv("final_project_preprocessed_data.csv")
            X = df.drop(columns=["Pack_SOH"], errors="ignore")

            # Drop U1..U21 if present, to match training
            for col in [f"U{i}" for i in range(1, 22)]:
                if col in X.columns:
                    X = X.drop(columns=[col])

            # Random sample prediction
            row = X.sample(1, random_state=random.randint(0, 9999))
            predicted_soh = model.predict(row)[0]
            status = "Healthy" if predicted_soh >= threshold else "Problem"

            st.session_state.predictions.append(
                {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "soh": predicted_soh,
                    "status": status,
                }
            )

        # Show latest prediction (if any)
        if st.session_state.predictions:
            latest = st.session_state.predictions[-1]
            soh = latest["soh"]
            status = latest["status"]

            if status == "Healthy":
                st.markdown(
                    f"### 🟢 Predicted SOH: **{soh:.2f}** — The Battery is **Healthy**!"
                )
            else:
                st.markdown(
                    f"### 🔴 Predicted SOH: **{soh:.2f}** — The Battery has a **Problem**!"
                )

            st.progress(min(soh / 5, 1.0))
            st.caption(
                "The progress bar is scaled to a nominal SOH range of 0–5. "
                "Values above the threshold are considered healthy."
            )
        else:
            st.info("Click **Check Battery SOH** to generate a prediction.")

    with col_right:
        st.markdown("#### ℹ️ About This Model")
        st.write(
            "- **Model**: Linear Regression (Scikit-Learn pipeline)\n"
            f"- **Train/test split**: 80/20\n"
            f"- **R² on test set**: `{r2:.3f}`\n"
            f"- **Features**: preprocessed pack-level measurements (no direct SOH leakage)\n"
            "- **Output**: Pack-level SOH, then thresholded into *Healthy* vs *Problem*."
        )

    st.markdown("---")

    # Trend + table
    if st.session_state.predictions:
        col_trend, col_controls = st.columns([3, 1])

        with col_trend:
            st.subheader("📉 SOH Trend (Last Predictions)")
            df_pred = pd.DataFrame(st.session_state.predictions)
            df_pred["attempt"] = range(1, len(df_pred) + 1)

            # Only show the last N predictions
            N = 10
            df_last = df_pred.tail(N)

            fig, ax = plt.subplots()
            ax.plot(df_last["attempt"], df_last["soh"], marker="o")
            ax.axhline(
                threshold,
                color="red",
                linestyle="--",
                label=f"Threshold = {threshold:.2f}",
            )
            ax.set_xlabel("Prediction Attempt")
            ax.set_ylabel("Predicted SOH")
            ax.set_title(f"SOH Trend (Last {len(df_last)} Predictions)")
            ax.set_xticks(df_last["attempt"])
            ax.legend()
            st.pyplot(fig)

        with col_controls:
            st.markdown("#### ⚙️ History Controls")
            if st.button("🧹 Clear Prediction History"):
                st.session_state.predictions = []
                st.rerun()  # 🔁 updated from experimental_rerun

            st.markdown("#### 📋 Data Preview")
            st.dataframe(
                df_last[["attempt", "time", "soh", "status"]],
                use_container_width=True,
                height=250,
            )

        # Download full history
        st.subheader("📄 Export Prediction Report")
        buffer = BytesIO()
        df_pred.to_csv(buffer, index=False)
        st.download_button(
            label="📥 Download All Predictions as CSV",
            data=buffer.getvalue(),
            file_name="battery_predictions.csv",
            mime="text/csv",
        )
    else:
        st.info("Prediction history will appear here after you run a few checks.")

# --------------------------------------------------
# TAB 2 – CHATBOT
# --------------------------------------------------
with tab_chat:
    st.subheader("🤖 Battery & Sustainability Chatbot")
    st.caption(
        "Ask questions about battery health, SOH, charging practices, or sustainability. "
        "Responses are generated with Google Gemini."
    )

    # Show previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input pinned at bottom
    prompt = st.chat_input("Type your question about batteries…")

    if prompt:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        if client is None:
            answer = "⚠️ Gemini client is not configured. Check your API key."
        else:
            try:
                system_prompt = (
                    "You are an assistant helping a student explain battery health, "
                    "State of Health (SOH), and sustainable battery use in simple terms. "
                    "Keep answers concise, clear, and non-technical unless asked."
                )
                resp = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=f"{system_prompt}\n\nUser question: {prompt}",
                )
                answer = resp.text
            except Exception as e:
                answer = f"⚠️ Gemini API unavailable: {e}"

        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)

    # Optional: clear chat history button
    if st.session_state.messages:
        if st.button("🧹 Clear Chat History"):
            st.session_state.messages = []
            st.rerun()  # 🔁 updated from experimental_rerun
