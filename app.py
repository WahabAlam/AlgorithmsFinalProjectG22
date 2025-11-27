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
r2 = metrics.get("r2", 0.0)
mae = metrics.get("mae", 0.0)
rmse = metrics.get("rmse", 0.0)

# Optional: feature info saved by train_linear.py
num_cols = metrics.get("numeric_columns", [])
cat_cols = metrics.get("categorical_columns", [])

# -------------------------------
# 🧠 MODEL CONTEXT FOR CHATBOT
# -------------------------------
MODEL_CONTEXT = f"""
You are part of a Streamlit dashboard that predicts lithium-ion battery pack State of Health (SOH).

The backend ML model is:
- A scikit-learn Pipeline
- Preprocessing: ColumnTransformer with StandardScaler for numeric features and OneHotEncoder for categorical features
- Estimator: LinearRegression

Training setup:
- Target variable: Pack_SOH (pack-level state of health)
- Train/test split: 80/20 with stratification on a health threshold
- Number of numeric features: {len(num_cols)}
- Number of categorical features: {len(cat_cols)}

Test-set performance:
- R² = {r2:.3f}
- MAE = {mae:.4f}
- RMSE = {rmse:.4f}

Operational logic in the app:
- The model outputs a continuous SOH value for a pack.
- If predicted SOH ≥ {threshold_default:.2f}, the pack is classified as 'Healthy'.
- If predicted SOH < {threshold_default:.2f}, the pack is classified as 'Problem'.

When users ask about "the model", "the algorithm", "accuracy", "threshold",
or "how this app works", you must use the details above in your explanation.

You do NOT see raw feature values for individual predictions, but you know the overall design,
metrics, and the fact that predictions come from preprocessed pack-level inputs (no direct SOH leakage).
"""

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
    "the **Chatbot** to ask battery/model questions using Google Gemini, "
    "and the **Model Evaluation** tab to inspect performance."
)

tab_dash, tab_chat, tab_eval = st.tabs(["📊 Dashboard", "🤖 Chatbot", "📈 Model Evaluation"])

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

            # 🧠 Explain this prediction using the model-aware chatbot
            if client is not None and st.button("🧠 Explain This Prediction"):
                explanation_prompt = (
                    f"The model just predicted an SOH of {soh:.2f}, which is classified as '{status}' "
                    f"using a threshold of {threshold:.2f}. "
                    "Explain what this means in simple terms for a non-technical user. "
                    "Also comment on how reliable this prediction might be, using the model's R², MAE, and RMSE. "
                    "Do not make up any new numbers; only reason from the metrics you know."
                )

                try:
                    resp = client.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=MODEL_CONTEXT + "\n\n" + explanation_prompt,
                    )
                    explanation = resp.text
                except Exception as e:
                    explanation = f"⚠️ Gemini API unavailable: {e}"

                st.info(explanation)

        else:
            st.info("Click **Check Battery SOH** to generate a prediction.")

    with col_right:
        st.markdown("#### ℹ️ About This Model")
        st.write(
            "- **Model**: Linear Regression (Scikit-Learn pipeline)\n"
            "- **Preprocessing**: StandardScaler for numeric features, OneHotEncoder for categoricals\n"
            f"- **Train/test split**: 80/20\n"
            f"- **R² on test set**: `{r2:.3f}`\n"
            "- **Output**: Continuous Pack SOH, then thresholded into *Healthy* vs *Problem*."
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
                st.rerun()

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
    st.subheader("🤖 Battery, SOH & Model-Aware Chatbot")
    st.caption(
        "Ask questions about battery health, SOH, charging practices, sustainability, "
        "or how this machine learning model and dashboard work."
    )

    # Show previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input pinned at bottom
    prompt = st.chat_input("Type your question about batteries or the model…")

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
                    MODEL_CONTEXT
                    + "\n\nYou are also helping a student explain battery health, "
                      "State of Health (SOH), and sustainable battery use in simple terms. "
                      "Keep answers concise, clear, and non-technical unless they ask for more detail. "
                      "If they ask about the model, its accuracy, or the threshold logic, "
                      "use the MODEL_CONTEXT details above."
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
            st.rerun()

# --------------------------------------------------
# TAB 3 – MODEL EVALUATION
# --------------------------------------------------
with tab_eval:
    st.subheader("📈 Model Evaluation")

    col_metrics, col_plot = st.columns([1, 2])

    with col_metrics:
        st.markdown("### 📊 Metrics")
        st.write(f"**R² (coefficient of determination):** `{r2:.3f}`")
        st.write(f"**MAE (mean absolute error):** `{mae:.4f}`")
        st.write(f"**RMSE (root mean squared error):** `{rmse:.4f}`")
        st.markdown(
            """
            - **R²** measures how much of the variation in SOH the model explains.  
            - **MAE** is the average absolute prediction error (in SOH units).  
            - **RMSE** penalizes larger errors more strongly than MAE.  
            """
        )

    with col_plot:
        st.markdown("### 📉 Predicted vs Actual Pack SOH")
        try:
            st.image("figs/pred_vs_actual.png", use_column_width=True)
            st.caption(
                "Each point compares the true Pack_SOH (x-axis) to the model prediction (y-axis). "
                "Points close to the diagonal line mean good predictions."
            )
        except Exception:
            st.info(
                "Predicted vs Actual plot not found. "
                "Re-run `train_linear.py` to generate `figs/pred_vs_actual.png`."
            )
