# 🔋 Battery Pack SOH Prediction & Chatbot Dashboard

This project is the final assignment for **Design & Analysis of Algorithms**.  
It combines:

- A **machine learning model** (Linear Regression) that predicts battery **State of Health (SOH)** from preprocessed pack-level data, and  
- A **Streamlit web app** with a **dashboard + chatbot** interface.

The goal is to show how algorithms and models can support **battery health monitoring** and **sustainability education** in an interactive way.

---

## 📁 Project Structure

```text
.
├── app.py                       # Streamlit dashboard & chatbot
├── train_linear.py              # Script to train & evaluate Linear Regression model
├── final_project_preprocessed_data.csv   # Preprocessed dataset (features + Pack_SOH)
├── models/
│   ├── linreg_soh.pkl           # Trained sklearn Pipeline (preprocessing + LinearRegression)
│   └── test_metrics.json        # Saved metrics (R², MAE, RMSE, threshold, confusion matrix, etc.)
├── figs/
│   └── pred_vs_actual.png       # Predicted vs Actual Pack_SOH plot
├── requirements.txt             # Python dependencies
├── .env                         # GEMINI_API_KEY (NOT committed – in .gitignore)
└── README.md                    # This file

```

## ⚙️ 1. Setup & Installation

1. Clone the repository

```text
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

2. Create and activate a virtual environment

```text
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# or
.\.venv\Scripts\activate         # Windows
```

3. Install dependencies

```text
pip install -r requirements.txt
```

4. Configure Google Gemini API (free tier)

    1. Go to Google AI Studio: https://aistudio.google.com

    2. Create an API key.

    3. At the project root, create a file named .env:

    ```text
    GEMINI_API_KEY=AIzaSyYOUR_REAL_KEY_HERE
    ```

    .env is ignored by Git (see .gitignore), so your key is not pushed to GitHub.

## 🧠 2. Training the Linear Regression Model

If models/linreg_soh.pkl and models/test_metrics.json already exist, you don’t have to retrain.
If you want to retrain from the CSV:
```text
python train_linear.py \
  --data final_project_preprocessed_data.csv \
  --target Pack_SOH \
  --threshold 0.6 \
  --test_size 0.2 \
  --seed 42
```

This script:
- Loads final_project_preprocessed_data.csv
- Strips column names and safely finds the target column
- Removes potential label leakage:
    * Drops SOH if predicting Pack_SOH, and vice versa
    * If target is Pack_SOH, also drops U1–U21 because they’re directly averaged into Pack_SOH
- Builds a Pipeline:
    * ColumnTransformer
        * StandardScaler for numeric columns
        * OneHotEncoder for categorical columns
    * LinearRegression model
- Splits data into train/test (80/20 by default), stratified by a health threshold
- Computes regression metrics:
    * R²
    * MSE / RMSE
    * MAE
    * 5-fold CV R² on training data
- Converts predictions into Healthy / Problem using the threshold and computes:
    * Confusion matrix
    * Classification accuracy

- It then saves:
    * models/linreg_soh.pkl – the trained pipeline
    * models/test_metrics.json – all key metrics + column lists
    * figs/pred_vs_actual.png – Predicted vs Actual Pack_SOH plot

## 🖥 3. Running the Streamlit App

Once dependencies and .env are set up:

```text
streamlit run app.py
```
A browser window will open (usually at http://localhost:8501).
The app has **two main tabs**:

## 📊 4. Dashboard Tab – SOH Prediction & Visualization
Tab: 📊 Dashboard

🔍 Predict Battery Pack SOH
- Click “Check Battery SOH”.
- The app:
    1. Loads final_project_preprocessed_data.csv
    2. Drops the target column Pack_SOH and any U1..U21 columns (to match the training setup)
    3. Samples one row (simulating a new measurement from a battery pack)
    4. Passes it through linreg_soh.pkl to get a predicted Pack SOH
    5. Compares the prediction to the current threshold (slider in the sidebar)
- The result is shown as:
    + 🟢 Predicted SOH: X.XX — The Battery is Healthy
    + 🔴 Predicted SOH: X.XX — The Battery has a Problem
- A progress bar shows SOH scaled to a 0–5 nominal range.

📉 SOH Trend (Last Predictions)
- Every time you click “Check Battery SOH”, the app stores:
    * time of prediction
    * predicted SOH
    * Healthy / Problem status
- The last 10 predictions are plotted as a line chart vs. Prediction Attempt (1, 2, 3, …), with a red dashed line at the current threshold.

This gives a quick sense of how predicted SOH is trending across multiple checks.

📋 Data preview & CSV export
- A small table shows the last few predictions (attempt, time, SOH, status).
- A “Download All Predictions as CSV” button exports the full history to battery_predictions.csv.

🧹 Clearing history
- **“Clear Prediction History”** wipes the stored predictions and refreshes the dashboard.

## 🤖 5. Chatbot Tab – Gemini Battery Assistant
Tab: 🤖 Chatbot
This tab lets you chat with a Gemini-powered assistant about:
* Battery health & SOH
* Good charging practices
* Sustainability / recycling of batteries
* Any related conceptual questions

How it works
- Chat interface uses Streamlit’s st.chat_message and st.chat_input:
    * Previous messages appear as bubbles
    * Input box stays pinned at the bottom, like a real chat app
- On each user message:
    1. The app builds a prompt with a system-style instruction:
        You are an assistant helping a student explain battery health, State of Health (SOH), and sustainable battery use in simple terms.
    2. Sends it to Google Gemini 2.0 Flash:
        ```text
        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=full_prompt,
        )
        answer = resp.text
        ```

    3. Displays the reply as an assistant message.
- “Clear Chat History” wipes the stored conversation and reruns the tab.

## 🧩 6. How This Could Be Used in Real Life
In this project, predictions are made by sampling rows from the historical dataset.

In a real-world deployment:
- The input features would come from a Battery Management System (BMS):
    * pack/cell voltages, currents, temperatures,
    * cycle count, time in service, etc.
- The same trained pipeline would run on each new measurement to estimate current SOH.
- The dashboard could be part of:
    * an EV or e-bike fleet maintenance portal,
    * a stationary storage monitoring system, or
    * a recycling / second-life screening tool.
The current Streamlit app demonstrates the full pipeline: **preprocessed measurements → ML model → SOH estimate → health classification → UX + explanations.**

## 🧪 7. Reproducibility Notes
- The app assumes the presence of:
    * ```final_project_preprocessed_data.csv```
    * ```models/linreg_soh.pkl```
    * ```models/test_metrics.json```
- If these are missing, run ```train_linear.py``` first.
- To change the health threshold, use:
    * the ```--threshold``` flag in ```train_linear.py```, and/or
    * the **Threshold slider** in the Streamlit sidebar for interactive experiments.