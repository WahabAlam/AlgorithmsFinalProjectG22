import json
import os
import random

import joblib
import pandas as pd
from dotenv import load_dotenv
from google import genai  # Gemini SDK

# === Load environment and configure Gemini ===
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = None
if GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"Error creating Gemini client: {e}")
else:
    print("Warning: GEMINI_API_KEY not set in .env. Chatbot answers will not work.")

# === Load your saved model and metrics ===
model = joblib.load("models/linreg_soh.pkl")
print("Loaded trained model from models/linreg_soh.pkl")

with open("models/test_metrics.json") as f:
    metrics = json.load(f)
threshold = metrics["threshold"]

# === Classification rule ===
def classify_battery(predicted_soh, threshold):
    if predicted_soh < threshold:
        return "The battery has a problem."
    else:
        return "The battery is healthy."

# === Predict SOH using your trained model ===
def predict_random_soh():
    df = pd.read_csv("final_project_preprocessed_data.csv")

    # Drop the target and leakage columns (same logic as your train script)
    if "Pack_SOH" in df.columns:
        X = df.drop(columns=["Pack_SOH"], errors="ignore")
    else:
        X = df.copy()

    for col in [f"U{i}" for i in range(1, 22)]:
        if col in X.columns:
            X = X.drop(columns=[col], errors="ignore")

    # Pick a random row
    random_row = X.sample(1, random_state=random.randint(0, 9999))
    predicted_soh = model.predict(random_row)[0]
    return predicted_soh

# === Chatbot logic ===
def chatbot_response(query):
    query_lower = query.lower()

    # If user asks to check SOH, use the local ML model
    if "check" in query_lower and "soh" in query_lower:
        predicted_soh = predict_random_soh()
        status = classify_battery(predicted_soh, threshold)
        return f"Predicted SOH = {predicted_soh:.2f}. {status}"

    # Otherwise, answer using Gemini
    if client is None:
        return "Gemini client is not configured. Please set GEMINI_API_KEY in your .env file."

    try:
        system_prompt = (
            "You are a helpful assistant knowledgeable about batteries, "
            "battery State of Health (SOH), and sustainability. "
            "Explain things clearly and simply."
        )

        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"{system_prompt}\n\nUser question: {query}",
        )
        # resp.text should contain the model's answer
        return resp.text
    except Exception as e:
        return f"Gemini API error: {e}"

# === Run chatbot in console ===
if __name__ == "__main__":
    print("Battery Chatbot is running in the terminal. Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye!")
            break
        print("Chatbot:", chatbot_response(user_input))
