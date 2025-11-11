import json
import pandas as pd
import joblib
from openai import OpenAI
import random

# === Load your saved model and metrics ===
model = joblib.load("models/linreg_soh.pkl")
print("✅ Loaded trained model from models/linreg_soh.pkl")

with open("models/test_metrics.json") as f:
    metrics = json.load(f)
threshold = metrics["threshold"]

# === Initialize OpenAI API ===
client = OpenAI(api_key="sk-your-real-api-key-here")

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

    if "check" in query_lower and "soh" in query_lower:
        predicted_soh = predict_random_soh()
        status = classify_battery(predicted_soh, threshold)
        return f"Predicted SOH = {predicted_soh:.2f}. {status}"

    # Otherwise, pass question to ChatGPT
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant knowledgeable about batteries and sustainability."},
            {"role": "user", "content": query}
        ]
    )
    return completion.choices[0].message.content

# === Run chatbot in console ===
if __name__ == "__main__":
    print("🔋 Battery Chatbot is running! Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye!")
            break
        print("Chatbot:", chatbot_response(user_input))
