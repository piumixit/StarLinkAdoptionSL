import joblib

try:
    model = joblib.load("starlink_final_model.pkl")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error: {e}")
