import joblib
MODEL_PATH = "models/house_price_random_forest.pkl"
model = joblib.load(MODEL_PATH)
print("Model loaded successfully.")
print(model)
