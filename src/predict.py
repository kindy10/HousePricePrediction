import joblib
import pandas as pd
MODEL_PATH = "models/house_price_random_forest.pkl"
model = joblib.load(MODEL_PATH)
print("Model loaded successfully.")
print(model)

#Example house
house = pd.DataFrame([{
    "MedInc": 5.0,
    "HouseAge": 20.0,
    "AveRooms": 6.0,
    "AveBedrms": 1.0,
    "Population": 1500.0,
    "AveOccup": 3.0,
    "Latitude": 34.0,
    "Longitude": -118.0
}])
prediction = model.predict(house)
print("Predicted house value:",prediction[0])
print("Predicted price:${:,.2f}".format(prediction[0]*100000))