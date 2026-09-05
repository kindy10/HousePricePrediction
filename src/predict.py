import joblib
import pandas as pd


MODEL_PATH = "models/house_price_random_forest.pkl"


def predict_house_price(house_data):
    """
    Predict the price of a house using the trained Random Forest model.

    Parameters:
        house_data (dict): House features.

    Returns:
        float: Predicted house price in dollars.
    """

    model = joblib.load(MODEL_PATH)

    house = pd.DataFrame([house_data])

    prediction = model.predict(house)[0]

    price = prediction * 100000

    return price

def get_float(prompt):
    """
    Ask the user for a number untill a valid number is entered.
    """
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Please enter a valid number.")


if __name__ == "__main__":

     house = {
        "MedInc": get_float("MedInc: "),
        "HouseAge": get_float("HouseAge: "),
        "AveRooms": get_float("AveRooms: "),
        "AveBedrms": get_float("AveBedrms: "),
        "Population": get_float("Population: "),
        "AveOccup": get_float("AveOccup: "),
        "Latitude": get_float("Latitude: "),
        "Longitude": get_float("Longitude: ")
     }

     predicted_price = predict_house_price(house)

     print()
     print("Predicted house price: ${:,.2f}".format(predicted_price))