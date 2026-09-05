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

def get_float(prompt,min_value =None,max_value = None):
    """
    Ask the user for a valid number within an optimal range.
    """
    while True:
        try:
            value = float(input(prompt))
            
            if min_value is not None and value < min_value:
                print(f"value must be at least {min_value}.")
                continue
            if max_value is not None and value > max_value:
                print(f"Value must be at most{max_value}.")
                continue
                
            return value
        except ValueError:
            print("Please enter a valid number.")


if __name__ == "__main__":

     house = {
    "MedInc": get_float("MedInc: ", min_value=0),
    "HouseAge": get_float("HouseAge: ", min_value=0),
    "AveRooms": get_float("AveRooms: ", min_value=0),
    "AveBedrms": get_float("AveBedrms: ", min_value=0),
    "Population": get_float("Population: ", min_value=0),
    "AveOccup": get_float("AveOccup: ", min_value=0),
    "Latitude": get_float("Latitude: ", min_value=32, max_value=42),
    "Longitude": get_float("Longitude: ", min_value=-125, max_value=-114)
    }

     predicted_price = predict_house_price(house)
    
     print()
     print("Predicted house price: ${:,.2f}".format(predicted_price))