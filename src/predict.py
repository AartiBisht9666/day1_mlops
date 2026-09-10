import pandas as pd
import numpy as np
import joblib

new_data = pd.DataFrame([[100,250,198]], columns=["TV", "radio", "newspaper"])

model= joblib.load("models/linear_model.pkl")
prediction = model.predict(new_data)

print(f"Predicted Sales is {prediction}")