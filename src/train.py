import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error

df = pd.read_csv("D:\\day1_mlops\\data\\data.csv")

# train-test-split
X = df[["TV", "radio", "newspaper"]]
y = df["sales"]

xtrain, xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=56
)

model = LinearRegression()

model.fit(xtrain, ytrain)

ypred = model.predict(xtest)

rmse = root_mean_squared_error(ytest, ypred)
r2 = r2_score(ytest, ypred)

print(f"RMSE: {rmse}")
print(f"R2: {r2}")

# Dump model
joblib.dump(model,"models/linear_model.pkl")
