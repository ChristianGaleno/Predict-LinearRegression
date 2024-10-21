import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
warnings.simplefilter(action="ignore", category=FutureWarning)

def wrangle(path):
    data = pd.read_csv(path)
    return data

df = wrangle("housing.csv")


X_train = df["Avg. Area Number of Rooms"].round().astype(int).to_frame()
y_train = df["Price"].astype(int)

y_mean = [y_train.mean()] * len(y_train)

model = LinearRegression()
model.fit(X_train, y_train)

plt.plot(X_train, model.predict(X_train), linestyle="-", color="red")
plt.scatter(X_train, y_train)
plt.show()