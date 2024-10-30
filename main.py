import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('Nairobi Office Price Ex.csv')
x = data['SIZE'].values
y = data['PRICE'].values

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def gradient_descent(x, y, m, c, learning_rate, epochs):
    n = len(y)
    for epoch in range(epochs):
        y_pred = m * x + c
        dm = (-2/n) * np.sum(x * (y - y_pred))
        dc = (-2/n) * np.sum(y - y_pred)
        m -= learning_rate * dm
        c -= learning_rate * dc
        error = mean_squared_error(y, y_pred)
        print(f"Epoch {epoch+1}/{epochs}: MSE = {error}")
    return m, c

m = np.random.rand()
c = np.random.rand()
learning_rate = 0.0001
epochs = 10

m, c = gradient_descent(x, y, m, c, learning_rate, epochs)

plt.scatter(x, y, color="blue", label="Data Points")
plt.plot(x, m * x + c, color="red", label="Best Fit Line")
plt.xlabel("Office Size (sq ft)")
plt.ylabel("Office Price")
plt.title("Predicting Office Price Based on Office Size in Nairobi")
plt.legend()
plt.show()

office_size = 100
predicted_price = m * office_size + c
print(f"Predicted price for office size {office_size} sq. ft.: {predicted_price}")