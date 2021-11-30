import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


x = np.array([1,2,3,4,5,6])
y = np.array([4,8,12,16,20,24])

x=x.reshape((-1,1))
y = y.reshape((-1,1))

mod = LinearRegression()

mod.fit(x,y)
r_sq_value = mod.score(x,y)
print("coefficient of determination:",r_sq_value)
print("intercept:",mod.intercept_)
print("slope",mod.coef_)
y_pred = mod.predict(x)
print("normal value:",y)
print("predict value:",y_pred)
plt.plot(x,y_pred)
plt.title("predictions")
plt.xlabel("x value")
plt.ylabel("y pred")
plt.show()
