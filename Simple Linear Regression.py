import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

data = {
    'EngineSize': [2.0, 2.4, 1.5, 3.5, 3.5, 3.5, 3.5, 3.7, 3.7, 2.4],
    'CO2Emissions': [196, 221, 136, 255, 244, 230, 232, 255, 267, 201.65]
}
df = pd.DataFrame(data)


X = df[['EngineSize']]
y = df[['CO2Emissions']]

model = LinearRegression()
model.fit(X, y)


X_future = np.linspace(0, 7, 100).reshape(-1, 1)
y_future = model.predict(X_future)
plt.figure(figsize=(10, 6))

plt.scatter(X, y, color='blue', label='real data')


plt.plot(X_future, y_future, color='red', label='Best fit line.') 

plt.xlim(0, 7)
plt.ylim(100, 401)

plt.title('Predict carbon dioxide emissions based on engine size')
plt.xlabel('Engine Size')
plt.ylabel('CO2 Emissions')
plt.legend()
plt.show()
