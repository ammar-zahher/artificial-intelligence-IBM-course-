from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

import warnings

warnings.filterwarnings("ignore")
# ==============================================
#               organize data
# ==============================================
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/pu9kbeSaAtRZ7RxdJKX9_A/yellow-tripdata.csv"
raw_data = pd.read_csv(url)
print(raw_data)
print("==============================================")
print(raw_data.corr())
print("==============================================")
print(raw_data.info())
# ==============================================
#               prepare data
correlation_values = raw_data.corr()["tip_amount"].drop("tip_amount").sort_values()
correlation_values.plot(kind="barh", figsize=(10, 8))
plt.show()
# ==============================================
#               split data
y = raw_data["tip_amount"].values.astype("float32")
X = raw_data.drop(
    ["tip_amount", "store_and_fwd_flag", "improvement_surcharge"], axis=1
).values.astype("float32")
# not an essential step but it is good to normalize the data before training the model
# X = normalize(X, axis=1, norm="l1", copy=False)
# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
A = DecisionTreeRegressor(criterion="squared_error", max_depth=10, random_state=35)
A.fit(X_train, y_train)
y_pred = A.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("MSE score : {0:.3f}".format(mse))
r2_score = A.score(X_test, y_test)
print("R2 score : {0:.3f}".format(r2_score))
