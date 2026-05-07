import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from keras.layers import Input
import random

random.seed(43200)
np.random.seed(43200)
tf.random.set_seed(43200)

import pandas as pd
import numpy as np
import keras

import warnings

warnings.simplefilter("ignore", FutureWarning)

import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
filepath = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv"
concrete_data = pd.read_csv(filepath)
#print("="*50,"pace of data","="*50)
#print(concrete_data.head())
#print("="*50,"shape","="*50)
#print(concrete_data.shape)
#print("="*50,"describe","="*50)
#print(concrete_data.describe())
#print("="*50,"isnull","="*50)
#print(concrete_data.isnull().sum())
concrete_data_columns = concrete_data.columns
#print("="*50,"columns","="*50)
#print(concrete_data_columns)
#select all columns except Strength
predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']]
target = concrete_data['Strength']
#print("="*50,"predictors","="*50)
#print(predictors.head())
#print("="*50,"target","="*50)
#print(target.head())
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
predictors_norm = scaler.fit_transform(predictors)
n_cols = predictors_norm.shape[1]
print(predictors_norm.shape[1])
def regression_model():
  model = Sequential()
  model.add(Input(shape=(n_cols,)))
  model.add(Dense(50, activation='relu'))
  model.add(Dense(50, activation='relu'))
  model.add(Dense(1))
  model.compile(optimizer='adam', loss='mean_squared_error')
  return model
model = regression_model()
model.fit(predictors_norm, target, validation_split=0.3, epochs=200, verbose=2)
scores = model.evaluate(predictors_norm, target, verbose=0)
print(f"final mse: {scores}")

y_pred = model.predict(predictors_norm)
accuracy_r2 = r2_score(target, y_pred)

print(f"R2 Score (Accuracy): {accuracy_r2 * 100:.2f}%")
###############################################
y_pred = model.predict(predictors_norm)
plt.figure(figsize=(10, 6))
plt.scatter(target, y_pred, alpha=0.5, color='blue')
plt.plot([target.min(), target.max()], [target.min(), target.max()], color='red', lw=2)
plt.xlabel('Actual Strength (MPa)')
plt.ylabel('Predicted Strength (MPa)')
plt.title('Actual vs Predicted Concrete Strength')
plt.show()

def predict_concrete_strength():
    print("\n--- Concrete Strength Prediction System ---")
    
    try:
        cement = float(input("Cement: "))
        slag = float(input("Blast Furnace Slag: "))
        ash = float(input("Fly Ash: "))
        water = float(input("Water: "))
        superplasticizer = float(input("Superplasticizer: "))
        coarse_agg = float(input("Coarse Aggregate: "))
        fine_agg = float(input("Fine Aggregate: "))
        age = float(input("Age (days): "))
        
        user_data = np.array([cement, slag, ash, water, superplasticizer, coarse_agg, fine_agg, age]).reshape(1, -1)
        
        user_data_norm = (user_data - predictors.mean().values) / predictors.std().values
        
        prediction = model.predict(user_data_norm, verbose=0)
        
        print("-" * 30)
        print(f"Predicted Strength: {prediction[0][0]:.2f} MPa")
        print("-" * 30)
        
    except ValueError:
        print("Error: Please enter numeric values only.")

predict_concrete_strength()
