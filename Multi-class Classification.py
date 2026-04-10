#take libraries that I need
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score
#not realy iportant
import warnings
warnings.filterwarnings("ignore")
#data prpparation
data = pd.read_csv(r"C:\Users\hesha\Downloads\Obesity_level_prediction_dataset.csv")
print(data.head(10))
#Exploratory Data Analysis
sns.countplot(y="NObeyesdad",data=data)
plt.title('Distribution of Obesity Levels')
plt.show()
#print("_"*100)
print(data["NObeyesdad"].value_counts())
print("_"*100)
print(data.isnull().sum())
print("_"*100)
print(data.info())
print("_"*100)  
print(data.describe())
print("_"*100)
print(data.isnull().sum())
continuous_columns=data.select_dtypes(include=["float64"]).columns.tolist()
scaler=StandardScaler()
scaled_features = scaler.fit_transform(data[continuous_columns])
#print(scaled_features)
sc=pd.DataFrame(scaled_features,columns=scaler.get_feature_names_out(continuous_columns))
#print(sc.head())
current_data = pd.concat([data.drop(columns=continuous_columns),sc],axis=1)
#categorical_columns turn into one hot encoding
categorical_columns=current_data.select_dtypes(include=["object"]).columns.tolist()
categorical_columns.remove("NObeyesdad")
encoder=OneHotEncoder(sparse_output=False,drop="first")
encoded_features=encoder.fit_transform(current_data[categorical_columns])
b=pd.DataFrame(encoded_features,columns=encoder.get_feature_names_out(categorical_columns))
final_data=pd.concat([current_data.drop(columns=categorical_columns),b],axis=1)
#print(final_data)
final_data["NObeyesdad"]=final_data["NObeyesdad"].astype('category').cat.codes
print(final_data.head())
X = final_data.drop('NObeyesdad', axis=1)
y= final_data['NObeyesdad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_ova = LogisticRegression(max_iter=1000)
model_ova.fit(X_train, y_train)
y_pred_ova = model_ova.predict(X_test)
print("One-vs-All (OvA) Strategy")
print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ova),2)}%")
