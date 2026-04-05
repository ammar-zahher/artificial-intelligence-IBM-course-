#Libaries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split



#----------------------organize data-------------------
url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
df=pd.read_csv(url)
print(df.describe())
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
viz = cdf[['CYLINDERS','ENGINESIZE','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
viz.hist()
plt.title("data")
plt.show()
#--------------------Plot FUELCONSUMPTION_COMB against CO2 Emission----------------------
plt.scatter(cdf.FUELCONSUMPTION_COMB,cdf.CO2EMISSIONS, color="yellow")
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("CO2 Emission")
plt.xlim(0.25)
plt.title("Plot FUELCONSUMPTION_COMB against CO2 Emission")
plt.show()
#----------------------Plot CYLINDER against CO2 Emission--------------------
plt.scatter(cdf.CYLINDERS,cdf.CO2EMISSIONS,color="blue")
plt.xlabel("CYLINDERS")
plt.ylabel("CO2 Emission")
plt.title("Plot CYLINDER against CO2 Emission")
plt.xlim(0,25)
plt.show()
#------------------------------------------
x=cdf.ENGINESIZE.to_numpy()
y=cdf.CO2EMISSIONS.to_numpy()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
type(x_train), np.shape(x_train), np.shape(x_train)
from sklearn import linear_model
regressor = linear_model.LinearRegression()
regressor.fit(x_train.reshape(-1,1),y_train)
print("Coefficients:",regressor.coef_[0])
print("regressor:",regressor.intercept_)
plt.scatter(x_train,y_train,color="red")
plt.plot(x_train,x_train*regressor.coef_+regressor.intercept_)
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.xlim(0,25)
plt.show()
