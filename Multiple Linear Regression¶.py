import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#___________________________________________________________________________
url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
df=pd.read_csv(url)
print("_"*100)
print(df.describe())
print("_"*100)

df=df.drop(['MODELYEAR', 'MAKE', 'MODEL', 'VEHICLECLASS', 'TRANSMISSION', 'FUELTYPE',],axis=1)
print("_"*100)
print(df.corr())
print("_"*100)

df=df.drop(["CYLINDERS","FUELCONSUMPTION_COMB","FUELCONSUMPTION_HWY","FUELCONSUMPTION_CITY",],axis=1)
print(df)
print("_"*100)
#_________________________________________________________________________
axes=pd.plotting.scatter_matrix(df,alpha=0.2)

for ax in axes.flatten():
    ax.xaxis.label.set_rotation(90)
    ax.yaxis.label.set_rotation(0)
    ax.yaxis.label.set_ha('right')

plt.tight_layout()
plt.gcf().subplots_adjust(wspace=0, hspace=0)
plt.show()
print("_"*100)
#__________________________________________________________________________
x=df.iloc[:,[0,1]].to_numpy()
y=df.iloc[:,[2]].to_numpy()

from sklearn import preprocessing
std_sclarr=preprocessing.StandardScaler()
x_std=std_sclarr.fit_transform(x)

print("_"*100)
print(pd.DataFrame(x_std).describe().round(2))
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_std,y,test_size=0.2,random_state=42)

#_____________________________________________________________________________
from sklearn import linear_model
regressor=linear_model.LinearRegression()
regressor.fit(x_train,y_train)
coef_ =  regressor.coef_
intercept_ = regressor.intercept_
#important focusing
print ('Coefficients: ',coef_)
print ('Intercept: ',intercept_)
#_____________________________________________________________________________
means_ = std_sclarr.mean_
std_devs_ = np.sqrt(std_sclarr.var_)
coef_original = coef_ / std_devs_
intercept_original = intercept_ - np.sum((means_ * coef_) / std_devs_)
print("_#"*100)
print ('Coefficients: ', coef_original)
print ('Intercept: ', intercept_original)
#_____________________________________________________________________________
from mpl_toolkits.mplot3d import Axes3D
X1 =x_test[:, 0] if x_test.ndim > 1 else X_test
X2 =x_test[:, 1] if x_test.ndim > 1 else np.zeros_like(X1)
x1_surf, x2_surf = np.meshgrid(np.linspace(X1.min(), X1.max(), 100), 
                               np.linspace(X2.min(), X2.max(), 100))

y_surf = intercept_ +  coef_[0,0] * x1_surf  +  coef_[0,1] * x2_surf
y_pred = regressor.predict(x_test.reshape(-1, 1)) if x_test.ndim == 1 else regressor.predict(x_test)
above_plane = y_test >= y_pred
below_plane = y_test < y_pred
above_plane = above_plane[:,0]
below_plane = below_plane[:,0]
fig = plt.figure(figsize=(20, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X1[above_plane], X2[above_plane], y_test[above_plane],  label="Above Plane",s=70,alpha=.7,ec='k')
ax.scatter(X1[below_plane], X2[below_plane], y_test[below_plane],  label="Below Plane",s=50,alpha=.3,ec='k')
ax.plot_surface(x1_surf, x2_surf, y_surf, color='k', alpha=0.21,label='plane')
ax.view_init(elev=10)

ax.legend(fontsize='x-large',loc='upper center')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_box_aspect(None, zoom=0.75)
ax.set_xlabel('ENGINESIZE', fontsize='xx-large')
ax.set_ylabel('FUELCONSUMPTION', fontsize='xx-large')
ax.set_zlabel('CO2 Emissions', fontsize='xx-large')
ax.set_title('Multiple Linear Regression of CO2 Emissions', fontsize='xx-large')
plt.tight_layout()
plt.show()

plt.scatter(x_train[:,0], y_train,  color='blue')
plt.plot(x_train[:,0], coef_[0,0] * X_train[:,0] + intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


plt.scatter(x_train[:,1], y_train,  color='blue')
plt.plot(x_train[:,1], coef_[0,1] * X_train[:,1] + intercept_[0], '-r')
plt.xlabel("FUELCONSUMPTION_COMB_MPG")
plt.ylabel("Emission")
plt.show()
