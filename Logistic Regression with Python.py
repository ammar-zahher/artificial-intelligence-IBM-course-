#_________________libraries__________________
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
#_______________________________________________
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"
churn_data=pd.read_csv(url)
print(churn_data)
churn_data = churn_data[['tenure', 'age','ed', 'employ', 'equip', 'churn']]

print("^"*100)
churn_data['churn'] = churn_data['churn'].astype('int')
print(churn_data)

X = np.asarray(churn_data[['tenure', 'age','ed', 'employ', 'equip']])
print(X[0:5])

y=np.asarray(churn_data["churn"])
print(y[0:5])


X_norm = StandardScaler().fit(X).transform(X)
print(X_norm[0:5])

X_train, X_test, y_train, y_test = train_test_split( X_norm, y, test_size=0.2, random_state=2010)
#_____________________________________________________________________
ammar=LogisticRegression(C=100, solver='liblinear').fit(X_train,y_train)
print("///////" *100)
y_hat = ammar.predict(X_test)
print(y_hat[:10])
#Probability
y_hat_prob = ammar.predict_proba(X_test)
c=y_hat_prob *100
for percient in c:
    print("%",percient[0])
    print("%",percient[1])
#______________________________________________________________________
coefficients = pd.Series(ammar.coef_[0], index=churn_data.columns[:-1])
coefficients.sort_values().plot(kind='barh')
plt.title("Feature Coefficients in Logistic Regression Churn Model")
plt.xlabel("Coefficient Value")
plt.show()
#_____________________________________________
print("%",log_loss(y_test, y_hat_prob)*100)
#_____________________________________________
from sklearn.metrics import accuracy_score
print("الدقة الحقيقية للنموذج هي: %", accuracy_score(y_test, y_hat) * 100)
