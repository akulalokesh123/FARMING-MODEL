import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
data1=pd.read_csv("C:/Users/akula/OneDrive/Desktop/PRECISION FARMING/datasets/irrigation_new_final.csv")
data1.head()
data1.isnull().sum()
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
Labels=labelencoder.fit_transform(data1['CropType'])
data1['CropType']=Labels
data1.std()
from sklearn.preprocessing import StandardScaler
X=data1.drop(columns='Irrigation',axis=1)
Y=data1['Irrigation']
scaler=StandardScaler()
stand=scaler.fit_transform(X)
X=stand
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
model=LogisticRegression()
model.fit(X_train,Y_train)
X_train_prediction=model.predict(X_train)
acc=accuracy_score(X_train_prediction,Y_train)
print(acc)
X_test_pred=model.predict(X_test)
acc=accuracy_score(X_test_pred,Y_test)
print(acc)
input=("7,548,18,85").split(",")
array=np.array(input)
reshaped=array.reshape(1,-1)
std_data=scaler.transform(reshaped)
prediction=model.predict(std_data)
print(prediction)