import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

data = pd.read_csv("C:/Users/akula/OneDrive/Desktop/PRECISION FARMING/datasets/rainfall_new_final.csv")

data = data.drop(columns=['maxtemp', 'mintemp', '         winddirection', 'windspeed', 'Last_24hr_Precipitation'])

label_encoder = LabelEncoder()
data['rainfall'] = label_encoder.fit_transform(data['rainfall'])

X = data.drop(columns='rainfall')
Y = data['rainfall']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
model = LogisticRegression()
model.fit(X_train, Y_train)

X_train_pred = model.predict(X_train)
train_acc = accuracy_score(X_train_pred, Y_train)
print("Training Accuracy:", train_acc)

X_test_pred = model.predict(X_test)
test_acc = accuracy_score(X_test_pred, Y_test)
print("Testing Accuracy:", test_acc)

input_values = list(map(float, input("Enter values separated by commas: ").split(",")))
array = np.array(input_values).reshape(1, -1)
std_data = scaler.transform(array)
prediction = model.predict(std_data)

predicted_label = label_encoder.inverse_transform(prediction)
print("Predicted Rainfall:", predicted_label[0]) 