import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ✅ Load the dataset
data = pd.read_csv("C:/Users/akula/OneDrive/Desktop/PRECISION FARMING/datasets/irrigation_new_final.csv")

# ✅ Handle missing values
print("\nMissing values in dataset:")
print(data.isnull().sum())

# ✅ Encode 'CropType' column
label_encoder = LabelEncoder()
data['CropType'] = label_encoder.fit_transform(data['CropType'])

# ✅ Features and target
X = data.drop(columns='Irrigation', axis=1)
Y = data['Irrigation']

# ✅ Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ✅ Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# ✅ Train the model
model = LogisticRegression()
model.fit(X_train, Y_train)

# ✅ Model accuracy
X_train_prediction = model.predict(X_train)
train_acc = accuracy_score(X_train_prediction, Y_train)
print("\nTraining Accuracy:", train_acc)

X_test_pred = model.predict(X_test)
test_acc = accuracy_score(X_test_pred, Y_test)
print("Testing Accuracy:", test_acc)

# 🚀 **User-Friendly Input**
print("\n🔹 Enter the following values:")
crop_type = int(input("Enter CropType (0, 1, 2... based on encoding): "))
soil_moisture = float(input("Enter SoilMoisture: "))
temperature = float(input("Enter Temperature: "))
humidity = float(input("Enter Humidity: "))

# ✅ Create input array
input_values = np.array([crop_type, soil_moisture, temperature, humidity]).reshape(1, -1)

# ✅ Scale the input data
std_data = scaler.transform(input_values)

# ✅ Make prediction
prediction = model.predict(std_data)

# ✅ Display the prediction
irrigation_result = "Irrigation Needed" if prediction[0] == 1 else "No Irrigation Needed"
print("\n🚜 Predicted Irrigation:", irrigation_result)
