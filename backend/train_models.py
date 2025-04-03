import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# ✅ RAIN PREDICTION MOD
print("\n🌧️ Training Rainfall Prediction Model...")
rain_data = pd.read_csv("C:/Users/akula/OneDrive/Desktop/PRECISION FARMING/datasets/rainfall_new_final.csv")

# Drop unnecessary columns
rain_data = rain_data.drop(columns=['maxtemp', 'mintemp', '         winddirection', 'windspeed', 'Last_24hr_Precipitation'])

# Encode 'rainfall'
rain_label_encoder = LabelEncoder()
rain_data['rainfall'] = rain_label_encoder.fit_transform(rain_data['rainfall'])

# Define features and target
X_rain = rain_data.drop(columns='rainfall')
Y_rain = rain_data['rainfall']

# Scale the features
rain_scaler = StandardScaler()
X_rain = rain_scaler.fit_transform(X_rain)

# Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X_rain, Y_rain, test_size=0.2, stratify=Y_rain, random_state=2)

# Train the model
rain_model = LogisticRegression()
rain_model.fit(X_train, Y_train)

# Save the rainfall model and scaler
joblib.dump(rain_model, "C:/Users/akula/OneDrive/Desktop/PRECISION FARMING/backend/models/rain_model.pkl")
joblib.dump(rain_scaler, "C:/Users/akula/OneDrive/Desktop/PRECISION FARMING/backend/models/rain_scaler.pkl")
joblib.dump(rain_label_encoder, "C:/Users/akula/OneDrive/Desktop/PRECISION FARMING/backend/models/rain_label_encoder.pkl")

print("✅ Rainfall model saved!")

# ✅ IRRIGATION PREDICTION MODEL
print("\n🚜 Training Irrigation Prediction Model...")
irrigation_data = pd.read_csv("C:/Users/akula/OneDrive/Desktop/PRECISION FARMING/datasets/irrigation_new_final.csv")

# Encode 'CropType'
irrigation_label_encoder = LabelEncoder()
irrigation_data['CropType'] = irrigation_label_encoder.fit_transform(irrigation_data['CropType'])

# Features and target
X_irrigation = irrigation_data.drop(columns='Irrigation', axis=1)
Y_irrigation = irrigation_data['Irrigation']

# Scale the features
irrigation_scaler = StandardScaler()
X_irrigation = irrigation_scaler.fit_transform(X_irrigation)

# Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X_irrigation, Y_irrigation, test_size=0.2, stratify=Y_irrigation, random_state=2)

# Train the model
irrigation_model = LogisticRegression()
irrigation_model.fit(X_train, Y_train)

# Save the irrigation model and scaler
joblib.dump(irrigation_model, "C:/Users/akula/OneDrive/Desktop/PRECISION FARMING/backend/models/irrigation_model.pkl")
joblib.dump(irrigation_scaler, "C:/Users/akula/OneDrive/Desktop/PRECISION FARMING/backend/models/irrigation_scaler.pkl")
joblib.dump(irrigation_label_encoder, "C:/Users/akula/OneDrive/Desktop/PRECISION FARMING/backend/models/irrigation_label_encoder.pkl")

print("✅ Irrigation model saved!")
