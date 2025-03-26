import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ✅ Load the dataset
data = pd.read_csv("C:/Users/akula/OneDrive/Desktop/PRECISION FARMING/datasets/rainfall_new_final.csv")

# ✅ Drop unnecessary columns
data = data.drop(columns=['maxtemp', 'mintemp', '         winddirection', 'windspeed', 'Last_24hr_Precipitation'])

# ✅ Encode the target variable
label_encoder = LabelEncoder()
data['rainfall'] = label_encoder.fit_transform(data['rainfall'])

# ✅ Define features and target
X = data.drop(columns='rainfall')
Y = data['rainfall']

# ✅ Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ✅ Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# ✅ Train the model
model = LogisticRegression()
model.fit(X_train, Y_train)

# ✅ Model accuracy
X_train_pred = model.predict(X_train)
train_acc = accuracy_score(X_train_pred, Y_train)
print("\nTraining Accuracy:", train_acc)

X_test_pred = model.predict(X_test)
test_acc = accuracy_score(X_test_pred, Y_test)
print("Testing Accuracy:", test_acc)

# 🚀 **User-Friendly Input**
print("\n🔹 Enter the following values:")
pressure = float(input("Enter Pressure: "))
temperature = float(input("Enter Temperature: "))
dewpoint = float(input("Enter Dewpoint: "))
humidity = float(input("Enter Humidity: "))
cloud = float(input("Enter Cloud: "))

# ✅ Create input array
input_values = np.array([pressure, temperature, dewpoint, humidity, cloud]).reshape(1, -1)

# ✅ Scale the input data
std_data = scaler.transform(input_values)

# ✅ Make prediction
prediction = model.predict(std_data)
predicted_label = label_encoder.inverse_transform(prediction)

# ✅ Display the prediction
print("\n🌧️ Predicted Rainfall:", predicted_label[0])
