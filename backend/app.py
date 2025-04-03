from flask import Flask, request, render_template
import joblib
import numpy as np
from flask import Flask
import os


# Get the absolute path to the 'models' folder
BASE_DIR = os.path.dirname(__file__)  # This gets the folder where app.py is located
MODELS_DIR = os.path.join(BASE_DIR, "models")  # Adjust if your models folder is elsewhere
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")  



app = Flask(__name__, template_folder=TEMPLATES_DIR)


# ✅ Load Rainfall Models
rain_model = joblib.load(os.path.join(MODELS_DIR, "rain_model.pkl"))
rain_scaler = joblib.load(os.path.join(MODELS_DIR, "rain_scaler.pkl"))
rain_label_encoder = joblib.load(os.path.join(MODELS_DIR, "rain_label_encoder.pkl"))

# ✅ Load Irrigation Models
irrigation_model = joblib.load(os.path.join(MODELS_DIR, "irrigation_model.pkl"))
irrigation_scaler = joblib.load(os.path.join(MODELS_DIR, "irrigation_scaler.pkl"))
irrigation_label_encoder = joblib.load(os.path.join(MODELS_DIR, "irrigation_label_encoder.pkl"))
app = Flask(__name__, template_folder='C:/Users/akula/OneDrive/Desktop/PRECISION FARMING/backend/templates')



# ✅ Home Route
@app.route('/')
def home():
    return render_template('index.html')

# ✅ Rainfall Prediction Route
@app.route('/predict_rain', methods=['POST'])
def predict_rain():
    try:
        # Extracting inputs
        pressure = float(request.form['pressure'])
        temperature = float(request.form['temperature'])
        dewpoint = float(request.form['dewpoint'])
        humidity = float(request.form['humidity'])
        cloud = float(request.form['cloud'])

        # Prepare input data
        input_values = np.array([pressure, temperature, dewpoint, humidity, cloud]).reshape(1, -1)

        # Scale the input
        scaled_data = rain_scaler.transform(input_values)

        # Prediction
        prediction = rain_model.predict(scaled_data)
        predicted_label = rain_label_encoder.inverse_transform(prediction)[0]

        return render_template('results.html', rain_output=f"Rainfall: {predicted_label}", irrigation_output="")

    except Exception as e:
        return render_template('results.html', rain_output="Error", irrigation_output=str(e))

# ✅ Irrigation Prediction Route
@app.route('/predict_irrigation', methods=['POST'])
def predict_irrigation():
    try:
        # Extracting inputs
        crop_type = int(request.form['crop_type'])
        soil_moisture = float(request.form['soil_moisture'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])

        # Prepare input data
        input_values = np.array([crop_type, soil_moisture, temperature, humidity]).reshape(1, -1)

        # Scale the input
        scaled_data = irrigation_scaler.transform(input_values)

        # Prediction
        prediction = irrigation_model.predict(scaled_data)
        
        # Display as "Irrigation Needed" or "No Irrigation Needed"
        result = "Irrigation Needed" if prediction[0] == 1 else "No Irrigation Needed"

        return render_template('results.html', rain_output="", irrigation_output=result)

    except Exception as e:
        return render_template('results.html', rain_output="", irrigation_output=str(e))

if __name__ == '__main__':
    app.run(debug=True)
