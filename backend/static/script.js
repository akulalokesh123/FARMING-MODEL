document.getElementById('prediction-form').onsubmit = async (e) => {
    e.preventDefault();

    const rainData = document.getElementById('rain_data').value;
    const irrigationData = document.getElementById('irrigation_data').value;

    const response = await fetch('/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/x-www-form-urlencoded'},
        body: `rain_data=${rainData}&irrigation_data=${irrigationData}`
    });

    const result = await response.json();
    document.getElementById('result').innerHTML = `
        <h3>Rain Prediction: ${result.rain_prediction}</h3>
        <h3>Irrigation Prediction: ${result.irrigation_prediction}</h3>
    `;
};
