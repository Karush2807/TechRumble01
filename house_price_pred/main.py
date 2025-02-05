import pickle
from flask import Flask, request, render_template
import numpy as np

app = Flask(__name__, template_folder='templates')

# Load the machine learning model
model = pickle.load(open('ran_reg_model.pkl', 'rb'))

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Helper function to convert 'yes'/'no' inputs to 1/0
            def convert_input(value):
                return 1 if value and value.lower() == 'yes' else 0

            # Extract and convert form data safely
            area = int(request.form.get('area', 0))
            bedrooms = int(request.form.get('bedrooms', 0))
            bathrooms = float(request.form.get('bathrooms', 0))
            stories = int(request.form.get('stories', 0))
            mainroad = convert_input(request.form.get('mainroad'))
            guestroom = convert_input(request.form.get('guestroom'))
            basement = convert_input(request.form.get('basement'))
            hotwaterheating = convert_input(request.form.get('hotwaterheating'))
            airconditioning = convert_input(request.form.get('airconditioning'))
            parking = int(request.form.get('parking', 0))
            prefarea = convert_input(request.form.get('prefarea'))
            furnishingstatus = request.form.get('furnishingstatus', '').lower()
            

            # Handle furnishingstatus as categorical
            furnishingstatus_encoded = 0
            if furnishingstatus == 'furnished':
                furnishingstatus_encoded = 2
            elif furnishingstatus == 'semi-furnished':
                furnishingstatus_encoded = 1

            # Prepare the feature vector for prediction
            features = np.array([[area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus_encoded]])

            # Predict using the model
            prediction = model.predict(features)

            # Return the prediction result
            return render_template('index.html', result=f"The predicted house price is ${prediction[0]:,.2f}")
        except Exception as e:
            return render_template('index.html', result=f"Error: {e}")
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
