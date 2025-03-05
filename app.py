import numpy as np
from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler
import pickle


#creating flask application
app = Flask(__name__)  

try:
    #loading pickel file
    model = pickle.load(open("model.pkl", "rb"))  
    sc = pickle.load(open("scaler_x.pkl", "rb"))  
    sc_y = pickle.load(open("scaler_y.pkl", "rb"))  
except FileNotFoundError:
    model = None
    

#Routes 
@app.route("/")
def page():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return "Model Not Loaded", 500

    try:
       
        # Get input features from the form
        area = float(request.form['area'])
        bedrooms = float(request.form['bedrooms'])
        bathrooms = float(request.form['bathrooms'])
        stories = float(request.form['stories'])
        parking = float(request.form['parking'])

        
        

        # Create a feature array
        features_array = np.array([[area, bedrooms, bathrooms, stories, parking]])

        features_scaled = sc.transform(features_array)

        # Make a prediction
        prediction_scaled = model.predict(features_scaled)

        prediction_text = sc_y.inverse_transform(prediction_scaled.reshape(-1, 1))[0][0]

        formatted_prediction = f"The Price is INR {int(prediction_text):,}"

        return render_template("index.html", prediction_text=formatted_prediction)
    
    except Exception as e:
        return f"An error occurred: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True, port=5000) 
