# HousePriceModel
 A house price prediction a web application developed using flask
 This project is a Flask web application that predicts house prices based on user input. The prediction model is built using Random Forest and trained on a dataset containing features like area, number of bedrooms, 
 bathrooms, stories, and parking.

Features
Machine Learning Model: Uses Random Forest Regressor for price prediction.
Feature Scaling: StandardScaler is used to normalize input features.
Web Interface: Users can input house details via an HTML form.
Flask Backend: Handles requests and returns predictions.
Pickle Storage: Saves the trained model and scalers for future use.

Data
The which we used to train this model was imported from kaggale.com(HousePricePrediction.csv).
Data shoud be cleaned and no null values should be present. Drop the columns having maximum null values.
Setup Instructions
Make Sure you are deploying in virtualenv:
1.pip install virtualenv
2.virtualenv env
3../env./Scripts.activate.ps1
1. Clone the Repository
   git clone https://github.com/RajeshwarPatil10/HousePriceModel.git cd house-price-prediction
2.Install Dependencies
   Ensure you have Python 3 installed. Then, install required libraries:
   Required Packages:
   Flask
   NumPy
   Pandas
   Scikit-learn
   Pickle
3.Train the Model
  Before running the web app, you need to train the model and generate required .pkl files.
  Run the following command:
  python train.py
  This will generate three files:
  model.pkl (Trained Machine Learning model)
  scaler_x.pkl (Feature scaler for input variables)
  scaler_y.pkl (Scaler for output variable)
4.Run the Flask App
  python app.py
  After training the model, start the Flask server:
  http://127.0.0.1:5000/
5.Usage Guide
  Open http://127.0.0.1:5000/ in your browser.
  Enter house details:
  Area
  Number of bedrooms
  Number of bathrooms
  Number of stories
  Parking spaces
  Click Predict Price.
  The estimated house price will be displayed in Indian Rupees (INR).
6.Troubleshooting
 1. Flask App Fails to Start?
    Ensure model.pkl, scaler_x.pkl, and scaler_y.pkl exist.
    If missing, retrain the model using python train_model.py.
 2. Prediction is Always INR 1?
    Ensure both input and output scalers are applied correctly.
    Verify scaler_y.inverse_transform() is used after prediction.
 3. Port 5000 Already in Use?
    Try running Flask on a different port:
    python app.py --port=5001
