Crop-Recommendation-Sys

Uses multiple Machine Learning Models to train and evaluate crop recommendation accuracy, eventually using Gaussian Naive Bayes due to high accuracy (~99.3%).

📌 Description

This project predicts the most suitable crop to plant based on environmental and soil conditions using machine learning techniques. It involves:

Dataset with soil and climate features

Training ML models

Saving trained model and scalers

A Flask web app to get predictions

📁 Files Included

Crop Recommendation Using Machine Learning.ipynb – Jupyter notebook demonstrating training and evaluation

Crop_recommendation.csv – Dataset with soil and environmental data

app.py – Flask application backend

index.html – Web UI

crop.png – Image used in interface

model.pkl – Pickled trained model

minmaxscaler.pkl & standscaler.pkl – Scalers for preprocessing

requirements.txt – Required Python packages

train_model.py – Script for training the ML model

test_predictions.py – Scripts for testing the model accuracy

🧠 How It Works

Load dataset of features:

Nitrogen (N), Phosphorus (P), Potassium (K)

Temperature

Humidity

pH

Rainfall

Train multiple ML models (Naive Bayes, etc.)

Select best performing model (GaussianNB)

Save model & scalers

Use flask app to serve predictions from the trained model

🚀 Using the Web App

Install dependencies:

pip install -r requirements.txt


Run Flask server:

python app.py


Open browser at: http://localhost:5000

Enter crop parameters and get the recommended crop.

📌 Notes

The project uses Gaussian Naive Bayes model due to its high accuracy score (around 99.3%).

Flask app provides a simple web interface for users to input values and receive predictions.
