from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

base_dir = os.path.abspath(os.path.dirname(__file__))
## Load the model
model=pickle.load(open(os.path.join(base_dir, 'XGB.pkl'),'rb'))
# Load the model from the pickle file

# Load the dataset to get the column names
relative_path = 'data/output.csv'
file_path = os.path.join(base_dir, relative_path)
df = pd.read_csv(file_path)

relative_path = 'data/trained.csv'
file_path = os.path.join(base_dir, relative_path)
df_trained = pd.read_csv(file_path)

# Extract column names for Brands, Fuel Types, Transmission, and Owner Type
brand_cols = [col.split("_")[1] for col in df.columns if "Brand" in col]
fuel_type_cols = [col.replace("Fuel_Type_", "") for col in df.columns if "Fuel_Type" in col]
transmission_cols = [col.replace("Transmission_", "") for col in df.columns if "Transmission" in col]
owner_type_cols = [col.replace("Owner_Type_", "") for col in df.columns if "Owner_Type" in col]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the values from the form
    form_values = request.form

    # Create a DataFrame with zeros for all columns in the trained dataset
    features = pd.DataFrame(np.zeros((1, len(df_trained.columns) - 2)), columns=df_trained.columns.drop(['Unnamed: 0', 'Price']))

    # Fill in the appropriate values based on the form input
    features['Year'] = form_values['Year']
    features['Kilometers_Driven'] = form_values['Kilometers_Driven']
    features['Mileage'] = form_values['Mileage']
    features['Engine'] = form_values['Engine']
    features['Power'] = form_values['Power']
    features['Seats'] = form_values['Seats']
    features['Brand_' + form_values['Brand']] = 1
    features['Fuel_Type_' + form_values['Fuel_Type']] = 1
    features['Transmission_' + form_values['Transmission']] = 1
    features['Owner_Type_' + form_values['Owner_Type']] = 1

    # Convert the features to a numpy array and make a prediction
    final_features = features.values
    prediction = model.predict(final_features)

    return render_template('index.html', prediction_text='Estimated car price is {}'.format(prediction))



if __name__ == "__main__":
    app.run(debug=True)
