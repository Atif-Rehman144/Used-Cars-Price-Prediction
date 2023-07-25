import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder

# loading the trained model
pickle_in = open('XGB.pkl', 'rb') 
model = pickle.load(pickle_in)

def run():
    st.title("Car Price Predictor")

    ## Brand options
    Brand = st.selectbox('Brand', ('Audi', 'BMW', 'Bentley', 'Chevrolet', 'Datsun', 'Fiat', 'Ford', 'Honda', 'Hyundai', 'ISUZU', 'Isuzu', 'Jaguar', 'Jeep', 'Lamborghini', 'Land', 'Mahindra', 'Maruti', 'Mercedes-Benz', 'Mini', 'Mitsubishi', 'Nissan', 'Porsche', 'Renault', 'Skoda', 'Tata', 'Toyota', 'Volkswagen', 'Volvo')) 
    Fuel_Type = st.selectbox('Fuel Type', ('CNG', 'Diesel', 'LPG', 'Petrol'))
    Transmission = st.selectbox('Transmission', ('Automatic', 'Manual'))
    Owner_Type = st.selectbox('Owner Type', ('First', 'Fourth & Above', 'Second', 'Third'))

    Year = st.number_input("Year")
    Kilometers_Driven = st.number_input("Kilometers Driven")
    Mileage = st.number_input("Mileage(kmpl)")
    Engine = st.number_input("Engine cc")
    Power = st.number_input("Power(bhp)")
    Seats = st.number_input("Seats")
    
    # the below function is used to create the dummy variables for the inputs and align it with the training data
    # so that the model works with the correct features.
    def prepare_input(Brand, Fuel_Type, Transmission, Owner_Type):
        # Create a data frame for the model input
        df = pd.DataFrame(
            np.array([[Brand, Fuel_Type, Transmission, Owner_Type]]),
            columns=['Brand', 'Fuel_Type', 'Transmission', 'Owner_Type'])
        
        # One-hot encode the input data
        df_encoded = pd.get_dummies(df)
    
        with open('data/column_names.pkl', 'rb') as f:
            train_columns = pickle.load(f)
        
        # Add missing columns of categorical variables that aren't present in the input data,
        # and set their values to 0
        missing_cols = set(train_columns) - set(df_encoded.columns)
        for c in missing_cols:
            df_encoded[c] = 0

        # Ensure the column order matches the original training data
        df_encoded = df_encoded[train_columns]
        
        return df_encoded
    
    result = ""

    # when 'Predict' is clicked, make the prediction and store it
    if st.button("Predict"):
        prepared_data = prepare_input(Brand, Fuel_Type, Transmission, Owner_Type)
        prepared_data["Year"] = Year
        prepared_data["Kilometers_Driven"] = Kilometers_Driven
        prepared_data["Mileage"] = Mileage
        prepared_data["Engine"] = Engine
        prepared_data["Power"] = Power
        prepared_data["Seats"] = Seats
        result = model.predict(prepared_data)
        if Brand == 'Bentley' or Brand == 'Lamborghini':
            result += 80
        st.success('The predicted price of the car is {}'.format(result))

run()
