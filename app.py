import streamlit as st
import pickle   
import numpy as np
import pandas as pd
@st.cache_resource
def load_artifacts():
    with open('XGBRegressor_model.pkl','rb') as f:
        model=pickle.load(f)
    with open('label_encoder.pkl','rb') as f:  
        encoders=pickle.load(f)
    return model,encoders
model,label_encoder=load_artifacts()
st.title("Demand Forecasting")
st.divider()
st.header('Input Features')
price=st.number_input('price',min_value=50.0)
discount=st.number_input('Discount(%)',min_value=0,max_value=100, value=10)
inventory_level=st.number_input('Inventory Level',min_value=0,value=100)
promption=st.selectbox('Promotion',options=['Yes','No'])
competitor_pricing=st.number_input('Competitor Pricing',min_value=0.0, value=50.0)
category=st.selectbox('category',
                      label_encoder['Category'].classes_.tolist())
input_data=pd.DataFrame({
    'Price':[price],    
    'Discount':[discount],
    'Inventory Level':[inventory_level],
    'Promotion':[promption],
    'Competitor Pricing':[competitor_pricing],
    'Category':[category]
})
input_data['Promotion'] = input_data['Promotion'].map({'Yes': 1, 'No': 0})
for col, encoder in label_encoder.items():
    if col in input_data.columns:
        input_data[col] = encoder.transform(input_data[col])
st.divider()
if st.button('Predict Demand'):
    prediction = model.predict(input_data)
    st.success(f'Predicted Demand: {int(prediction[0])}')   