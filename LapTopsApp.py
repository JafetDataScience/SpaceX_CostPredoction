import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np
import pandas as pd
import joblib as jl

# Suponiendo que tienes un modelo entrenado para predecir el precio de la laptop.
# Aquí estamos utilizando un modelo ficticio para la demostración. 
# Sustituye esto con la carga de tu modelo real.
model  = jl.load("RegrssionTree_LaptopsPrice.pkl")
scaler = jl.load("Scaler_LaptopsPrice.pkl")
def predict_price(inputs):
    # Aquí deberías cargar tu modelo y usarlo para predecir el precio.
    # Este es solo un ejemplo ficticio.     
    Manufacturers = ['Acer', 'Asus', 'Dell', "HP", 'Huawei', 'Lenovo', 'MSI', 'Razer', 'Samsung', 'Toshiba', 'Xiaomi']
    Screens = ['Full HD', 'IPS panel']
    for i in enumerate(Manufacturers):
        if inputs[-2] == i[1]:
            Manufacturers_0 = np.zeros(len(Manufacturers))
            Manufacturers_0[i[0]] = 1
            if inputs[-1] == 'Full HD':           
                inputs_1 = np.concatenate((np.array(inputs[0:-2]),Manufacturers_0,np.array([1,0])))
            elif inputs[-1] == 'IPS panel':           
                inputs_1 = np.concatenate((np.array(inputs[0:-2]),Manufacturers_0,np.array([0,1])))
    # Esta función debería utilizar el modelo entrenado para realizar una predicción basada en las entradas
    # Aquí simplemente devolvemos un valor ficticio
#    print(inputs_1)
    df = pd.DataFrame([inputs_1],columns = ['Category', 'GPU', 'OS', 'CPU_core',
       'CPU_frequency', 'RAM_GB', 'Storage_GB_SSD','Screen-Full_HD', 'Screen-IPS_panel', 'Manufacturer_Acer',
       'Manufacturer_Asus', 'Manufacturer_Dell', 'Manufacturer_HP','Manufacturer_Huawei', 'Manufacturer_Lenovo',
       'Manufacturer_MSI', 'Manufacturer_Razer', 'Manufacturer_Samsung', 'Manufacturer_Toshiba','Manufacturer_Xiaomi'])
    New_data = scaler.transform(df)
    prediction = model.predict(New_data)
    return prediction[0]

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Laptop Price Prediction"),
    html.Div([
        html.Label("CPU Frequency (GHz)"),
        dcc.Input(id='cpu-frequency', type='number', step=0.1),
        
        html.Label("Category"),
        dcc.Dropdown(id='category', options=[
            {'label': str(i), 'value': i} for i in range(1, 6)
        ]),
        
        html.Label("GPU"),
        dcc.Dropdown(id='gpu', options=[
            {'label': str(i), 'value': i} for i in range(1, 4)
        ]),
        
        html.Label("OS"),
        dcc.Dropdown(id='os', options=[
            {'label': str(i), 'value': i} for i in range(1, 3)
        ]),
        
        html.Label("CPU Cores"),
        dcc.Dropdown(id='cpu-core', options=[
            {'label': str(i), 'value': i} for i in [3, 5, 7]
        ]),
        
        html.Label("RAM (GB)"),
        dcc.Dropdown(id='ram-gb', options=[
            {'label': str(i), 'value': i} for i in [4, 6, 8, 12, 16]
        ]),
        
        html.Label("Storage (GB SSD)"),
        dcc.Dropdown(id='storage-gb', options=[
            {'label': str(i), 'value': i} for i in [128, 256]
        ]),
        
        html.Label("Manufacturer"),
        dcc.Dropdown(id='manufacturer', options=[
            {'label': i, 'value': i} for i in ['Acer', 'Dell', 'HP', 'Asus', 'Lenovo', 'Huawei', 'Toshiba', 'MSI', 'Razer', 'Samsung', 'Xiaomi']
        ]),
        
        html.Label("Screen Type"),
        dcc.Dropdown(id='screen-type', options=[
            {'label': i, 'value': i} for i in ['Full HD', 'IPS panel']
        ]),
    ], style={'columnCount': 2}),
    html.Button('Predict Price', id='predict-button', n_clicks=0),
    html.Div(id='prediction-output')
])

@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [Input('cpu-frequency', 'value'),
     Input('category', 'value'),
     Input('gpu', 'value'),
     Input('os', 'value'),
     Input('cpu-core', 'value'),
     Input('ram-gb', 'value'),
     Input('storage-gb', 'value'),
     Input('manufacturer', 'value'),
     Input('screen-type', 'value')]
)
def update_prediction(n_clicks, cpu_frequency, category, gpu, os, cpu_core, ram_gb, storage_gb, manufacturer, screen_type):
    if n_clicks > 0:
        inputs = [cpu_frequency, category, gpu, os, cpu_core, ram_gb, storage_gb, manufacturer, screen_type]
        # Converting categorical variables to numeric values (dummy encoding for simplicity)
        inputs_encoded = [cpu_frequency, category, gpu, os, cpu_core, ram_gb, storage_gb, manufacturer, screen_type]
        prediction = predict_price(inputs_encoded)
        return f"The predicted price for the laptop is: {prediction} Dollars"
    return ""

if __name__ == '__main__':
    app.run_server(debug=True)
