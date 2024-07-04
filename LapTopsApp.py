import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np
import pandas as pd
import joblib as jl

# Load the model and scaler
model = jl.load("RegrssionTree_LaptopsPrice.pkl")
scaler = jl.load("Scaler_LaptopsPrice.pkl")

def predict_price(inputs):
    Manufacturers = ['Acer', 'Asus', 'Dell', "HP", 'Huawei', 'Lenovo', 'MSI', 'Razer', 'Samsung', 'Toshiba', 'Xiaomi']
    Screens = ['Full HD', 'IPS panel']
    
    for i in enumerate(Manufacturers):
        if inputs[-2] == i[1]:
            Manufacturers_0 = np.zeros(len(Manufacturers))
            Manufacturers_0[i[0]] = 1
            if inputs[-1] == 'Full HD':           
                inputs_1 = np.concatenate((np.array(inputs[0:-2]), np.array([1, 0]), Manufacturers_0))
            elif inputs[-1] == 'IPS panel':           
                inputs_1 = np.concatenate((np.array(inputs[0:-2]), np.array([0, 1]), Manufacturers_0))

    df = pd.DataFrame([inputs_1], columns=[
        'Category', 'GPU', 'OS', 'CPU_core', 'CPU_frequency', 'RAM_GB', 'Storage_GB_SSD',
        'Screen-Full_HD', 'Screen-IPS_panel', 'Manufacturer_Acer', 'Manufacturer_Asus',
        'Manufacturer_Dell', 'Manufacturer_HP', 'Manufacturer_Huawei', 'Manufacturer_Lenovo',
        'Manufacturer_MSI', 'Manufacturer_Razer', 'Manufacturer_Samsung', 'Manufacturer_Toshiba',
        'Manufacturer_Xiaomi'
    ])
    New_data = scaler.transform(df)
    df = pd.DataFrame([New_data[0]], columns=[
        'Category', 'GPU', 'OS', 'CPU_core', 'CPU_frequency', 'RAM_GB', 'Storage_GB_SSD',
        'Screen-Full_HD', 'Screen-IPS_panel', 'Manufacturer_Acer', 'Manufacturer_Asus',
        'Manufacturer_Dell', 'Manufacturer_HP', 'Manufacturer_Huawei', 'Manufacturer_Lenovo',
        'Manufacturer_MSI', 'Manufacturer_Razer', 'Manufacturer_Samsung', 'Manufacturer_Toshiba',
        'Manufacturer_Xiaomi'
    ])
    prediction = model.predict(df)
    return prediction[0]

app = dash.Dash(__name__)
server = app.server
app.layout = html.Div([
    html.H1("Laptop Price Prediction", style={'text-align': 'center'}),
    html.Div([
        html.Div([
		html.Label("CPU Frequency (GHz)"),
            dcc.Input(id='cpu-frequency', type='number', placeholder="Enter CPU Frequency", style={'width': '80%'}),
            
            html.Label("Category", style={'display':'block'}),
            dcc.Dropdown(id='category', options=[
                {'label': str(i), 'value': i} for i in range(1, 6)
            ], placeholder="Select Category", style={'width': '80%'}),
            
            html.Label("GPU"),
            dcc.Dropdown(id='gpu', options=[
                {'label': str(i), 'value': i} for i in range(1, 4)
            ], placeholder="Select GPU", style={'width': '80%'}),
            
            html.Label("OS"),
            dcc.Dropdown(id='os', options=[
                {'label': str(i), 'value': i} for i in range(1, 3)
            ], placeholder="Select OS", style={'width': '80%'}),
            
            html.Label("CPU Cores"),
            dcc.Dropdown(id='cpu-core', options=[
                {'label': str(i), 'value': i} for i in [3, 5, 7]
            ], placeholder="Select CPU Cores", style={'width': '80%'}),
		            html.Label("RAM (GB)"),
            dcc.Dropdown(id='ram-gb', options=[
                {'label': str(i), 'value': i} for i in [4, 6, 8, 12, 16]
            ], placeholder="Select RAM (GB)", style={'width': '80%'}),
            
            html.Label("Storage (GB SSD)"),
            dcc.Dropdown(id='storage-gb', options=[
                {'label': str(i), 'value': i} for i in [128, 256]
            ], placeholder="Select Storage (GB SSD)", style={'width': '80%'}),
            
            html.Label("Manufacturer"),
            dcc.Dropdown(id='manufacturer', options=[
                {'label': i, 'value': i} for i in ['Acer', 'Dell', 'HP', 'Asus', 'Lenovo', 'Huawei', 'Toshiba', 'MSI', 'Razer', 'Samsung', 'Xiaomi']
            ], placeholder="Select Manufacturer", style={'width': '80%'}),
            
            html.Label("Screen Type"),
            dcc.Dropdown(id='screen-type', options=[
                {'label': i, 'value': i} for i in ['Full HD', 'IPS panel']
            ], placeholder="Select Screen Type", style={'width': '80%'}),
        ], style={'padding': '10px', 'position':'relative', 'height': 'fit-content', 'display':'grid', 'gap':'1rem'}),
        
        html.Button('Predict Price', id='predict-button', n_clicks=0, style={'margin-top': '20px'}),
    ], style={'max-width': '800px', 'margin': '0 auto'}),
    
    html.Div(id='prediction-output', style={'text-align': 'center', 'margin-top': '20px', 'font-size': '24px'})
], style={'font-family': 'Arial, sans-serif'})

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
        inputs = [category, gpu, os, cpu_core, cpu_frequency, ram_gb, storage_gb, screen_type, manufacturer]
        prediction = predict_price(inputs)
        return f"The predicted price for the laptop is: {prediction} Dollars"
    return ""

if __name__ == '__main__':
    app.run_server(debug=True)
