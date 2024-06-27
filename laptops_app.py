import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import joblib as jl
import pandas as pd

# Inicializar la aplicación Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
# Definir el diseño de la aplicación
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Laptop Price Prediction"), className="mb-2")
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Label("CPU Frequency"),
            dcc.Input(id="cpu-frequency", type="number", step=0.1, className="mb-2")
        ])
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Label("Category"),
            dcc.Dropdown(id="category", options=[{'label': str(i), 'value': i} for i in range(1, 6)], className="mb-2")
        ])
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Label("GPU"),
            dcc.Dropdown(id="gpu", options=[{'label': str(i), 'value': i} for i in range(1, 4)], className="mb-2")
        ])
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Label("OS"),
            dcc.Dropdown(id="os", options=[{'label': str(i), 'value': i} for i in range(1, 3)], className="mb-2")
        ])
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Label("CPU Core"),
            dcc.Dropdown(id="cpu-core", options=[{'label': str(i), 'value': i} for i in [3, 5, 7]], className="mb-2")
        ])
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Label("RAM (GB)"),
            dcc.Dropdown(id="ram-gb", options=[{'label': str(i), 'value': i} for i in [4, 6, 8, 12, 16]], className="mb-2")
        ])
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Label("Storage (GB SDD)"),
            dcc.Dropdown(id="storage-gb", options=[{'label': str(i), 'value': i} for i in [128, 256]], className="mb-2")
        ])
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Label("Manufacturer"),
            dcc.Dropdown(id="manufacturer", options=[{'label': m, 'value': m} for m in ['Acer', 'Dell', 'HP', 'Asus', 'Lenovo', 'Huawei', 'Toshiba', 'MSI', 'Razer', 'Samsung', 'Xiaomi']], className="mb-2")
        ])
    ]),                
    dbc.Row([
        dbc.Col([
            dbc.Label("Screen type"),
            dcc.Dropdown(id="Screen_type", options=[{'label': m, 'value': m} for m in ['Full HD', 'IPS panel']], className="mb-2")
        ])
    ]),                                 
    dbc.Row([
        dbc.Col(
            dbc.Button("Predict Price", id="predict-button", className="mb-2")
        )
    ]),
    dbc.Row([
        dbc.Col(html.Div(id="prediction-output", className="mb-2"))
    ])
])

# Definir la función de predicción (ejemplo simplificado)
def predict_price(inputs):
    model  = jl.load("RegrssionTree_LaptopsPrice.pkl")
    scaler = jl.load("Scaler_LaptopsPrice.pkl")
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
    df = pd.DataFrame([inputs_1], columns = ['Category', 'GPU', 'OS', 'CPU_core',
       'CPU_frequency', 'RAM_GB', 'Storage_GB_SSD',
       'Screen-Full_HD', 'Screen-IPS_panel', 'Manufacturer_Acer',
       'Manufacturer_Asus', 'Manufacturer_Dell', 'Manufacturer_HP',
       'Manufacturer_Huawei', 'Manufacturer_Lenovo', 'Manufacturer_MSI',
       'Manufacturer_Razer', 'Manufacturer_Samsung', 'Manufacturer_Toshiba',
       'Manufacturer_Xiaomi'])
    New_data = scaler.transform(df)
    predict = model.predict(New_data)
#    print(prediction,type(prediction))
    return predict[0]

# Definir la callback para la predicción
@app.callback(
    Output("prediction-output", "children"),
    Input("predict-button", "n_clicks"),
    State("cpu-frequency", "value"),
    State("category", "value"),
    State("gpu", "value"),
    State("os", "value"),
    State("cpu-core", "value"),
    State("ram-gb", "value"),
    State("storage-gb", "value"),
    State("manufacturer", "value"),
    State("Screen_type", "value")
)
def update_prediction(n_cliks, cpu_freq, category, gpu, os, cpu_core, ram_gb, storage_gb, manufacturer, Screen_type):
    inputs = [cpu_freq, category, gpu, os, cpu_core, ram_gb, storage_gb, manufacturer, Screen_type]
    prediction = round(predict_price(inputs),2)
    return f"The predicted price for the laptop is: {prediction}"

# Ejecutar la aplicación
if __name__ == "__main__":
    app.run_server(debug=True)

