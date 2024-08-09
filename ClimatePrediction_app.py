import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import joblib as jl
import numpy as np
import pandas as pd

# Simulando un modelo de clasificación
def dummy_model(inputs):
    model  = jl.load("LinearRegression_ClimatePrediction.pkl")
    scaler = jl.load("Scaler_ClimatePrediction.pkl")
    directions = ['WSW', 'WNW', 'W', 'SW', 'SSW', 'SSE', 'SE', 'S', 'NW', 'NNW', 'NNE', 'NE', 'N', 'ESE', 'ENE', 'E']
#    Screens = ['Full HD', 'IPS panel']	
    for i in enumerate(directions):
        print(inputs[-3], i[1])
        if inputs[-3] == i[1]:
            WindGustDir_0 = np.zeros(len(directions))
            WindGustDir_0[i[0]] = 1
            
        if inputs[-2] == i[1]:
            Wind9amDir_0 = np.zeros(len(directions))
            Wind9amDir_0[i[0]] = 1
        
        if inputs[-1] == i[1]:
            Wind3pmDir_0 = np.zeros(len(directions))
            Wind3pmDir_0[i[0]] = 1
            
    if inputs[-4] == 'yes':           
        inputs_1 = np.concatenate((np.array(inputs[0:-4]), np.array([0,1]), WindGustDir_0, Wind9amDir_0, Wind3pmDir_0))
    else :
        inputs_1 = np.concatenate((np.array(inputs[0:-4]), np.array([1,0]), WindGustDir_0, Wind9amDir_0, Wind3pmDir_0))
            
    df = pd.DataFrame([inputs_1], columns = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
       'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am',
       'Humidity3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm',
       'RainToday_No', 'RainToday_Yes', 'WindGustDir_E', 'WindGustDir_ENE',
       'WindGustDir_ESE', 'WindGustDir_N', 'WindGustDir_NE', 'WindGustDir_NNE',
       'WindGustDir_NNW', 'WindGustDir_NW', 'WindGustDir_S', 'WindGustDir_SE',
       'WindGustDir_SSE', 'WindGustDir_SSW', 'WindGustDir_SW', 'WindGustDir_W',
       'WindGustDir_WNW', 'WindGustDir_WSW', 'WindDir9am_E', 'WindDir9am_ENE',
       'WindDir9am_ESE', 'WindDir9am_N', 'WindDir9am_NE', 'WindDir9am_NNE',
       'WindDir9am_NNW', 'WindDir9am_NW', 'WindDir9am_S', 'WindDir9am_SE',
       'WindDir9am_SSE', 'WindDir9am_SSW', 'WindDir9am_SW', 'WindDir9am_W',
       'WindDir9am_WNW', 'WindDir9am_WSW', 'WindDir3pm_E', 'WindDir3pm_ENE',
       'WindDir3pm_ESE', 'WindDir3pm_N', 'WindDir3pm_NE', 'WindDir3pm_NNE',
       'WindDir3pm_NNW', 'WindDir3pm_NW', 'WindDir3pm_S', 'WindDir3pm_SE',
       'WindDir3pm_SSE', 'WindDir3pm_SSW', 'WindDir3pm_SW', 'WindDir3pm_W',
       'WindDir3pm_WNW', 'WindDir3pm_WSW'])
    New_data = scaler.transform(df)
    predict = model.predict(New_data)
#    print(prediction,type(prediction))
    if predict[0] >= 0.5:
        outcome = "Yes it will rain"
    elif predict[0] < 0.5:
        outcome = "No it will not rain"
        
    return outcome
app = dash.Dash(__name__)
server = app.server
app.layout = html.Div([
    html.H1('Weather Prediction App'),
    
    dbc.Row([
        dbc.Col([
            html.Label('Minimum Temperature'),
            dcc.Input(id='min_temp', type='number', placeholder='Enter min temp [Celsius]'),
        ]),
        dbc.Col([
            html.Label('Maximum Temperature'),
            dcc.Input(id='max_temp', type='number', placeholder='Enter max temp [Celsius]'),
        ]),
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Label('Temperature at 9 a.m.'),
            dcc.Input(id='temp_9am', type='number', placeholder='Enter temp at 9 a.m. [Celsius]'),
        ]),
        dbc.Col([
            html.Label('Temperature at 3 p.m.'),
            dcc.Input(id='temp_3pm', type='number', placeholder='Enter temp at 3 p.m. [Celsius]'),
        ]),
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Label('Rainfall'),
            dcc.Input(id='rainfall', type='number', placeholder='Enter rainfall [mm]'),
        ]),
        dbc.Col([
            html.Label('Evaporation'),
            dcc.Input(id='evaporation', type='number', placeholder='Enter evaporation [mm]'),
        ]),
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Label('Sunshine'),
            dcc.Input(id='sunshine', type='number', placeholder='Enter sunshine [Hours]'),
        ]),
        dbc.Col([
            html.Label('Wind Gust Direction'),
            dcc.Dropdown(
                id='wind_gust_dir',
                options=[
                    {'label': dir, 'value': dir} for dir in ['WSW', 'WNW', 'W', 'SW', 'SSW', 'SSE', 'SE', 'S', 'NW', 'NNW', 'NNE', 'NE', 'N', 'ESE', 'ENE', 'E']
                ],
                placeholder='Select direction'
            ),
        ]),
    ]),
    
    dbc.Row([        
        dbc.Col([
            html.Label('Wind Direction at 9 a.m.'),
            dcc.Dropdown(
                id='wind_dir_9am',
                options=[
                    {'label': dir, 'value': dir} for dir in ['WSW', 'WNW', 'W', 'SW', 'SSW', 'SSE', 'SE', 'S', 'NW', 'NNW', 'NNE', 'NE', 'N', 'ESE', 'ENE', 'E']
                ],
                placeholder='Select direction'
            ),
        ]),
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Label('Wind Direction at 3 p.m.'),
            dcc.Dropdown(
                id='wind_dir_3pm',
                options=[
                    {'label': dir, 'value': dir} for dir in ['WSW', 'WNW', 'W', 'SW', 'SSW', 'SSE', 'SE', 'S', 'NW', 'NNW', 'NNE', 'NE', 'N', 'ESE', 'ENE', 'E']
                ],
                placeholder='Select direction'
            ),
        ]),
        dbc.Col([
            html.Label('Wind Gust Speed'),
            dcc.Input(id='wind_gust_speed', type='number', placeholder='Enter gust speed [km/h]'),
        ]),
        dbc.Col([
            html.Label('Wind Speed at 9 a.m.'),
            dcc.Input(id='wind_speed_9am', type='number', placeholder='Enter speed at 9 a.m. [km/h]'),
        ]),
    ]),
    
    dbc.Row([    
        dbc.Col([
            html.Label('Wind Speed at 3 p.m.'),
            dcc.Input(id='wind_speed_3pm', type='number', placeholder='Enter speed at 3 p.m. [km/h]'),
        ]),
        dbc.Col([
            html.Label('Humidity at 9 a.m.'),
            dcc.Input(id='humidity_9am', type='number', placeholder='Enter humidity at 9 a.m. [Percent]'),
        ]),
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Label('Humidity at 3 p.m.'),
            dcc.Input(id='humidity_3pm', type='number', placeholder='Enter humidity at 3 p.m. [Percent]'),
        ]),
        dbc.Col([
            html.Label('Cloud at 9 a.m.'),
            dcc.Input(id='cloud_9am', type='number', placeholder='Enter cloud at 9 a.m. [Eights of sky surface]'),
        ]),
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Label('Cloud at 3 p.m.'),
            dcc.Input(id='cloud_3pm', type='number', placeholder='Enter cloud at 3 p.m. [Eights of sky surface]'),
        ]),
        dbc.Col([
            html.Label('Rain Today'),
            dcc.Dropdown(
                id='rain_today',
                options=[
                    {'label': 'Yes', 'value': 'yes'},
                    {'label': 'No', 'value': 'no'}
                ],
                placeholder='Select yes or no'
            ),
        ]),
    ]),
    
    html.Br(),
    
    dbc.Button('Submit', id='submit-button', color='primary'),
    
    html.Br(), html.Br(),
    
    html.Div(id='output-container')
])

@app.callback(
    Output('output-container', 'children'),
    [Input('submit-button', 'n_clicks')],
    [State('min_temp', 'value'), State('max_temp', 'value'), State('temp_9am', 'value'), State('temp_3pm', 'value'),
     State('rainfall', 'value'), State('evaporation', 'value'), State('sunshine', 'value'), State('wind_gust_dir', 'value'),
     State('wind_gust_speed', 'value'), State('wind_dir_9am', 'value'), State('wind_dir_3pm', 'value'), State('wind_speed_9am', 'value'),
     State('wind_speed_3pm', 'value'), State('humidity_9am', 'value'), State('humidity_3pm', 'value'), State('cloud_9am', 'value'),
     State('cloud_3pm', 'value'), State('rain_today', 'value')]
)
def update_output(n_clicks, min_temp, max_temp, temp_9am, temp_3pm, rainfall, evaporation, sunshine, wind_gust_dir, wind_gust_speed,
                  wind_dir_9am, wind_dir_3pm, wind_speed_9am, wind_speed_3pm, humidity_9am, humidity_3pm, cloud_9am, cloud_3pm, rain_today):
    if n_clicks is None:
        return ''
    else:
        # Recolectar los inputs en una lista
        inputs = [min_temp, max_temp, rainfall, evaporation, sunshine, wind_gust_speed, wind_speed_9am, wind_speed_3pm, humidity_9am, 
                  humidity_3pm, cloud_9am, cloud_3pm, temp_9am, temp_3pm, rain_today, wind_gust_dir, wind_dir_9am, wind_dir_3pm]
        # Simular el procesamiento del modelo
#        inputs = [19.0,24.8,0.0,5.2,7.5,41,7,17,79,65,7,7,20.8,23.4,"no","W","WNW","E"]
        prediction = dummy_model(inputs)
        
        return f"Prediction: {prediction}"

if __name__ == '__main__':
    app.run_server(debug=True)

