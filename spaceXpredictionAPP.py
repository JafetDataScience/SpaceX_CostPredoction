import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import joblib as jl
import numpy as np
import pandas as pd

app = dash.Dash(__name__)
server = app.server
app.layout = html.Div([
    html.H1("SpaceX landing prediction"),

    html.Label("Number of flights with the core"),
    dcc.Input(id='flight-number', type='number', value=0),
    html.Br(),

    html.Label("Payload Mass"),
    dcc.Input(id='payload-mass', type='number', value=0),
    html.Br(),

    html.Label("Flights"),
    dcc.Input(id='flights', type='number', value=0),
    html.Br(),

    html.Label("Block (nucle version)"),
    dcc.Input(id='block', type='number', value=0),
    html.Br(),

    html.Label("Reused Count"),
    dcc.Input(id='reused-count', type='number', value=0),
    html.Br(),
    
    html.Label("Orbit Type"),
    dcc.Dropdown(
        id='orbit-type',
        options=[
            {'label': 'Orbit_ES-L1', 'value': 'Orbit_ES-L1'},
            {'label': 'Orbit_GEO', 'value': 'Orbit_GEO'},
            {'label': 'Orbit_GTO', 'value': 'Orbit_GTO'},
            {'label': 'Orbit_HEO', 'value': 'Orbit_HEO'},
            {'label': 'Orbit_ISS', 'value': 'Orbit_ISS'},
            {'label': 'Orbit_LEO', 'value': 'Orbit_LEO'},
            {'label': 'Orbit_MEO', 'value': 'Orbit_MEO'},
            {'label': 'Orbit_PO', 'value': 'Orbit_PO'},
            {'label': 'Orbit_SO', 'value': 'Orbit_SO'},
            {'label': 'Orbit_SSO', 'value': 'Orbit_SSO'},
            {'label': 'Orbit_VLEO', 'value': 'Orbit_VLEO'}
        ],
        value='Orbit_LEO'
    ),
    html.Br(),

    html.Label("Launch Site"),
    dcc.Dropdown(
        id='launch-site',
        options=[
            {'label': 'LaunchSite_CCAFS SLC 40', 'value': 'LaunchSite_CCAFS SLC 40'},
            {'label': 'LaunchSite_KSC LC 39A', 'value': 'LaunchSite_KSC LC 39A'},
            {'label': 'LaunchSite_VAFB SLC 4E', 'value': 'LaunchSite_VAFB SLC 4E'}
        ],
        value='LaunchSite_CCAFS SLC 40'
    ),
    html.Br(),
    
    html.Label("Landing Site"),
    dcc.Dropdown(
        id='landing-site',
        options=[
            {'label': 'LandingPad_5e9e3032383ecb267a34e7c7', 'value': 'LandingPad_5e9e3032383ecb267a34e7c7'},
            {'label': 'LandingPad_5e9e3032383ecb554034e7c9', 'value': 'LandingPad_5e9e3032383ecb554034e7c9'},
            {'label': 'LandingPad_5e9e3032383ecb6bb234e7ca', 'value': 'LandingPad_5e9e3032383ecb6bb234e7ca'},
            {'label': 'LandingPad_5e9e3032383ecb761634e7cb', 'value': 'LandingPad_5e9e3032383ecb761634e7cb'},
            {'label': 'LandingPad_5e9e3033383ecbb9e534e7cc', 'value': 'LandingPad_5e9e3033383ecbb9e534e7cc'},
            {'label': 'N/A', 'value': 'N/A'}
        ],
        value='LandingPad_5e9e3032383ecb267a34e7c7'
    ),
    html.Br(),

    html.Label("#Serie"),
    dcc.Dropdown(
        id='version',
        options=[
            {'label': f'Serial_{i}', 'value': f'Serial_{i}'} for i in [
                'B0003', 'B0005', 'B0007', 'B1003', 'B1004', 'B1005',
                'B1006', 'B1007', 'B1008', 'B1010', 'B1011', 'B1012',
                'B1013', 'B1015', 'B1016', 'B1017', 'B1018', 'B1019',
                'B1020', 'B1021', 'B1022', 'B1023', 'B1025', 'B1026',
                'B1028', 'B1029', 'B1030', 'B1031', 'B1032', 'B1034',
                'B1035', 'B1036', 'B1037', 'B1038', 'B1039', 'B1040',
                'B1041', 'B1042', 'B1043', 'B1044', 'B1045', 'B1046',
                'B1047', 'B1048', 'B1049', 'B1050', 'B1051', 'B1054',
                'B1056', 'B1058', 'B1059', 'B1060', 'B1062'
            ]
        ],
        value='B1003'
    ),
    html.Br(),

    html.Label("Grid"),
    dcc.Dropdown(
        id='grid',
        options=[
            {'label': 'True', 'value': 'True'},
            {'label': 'False', 'value': 'False'}
        ],
        value='True'
    ),
    html.Br(),

    html.Label("Reused"),
    dcc.Dropdown(
        id='reused',
        options=[
            {'label': 'True', 'value': 'True'},
            {'label': 'False', 'value': 'False'}
        ],
        value='True'
    ),
    html.Br(),

    html.Label("Legs"),
    dcc.Dropdown(
        id='legs',
        options=[
            {'label': 'True', 'value': 'True'},
            {'label': 'False', 'value': 'False'}
        ],
        value='True'
    ),
    html.Br(),

    html.Button('Submit', id='submit-button', n_clicks=0),
    html.Div(id='output-container')
])

@app.callback(
    Output('output-container', 'children'),
    [Input('submit-button', 'n_clicks')],
    [Input('flight-number', 'value'),
     Input('payload-mass', 'value'),
     Input('flights', 'value'),
     Input('block', 'value'),
     Input('reused-count', 'value'),
     Input('orbit-type', 'value'),
     Input('launch-site', 'value'),
     Input('landing-site', 'value'),
     Input('version', 'value'),
     Input('grid', 'value'),
     Input('reused', 'value'),
     Input('legs', 'value')]
)
def predict_landing(n_clicks, flight_number, payload_mass, flights, block, reused_count, orbit_type, launch_site, landing_site, version, grid, reused, legs):
    if n_clicks > 0:
        model = jl.load("LogisticReg_SpaceX.pkl")
        scaler = jl.load("Scaler_SpaceX_LR.pkl")
        new_data = [flight_number, payload_mass, flights, block, reused_count]
        
        # Processing inputs (as in your original code)
        # ...

        # Prediction logic
        prediction = model.predict(new_data)
        probability = model.predict_proba(new_data)

        if prediction == 1:
            return f"A Successful landing of the first stage is predicted with a probability of {100*round(probability[0,1], 2)}%"
        else:
            return f"An Unsuccessful landing of the first stage is predicted with a probability of {100*round(probability[0,0], 2)}%"
    return "Please click Submit to make a prediction."

if __name__ == '__main__':
    app.run_server(debug=True)
