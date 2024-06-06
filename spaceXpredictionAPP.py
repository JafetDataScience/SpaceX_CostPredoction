import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import joblib as jl
import numpy as np
import pandas as pd

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("SpaceX landing prediction"),

    html.Label("Number of flights with the core"),
    dcc.Input(id='flight-number', type='number', value=0),

    html.Label("Payload Mass"),
    dcc.Input(id='payload-mass', type='number', value=0),

    html.Label("Flights"),
    dcc.Input(id='flights', type='number', value=0),

    html.Label("Block (nucle version)"),
    dcc.Input(id='block', type='number', value=0),

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

    html.Label("Grid"),
    dcc.Dropdown(
        id='grid',
        options=[
            {'label': 'True', 'value': 'True'},
            {'label': 'False', 'value': 'False'}
        ],
        value='True'
    ),

    html.Label("Reused"),
    dcc.Dropdown(
        id='reused',
        options=[
            {'label': 'True', 'value': 'True'},
            {'label': 'False', 'value': 'False'}
        ],
        value='True'
    ),

    html.Label("Legs"),
    dcc.Dropdown(
        id='legs',
        options=[
            {'label': 'True', 'value': 'True'},
            {'label': 'False', 'value': 'False'}
        ],
        value='True'
    ),

    html.Button('Submit', id='submit-button', n_clicks=0),
    html.Div(id='output-container')
])

@app.callback(
    Output('output-container', 'children'),
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
def predict_landing(flight_number, payload_mass, flights, block, reused_count, orbit_type, launch_site, landing_site, version, grid, reused, legs):
    model = jl.load("LogisticReg_SpaceX.pkl")
    scaler = jl.load("Scaler_SpaceX_LR.pkl")
    new_data = [flight_number, payload_mass, flights, block, reused_count]
    
    orbits = np.array(['Orbit_ES-L1', 'Orbit_GEO', 'Orbit_GTO', 'Orbit_HEO', 'Orbit_ISS', 'Orbit_LEO', 'Orbit_MEO', 'Orbit_PO', 'Orbit_SO', 'Orbit_SSO', 'Orbit_VLEO'])
    Meta_Orbit = np.zeros(len(orbits))
    Choosen_orbit = np.where(orbits == orbit_type)
    Meta_Orbit[Choosen_orbit] = 1
    
    L_sites = np.array(['LaunchSite_CCAFS SLC 40', 'LaunchSite_KSC LC 39A','LaunchSite_VAFB SLC 4E'])
    Meta_site = np.zeros(len(L_sites))
    Choosen_site = np.where(L_sites == launch_site)
    Meta_site[Choosen_site] = 1
    
    Landing_sites = np.array(['LandingPad_5e9e3032383ecb267a34e7c7','LandingPad_5e9e3032383ecb554034e7c9','LandingPad_5e9e3032383ecb6bb234e7ca','LandingPad_5e9e3032383ecb761634e7cb','LandingPad_5e9e3033383ecbb9e534e7cc'])
    Meta_Land_site = np.zeros(len(Landing_sites))
    if landing_site != "N/A":
        Choosen_landing_site = np.where(Landing_sites == landing_site)
        Meta_Land_site[Choosen_landing_site] = 1
    
    versions = np.array(['Serial_B0003', 'Serial_B0005',
       'Serial_B0007', 'Serial_B1003', 'Serial_B1004', 'Serial_B1005',
       'Serial_B1006', 'Serial_B1007', 'Serial_B1008', 'Serial_B1010',
       'Serial_B1011', 'Serial_B1012', 'Serial_B1013', 'Serial_B1015',
       'Serial_B1016', 'Serial_B1017', 'Serial_B1018', 'Serial_B1019',
       'Serial_B1020', 'Serial_B1021', 'Serial_B1022', 'Serial_B1023',
       'Serial_B1025', 'Serial_B1026', 'Serial_B1028', 'Serial_B1029',
       'Serial_B1030', 'Serial_B1031', 'Serial_B1032', 'Serial_B1034',
       'Serial_B1035', 'Serial_B1036', 'Serial_B1037', 'Serial_B1038',
       'Serial_B1039', 'Serial_B1040', 'Serial_B1041', 'Serial_B1042',
       'Serial_B1043', 'Serial_B1044', 'Serial_B1045', 'Serial_B1046',
       'Serial_B1047', 'Serial_B1048', 'Serial_B1049', 'Serial_B1050',
       'Serial_B1051', 'Serial_B1054', 'Serial_B1056', 'Serial_B1058',
       'Serial_B1059', 'Serial_B1060', 'Serial_B1062'])
    Meta_version = np.zeros(len(versions))
    Choosen_version = np.where(versions == version)
    Meta_version[Choosen_version] = 1
    
    Grids = np.array([False,True])
    Meta_grid = np.zeros(len(Grids))
    Choosen_grid = np.where(Grids == grid)
    Meta_grid[Choosen_grid] = 1
    
    Reuseds = np.array([False,True])
    Meta_Reused = np.zeros(len(Reuseds))
    Choosen_Reused = np.where(Reuseds == reused)
    Meta_Reused[Choosen_Reused] = 1
    
    Legs = np.array([False,True])
    Meta_Legs = np.zeros(len(Legs))
    Choosen_Legs = np.where(Legs == legs)
    Meta_Legs[Choosen_Legs] = 1
    
    New_data = np.concatenate((np.array(new_data),Meta_Orbit,Meta_site,Meta_Land_site,Meta_version,Meta_grid,Meta_Reused,Meta_Legs))
    print(len(New_data))
    df = pd.DataFrame([New_data],columns = ['FlightNumber', 'PayloadMass', 'Flights', 'Block', 'ReusedCount',
       'Orbit_ES-L1', 'Orbit_GEO', 'Orbit_GTO', 'Orbit_HEO', 'Orbit_ISS',
       'Orbit_LEO', 'Orbit_MEO', 'Orbit_PO', 'Orbit_SO', 'Orbit_SSO',
       'Orbit_VLEO', 'LaunchSite_CCAFS SLC 40', 'LaunchSite_KSC LC 39A',
       'LaunchSite_VAFB SLC 4E', 'LandingPad_5e9e3032383ecb267a34e7c7',
       'LandingPad_5e9e3032383ecb554034e7c9',
       'LandingPad_5e9e3032383ecb6bb234e7ca',
       'LandingPad_5e9e3032383ecb761634e7cb',
       'LandingPad_5e9e3033383ecbb9e534e7cc', 'Serial_B0003', 'Serial_B0005',
       'Serial_B0007', 'Serial_B1003', 'Serial_B1004', 'Serial_B1005',
       'Serial_B1006', 'Serial_B1007', 'Serial_B1008', 'Serial_B1010',
       'Serial_B1011', 'Serial_B1012', 'Serial_B1013', 'Serial_B1015',
       'Serial_B1016', 'Serial_B1017', 'Serial_B1018', 'Serial_B1019',
       'Serial_B1020', 'Serial_B1021', 'Serial_B1022', 'Serial_B1023',
       'Serial_B1025', 'Serial_B1026', 'Serial_B1028', 'Serial_B1029',
       'Serial_B1030', 'Serial_B1031', 'Serial_B1032', 'Serial_B1034',
       'Serial_B1035', 'Serial_B1036', 'Serial_B1037', 'Serial_B1038',
       'Serial_B1039', 'Serial_B1040', 'Serial_B1041', 'Serial_B1042',
       'Serial_B1043', 'Serial_B1044', 'Serial_B1045', 'Serial_B1046',
       'Serial_B1047', 'Serial_B1048', 'Serial_B1049', 'Serial_B1050',
       'Serial_B1051', 'Serial_B1054', 'Serial_B1056', 'Serial_B1058',
       'Serial_B1059', 'Serial_B1060', 'Serial_B1062', 'GridFins_False',
       'GridFins_True', 'Reused_False', 'Reused_True', 'Legs_False',
       'Legs_True'])
    New_data = scaler.transform(df)
    prediction = model.predict(New_data)
    probability= model.predict_proba(New_data)
    
    if prediction == 1:
           return "A Successful landing of the first stage is predicted with a probability "+str(100*round(probability[0,1],2))+"%"
    else:
            return "An Unsuccessful landing of the first stage is predicted with a probability of"+str(100*round(probability[0,1],2))+"%"

if __name__ == '__main__':
    app.run_server(debug=True)
