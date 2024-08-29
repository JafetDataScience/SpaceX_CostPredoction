import pandas as pd
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
#import plotly.graph_objects as go
import plotly.express as px
from dash import no_update
#import datetime as dt
import Cosmology as Cosmo


#Create app
app = dash.Dash(__name__)
#Clear the layout and do not display exception till callback gets executed
app.config.suppress_callback_exceptions = True

#Layout Section of Dash
#Task 1 Add the Title to the Dashboard
app.layout = html.Div(children=[html.H1('Universo tipo Fridman-Lemetre',
                                style={'textAlign': 'center', 'color': '#503D36',
                                'font-size': 26}),
# TASK 2: Add the radio items and a dropdown right below the first inner division
     #outer division starts
     html.Div([
                   # First inner divsion for  adding dropdown helper text for Selected Drive wheels
                    html.Div([
                            html.H2("Hubble constant [km/Mpc*s]"),
                        dcc.Input(placeholder=69.60, value = 69.60, id="H_0")]),
 
                    html.Div([
                            html.H2('Relativistic particles'),
                        dcc.Input(placeholder=1e-16, value=1e-16, id="Wr")]),

                    html.Div([
                            html.H2('Barionic matter'),
                        dcc.Input(placeholder=0.28,value=0.28, id="Wm")]),

                    html.Div([
                            html.H2('Dark energy'),
                        dcc.Input(placeholder=1-0.28-1e-16,value=1-0.28-1e-16, id="Wv")]),
                    html.Br(),
                    dbc.Button("Submit", id = "submit_button", color = "primary"),
                    html.Br(), html.Br(),
#TASK 3: Add one empty division for output inside the next inner division.
                    html.Div(dcc.Graph(id='plot1'), style={'width':'65%'})
    #outer division ends
     ])
                                ])
#layout ends
#TASK 4: Add the Ouput and input components inside the app.callback decorator.
#Place to add @app.callback Decorator
@app.callback(Output(component_id='plot1', component_property='figure'),
               [Input('submit_button', 'n_clicks')],
               [State(component_id='H_0'  , component_property='value'),
                State(component_id='Wr'   , component_property='value'),
                State(component_id='Wm'   , component_property='value'),
                State(component_id='Wv'   , component_property="value")])
#TASK 5: Add the callback function.
#Place to define the callback function .
def line(n_clicks,H_0,Wr,Wm,Wv):
  if n_clicks == None:
    return ""
  else:
    df = Cosmo.Universe_R_DataFrame(float(H_0),float(Wm),float(Wr),float(Wv))
    t_present, R_present = Cosmo.Universe_Age(float(H_0),float(Wm),float(Wr),float(Wv))
    present = pd.DataFrame({'time': [t_present], 'radi': [R_present]})
  
    df["type"] = f"Ωm= {round(float(Wm),2)}\Ωr= {round(float(Wr),2)}\Ωv= {round(float(Wv),2)}"
    present["type"] = f"t_p= {round(t_present,2)}"
    df_combined = pd.concat([df,present])
    #print(df_combined)

    fig1 = px.scatter(df_combined, x="time", y="radi",color="type",title="Evolución del universo",labels={"time":"Time [Gy]","radi":"Normal scale factor R(t)"})
    return fig1
  #fig1.scatter(present, x="x", y="y")
  #fig1.ylabel("Factor de escala normalizado",fontsize = 20)
#  fig1.grid()
if __name__ == '__main__':
    app.run_server()
#Recuerda que tienes tres dx's comentados
