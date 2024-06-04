# Import required libraries
import pandas as pd
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.express as px
URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_dash.csv"
# Read the airline data into pandas dataframe
spacex_df = pd.read_csv(URL)
max_payload = spacex_df['Payload Mass (kg)'].max()
min_payload = spacex_df['Payload Mass (kg)'].min()
print(spacex_df.head())
# Create a dash application
app = dash.Dash(__name__)
server = app.server


# Create an app layout
app.layout = html.Div(children=[html.H1('SpaceX Launch Records Dashboard',
                                        style={'textAlign': 'center', 'color': '#503D36',
                                               'font-size': 40}),
                                # TASK 1: Add a dropdown list to enable Launch Site selection
                                # The default select value is for ALL sites
                                dcc.Dropdown(id='site-dropdown',
                                            options=[
                                                {"label":"All sites"   , "value":"ALL"},
                                                {"label":"CCAFS LC-40" , "value":"CCAFS LC-40"},
                                                {"label":"KSC LC-39A"  , "value":"KSC LC-39A"},
                                                {"label":"VAFB SLC-4E ", "value":"VAFB SLC-4E"},
                                                {"label":"CCAFS SLC-40", "value":"CCAFS SLC-40"}
                                                ],
                                            value="ALL",
                                            placeholder="Select a Launch site here",
                                            searchable=True
                                            ),
                                html.Br(),

                                # TASK 2: Add a pie chart to show the total successful launches count for all sites
                                # If a specific launch site was selected, show the Success vs. Failed counts for the site
                                html.Div(dcc.Graph(id='success-pie-chart')),
                                html.Br(),

                                html.P("Payload range (Kg):"),
                                # TASK 3: Add a slider to select payload range
                                dcc.RangeSlider(id   ='payload-slider',
                                		min=0, max=1e4, step=1e3,
                                		marks={0:"0",1000:"100",500:"500",1000:"1000",5000:"5000",10000:"$1e4$"},
                                		value=[min_payload, max_payload]),

                                # TASK 4: Add a scatter chart to show the correlation between payload and launch success
                                html.Div(dcc.Graph(id='success-payload-scatter-chart')),
                                ])

# TASK 2:
# Add a callback function for `site-dropdown` as input, `success-pie-chart` as output
# Function decorator to specify function input and output
@app.callback(Output(component_id='success-pie-chart', component_property='figure'),
              Input(component_id='site-dropdown', component_property='value'))
def get_pie_chart(site_dropdown):
    if site_dropdown == 'ALL':
        fig = px.pie(values= spacex_df["class"],                    
                     names = spacex_df["Launch Site"],
                     title="Succes percentage by launch site")
        return fig
    else:
        filtered_df = spacex_df[spacex_df["Launch Site"]==site_dropdown]
        print(filtered_df[["class","Launch Site"]])
        print("########################33")
        print(filtered_df["class"])
        print("##########################")
        print(filtered_df)
        print("##############################swsd")
        print("hola\n",filtered_df["class"].value_counts().index)
        fig = px.pie(values= filtered_df["class"].value_counts(),
                     names = filtered_df["class"].unique(),
                     title = "Succes percentage by launch site")
        return fig

# TASK 4:
# Add a callback function for `site-dropdown` and `payload-slider` as inputs, `success-payload-scatter-chart` as output
@app.callback(Output(component_id="success-payload-scatter-chart", component_property="figure"),
	      [Input(component_id="payload-slider", component_property="value"),
               Input(component_id="site-dropdown" , component_property="value")])
def get_scatter_chart(payload_slider,site_dropdown):
	if site_dropdown == 'ALL':
		scatter_data = spacex_df[(spacex_df["Payload Mass (kg)"]>=payload_slider[0]) & (spacex_df["Payload Mass (kg)"]<=payload_slider[1])]# and spacex_df["Payload Mass (kg)"]<=payload_slider[1]]
		scatter_data = scatter_data[["Payload Mass (kg)","class","Booster Version Category"]]
		print(scatter_data)
		fig = px.scatter(scatter_data,x="Payload Mass (kg)", y="class", color="Booster Version Category")
#               names = spacex_df["Launch Site"],
#               title="Succes percentage by launch site")
	return fig
# Run the app
if __name__ == '__main__':
    app.run_server()
