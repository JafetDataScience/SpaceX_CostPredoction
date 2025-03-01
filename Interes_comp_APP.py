import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import numpy as np
#import seaborn as sns

app = dash.Dash(__name__)
server = app.server
# Layout
app.layout = html.Div([
    html.H1("Investment Growth Dashboard"),
    
    # Inputs
    html.Label("Investment Period"),
    dcc.Dropdown(
        id='period',
        options=[
            {'label': 'Monthly', 'value': 'monthly'},
            {'label': 'Quarterly', 'value': 'quarterly'},
            {'label': 'Semi-Annually', 'value': 'semiannual'},
            {'label': 'Annually', 'value': 'annual'}
        ],
        value='monthly'
    ),
    
    html.Label("Time (Years)"),
    dcc.Input(id='time', type='number', value=10),
    
    html.Label("Investment Return (%)"),
    dcc.Input(id='return', type='number', value=5, step=0.01),
    
    html.Label("Initial Investment ($)"),
    dcc.Input(id='initial-investment', type='number', value=10000),
    
    html.Label("Monthly Investment ($)"),
    dcc.Input(id='monthly-investment', type='number', value=200),
    
    # Outputs: Graphs
    dcc.Graph(id='total-value-graph'),
    dcc.Graph(id='monthly-investment-graph')
])

# Function M(t)
def M(t, I_0, alpha, I):
    return I_0*(1+alpha)**t+(1+alpha)*I*((1+alpha)**t-1)/alpha

# Function DeltaM(t)
def DeltaM(t, I_0, alpha, I):
    return (I_0+(1+alpha)*I/alpha)*((1+alpha)**t-1)-I*t

# Callback
@app.callback(
    [Output('total-value-graph', 'figure'),
     Output('monthly-investment-graph', 'figure')],
    [Input('period', 'value'),
     Input('time', 'value'),
     Input('return', 'value'),
     Input('initial-investment', 'value'),
     Input('monthly-investment', 'value')]
)
def update_graphs(period, time, investment_return, initial_investment, monthly_investment):
    # Time period mapping
    time_dict = {"annual": 1, "semiannual": 2, "quarterly": 4, "monthly": 12}

    # Calculate alpha
    alpha = investment_return / (100*time_dict[period])
    
    # Convert time to periods
    t_values = np.arange(0, time * time_dict[period] + 1) 
    
    # Calculate M(t) and Delta
    M_values = M(t_values, initial_investment, alpha, monthly_investment)
    DeltaM_values = DeltaM(t_values, initial_investment, alpha, monthly_investment)
    # Calculate monthly investment accumulation (optional second graph)
    accumulated_investment = [monthly_investment * t for t in t_values]

    # Generate figures
    total_value_fig = go.Figure(
        data=go.Scatter(x=t_values/time_dict[period], y=M_values, mode='lines', name='M(t)'),
        layout=go.Layout(
         title='Total Investment Value Over Time (M(t))',
         xaxis={'title': 'Time [Yrs]', "title_font":{'size': 22, 'color': 'black', 'family': 'Arial'}},
         yaxis={'title': 'M(t) [$]', "title_font":{'size': 22, 'color': 'black', 'family': 'Arial'}}
    ))

    investment_fig = go.Figure(
        data=go.Scatter(x=t_values/time_dict[period], y=DeltaM_values, mode='lines', name='DeltaM(t)'),
        layout=go.Layout(
            title='Interest Growth Over Time (ΣΔM)',
            xaxis={'title': 'Time [Yrs]', "title_font":{'size': 22, 'color': 'black', 'family': 'Arial'}},
            yaxis={'title': 'ΣΔM [$]', "title_font":{'size': 22, 'color': 'black', 'family': 'Arial'}}
    ))

    return total_value_fig, investment_fig

if __name__ == '__main__':
    app.run_server(debug=True)
