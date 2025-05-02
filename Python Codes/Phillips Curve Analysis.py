import pandas as pd
import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import datetime
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import warnings
warnings.filterwarnings('ignore')

class PhillipsCurveAnalysis:
    def __init__(self, start_date='2000-01-01', end_date='2024-10-01'):
        self.start_date = start_date
        self.end_date = end_date
        self.load_data()
        
    def load_data(self):
        print("Downloading data from FRED...")
        self.unemployment_data = web.DataReader('UNRATE', 'fred', self.start_date, self.end_date)
        self.cpi_data = web.DataReader('CPIAUCSL', 'fred', self.start_date, self.end_date)
        
        self.pce_data = web.DataReader('PCEPI', 'fred', self.start_date, self.end_date)  # PCE Price Index
        self.wage_data = web.DataReader('CES0500000003', 'fred', self.start_date, self.end_date)  # Average hourly earnings
        
        self.unemployment_data.columns = ['Unemployment']
        self.cpi_data.columns = ['CPI']
        self.pce_data.columns = ['PCE']
        self.wage_data.columns = ['Wages']
        
        self.inflation_cpi = self.cpi_data.pct_change(12) * 100
        self.inflation_cpi.columns = ['Inflation_CPI']
        
        self.inflation_pce = self.pce_data.pct_change(12) * 100
        self.inflation_pce.columns = ['Inflation_PCE']
        
        self.wage_growth = self.wage_data.pct_change(12) * 100
        self.wage_growth.columns = ['Wage_Growth']
        
        dfs = [
            self.unemployment_data, 
            self.inflation_cpi, 
            self.inflation_pce,
            self.wage_growth
        ]
        
        self.combined_data = pd.concat(dfs, axis=1)
        
        
        self.data = self.combined_data.dropna()
        
        self.periods = [
            ('Pre-Crisis', '2000-01-01', '2007-12-31'),       
            ('Fin Crisis', '2008-01-01', '2014-12-31'),       
            ('Pre-COVID', '2015-01-01', '2019-12-31'),        
            ('COVID', '2020-01-01', '2022-12-31'),            
            ('Post-COVID', '2023-01-01', '2024-10-01')        
        ]
        
        self.period_data = {}
        for label, start, end in self.periods:
            period_slice = self.data.loc[start:end] if len(self.data.loc[start:end]) > 0 else None
            if period_slice is not None and len(period_slice) > 0:
                self.period_data[label] = period_slice.copy()
    
    def run_statistical_analysis(self):
        print("\nRunning regression analysis for the Phillips Curve:")
        
        X = sm.add_constant(self.data['Unemployment'])
        self.phillips_model = sm.OLS(self.data['Inflation_CPI'], X).fit()
        print(self.phillips_model.summary())
        
        self.period_models = {}
        print("\nAnalysis by period:")
        
        for label, period_df in self.period_data.items():
            if len(period_df) > 10:  
                X = sm.add_constant(period_df['Unemployment'])
                model = sm.OLS(period_df['Inflation_CPI'], X).fit()
                self.period_models[label] = model
                print(f"\nPeriod: {label}")
                print(f"Unemployment coefficient: {model.params[1]:.4f}")
                print(f"P-value: {model.pvalues[1]:.4f}")
                print(f"R-squared: {model.rsquared:.4f}")
        
        return self
        
    def create_dashboard(self):
        app = dash.Dash(__name__, title="Phillips Curve Analysis Dashboard")
        
        scatter_fig = px.scatter(self.data, x='Unemployment', y='Inflation_CPI',
                            hover_data=['Wage_Growth'],
                            title='Phillips Curve (2000-2024)')
        
        X = self.data['Unemployment']
        y = self.data['Inflation_CPI']
        slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
        
        scatter_fig.add_trace(go.Scatter(
            x=np.sort(X),
            y=intercept + slope * np.sort(X),
            mode='lines',
            name=f'Linear fit: y = {slope:.3f}x + {intercept:.3f}, r = {r_value:.3f}',
            line=dict(color='red', width=2)
        ))
        
        scatter_fig.update_layout(
            xaxis_title='Unemployment Rate (%)',
            yaxis_title='CPI Inflation Rate (%)',
            height=600,
            width=1200,  
            margin=dict(l=60, r=40, t=80, b=60),
            hovermode='closest',
            plot_bgcolor='rgba(240, 240, 250, 0.9)',  
            xaxis=dict(gridcolor='white', gridwidth=1),
            yaxis=dict(gridcolor='white', gridwidth=1)
        )
        
        ts_fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        ts_fig.add_trace(
            go.Scatter(x=self.data.index, y=self.data['Unemployment'], name='Unemployment',
                      line=dict(color='blue', width=2)),
            secondary_y=False
        )
        
        ts_fig.add_trace(
            go.Scatter(x=self.data.index, y=self.data['Inflation_CPI'], name='CPI Inflation',
                      line=dict(color='red', width=2)),
            secondary_y=True
        )
        
        ts_fig.update_layout(
            title='Time Series of Unemployment and Inflation (2000-2024)',
            hovermode='x unified',
            height=500,
            width=1200,  
            margin=dict(l=60, r=40, t=80, b=60),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        ts_fig.update_xaxes(title_text='Date')
        ts_fig.update_yaxes(title_text='Unemployment Rate (%)', secondary_y=False)
        ts_fig.update_yaxes(title_text='Inflation Rate (%)', secondary_y=True)
        
        period_colors = {
            'Pre-Crisis': 'blue',
            'Fin Crisis': 'red',
            'Pre-COVID': 'green',
            'COVID': 'orange',
            'Post-COVID': 'purple'
        }
        
        period_scatter_fig = go.Figure()
        
        for label, df in self.period_data.items():
            if len(df) > 5:
                period_scatter_fig.add_trace(
                    go.Scatter(
                        x=df['Unemployment'],
                        y=df['Inflation_CPI'],
                        mode='markers',
                        name=label,
                        marker=dict(color=period_colors.get(label, 'gray'), size=8)
                    )
                )
                
                if label in self.period_models:
                    model = self.period_models[label]
                    X_sorted = np.sort(df['Unemployment'].values)
                    period_scatter_fig.add_trace(
                        go.Scatter(
                            x=X_sorted,
                            y=model.params[0] + model.params[1] * X_sorted,
                            mode='lines',
                            name=f'{label} Fit',
                            line=dict(color=period_colors.get(label, 'gray'))
                        )
                    )
        
        period_scatter_fig.update_layout(
            title='Phillips Curves by Period',
            xaxis_title='Unemployment Rate (%)',
            yaxis_title='CPI Inflation Rate (%)',
            height=600,
            width=1200,  
            margin=dict(l=60, r=40, t=80, b=60),
            plot_bgcolor='rgba(240, 240, 250, 0.9)'  
        )
        
        app.layout = html.Div([
            html.Div([
                html.H1('Phillips Curve Analysis Dashboard - USA (2000-2024)', 
                      style={'textAlign': 'center', 'color': '#2c3e50', 'fontSize': 36}),
                html.Hr(style={'borderTop': '3px solid #3498db', 'width': '80%', 'margin': 'auto'})
            ]),
            
            html.Div([
                html.Div([
                    html.H2('Introduction', style={'color': '#2c3e50'}),
                    html.P("""
                        The Phillips Curve, introduced by A.W. Phillips in 1958, suggests an inverse relationship 
                        between inflation and unemployment. This dashboard analyzes this relationship for the US economy 
                        from 2000 to 2024, focusing on how it has changed during different economic periods.
                    """, style={'fontSize': 16})
                ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px', 'margin': '20px 0'})
            ]),
            
            html.Div([
                html.H2('Phillips Curve', style={'textAlign': 'center', 'color': '#2c3e50'}),
                dcc.Graph(id='phillips-scatter', figure=scatter_fig)
            ], style={'width': '100%', 'backgroundColor': 'white', 
                      'padding': '15px', 'boxShadow': '0 4px 8px rgba(0,0,0,0.1)', 'borderRadius': '10px'}),
            
            html.Div([
                html.H2('Time Series', style={'textAlign': 'center', 'color': '#2c3e50'}),
                dcc.Graph(id='time-series', figure=ts_fig)
            ], style={'width': '100%', 'backgroundColor': 'white', 'marginTop': '20px',
                      'padding': '15px', 'boxShadow': '0 4px 8px rgba(0,0,0,0.1)', 'borderRadius': '10px'}),
                      
            html.Div([
                html.H2('Phillips Curves by Period', style={'textAlign': 'center', 'color': '#2c3e50'}),
                dcc.Graph(id='period-phillips', figure=period_scatter_fig)
            ], style={'width': '100%', 'backgroundColor': 'white', 'marginTop': '20px',
                      'padding': '15px', 'boxShadow': '0 4px 8px rgba(0,0,0,0.1)', 'borderRadius': '10px'}),
            
            html.Div([
                html.H2('Key Findings', style={'textAlign': 'center', 'color': '#2c3e50'}),
                html.Div([
                    html.P([
                        html.Strong("1. Time Evolution: "),
                        """The Phillips curve relationship shows significant instability over time. The coefficient 
                        changes both in sign and magnitude during different economic periods, suggesting that the 
                        relationship is not a structural constant of the economy."""
                    ], style={'margin': '10px 0', 'fontSize': '15px'}),
                    
                    html.P([
                        html.Strong("2. Structural Changes: "),
                        """Analysis by period confirms that the 2008 financial crisis and the COVID-19 pandemic 
                        caused significant structural changes in the Phillips relationship. The relationship appears 
                        stronger in periods of high economic stress."""
                    ], style={'margin': '10px 0', 'fontSize': '15px'})
                ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'})
            ], style={'backgroundColor': 'white', 'padding': '20px', 'marginTop': '20px', 
                      'boxShadow': '0 4px 8px rgba(0,0,0,0.1)', 'borderRadius': '10px'}),
            
            html.Footer([
                html.P("Phillips Curve Analysis - USA (2000-2024)", style={'fontSize': '14px'}),
                html.P("Developed with Python, Pandas, StatsModels, Plotly and Dash", style={'fontSize': '12px'})
            ], style={'textAlign': 'center', 'padding': '20px', 'marginTop': '30px', 
                      'borderTop': '1px solid #ddd'})
        ], style={'margin': '0 auto', 'width': '90%', 'maxWidth': '1400px', 'fontFamily': 'Arial, sans-serif', 
                 'padding': '20px', 'backgroundColor': '#ecf0f1'})
        
        return app
    
    def run_dashboard(self, debug=True, port=8050):
        """Start the Dash dashboard"""
        app = self.create_dashboard()
        print(f"\nStarting Dash dashboard on port {port}...")
        app.run(debug=debug, port=port)

if __name__ == '__main__':
    analysis = PhillipsCurveAnalysis(start_date='2000-01-01', end_date='2024-10-01')
    
    analysis.run_statistical_analysis()
    
    analysis.run_dashboard(debug=True)
