# app/ai_infra_dashboard/app.py
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
from pathlib import Path
import flask
import io
import base64

ROOT = Path.cwd()
DATA = ROOT / "data" / "processed" / "features_lean.csv"
REPORT = ROOT / "reports" / "onepager.png"

if not DATA.exists():
    raise SystemExit("Missing features_lean.csv - run pipeline first")

df = pd.read_csv(DATA, low_memory=False)
latest = df.sort_values(['iso3','year']).groupby('iso3').tail(1)

app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

metric_options = [
    {'label':'Governance composite (gov_index_zmean)','value':'gov_index_zmean'},
    {'label':'GDP growth (%)','value':'gdp_growth_pct'},
    {'label':'GDP (ln)','value':'gdp_usd_ln_safe'},
    {'label':'Trade exposure','value':'trade_exposure'},
    {'label':'Temp anomaly','value':'temp_anom_roll'}
]

def pilto_bytes(path):
    b = open(path,'rb').read()
    return "data:image/png;base64," + base64.b64encode(b).decode()

app.layout = html.Div([
    html.H2("AI-Infra Economic Dashboard — TUM Demo"),
    html.Div([
        dcc.Dropdown(id='color-metric', options=metric_options, value='gov_index_zmean', style={'width':'40%'})
    ]),
    dcc.Graph(id='world-map'),
    html.Div(id='country-panel', children=[
        html.Div(id='selected-country', style={'fontWeight':'bold','marginTop':8}),
        html.Div([
            dcc.Graph(id='ts-plot', style={'width':'66%','display':'inline-block'}),
            dcc.Graph(id='pillars', style={'width':'33%','display':'inline-block'})
        ]),
        html.Button("Download one-page report (PNG)", id='download-report-button'),
        html.A("Open full manifest", href="/data_manifest.json", target="_blank")
    ]),
    html.Div(id='report-image', children=html.Img(src=pilto_bytes(REPORT), style={'width':'600px','marginTop':20}))
])

@app.callback(
    Output('world-map','figure'),
    Input('color-metric','value')
)
def update_map(metric):
    col = metric
    if col not in latest.columns:
        fig = px.choropleth(latest, locations='iso3', color=latest.columns[3], hover_name='country', projection='natural earth')
    else:
        fig = px.choropleth(latest, locations='iso3', color=col, hover_name='country', projection='natural earth')
    fig.update_layout(margin=dict(l=0,r=0,t=30,b=0))
    return fig

@app.callback(
    Output('ts-plot','figure'),
    Output('pillars','figure'),
    Output('selected-country','children'),
    Input('world-map','clickData'),
    Input('color-metric','value')
)
def country_drill(clickData, metric):
    if clickData and 'location' in clickData['points'][0]:
        iso = clickData['points'][0]['location']
    else:
        iso = latest['iso3'].iloc[0]
    cdf = df[df['iso3']==iso].sort_values('year')
    title = f"{cdf['country'].iloc[0]} ({iso})"
    ts_vars = [v for v in ['gdp_usd_ln_safe', metric] if v in cdf.columns]
    if ts_vars:
        ts_fig = px.line(cdf, x='year', y=ts_vars, markers=True, title=f"{title} — time series")
    else:
        ts_fig = px.line(title="No time series available")
    pillar_cols = [c for c in ['voice_accountability_imputed','political_stability_imputed','gov_effectiveness_imputed','reg_quality_imputed','rule_of_law_imputed','control_corruption_imputed'] if c in cdf.columns]
    if pillar_cols:
        pill = cdf.melt(id_vars=['year'], value_vars=pillar_cols, var_name='pillar', value_name='value')
        pfig = px.line(pill, x='year', y='value', color='pillar', title='Governance pillars')
    else:
        pfig = px.line(title='No pillar data available')
    return ts_fig, pfig, f"Selected: {title}"

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
