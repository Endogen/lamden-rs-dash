# -*- coding: utf-8 -*-

import time
import logging
import requests

import dash
import dash_core_components as dcc
import dash_html_components as html

import plotly.express as px
import plotly.graph_objs as go

import pandas as pd

from pandas import DataFrame

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

timeframe = 1
end_secs = int(time.time() - (timeframe * 24 * 60 * 60))

token_symbol = "RSWP"
token_contract = None

skip = 0
take = 50
call = True
data = list()

while call:
    try:
        res = requests.get(
            "https://stats.rocketswap.exchange:2053/api/get_trade_history",
            params={"take": take, "skip": skip})
    except Exception as e:
        logging.error(f"Can't retrieve trade history: {e}")
        # TODO

    skip += take

    for tx in res.json():
        if tx["token_symbol"] == token_symbol:
            if not token_contract:
                token_contract = tx["contract_name"]
            if int(tx["time"]) > end_secs:
                data.append([tx["time"], float(tx["price"])])
            else:
                call = False
                break

    if len(res.json()) != take:
        call = False

if not data:
    logging.error(f"No data for {token_symbol}")
    # TODO

df_price = DataFrame(reversed(data), columns=["DateTime", "Price"])
df_price["DateTime"] = pd.to_datetime(df_price["DateTime"], unit="s")
price = go.Scatter(x=df_price.get("DateTime"), y=df_price.get("Price"))

layout = go.Layout(
    title=dict(
        text=f"{token_symbol}-TAU",
        x=0.5,
        font=dict(
            size=24
        )
    ),
    paper_bgcolor='rgb(233,233,233)',
    plot_bgcolor='rgb(233,233,233)',
    xaxis=dict(
        gridcolor="rgb(215, 215, 215)"
    ),
    yaxis=dict(
        gridcolor="rgb(215, 215, 215)",
        zerolinecolor="rgb(233, 233, 233)",
        tickprefix="",
        ticksuffix=" "
    ),
    shapes=[{
        "type": "line",
        "xref": "paper",
        "yref": "y",
        "x0": 0,
        "x1": 1,
        "y0": data[0][1],
        "y1": data[0][1],
        "line": {
            "color": "rgb(50, 171, 96)",
            "width": 1,
            "dash": "dot"
        }
    }]
)

fig = go.Figure(data=[price], layout=layout)

app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),

    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
