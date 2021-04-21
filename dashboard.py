# -*- coding: utf-8 -*-

import sqlite3
import logging
import requests

import dash
import dash_core_components as dcc
import dash_html_components as html

import plotly.graph_objs as go

import pandas as pd
from dash.dependencies import Output, Input
from pandas import DataFrame

from pathlib import Path

# TODO: Add possibility to update DB periodically

database = "database.db"

table_exists_sql = \
    "SELECT name " \
    "FROM sqlite_master " \
    "WHERE type = 'table' AND name = ?"

trade_history_sql = \
    "CREATE TABLE trade_history (" \
    "   contract_name TEXT NOT NULL," \
    "   token_symbol TEXT NOT NULL," \
    "   price REAL NOT NULL," \
    "   time INTEGER NOT NULL," \
    "   type TEXT NOT NULL" \
    ")"

token_list_sql = \
    "CREATE TABLE token_list (" \
    "   contract_name TEXT NOT NULL," \
    "   has_market INTEGER NOT NULL," \
    "   token_base64_png TEXT NOT NULL," \
    "   token_base64_svg TEXT NOT NULL," \
    "   logo_type TEXT NOT NULL," \
    "   logo_data TEXT NOT NULL," \
    "   token_logo_url TEXT NOT NULL," \
    "   token_name TEXT NOT NULL," \
    "   token_symbol TEXT NOT NULL" \
    ")"

select_trade_history_sql = \
    "SELECT * " \
    "FROM trade_history " \
    "WHERE token_symbol = ? " \
    "ORDER BY time ASC"

select_token_sql = \
    "SELECT * " \
    "FROM token_list " \
    "WHERE token_symbol = ?"

select_last_trade_sql = \
    "SELECT * " \
    "FROM trade_history " \
    "ORDER BY time DESC LIMIT 1"

insert_trade_history_sql = \
    "INSERT INTO trade_history (contract_name, token_symbol, price, time, type) " \
    "VALUES (?, ?, ?, ?, ?)"

insert_token_sql = \
    "INSERT INTO token_list (contract_name, has_market, token_base64_png, token_base64_svg, " \
    "            logo_type, logo_data, token_logo_url, token_name, token_symbol)" \
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"


def db_exists(table_name):
    if not Path(database).is_file():
        return False

    con = sqlite3.connect(database)
    cur = con.cursor()
    exists = False

    try:
        if cur.execute(table_exists_sql, [table_name]).fetchone():
            exists = True
    except Exception as e:
        logging.error(e)

    con.close()
    return exists


def db_exec(sql_statement, *args):
    res = {"success": None, "data": None}

    con = None
    cur = None

    try:
        con = sqlite3.connect(database)
        cur = con.cursor()
        cur.execute(sql_statement, args)
        con.commit()

        res["data"] = cur.fetchall()
        res["success"] = True
    except Exception as e:
        res["data"] = str(e)
        res["success"] = False
        logging.error(e)
    finally:
        if cur:
            cur.close()
        if con:
            con.close()

    return res


# TODO: Include reading out available tokens
def update_db():
    history_res = db_exec(select_last_trade_sql)

    if history_res and history_res["data"]:
        last_time = history_res["data"][0][3]
    else:
        last_time = 0

    skip = 0
    take = 50
    call = True

    while call:
        try:
            token_res = requests.get(
                "https://stats.rocketswap.exchange:2053/api/token_list")
        except Exception as e:
            logging.error(e)

        for t in token_res.json():
            # TODO
            pass

        try:
            history_res = requests.get(
                "https://stats.rocketswap.exchange:2053/api/get_trade_history",
                params={"take": take, "skip": skip})
        except Exception as e:
            logging.error(e)

        skip += take

        for tx in history_res.json():
            if tx["time"] > last_time:
                db_exec(
                    insert_trade_history_sql,
                    tx["contract_name"],
                    tx["token_symbol"],
                    tx["price"],
                    tx["time"],
                    tx["type"])
            else:
                call = False
                break

        if len(history_res.json()) != take:
            call = False


# TODO: Sollte hier der Zeitraum als Argument vorhanden sein?
def get_trade_history(token_symbol):
    res = db_exec(select_trade_history_sql, token_symbol)

    data = list()
    for trade in res["data"]:
        data.append([trade[3], trade[2]])

    return data


def get_fig(data):
    df_price = DataFrame(reversed(data), columns=["DateTime", "Price"])
    df_price["DateTime"] = pd.to_datetime(df_price["DateTime"], unit="s")
    price = go.Scatter(x=df_price.get("DateTime"), y=df_price.get("Price"))

    layout = go.Layout(
        title=dict(
            text=f"RSWP-TAU",  # TODO: Dynamisch setzen
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
        )
    )

    fig = go.Figure(data=[price], layout=layout)

    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1h", step="hour", stepmode="todate"),
                dict(count=1, label="1d", step="day", stepmode="todate"),
                dict(count=5, label="5d", step="day", stepmode="todate"),
                dict(count=1, label="1m", step="month", stepmode="todate"),
                dict(count=1, label="YTD", step="month", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="todate"),
                dict(step="all")
            ])
        )
    )

    return fig


if __name__ == '__main__':
    if not db_exists("trade_history"):
        db_exec(trade_history_sql)
    if not db_exists("token_list"):
        db_exec(token_list_sql)

    update_db()

    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    app.layout = html.Div(children=[
        html.H1(children='Hello Dash'),

        html.Div(children='''
            Dash: A web application framework for Python.
        '''),

        dcc.Dropdown(
            options=[
                {'label': 'RSWP', 'value': 'RSWP'},  # TODO: Dynamisch setzen
                {'label': 'DOUG', 'value': 'DOUG'}
            ],
            value='RSWP',
            multi=False,
            id="token_symbol_input"
        ),

        dcc.Graph(
            id='price-graph',
            figure=get_fig(get_trade_history("RSWP"))  # TODO: Dynamisch setzen
        )
    ])

    @app.callback(Output('price-graph', 'figure'), Input('token_symbol_input', 'value'))
    def update_figure(token_symbol):
        data = get_trade_history(token_symbol)

        fig = get_fig(data)
        fig.update_layout(transition_duration=500)

        return fig

    app.run_server(debug=True)
