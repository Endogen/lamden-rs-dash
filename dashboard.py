import sqlite3
import logging

import requests
import configparser

import dash
import dash_core_components as dcc
import dash_html_components as html

import plotly.graph_objs as go
import pandas as pd

from dash.dependencies import Output, Input

from pandas import DataFrame
from pathlib import Path
from distutils.util import strtobool
from datetime import datetime, timedelta

app = dash.Dash(
    __name__,
    external_stylesheets=['style.css'],
    title="Rocketswap Dashboard",
    meta_tags=[{
        "name": "viewport",
        "content": "width=device-width, initial-scale=1.0, maximum-scale=1.2, minimum-scale=0.5"
    }]
)


class Config:
    def __init__(self, cfg_path: str = None):
        cfg_path = cfg_path if cfg_path else app.get_asset_url("config.cfg")[1:]
        self._cfg_parser = configparser.RawConfigParser()
        self._cfg_parser.read(cfg_path)

    def get(self, param_name):
        value = self._cfg_parser.get("Dashboard-Config", param_name)

        try:
            value = float(value)
            return int(value) if value.is_integer() else value
        except:
            if value.lower() in ("true", "false"):
                return bool(strtobool(value))
            return value


cfg = Config()


class Database:
    def __init__(self, db_path: str = None):
        self._db_path = db_path if db_path else app.get_asset_url("database.db")[1:]

    def execute(self, sql_statement, *args):
        res = {"success": None, "data": None}

        con = None
        cur = None

        try:
            con = sqlite3.connect(self._db_path)
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

    def table_exists(self, table_name):
        if not Path(self._db_path).is_file():
            return False

        sql = \
            "SELECT name " \
            "FROM sqlite_master " \
            "WHERE type = 'table' AND name = ?"

        if self.execute(sql, table_name)["data"]:
            return True
        return False

    def token_exists(self, contract_name):
        sql = \
            "SELECT EXISTS (" \
            "   SELECT 1 " \
            "   FROM token_list " \
            "   WHERE contract_name = ?" \
            ")"

        result = self.execute(sql, contract_name)
        return True if result["data"][0][0] == 1 else False

    def create_trade_history(self):
        if self.table_exists("trade_history"):
            return

        sql = \
            "CREATE TABLE trade_history (" \
            "   contract_name TEXT NOT NULL," \
            "   token_symbol TEXT NOT NULL," \
            "   price REAL NOT NULL," \
            "   time INTEGER NOT NULL," \
            "   type TEXT NOT NULL" \
            ")"

        return self.execute(sql)

    def create_token_list(self):
        if self.table_exists("token_list"):
            return

        sql = \
            "CREATE TABLE token_list (" \
            "   contract_name TEXT NOT NULL PRIMARY KEY," \
            "   has_market TEXT," \
            "   token_base64_png TEXT," \
            "   token_base64_svg TEXT," \
            "   logo_type TEXT," \
            "   logo_data TEXT," \
            "   token_logo_url TEXT," \
            "   token_name TEXT NOT NULL," \
            "   token_symbol TEXT NOT NULL" \
            ")"

        return self.execute(sql)

    def select_last_trade(self):
        sql = \
            "SELECT * " \
            "FROM trade_history " \
            "ORDER BY time DESC LIMIT 1"

        return self.execute(sql)

    def select_trades(self, token_symbol, start_secs=0):
        sql = \
            "SELECT * " \
            "FROM trade_history " \
            f"WHERE token_symbol = ? AND time >= {start_secs} " \
            "ORDER BY time ASC"

        return self.execute(sql, token_symbol)

    def select_trade_count(self, start_secs: int = 0):
        sql = \
            "SELECT token_symbol, count(token_symbol) " \
            "FROM trade_history " \
            f"WHERE time > {start_secs} " \
            "GROUP BY token_symbol " \
            "ORDER BY count(token_symbol) DESC"

        return self.execute(sql)

    def select_contract(self, token_symbol):
        sql = \
            "SELECT * " \
            "FROM token_list " \
            "WHERE token_symbol = ?"

        return self.execute(sql, token_symbol)

    def insert_trade(self, contract_name, token_symbol, price, time, type):
        sql = \
            "INSERT INTO trade_history (contract_name, token_symbol, price, time, type) " \
            "VALUES (?, ?, ?, ?, ?)"

        return self.execute(sql, contract_name, token_symbol, price, time, type)

    def insert_token(self, contract_name, has_market, base64_png, base64_svg, logo_type,
                     logo_data, token_logo_url, token_name, token_symbol):

        if self.token_exists(contract_name):
            return

        sql = \
            "INSERT INTO token_list (contract_name, has_market, token_base64_png, token_base64_svg, " \
            "            logo_type, logo_data, token_logo_url, token_name, token_symbol)" \
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"

        return self.execute(
            sql,
            contract_name,
            has_market,
            base64_png,
            base64_svg,
            logo_type,
            logo_data,
            token_logo_url,
            token_name,
            token_symbol)

    def select_symbols(self):
        sql = \
            "SELECT DISTINCT token_symbol " \
            "FROM trade_history"

        return self.execute(sql)


db = Database()
db.create_token_list()
db.create_trade_history()


def to_unix_time(date_time):
    return int((date_time - datetime(1970, 1, 1)).total_seconds())


d1 = to_unix_time(datetime.utcnow() - timedelta(days=1))
d5 = to_unix_time(datetime.utcnow() - timedelta(days=5))
m1 = to_unix_time(datetime.utcnow() - timedelta(days=30))


class Dashboard:
    def __init__(self):
        self.selected_token = cfg.get("default_token")

    def update_trades(self):
        res = db.select_last_trade()

        if res and res["data"]:
            last_secs = res["data"][0][3]
        else:
            last_secs = 0

        trades = list()

        skip = 0
        take = 10
        call = True

        while call:
            try:
                url = f"{cfg.get('rocketswap_url')}get_trade_history"
                res = requests.get(url, params={"take": take, "skip": skip}, timeout=2)
            except Exception as e:
                logging.error(e)
                return

            skip += take

            for tx in res.json():
                if tx["time"] > last_secs:
                    if tx not in trades:
                        trade = [
                            tx["contract_name"],
                            tx["token_symbol"],
                            tx["price"],
                            tx["time"],
                            tx["type"]
                        ]

                        trades.append(trade)
                        db.insert_trade(*trade)
                        print("New trade:", tx)
                        logging.info(f"New trade: {tx}")
                else:
                    call = False
                    break

            if len(res.json()) != take:
                call = False

        return trades

    def update_tokens(self):
        try:
            url = f"{cfg.get('rocketswap_url')}token_list"
            res = requests.get(url, timeout=2)
        except Exception as e:
            logging.error(e)
            return

        for tk in res.json():
            db.insert_token(
                tk["contract_name"],
                tk["has_market"],
                tk["token_base64_png"],
                tk["token_base64_svg"],
                tk["logo"]["type"],
                tk["logo"]["data"],
                tk["token_logo_url"],
                tk["token_name"],
                tk["token_symbol"])

    def get_trades(self, start_secs: int = 0):
        data = list()

        for trade in db.select_trades(self.selected_token, start_secs)["data"]:
            data.append([trade[3], trade[2]])

        return data

    def get_trade_count(self, start_secs: int = 0):
        res = db.select_trade_count(start_secs)

        data = list()
        for trade in res["data"]:
            data.append([trade[0], trade[1]])

        return data

    def get_symbols(self):
        res = db.select_symbols()

        data = list()
        for ts in res["data"]:
            data.append({"label": ts[0], "value": ts[0]})

        return data

    def get_contract(self):
        res = db.select_contract(self.selected_token)

        if res["data"] and res["data"][0]:
            return res["data"][0][0]
        else:
            return "-"

    def get_price_graph(self, data):
        df = DataFrame(reversed(data), columns=["DateTime", "Price"])
        df["DateTime"] = pd.to_datetime(df["DateTime"], unit="s")
        df.sort_index(ascending=True, inplace=True)

        graph = go.Scatter(x=df.get("DateTime"), y=df.get("Price"))

        layout = go.Layout(
            height=350,
            plot_bgcolor='rgb(255,255,255)',
            xaxis=dict(
                gridcolor="rgb(215, 215, 215)"
            ),
            yaxis=dict(
                gridcolor="rgb(215, 215, 215)",
                zerolinecolor="rgb(233, 233, 233)",
                ticksuffix=" ",
                autorange=True,
                fixedrange=False
            ),
            margin=dict(
                r=25,
                t=5,
                b=40)
        )

        return go.Figure(data=[graph], layout=layout)

    def get_trades_graph(self, data):
        df = DataFrame(reversed(data), columns=["Tokens", "Trades"])
        df.sort_index(ascending=True, inplace=True)

        colors = ["rgb(84, 95, 249)", ] * len(data)

        for i, d in enumerate(reversed(data)):
            if d[0] == self.selected_token:
                colors[i] = "crimson"

        graph = go.Bar(x=df.get("Tokens"), y=df.get("Trades"), marker_color=colors)

        layout = go.Layout(
            plot_bgcolor='rgb(255,255,255)',
            xaxis=dict(
                gridcolor="rgb(215, 215, 215)"
            ),
            yaxis=dict(
                gridcolor="rgb(215, 215, 215)",
                zerolinecolor="rgb(233, 233, 233)",
                ticksuffix=" "
            ),
            margin=dict(
                r=25,
                t=5,
                b=40)
        )

        return go.Figure(data=[graph], layout=layout)


ds = Dashboard()


app.layout = html.Div(children=[
    dcc.Store(
        id='storage',
        storage_type='local'
    ),

    dcc.Interval(
        id='interval-component',
        interval=cfg.get("update_interval"),
        n_intervals=0
    ),

    html.Div([
        html.Div([
            dcc.Dropdown(
                options=ds.get_symbols(),
                value=ds.selected_token,
                multi=False,
                id="token-input"
            )
        ], id="div-token-input", className="four columns"),

        html.Div([
            html.Label(
                children=ds.get_contract(),
                id="contract-label",
                style={"text-align": "center"}
            )
        ], id="div-contract-label", className="four columns"),

        html.Div([
            dcc.Checklist(
                id="select-visible",
                options=[
                    {'label': '1D', 'value': '1D'},
                    {'label': '5D', 'value': '5D'},
                    {'label': '30D', 'value': '30D'},
                    {'label': 'Trades', 'value': 'TRADES'}
                ],
                value=['1D', '5D', '30D', 'TRADES'],
                labelStyle={'display': 'inline-block'},
                style={"text-align": "right"}
            )
        ], id="div-select-visible", className="four columns"),

    ], id="div-top", className="row"),

    html.Br(),

    html.Div([
        html.Div([
            html.H5(
                id="price-title-1d",
                style={"text-align": "center"},
                children=f"{ds.selected_token}-TAU 1 Day"
            ),
            dcc.Graph(
                id='price-graph-1d',
                figure=ds.get_price_graph(ds.get_trades(d1))
            )
        ], id="div-1d", className="four columns"),

        html.Div([
            html.H5(
                id="price-title-5d",
                style={"text-align": "center"},
                children=f"{ds.selected_token}-TAU 5 Days"
            ),
            dcc.Graph(
                id='price-graph-5d',
                figure=ds.get_price_graph(ds.get_trades(d5))
            )
        ], id="div-5d", className="four columns"),

        html.Div([
            html.H5(
                id="price-title-30d",
                style={"text-align": "center"},
                children=f"{ds.selected_token}-TAU 30 Days"
            ),
            dcc.Graph(
                id='price-graph-30d',
                figure=ds.get_price_graph(ds.get_trades(m1))
            )
        ], id="div-30d", className="four columns"),
    ], id="div-1d5d30d", className="row"),

    html.Br(),

    html.Div([
        html.H5(
            children="Number of Trades per Token in last 5 Days",
            id="trades-per-token-title",
            style={"text-align": "center"}),

        dcc.Graph(
            id='trades-graph',
            figure=ds.get_trades_graph(ds.get_trade_count(d5))
        )
    ], id="div-trades")
])


@app.callback(
    Output('price-title-1d', 'children'),
    Output('price-graph-1d', 'figure'),
    Output('price-title-5d', 'children'),
    Output('price-graph-5d', 'figure'),
    Output('price-title-30d', 'children'),
    Output('price-graph-30d', 'figure'),
    Output('trades-graph', 'figure'),
    Output('token-input', 'options'),
    Output('contract-label', 'children'),
    Input('interval-component', 'n_intervals'),
    Input('token-input', 'value'))
def update_price(n, token_symbol):
    caller_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0]

    if caller_id == "token-input" and token_symbol:
        ds.selected_token = token_symbol

    elif caller_id == "interval-component" and n:
        ds.update_trades()

        if n % 60 == 0:
            ds.update_tokens()

    return [
        f"{ds.selected_token}-TAU 1 Day",
        ds.get_price_graph(ds.get_trades(d1)).update_layout(transition_duration=500),
        f"{ds.selected_token}-TAU 5 Days",
        ds.get_price_graph(ds.get_trades(d5)).update_layout(transition_duration=500),
        f"{ds.selected_token}-TAU 30 Days",
        ds.get_price_graph(ds.get_trades(m1)).update_layout(transition_duration=500),
        ds.get_trades_graph(ds.get_trade_count(d5)).update_layout(transition_duration=500),
        ds.get_symbols(),
        ds.get_contract()
    ]


@app.callback(
    Output('div-1d', 'style'),
    Output('div-1d', 'className'),
    Output('div-5d', 'className'),
    Output('div-30d', 'className'),
    Input('select-visible', 'value'))
def update_visibility_1d(value):
    if "1D" in value:
        return [
            {'display': 'inline'},
            "four columns",
            "four columns",
            "four columns"
        ]
    else:
        return [
            {'display': 'none'},
            "six columns",
            "six columns",
            "six columns"
        ]


@app.callback(
    Output('div-5d', 'style'),
    Input('select-visible', 'value'))
def update_visibility_5d(value):
    if "5D" in value:
        return {'display': 'inline'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('div-30d', 'style'),
    Input('select-visible', 'value'))
def update_visibility_30d(value):
    if "30D" in value:
        return {'display': 'inline'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('div-trades', 'style'),
    Input('select-visible', 'value'))
def update_visibility_trades(value):
    if "TRADES" in value:
        return {'display': 'inline'}
    else:
        return {'display': 'none'}


if __name__ == '__main__':
    app.run_server(debug=cfg.get("debug"), threaded=True, port=cfg.get("dashboard_port"), host="0.0.0.0")
