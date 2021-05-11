import os
import sqlite3
import logging
import requests
import configparser

import dash
import dash_core_components as dcc
import dash_html_components as html

import plotly.graph_objs as go
import pandas as pd

from dash.dependencies import Output, Input, State
from plotly.subplots import make_subplots

from pandas import DataFrame
from pathlib import Path
from distutils.util import strtobool
from datetime import datetime, timedelta
from threading import Thread, Event


# TODO: Check what to do with DOUG / DOUGLAS - Remove DOUG
# TODO: When interval to check is very long then we get less data points in the graph
# TODO: Titles of graphs are not always changing correctly
# TODO: Use websockets to get messages over to users - see example

def create_kill_script(filename):
    shebang = "#!/bin/bash"
    command = f"kill -9 {os.getpid()}"

    with open(filename, "w") as f:
        f.truncate(0)
        f.write(f"{shebang}\n\n{command}")


create_kill_script("stop")

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
            "   amount REAL NOT NULL," \
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

    def select_count_trades(self):
        sql = \
            "SELECT COUNT (*) " \
            "FROM trade_history"

        return self.execute(sql)

    def select_count_tokens(self):
        sql = \
            "SELECT COUNT (*) " \
            "FROM token_list"

        return self.execute(sql)

    # TODO: Change SELECT * to only what we need
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

    def insert_trade(self, contract_name, token_symbol, price, time, amount, type):
        sql = \
            "INSERT INTO trade_history (contract_name, token_symbol, price, time, amount, type) " \
            "VALUES (?, ?, ?, ?, ?, ?)"

        return self.execute(sql, contract_name, token_symbol, price, time, amount, type)

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


class Dashboard:
    def update_trades(self):
        res = db.select_last_trade()

        if res and res["data"]:
            last_secs = res["data"][0][3]
        else:
            last_secs = 0

        trades = list()

        skip = 0
        take = 50
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
                            tx["amount"],
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

    def get_trades(self, token, start_secs: int = 0):
        data = list()

        for trade in db.select_trades(token, start_secs)["data"]:
            data.append([trade[3], trade[2], trade[4]])

        return data

    def get_trade_count(self, start_secs: int = 0):
        res = db.select_trade_count(start_secs)

        data = list()
        for trade in res["data"]:
            data.append([trade[0], trade[1]])

        return data

    # TODO: Sort alphabetically
    def get_symbols(self):
        res = db.select_symbols()

        data = list()
        for ts in res["data"]:
            data.append({"label": ts[0], "value": ts[0]})

        return data

    def get_contract(self, token):
        res = db.select_contract(token)

        if res["data"] and res["data"][0]:
            return res["data"][0][0]
        else:
            return "-"

    # TODO: Range Slider? https://community.plotly.com/t/get-min-and-max-values-of-x-axis-which-is-time-series/6898
    # TODO: Add ATH and ATL
    # TODO: Add slider to change graph height
    # TODO: Remove while line one y-axis of top graph
    def get_price_graph(self, data, token):
        df = DataFrame(reversed(data), columns=["DateTime", "Price", "Volume"])
        df["DateTime"] = pd.to_datetime(df["DateTime"], unit="s")
        df.sort_index(ascending=True, inplace=True)

        fig = make_subplots(rows=2, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.05,
                            row_heights=[0.8, 0.2])

        fig.add_trace(go.Scatter(x=df.get("DateTime"), y=df.get("Price"), name="Price", yaxis="y1"), row=1, col=1)
        fig.add_trace(go.Bar(x=df.get("DateTime"), y=df.get("Volume"), name="Volume", yaxis="y2"), row=2, col=1)

        """
        max_index = df["Price"].idxmax()
        min_index = df["Price"].idxmin()

        max_price = df["Price"].max()
        min_price = df["Price"].min()

        max_datetime = df["DateTime"][max_index]
        min_datetime = df["DateTime"][min_index]

        fig.add_annotation(x=max_datetime, y=max_price, text="ATH", showarrow=False)
        fig.add_annotation(x=min_datetime, y=min_price, text="ATL", showarrow=False)
        """

        fig.update_layout(
            height=600,
            showlegend=False,
            paper_bgcolor='rgb(34,34,34)',
            plot_bgcolor='rgb(34,34,34)',
            font=dict(
                family='Open Sans',
                color='rgb(255,255,255)'
            ),
            title=dict(
                text=token,
                font=dict(
                    size=20
                )
            ),
            xaxis=dict(
                gridcolor="rgb(90,90,90)"
            ),
            yaxis=dict(
                gridcolor="rgb(90,90,90)",
                zerolinecolor="rgb(90,90,90)",
                domain=[0.25, 1],
                ticksuffix="  "
            ),
            xaxis2=dict(
                gridcolor="rgb(90,90,90)"
            ),
            yaxis2=dict(
                gridcolor="rgb(90,90,90)",
                zerolinecolor="rgb(90,90,90)",
                ticksuffix=" "
            ),
            bargap=0
        )

        fig.update_traces(marker=dict(line=dict(width=0)), row=2, col=1)

        # TODO: Change margin?
        """
            margin=dict(
                r=25,
                t=5,
                b=40)
        )
        """

        return fig

    def get_price_graph_1d(self, token):
        d1 = to_unix_time(datetime.utcnow() - timedelta(days=1))

        graph = self.get_price_graph(ds.get_trades(token, d1), token)
        graph.update_layout(title_text=f"{token}-TAU 1 Day")
        graph.update_traces(mode='lines+markers', row=1, col=1)

        return graph

    def get_price_graph_5d(self, token):
        d5 = to_unix_time(datetime.utcnow() - timedelta(days=5))

        graph = self.get_price_graph(ds.get_trades(token, d5), token)
        graph.update_layout(title_text=f"{token}-TAU 5 Days")

        return graph

    def get_price_graph_1m(self, token):
        m1 = to_unix_time(datetime.utcnow() - timedelta(days=30))

        graph = self.get_price_graph(ds.get_trades(token, m1), token)
        graph.update_layout(title_text=f"{token}-TAU 30 Days")

        return graph

    # TODO: Add Range Selector Buttons https://plotly.com/python/time-series/
    def get_trades_graph(self, data, token):
        df = DataFrame(reversed(data), columns=["Tokens", "Trades"])
        df.sort_index(ascending=True, inplace=True)

        colors = ["rgb(84, 95, 249)", ] * len(data)

        for i, d in enumerate(reversed(data)):
            if d[0] == token:
                colors[i] = "crimson"

        graph = go.Bar(x=df.get("Tokens"), y=df.get("Trades"), marker_color=colors)

        layout = go.Layout(
            height=600,
            showlegend=False,
            paper_bgcolor='rgb(34,34,34)',
            plot_bgcolor='rgb(34,34,34)',
            font=dict(
                family='Open Sans',
                color='rgb(255,255,255)'
            ),
            yaxis=dict(
                gridcolor="rgb(90,90,90)",
                zerolinecolor="rgb(90,90,90)",
                ticksuffix="  "
            ),
            title=dict(
                text=f"Trades per Token in last 5 Days",
                font=dict(
                    size=20
                )
            )
        )

        return go.Figure(data=[graph], layout=layout)


ds = Dashboard()


# TODO: Runs twice?
class UpdateTrades(Thread):
    def __init__(self, event):
        Thread.__init__(self)
        self.stopped = event

    def run(self):
        seconds = cfg.get("update_interval")
        while not self.stopped.wait(seconds):
            try:
                ds.update_trades()
                print("Updated trades")
            except Exception as e:
                logging.error(f"Error while updating trades: {e}")


class UpdateTokens(Thread):
    def __init__(self, event):
        Thread.__init__(self)
        self.stopped = event

    def run(self):
        seconds = cfg.get("update_interval") * 10
        while not self.stopped.wait(seconds):
            try:
                ds.update_tokens()
                print("Updated tokens")
            except Exception as e:
                logging.error(f"Error while updating tokens: {e}")


app.layout = html.Div(
    children=[
        dcc.Interval(
            id='interval-component',
            interval=cfg.get("update_interval") * 1000,
            n_intervals=0
        ),

        dcc.Store(id='session', storage_type='session'),

        html.Div(
            className="row",
            children=[
                # Column for user controls
                html.Div(
                    className="four columns div-user-controls",
                    children=[
                        html.H2("🚀 ROCKETSWAP DASHBOARD"),
                        html.P(
                            """Choose a token to show live data. The graphs will update 
                            within 5 seconds after a trade happens."""
                        ),
                        html.Div(
                            className="row",
                            children=[
                                html.Div(
                                    className="div-for-dropdown",
                                    children=[
                                        dcc.Dropdown(
                                            multi=False,
                                            id="token-input"
                                        )
                                    ],
                                ),
                                html.Br(),
                                html.P(
                                    """Select graphs to display"""
                                ),
                                html.Div(
                                    className="div-for-dropdown",
                                    children=[
                                        dcc.Checklist(
                                            id="select-visible",
                                            options=[
                                                {'label': '1 Day', 'value': '1D'},
                                                {'label': '5 Days', 'value': '5D'},
                                                {'label': '30 Days', 'value': '1M'},
                                                {'label': 'Trades', 'value': 'TR'}
                                            ],
                                            value=['1D', '5D', '1M', 'TR'],
                                            persistence_type="local",
                                            persistence="true"
                                        )
                                    ],
                                ),
                                html.Div(
                                    className="div-for-dropdown",
                                    children=[
                                        html.Label(id="contract-label")
                                    ],
                                ),
                            ],
                        ),
                        dcc.Markdown(
                            children=[
                                "View Source on [GitHub](https://github.com/Endogen/lamden-rs-dash)"
                            ]
                        ),
                    ],
                ),
                # Column for app graphs and plots
                html.Div(
                    className="eight columns div-for-charts bg-grey",
                    children=[
                        dcc.Graph(id='price-graph-1d'),
                        dcc.Graph(id='price-graph-5d'),
                        dcc.Graph(id='price-graph-1m'),
                        dcc.Graph(id='trades-graph')
                    ],
                ),
            ],
        )
    ]
)


# TODO: Add html.Table
# TODO: Find similarities and remove duplicates stuff
# TODO: Fetch trade data only once from DB and send it to all graphs
@app.callback(
    Output('token-input', 'options'),
    Output('token-input', 'value'),
    Output('contract-label', 'children'),
    Output('price-graph-1d', 'figure'),
    Output('price-graph-5d', 'figure'),
    Output('price-graph-1m', 'figure'),
    Output('trades-graph', 'figure'),
    Output('session', 'data'),
    Input('interval-component', 'n_intervals'),
    Input('token-input', 'value'),
    State('session', 'data'))
def update_price(counter, token, data):
    token_input_options = dash.no_update
    token_input_value = dash.no_update
    token_contract = dash.no_update

    graph_1d = dash.no_update
    graph_5d = dash.no_update
    graph_1m = dash.no_update
    graph_tr = dash.no_update

    data = data or {
        "token": cfg.get("default_token"),
        "token_count": db.select_count_tokens(),
        "trade_count": db.select_count_trades()
    }

    caller_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0]

    # Token was manually changed in Dropdown
    if caller_id == "token-input" and token:
        data["token"] = token
        token_contract = ds.get_contract(token)

        d5 = to_unix_time(datetime.utcnow() - timedelta(days=5))
        graph_1d = ds.get_price_graph_1d(token).update_layout(transition_duration=500)
        graph_5d = ds.get_price_graph_5d(token).update_layout(transition_duration=500)
        graph_1m = ds.get_price_graph_1m(token).update_layout(transition_duration=500)
        graph_tr = ds.get_trades_graph(ds.get_trade_count(d5), token).update_layout(transition_duration=500)

    # Every X seconds - check for updates
    elif caller_id == "interval-component" and counter:
        if db.select_count_tokens() != data["token_count"]:
            token_input_options = ds.get_symbols()

        if db.select_count_trades() != data["trade_count"]:
            d5 = to_unix_time(datetime.utcnow() - timedelta(days=5))
            graph_1d = ds.get_price_graph_1d(data["token"]).update_layout(transition_duration=500)
            graph_5d = ds.get_price_graph_5d(data["token"]).update_layout(transition_duration=500)
            graph_1m = ds.get_price_graph_1m(data["token"]).update_layout(transition_duration=500)
            graph_tr = ds.get_trades_graph(ds.get_trade_count(d5), data["token"]).update_layout(transition_duration=500)

    # Initial opening of website
    else:
        token_input_options = ds.get_symbols()
        token_input_value = data["token"]
        token_contract = ds.get_contract(data["token"])

        d5 = to_unix_time(datetime.utcnow() - timedelta(days=5))
        graph_1d = ds.get_price_graph_1d(data["token"]).update_layout(transition_duration=500)
        graph_5d = ds.get_price_graph_5d(data["token"]).update_layout(transition_duration=500)
        graph_1m = ds.get_price_graph_1m(data["token"]).update_layout(transition_duration=500)
        graph_tr = ds.get_trades_graph(ds.get_trade_count(d5), data["token"]).update_layout(transition_duration=500)

    return [
        token_input_options,
        token_input_value,
        token_contract,
        graph_1d,
        graph_5d,
        graph_1m,
        graph_tr,
        data
    ]


@app.callback(
    Output('price-graph-1d', 'style'),
    Input('select-visible', 'value'))
def update_visibility_1d(value):
    if "1D" in value:
        return {'display': 'inline'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('price-graph-5d', 'style'),
    Input('select-visible', 'value'))
def update_visibility_5d(value):
    if "5D" in value:
        return {'display': 'inline'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('price-graph-1m', 'style'),
    Input('select-visible', 'value'))
def update_visibility_30d(value):
    if "1M" in value:
        return {'display': 'inline'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('trades-graph', 'style'),
    Input('select-visible', 'value'))
def update_visibility_trades(value):
    if "TR" in value:
        return {'display': 'inline'}
    else:
        return {'display': 'none'}


# TODO: Writing in dropdown field: font is black, needs to be white
if __name__ == '__main__':
    UpdateTrades(Event()).start()
    UpdateTokens(Event()).start()

    app.run_server(debug=cfg.get("debug"), threaded=True, port=cfg.get("dashboard_port"), host="0.0.0.0")
