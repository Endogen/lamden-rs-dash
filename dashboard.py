import sqlite3
import logging
import requests

import dash
import dash_core_components as dcc
import dash_html_components as html

import plotly.graph_objs as go
import pandas as pd

from dash.dependencies import Output, Input
from dash.exceptions import PreventUpdate
from pandas import DataFrame
from pathlib import Path

from datetime import datetime, timedelta


# TODO: Add possibility to update DB periodically
# TODO: Read config from file
class Config:
    rs_url = "https://stats.rocketswap.exchange:2053/api/"
    default_token = "RSWP"
    update_interval = 5000  # milliseconds


# TODO: Rename to 'dashboard.db'
class Database:
    def __init__(self, db: str = "database.db"):
        self.db = db

    def execute(self, sql_statement, *args):
        res = {"success": None, "data": None}

        con = None
        cur = None

        try:
            con = sqlite3.connect(self.db)
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
        if not Path(self.db).is_file():
            return False

        sql = \
            "SELECT name " \
            "FROM sqlite_master " \
            "WHERE type = 'table' AND name = ?"

        exists = False

        if self.execute(sql, table_name)["data"]:
            exists = True

        return exists

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
            "   contract_name TEXT NOT NULL," \
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

    def select_token_trades(self, token_symbol, start_secs=0):
        sql = f"" \
            "SELECT * " \
            "FROM trade_history " \
            f"WHERE token_symbol = ? AND time >= {start_secs} " \
            "ORDER BY time ASC"

        return self.execute(sql, token_symbol)

    def select_token_details(self, token_symbol):
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

    def get_token_symbols(self):
        sql = \
            "SELECT DISTINCT token_symbol " \
            "FROM trade_history"

        return self.execute(sql)


class Rocketswap:
    def __init__(self, cfg: Config, db: Database):
        self.cfg = cfg
        self.db = db

        self.last_update = None
        self.last_update_id = None
        self.selected_token = None

    def update_trades(self):
        res = self.db.select_last_trade()

        if res and res["data"]:
            last_secs = res["data"][0][3]
        else:
            last_secs = 0

        skip = 0
        take = 50
        call = True

        while call:
            try:
                url = f"{self.cfg.rs_url}get_trade_history"
                res = requests.get(url, params={"take": take, "skip": skip})
            except Exception as e:
                logging.error(e)
                return

            skip += take

            new_trades = tuple()
            for tx in res.json():
                if tx["time"] > last_secs:
                    trade = (
                        tx["contract_name"],
                        tx["token_symbol"],
                        tx["price"],
                        tx["time"],
                        tx["type"]
                    )

                    new_trades += (trade,)
                    self.db.insert_trade(*trade)  # TODO: Geht das?
                else:
                    call = False
                    break

            if new_trades:
                self.last_update = new_trades

            if len(res.json()) != take:
                call = False

            print(self.last_update, "\n")  # TODO: TEMP

    def update_token_list(self):
        try:
            res = requests.get(f"{self.cfg.rs_url}token_list")
        except Exception as e:
            logging.error(e)
            return

        for tk in res.json():
            self.db.insert_token(
                tk["contract_name"],
                tk["has_market"],
                tk["token_base64_png"],
                tk["token_base64_svg"],
                tk["logo"]["type"],
                tk["logo"]["data"],
                tk["token_logo_url"],
                tk["token_name"],
                tk["token_symbol"])

    def get_last_update(self):
        return self.last_update

    def get_token_trades(self, token_symbol, start_secs: int = 0):
        self.selected_token = token_symbol

        data = list()
        for trade in self.db.select_token_trades(token_symbol, start_secs)["data"]:
            data.append([trade[3], trade[2]])

        return data

    def get_token_symbols(self):
        res = db.get_token_symbols()

        data = list()
        for ts in res["data"]:
            data.append({"label": ts[0], "value": ts[0]})

        return data

    def get_graph(self, data):
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
                tickprefix="",
                ticksuffix=" ",
                autorange=True,
                fixedrange=False
            ),
            margin=dict(
                r=25,
                t=5,
                b=40)
        )

        fig = go.Figure(data=[graph], layout=layout)
        return fig


def to_unix_time(date_time):
    return int((date_time - datetime(1970, 1, 1)).total_seconds())


if __name__ == '__main__':
    db = Database()
    db.create_trade_history()
    db.create_token_list()

    cf = Config()

    rs = Rocketswap(cf, db)
    rs.update_token_list()
    rs.update_trades()

    now = datetime.utcnow()

    one_day = to_unix_time(now - timedelta(days=1))
    five_days = to_unix_time(now - timedelta(days=5))
    one_month = to_unix_time(now - timedelta(days=30))

    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    app.layout = html.Div(children=[
        html.H1(children='🚀 Rocketswap Dashboard'),

        html.Div(children='''
            Choose Lamden Token
        '''),

        dcc.Dropdown(
            options=rs.get_token_symbols(),
            value=cf.default_token,
            multi=False,
            id="token_symbol_input"
        ),

        html.Div([
            dcc.Interval(
                id='interval-component',
                interval=cf.update_interval,
                n_intervals=0
            ),

            html.Div([
                html.H5(children=f"{cf.default_token}-TAU 1 Day", id="price-title-1d", style={"text-align": "center"}),

                dcc.Graph(
                    id='price-graph-1d',
                    figure=rs.get_graph(rs.get_token_trades(cf.default_token, one_day)))  # TODO: Vereinfachen! Beide Objekte sind in 'rs'
            ], className="four columns"),

            html.Div([
                html.H5(children=f"{cf.default_token}-TAU 5 Days", id="price-title-5d", style={"text-align": "center"}),

                dcc.Graph(
                    id='price-graph-5d',
                    figure=rs.get_graph(rs.get_token_trades(cf.default_token, five_days)))
            ], className="four columns"),

            html.Div([
                html.H5(children=f"{cf.default_token}-TAU 30 Days", id="price-title-30d", style={"text-align": "center"}),

                dcc.Graph(
                    id='price-graph-30d',
                    figure=rs.get_graph(rs.get_token_trades(cf.default_token, one_month)))
            ], className="four columns"),
        ], className="row"),

        html.Div([
            html.H5(children=f"{cf.default_token}-TAU Overall", id="price-title-all", style={"text-align": "center"}),

            dcc.Graph(
                id='price-graph-all',
                figure=rs.get_graph(rs.get_token_trades(cf.default_token)))
        ]),

        html.Div([
            html.Label(children=f"Last Update: {str(datetime.now())}", id="last-update", style={"text-align": "center"}),
            html.Label(children=f"{cf.default_token}-TAU Overall", id="label-test", style={"text-align": "center"})
        ]),
    ])

    @app.callback(
        Output('price-graph-1d', 'figure'),
        Output('price-title-1d', 'children'),
        Output('price-graph-5d', 'figure'),
        Output('price-title-5d', 'children'),
        Output('price-graph-30d', 'figure'),
        Output('price-title-30d', 'children'),
        Output('price-graph-all', 'figure'),
        Output('price-title-all', 'children'),
        [Input('interval-component', 'n_intervals'),
         Input('token_symbol_input', 'value')])
    def update_graph_on_trade(n, token_symbol):
        caller_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0]

        if caller_id == "token_symbol_input" and token_symbol:
            fig_1d = rs.get_graph(rs.get_token_trades(token_symbol, one_day))
            fig_1d.update_layout(transition_duration=500)
            fig_5d = rs.get_graph(rs.get_token_trades(token_symbol, five_days))
            fig_5d.update_layout(transition_duration=500)
            fig_30d = rs.get_graph(rs.get_token_trades(token_symbol, one_month))
            fig_30d.update_layout(transition_duration=500)
            fig_all = rs.get_graph(rs.get_token_trades(token_symbol))
            fig_all.update_layout(transition_duration=500)

            return [
                fig_1d, f"{token_symbol}-TAU 1 Day",
                fig_5d, f"{token_symbol}-TAU 5 Days",
                fig_30d, f"{token_symbol}-TAU 30 Days",
                fig_all, f"{token_symbol}-TAU Overall"
            ]

        # TODO: Add update of token list if n_interval is 10, 20, 30, ...
        if caller_id == "interval-component" and n:
            rs.update_trades()

            if not rs.last_update:
                raise PreventUpdate

            if hash(rs.last_update) == rs.last_update_id:
                print("Update aborted")
                raise PreventUpdate

            print("Hash doesn't match!")
            rs.last_update_id = hash(rs.last_update)

            for tx in rs.last_update:
                if tx[1] == rs.selected_token:
                    print(f"Found new trade with token {rs.selected_token}")
                    fig_1d = rs.get_graph(rs.get_token_trades(rs.selected_token, one_day))
                    fig_1d.update_layout(transition_duration=500)
                    fig_5d = rs.get_graph(rs.get_token_trades(rs.selected_token, five_days))
                    fig_5d.update_layout(transition_duration=500)
                    fig_30d = rs.get_graph(rs.get_token_trades(rs.selected_token, one_month))
                    fig_30d.update_layout(transition_duration=500)
                    fig_all = rs.get_graph(rs.get_token_trades(rs.selected_token))
                    fig_all.update_layout(transition_duration=500)

                    return [
                        fig_1d, f"{rs.selected_token}-TAU 1 Day",
                        fig_5d, f"{rs.selected_token}-TAU 5 Days",
                        fig_30d, f"{rs.selected_token}-TAU 30 Days",
                        fig_all, f"{rs.selected_token}-TAU Overall"
                    ]

            print("Update aborted")
            raise PreventUpdate

        else:
            print("Update aborted")
            raise PreventUpdate

    app.run_server(debug=True)
