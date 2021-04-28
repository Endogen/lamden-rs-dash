import os
import base64
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
from dash.exceptions import PreventUpdate

from pandas import DataFrame
from pathlib import Path
from distutils.util import strtobool
from datetime import datetime, timedelta


class Config:
    def __init__(self, cfg_path: str = None):
        cfg_path = cfg_path if cfg_path else os.path.join("assets", "config.cfg")
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


class Database:
    def __init__(self, db_path: str = None):
        self._db_path = db_path if db_path else os.path.join("assets", "database.db")

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

    def token_exists(self, contract_name):
        sql = \
            "SELECT EXISTS (" \
            "   SELECT 1 " \
            "   FROM token_list " \
            "   WHERE contract_name = ?" \
            ")"

        result = self.execute(sql, contract_name)
        return True if result["data"][0][0] == 1 else False

    def select_last_trade(self):
        sql = \
            "SELECT * " \
            "FROM trade_history " \
            "ORDER BY time DESC LIMIT 1"

        return self.execute(sql)

    def select_token_trades(self, token_symbol, start_secs=0):
        sql = \
            "SELECT * " \
            "FROM trade_history " \
            f"WHERE token_symbol = ? AND time >= {start_secs} " \
            "ORDER BY time ASC"

        return self.execute(sql, token_symbol)

    def select_token_trade_count(self, start_secs: int = 0):
        sql = \
            "SELECT token_symbol, count(token_symbol) " \
            "FROM trade_history " \
            f"WHERE time > {start_secs} " \
            "GROUP BY token_symbol " \
            "ORDER BY count(token_symbol) DESC"

        return self.execute(sql)

    def select_token_data(self, token_symbol):
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

    def get_token_symbols(self):
        sql = \
            "SELECT DISTINCT token_symbol " \
            "FROM trade_history"

        return self.execute(sql)


class Rocketswap:
    def __init__(self, cfg: Config, db: Database):
        self._cfg = cfg
        self._db = db

        self._last_update = tuple()
        self._last_update_id = None

        self._selected_token = cfg.get("default_token")
        self._selected_token_data = None

        self._last_trade_date = "N/A"

    def set_selected_token(self, token_symbol):
        self._selected_token = token_symbol

        res = db.select_token_data(token_symbol)

        if res and res["data"]:
            self._selected_token_data = res["data"][0]
        else:
            self._selected_token_data = None

    def get_selected_token(self):
        return self._selected_token

    def get_last_trade_date(self):
        return self._last_trade_date

    def set_last_trade_date(self, new_date):
        self._last_trade_date = new_date

    def update_trades(self):
        res = self._db.select_last_trade()

        if res and res["data"]:
            last_secs = res["data"][0][3]
        else:
            last_secs = 0

        skip = 0
        take = 50
        call = True

        while call:
            try:
                url = f"{self._cfg.get('rocketswap_url')}get_trade_history"
                res = requests.get(url, params={"take": take, "skip": skip}, timeout=3)
            except Exception as e:
                logging.error(e)
                return

            trades_api = list()
            for t in res.json():
                if t not in trades_api:
                    trades_api.append(t)

            skip += take

            new_trades = tuple()
            for tx in trades_api:
                if tx["time"] > last_secs:
                    trade = (
                        tx["contract_name"],
                        tx["token_symbol"],
                        tx["price"],
                        tx["time"],
                        tx["type"]
                    )

                    new_trades += (trade,)
                    self._db.insert_trade(*trade)
                    print("New trade:", tx)
                    logging.info(f"New trade: {tx}")
                else:
                    call = False
                    break

            if new_trades:
                self._last_update = new_trades
                print("last_update set to:", self._last_update)

            if len(res.json()) != take:
                call = False

    def update_tokens(self):
        try:
            url = f"{self._cfg.get('rocketswap_url')}token_list"
            res = requests.get(url, timeout=3)
        except Exception as e:
            logging.error(e)
            return

        for tk in res.json():
            self._db.insert_token(
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
        return self._last_update

    def get_last_update_id(self):
        return self._last_update_id

    def set_last_update_id(self, new_id):
        self._last_update_id = new_id

    def get_token_trades(self, token_symbol, start_secs: int = 0):
        data = list()

        for trade in self._db.select_token_trades(token_symbol, start_secs)["data"]:
            data.append([trade[3], trade[2]])

        return data

    def get_token_trade_count(self, start_secs: int = 0):
        res = self._db.select_token_trade_count(start_secs)

        data = list()
        for trade in res["data"]:
            data.append([trade[0], trade[1]])

        return data

    def get_all_token_symbols(self):
        res = db.get_token_symbols()

        data = list()
        for ts in res["data"]:
            data.append({"label": ts[0], "value": ts[0]})

        return data

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

    def get_trades_graph(self, data, selected_token):
        df = DataFrame(reversed(data), columns=["Tokens", "Trades"])
        df.sort_index(ascending=True, inplace=True)

        colors = ["rgb(84, 95, 249)", ] * len(data)

        for i, d in enumerate(reversed(data)):
            if d[0] == selected_token:
                colors[i] = "crimson"

        # TODO: Hier gab es bei der alten Version keine Warning?
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


class Utils:
    @staticmethod
    def to_unix_time(date_time):
        return int((date_time - datetime(1970, 1, 1)).total_seconds())

    # TODO: Needed?
    @staticmethod
    def b64_image(image_data):
        #return 'data:image/png;base64,' + base64.b64encode(image_data).decode('utf-8')
        return 'data:image/png;base64,' + image_data


class Dashboard:

    def __init__(self, cf: Config, rs: Rocketswap):
        now = datetime.utcnow()

        one_day = Utils.to_unix_time(now - timedelta(days=1))
        five_days = Utils.to_unix_time(now - timedelta(days=5))
        one_month = Utils.to_unix_time(now - timedelta(days=30))

        app = dash.Dash(
            __name__,
            external_stylesheets=['style.css'],
            title="Rocketswap Dashboard",
            meta_tags=[{
                "name": "viewport",
                "content": "width=device-width, initial-scale=1.0, maximum-scale=1.2, minimum-scale=0.5"
            }]
        )

        app.layout = html.Div(children=[
            dcc.Store(id='local', storage_type='local'),

            dcc.Interval(
                id='interval-component',
                interval=cf.get("update_interval"),
                n_intervals=0
            ),

            dcc.Checklist(
                id="select-visible",
                options=[
                    {'label': '1D', 'value': '1D'},
                    {'label': '5D', 'value': '5D'},
                    {'label': '30D', 'value': '30D'},
                    {'label': 'Max', 'value': 'MAX'},
                    {'label': 'Trades', 'value': 'TRADES'}
                ],
                value=['1D', '5D', '30D', 'MAX', 'TRADES'],
                labelStyle={'display': 'inline-block'}
            ),

            html.Div([
                html.Label(children=f"Last Trade: N/A", id="last-trade", style={"text-align": "center"})
            ]),

            html.Div([
                html.Div([
                    dcc.Dropdown(
                        options=rs.get_all_token_symbols(),
                        value=rs.get_selected_token(),
                        multi=False,
                        id="token_symbol_input"
                    )
                ], className="four columns"),


            ], className="row"),

            html.Br(),

            html.Div([
                html.Div([
                    html.H5(
                        children=f"{rs.get_selected_token()}-TAU 1 Day",
                        id="price-title-1d",
                        style={"text-align": "center"}),

                    dcc.Graph(
                        id='price-graph-1d',
                        figure=rs.get_price_graph(rs.get_token_trades(rs.get_selected_token(), one_day)))
                ], id="div-1d", className="four columns"),

                html.Div([
                    html.H5(
                        children=f"{rs.get_selected_token()}-TAU 5 Days",
                        id="price-title-5d",
                        style={"text-align": "center"}),

                    dcc.Graph(
                        id='price-graph-5d',
                        figure=rs.get_price_graph(rs.get_token_trades(rs.get_selected_token(), five_days)))
                ], id="div-5d", className="four columns"),

                html.Div([
                    html.H5(
                        children=f"{rs.get_selected_token()}-TAU 30 Days",
                        id="price-title-30d",
                        style={"text-align": "center"}),

                    dcc.Graph(
                        id='price-graph-30d',
                        figure=rs.get_price_graph(rs.get_token_trades(rs.get_selected_token(), one_month)))
                ], id="div-30d", className="four columns"),
            ], className="row"),

            html.Div([
                html.H5(
                    children=f"{rs.get_selected_token()}-TAU Max",
                    id="price-title-all",
                    style={"text-align": "center"}),

                dcc.Graph(
                    id='price-graph-all',
                    figure=rs.get_price_graph(rs.get_token_trades(rs.get_selected_token())))
            ], id="div-all"),

            html.Div([
                html.H5(
                    children="Trades per Token",
                    id="trades-per-token-title",
                    style={"text-align": "center"}),

                dcc.Graph(
                    id='trades-graph',
                    figure=rs.get_trades_graph(rs.get_token_trade_count(0), rs.get_selected_token()))
            ], id="div-trades")
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
            Output('last-trade', 'children'),
            Output('trades-graph', 'figure'),
            [Input('interval-component', 'n_intervals'),
             Input('token_symbol_input', 'value')])
        def update_graph(n, token_symbol):
            caller_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
            print(f"In update. caller {caller_id} with params: ", n, token_symbol)

            if caller_id == "token_symbol_input" and token_symbol:
                rs.set_selected_token(token_symbol)
                print("Set selected token to:", token_symbol)

                fig_1d = rs.get_price_graph(rs.get_token_trades(token_symbol, one_day))
                fig_1d.update_layout(transition_duration=500)
                fig_5d = rs.get_price_graph(rs.get_token_trades(token_symbol, five_days))
                fig_5d.update_layout(transition_duration=500)
                fig_30d = rs.get_price_graph(rs.get_token_trades(token_symbol, one_month))
                fig_30d.update_layout(transition_duration=500)
                fig_all = rs.get_price_graph(rs.get_token_trades(token_symbol))
                fig_all.update_layout(transition_duration=500)
                fig_trades = rs.get_trades_graph(rs.get_token_trade_count(), rs.get_selected_token())
                fig_trades.update_layout(transition_duration=500)

                return [
                    fig_1d, f"{token_symbol}-TAU 1 Day",
                    fig_5d, f"{token_symbol}-TAU 5 Days",
                    fig_30d, f"{token_symbol}-TAU 30 Days",
                    fig_all, f"{token_symbol}-TAU Max",
                    f"Last Trade: {rs.get_last_trade_date()}",
                    fig_trades
                ]

            # TODO: Add update of token list if n_interval is 10, 20, 30, ...
            elif caller_id == "interval-component" and n:
                rs.update_trades()

                if not rs.get_last_update():
                    print("No last_update, exiting update")
                    raise PreventUpdate

                if hash(rs.get_last_update()) == rs.get_last_update_id():
                    print("Hash matches, exiting update")
                    raise PreventUpdate

                rs.set_last_update_id(hash(rs.get_last_update()))
                print("set new update to:", rs.get_last_update())

                for tx in rs.get_last_update():
                    print("Checking trade:", tx)

                    if tx[1] == rs.get_selected_token():
                        print("Trade matches selected token, updating and returning graphs")
                        logging.info(f"Graphs for token {rs.get_selected_token()} updated")

                        fig_1d = rs.get_price_graph(rs.get_token_trades(rs.get_selected_token(), one_day))
                        fig_1d.update_layout(transition_duration=500)
                        fig_5d = rs.get_price_graph(rs.get_token_trades(rs.get_selected_token(), five_days))
                        fig_5d.update_layout(transition_duration=500)
                        fig_30d = rs.get_price_graph(rs.get_token_trades(rs.get_selected_token(), one_month))
                        fig_30d.update_layout(transition_duration=500)
                        fig_all = rs.get_price_graph(rs.get_token_trades(rs.get_selected_token()))
                        fig_all.update_layout(transition_duration=500)
                        fig_trades = rs.get_trades_graph(rs.get_token_trade_count(), rs.get_selected_token())
                        fig_trades.update_layout(transition_duration=500)

                        rs.set_last_trade_date(str(datetime.now()))

                        return [
                            fig_1d, f"{rs.get_selected_token()}-TAU 1 Day",
                            fig_5d, f"{rs.get_selected_token()}-TAU 5 Days",
                            fig_30d, f"{rs.get_selected_token()}-TAU 30 Days",
                            fig_all, f"{rs.get_selected_token()}-TAU Max",
                            f"Last Trade: {rs.get_last_trade_date()}",
                            fig_trades
                        ]

                print("No trade matches selected token, existing")
                raise PreventUpdate
            else:
                print("Unknown caller, exiting")
                raise PreventUpdate

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
            Output('div-all', 'style'),
            Input('select-visible', 'value'))
        def update_visibility_all(value):
            if "MAX" in value:
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

        app.run_server(debug=cf.get('debug'), threaded=True, port=cf.get('dashboard_port'))


if __name__ == '__main__':
    db = Database()
    db.create_trade_history()
    db.create_token_list()

    cf = Config()

    rs = Rocketswap(cf, db)
    rs.update_tokens()
    rs.update_trades()

    Dashboard(cf, rs)
