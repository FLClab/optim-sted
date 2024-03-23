
import yaml
import os
import glob
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import plotly.graph_objects as go
import json
import shutil
import time
import numpy
import dash_renderjson

from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from .utils import load_data, load_images, get_data
from .plotting import plot_failures, plot_objectives, plot_parameters, plot_mean_failures, show_images

def create_app(PATH):

    SLC = slice(None, 1000)

    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME], suppress_callback_exceptions=True)

    text_style = {
        "color" : "white",
        "height" : "100%",
        "align" : "center",
        "text-align" : "center",
        "display" : "flex",
        "justify-content":"center",
        "align-items" : "center",
    }
    text_style_left = {
        "color" : "white",
        # "height" : "100%",
        "align" : "left",
        "text-align" : "left",
        "display" : "flex",
        "justify-content":"left",
        "align-items" : "top",
        "padding" : "2pt",
        "font-size" : 12,
    }
    tabs_styles = {
        "height" : "50%",
    }
    tab_style = {
        "margin" : "5pt",
        'border': '2px solid #666666',
        "border-radius" : "10px",
        "align" : "center",
        "text-align" : "center",
        "display" : "flex",
        "justify-content":"center",
        "align-items" : "center",
        "background-color" : "#666666",
        "color" : "white",
    }
    tab_selected_style = {
        "margin" : "5pt",
        # 'borderTop': '1px solid #d6d6d6',
        # 'borderBottom': '1px solid #d6d6d6',
        "border" : "2px solid white",
        "border-radius" : "10px",
        'background-color': "#ffcc00",
        "color": "white",
        "align" : "center",
        "text-align" : "center",
        "display" : "flex",
        "justify-content":"center",
        "align-items" : "center"
    }

    def get_folders():
        folders = list(sorted(filter(lambda item: os.path.isdir(item) and ("optim.hdf5" in os.listdir(item)),
                                    glob.glob(os.path.join(PATH, "optim-sted", "data", "**", "*"),
                                            recursive=True))))
        folders = list(sorted(filter(lambda item: os.path.isdir(item) and ("optim.hdf5" in os.listdir(item)),
                                    glob.glob(os.path.join(PATH, "**", "*"),
                                            recursive=True))))
        return folders

    def split_folders(folders):
        return [{"label": folder.split("data" + os.path.sep)[-1], "value": folder} for folder in folders]


    # PATH = os.path.dirname(__file__).split("optim-sted")[0]
    # PATH = os.path.join("C:", os.sep, "Users", "abberior", "Desktop", "DATA", "abilodeau")
    # PATH = None
    # print(PATH)
    # folders = list(sorted(filter(lambda item: os.path.isdir(item) and ("optim.hdf5" in os.listdir(item)), glob.glob(f"/{PATH}/data/**/*", recursive=True))))
    # print(glob.glob(os.path.join(PATH, "optim-sted", "data", "**", "*"), recursive=True))
    folders = get_folders()
    # print(folders)

    app.layout = dbc.Container([
        dcc.Store(id="cached-values"),
        dcc.Store(id="cached-folders-values"),
        dcc.Interval(
            id='interval',
            interval=30*1000, # in milliseconds
            n_intervals=0
        ),
        dbc.Row([
            dbc.Col([
                dbc.Row([
                    dbc.Col([dcc.Clipboard(id="folder-copy", title="copy", style=text_style,)], width=1),
                    # dbc.Col([dcc.Dropdown([{"label" : folder.split("data/")[-1], "value" : folder} for folder in folders], folders[-1], id="folder", clearable=False, style={"font-size" : 12}),], width=11)])
                    dbc.Col([dmc.MultiSelect(data=split_folders(folders), value=[folders[-1]], id="folder", clearable=True, limit=100, searchable=True, nothingFound="No options found...", placeholder="Select model...", maxSelectedValues=1, style={"font-size" : 12}),], width=11)])
            ], width=6),
            dbc.Col([
                dbc.Row([
                    dbc.Col([html.Div("Trials", style=text_style)], width=2),
                    dbc.Col([dcc.Slider(0, 10, 1, id="trial", value=0, marks=None, tooltip={"placement": "left", "always_visible": False})], width=10, style={"padding-top":"15pt"})
                ])
            ], width=5),
            dbc.Col([
                dbc.Row([
                    dbc.Col([html.I(id="refresh-button", className="fa-solid fa-arrows-rotate", style=text_style, title="Refresh...")], width=12)
                ])
            ], width=1),
        ], style={"height": "8%", "background-color" : "#87cdde"}, align="center"),
        dbc.Row([dcc.Tabs(id="tabs", value="run-tab", children=[
                dcc.Tab(label="Optimization", value="run-tab", style=tab_style, selected_style=tab_selected_style),
                dcc.Tab(label="Configuration", value="configuration-tab", style=tab_style, selected_style=tab_selected_style),
                dcc.Tab(label="Compare", value="compare-tab", style=tab_style, selected_style=tab_selected_style),
            ], style=tabs_styles)
        ], style={"height" : "8%", "background-color" : "#87cdde"}),
        dbc.Row([
            html.Div([
                html.Div([
                    dcc.Slider(0, 100, 1, value=0, marks=None, tooltip={"placement": "bottom", "always_visible": True}, id="slider"),
                    dcc.Checklist(folders, [], id="compare-checklist")
                ], style={"display":"none"})
            ], id="tabs-content", style={"height" : "100%", "width" : "100%"}),
        ], style={"height" : "84%"}),
    ], style={"height" : "100vh", "background-color" : "#343434"}, fluid=True)

    @app.callback(Output('tabs-content', 'children'),
                Input('tabs', 'value'))
    def render_content(tab):
        slider = dcc.Slider(0, 100, 1, value=0, marks=None, tooltip={"placement": "bottom", "always_visible": True}, id="slider")
        checklist = dcc.Checklist(folders, [], id="compare-checklist", labelStyle=text_style_left, inputStyle={"margin-right": "10px"}, style={"width" : "100%"})
        search_input = dcc.Input(placeholder="Search...", type="search", debounce=False, id="search-input", style={"height" : "10%", "width" : "100%", "padding" : "10px", "margin" : "10px", })
        if tab == 'run-tab':
            return html.Div([
                html.Div([search_input, checklist], style={"display":"none"}),
                dbc.Row([
                    dbc.Col([html.Div([dcc.Graph(id="parameters", style={"height" : "100%"})], style={"height" : "100%"})], style={"height": "100%"}),
                ], style={"height": "50%"}, align="center"),
                dbc.Row([
                    dbc.Col([
                        dbc.Row([
                            dbc.Col([html.Div("Image", style=text_style)], width=3),
                            dbc.Col([slider], width=9)
                        ], style={"height" : "10%"}),
                        dbc.Row([dcc.Graph(id="images", style={"height" : "100%"})], style={"height" : "90%"}),
                    ], width=5, style={"height": "100%"}),
                    dbc.Col([
                        dbc.Row([dcc.Dropdown(["line", "boxplot"], "line", id="objectives-graph", clearable=False)], style={"height": "10%"}),
                        dbc.Row([dcc.Graph(id="objectives", style={"height" : "100%"})], style={"height": "90%"})
                    ], width=4, style={"height": "100%"}),
                    dbc.Col([
                        dbc.Row([dcc.Checklist(["Show All"], ["Show All"], id="checklist", labelStyle=text_style)], style={"height": "10%"}),
                        dbc.Row([dcc.Graph(id='failures', style={"height" : "100%"})], style={"height": "90%"})
                    ], width=3, style={"height": "100%"})
                ], style={"height": "50%"}, align="center"),
            ], style={"height" : "100%", "width" : "100%"})
        elif tab == 'configuration-tab':
            return html.Div([
                html.Div([search_input, slider, checklist], style={"display":"none"}),
                html.Div([
                    dash_renderjson.DashRenderjson(id="configuration-json", max_depth=1, theme={
                    "scheme": 'default',
                    "base00": '#343434',
                    "base01": '#343434',
                    "base02": '#343434',
                    "base03": '#ababab',
                    "base04": '#a59f85',
                    "base05": '#f8f8f2',
                    "base06": '#f5f4f1',
                    "base07": '#f9f8f5',
                    "base08": '#f92672',
                    "base09": '#d19a66',
                    "base0A": '#f4bf75',
                    "base0B": '#93bd76',
                    "base0C": '#a1efe4',
                    "base0D": '#66d9ef',
                    "base0E": '#ae81ff',
                    "base0F": '#cc6633'}
                )
                ], style={"maxHeight" : "100%", "overflow" : "scroll"})
            ], style={"height" : "100%", "width" : "100%"})
        elif tab == "compare-tab":
            return html.Div([
                html.Div([slider], style={"display":"none"}),
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            dbc.Row([search_input], style={"width" : "100%"}),
                            dbc.Row([checklist], style={"overflow-y" : "scroll", "height" : "90%", "width" : "100%"}),
                        ], width=3, style={"background-color": "#404040", "height" : "100%"}),
                        dbc.Col([
                            dcc.Graph(id='compare-graph', style={"height" : "100%"})
                        ], width=9, style={"height" : "100%"})
                    ], style={"height" : "100%"})
                ], style={"height" : "100%"})
            ], style={"height" : "100%", "width" : "100%"})

    @app.callback(
        Output("folder-copy", "content"),
        Input("folder-copy", "n_clicks"),
        State("folder", "value"),
        State("folder", "data"),
    )
    def copy_selected_folder(i, value, folders):
        if not value:
            return ""
        value = value[0]
        for folder in folders:
            if value == folder["value"]:
                return folder["label"]
        return ""

    @app.callback(
        Output("cached-values", "data"),
        Input("refresh-button", "n_clicks"),
        Input("interval", "n_intervals"),
        Input("folder", "value"),
        Input("slider", "value"),
        Input("trial", "value"),
        Input("cached-values", "data"),
    )
    def update_cache(n, _, path, idx, trial, previous_cache):

        if isinstance(previous_cache, str):
            previous_cache = json.loads(previous_cache)
        else:
            previous_cache = {}

        if not path:
            return json.dumps(previous_cache)
        else:
            path = path[0]

        config = yaml.load(open(os.path.join(path, "config.yml"), "r"), Loader=yaml.Loader)
        ndims = []
        for param_name in config["param_names"]:
            if (param_name in ["decision_time", "threshold_count"]) and (config["microscope"] == "DyMIN"):
                ndims.append(2)
            else:
                ndims.append(1)
        N_POINTS = [config["n_divs_default"]]*sum(ndims)

        try:
            # Case where data can be read
            X, y, all_X, all_y = load_data(config, path, trial=str(trial), slc=SLC, ndims=ndims)
            conf1, sted_image, conf2 = load_images(config, path, trial=str(trial), idx=idx)

            data = {
                "config" : config,
                "X" : {key : value.tolist() for key, value in X.items()},
                "y" : {key : value.tolist() for key, value in y.items()},
                "conf1" : conf1.tolist(),
                "sted_image" : sted_image.tolist(),
                "conf2" : conf2.tolist(),
                "ndims" : ndims,
                "all_X" : [{key : value.tolist() for key, value in a_X.items()} for a_X in all_X],
                "all_y" : [{key : value.tolist() for key, value in a_y.items()} for a_y in all_y]
            }
        except (KeyError) as othererr:
            # Case where a key cannot be read in file, e.g. trial does not exist
            data = {
                "config" : config
            }
        except (OSError, BlockingIOError) as err:
            # Case where the file could not be read
            previous_cache["config"] = config
            data = previous_cache

        return json.dumps(data)

    @app.callback(
        Output("folder", "data"),
        Output("cached-folders-values", "data"),
        Input("interval", "n_intervals"),
    )
    def update_folder_cache(n):
        folders = get_folders()
        folders = split_folders(folders)
        # folders = list(´sorted(filter(lambda item: os.path.isdir(item) and ("optim.hdf5" in os.listdir(item)), glob.glob(f"/{PATH}/data/**/*", recursive=True))))
        # folders = [{"label" : folder.split("data/")[-1], "value" : folder} for folder in folders]
        return folders, json.dumps(folders)

    # @app.callback(
    #     Output("cached-folders-values", "data"),
    #     Input("interval", "n_intervals"),
    # )
    # def update_folder_cache(n):
    #     folders = list(sorted(filter(lambda item: os.path.isdir(item) and ("optim.hdf5" in os.listdir(item)), glob.glob(f"/{PATH}/data/**/*", recursive=True))))
    #     folders = [{"label" : folder.split("data/")[-1], "value" : folder} for folder in folders]
    #     return json.dumps(folders)

    # @app.callback(
    #     Output("folder", "value"),
    #     Input("cached-folders-values", "data"),
    #     Input("folder", "value")
    # )
    # def update_dropdown(cache, search_value):
    #     print(search_value)
    #     folders = json.loads(cache)
    #     if not search_value:
    #         raise PreventUpdate
    #         return folders
    #
    #     # folders = list(sorted(filter(lambda item: os.path.isdir(item) and ("optim.hdf5" in os.listdir(item)), glob.glob(f"/{PATH}/data/**/*", recursive=True))))
    #     # folders = [{"label" : folder.split("data/")[-1], "value" : folder} for folder in folders]
    #     folders = [o for o in folders if search_value in o["label"]]
    #     return folders
    #     # return folders

    @app.callback(
        Output("compare-checklist", "options"),
        Input("cached-folders-values", "data"),
        Input("search-input", "value")
    )
    def update_checklist(cache, search_value):
        folders = json.loads(cache)
        if search_value:
            folders = [
                folder for folder in folders
                if search_value in folder["label"]
            ]
        return folders

    @app.callback(
        Output("configuration-json", "data"),
        Input("interval", "n_intervals"),
        Input("cached-values", "data")
    )
    def update_configuration(n, data):
        if isinstance(data, str):
            data = json.loads(data)
        config = data.get("config")
        return config

    @app.callback(
        Output("trial", "max"),
        Output('slider', 'max'),
        Input("interval", "n_intervals"),
        Input('cached-values', 'data'),
    )
    def update_slider(n, data):
        if isinstance(data, str):
            data = json.loads(data)
        config = data.get("config")
        X = data.get("X", {"tmp" : [1]})
        return config["nbre_trials"] - 1, len(X[list(X.keys())[0]]) - 1

    @app.callback(
        Output('images', 'figure'),
        Input("slider", "value"),
        Input("trial", "value"),
        Input("cached-values", "data"),
    )
    def update_images(value, trial, data):

        if isinstance(data, str):
            data = json.loads(data)

        config = data.get("config")
        conf1 = data.get("conf1", None)
        sted_image = data.get("sted_image", None)
        conf2 = data.get("conf2", None)

        if isinstance(conf1, type(None)):
            images = go.Figure()
        else:
            images = show_images(config, conf1, sted_image, conf2)

        for fig in [images]:
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                margin=go.layout.Margin(
                    l=10, #left margin
                    r=10, #right margin
                    b=50, #bottom margin
                    t=50  #top margin
                )
            )
            fig.update_xaxes(
                color="white", showgrid=False
            )
            fig.update_yaxes(
                color="white", showgrid=False
            )
        return images

    @app.callback(
        Output("parameters", "figure"),
        Output("objectives", "figure"),
        Output("failures", "figure"),
        Input("interval", "n_intervals"),
        Input('folder', 'value'),
        Input("slider", "value"),
        Input("objectives-graph", "value"),
        Input("trial", "value"),
        Input("checklist", "value"),
        Input("cached-values", "data"),
    )
    def update_plots(n, path, value, obj_graph, trial, checklist, data):

        if isinstance(data, str):
            data = json.loads(data)

        config = data.get("config")
        X = data.get("X", None)
        y = data.get("y", None)
        all_X = data.get("all_X", None)
        all_y = data.get("all_y", None)
        ndims = data.get("ndims", None)

        if isinstance(X, type(None)):
            failures, objectives, parameters = go.Figure(), go.Figure(), go.Figure()
        else:
            X = {key : numpy.array(value) for key, value in X.items()}
            y = {key : numpy.array(value) for key, value in y.items()}
            all_X = [{key : numpy.array(value) for key, value in a_X.items()} for a_X in all_X]
            all_y = [{key : numpy.array(value) for key, value in a_y.items()} for a_y in all_y]

            failures = plot_failures(config, all_X, all_y, idx=trial, total=len(all_X), show_all="Show All" in checklist)
            objectives = plot_objectives(config, X, y, _type=obj_graph)
            parameters = plot_parameters(config, X, y, ndims=ndims)

        for fig in [failures, objectives, parameters]:
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                margin=go.layout.Margin(
                    l=10, #left margin
                    r=10, #right margin
                    b=50, #bottom margin
                    t=50  #top margin
                )
            )
            fig.update_xaxes(
                color="white", showgrid=False
            )
            fig.update_yaxes(
                color="white", showgrid=False
            )
        return parameters, objectives, failures

    @app.callback(
        Output("compare-graph", "figure"),
        Input("interval", "n_intervals"),
        Input("trial", "value"),
        Input("compare-checklist", "value")
    )
    def update_compare_plots(n, trial, checklist):

        failures = go.Figure()

        lim = 0
        checklist = list(sorted(checklist))
        for idx, path in enumerate(checklist):
            data = get_data(path, 0)

            config = data.get("config")
            X = data.get("X", None)
            y = data.get("y", None)
            all_X = data.get("all_X", None)
            all_y = data.get("all_y", None)

            lim = min(max(lim, config["optim_length"]), SLC.stop)

            if isinstance(X, type(None)):
                failures = go.Figure()
            else:
                X = {key : numpy.array(value) for key, value in X.items()}
                y = {key : numpy.array(value) for key, value in y.items()}
                all_X = [{key : numpy.array(value) for key, value in a_X.items()} for a_X in all_X]
                all_y = [{key : numpy.array(value) for key, value in a_y.items()} for a_y in all_y]

                failures = plot_mean_failures(config, all_X, all_y, idx=idx, total=len(checklist), fig=failures, lim=lim, name="_".join(path.split("/")[-1].split("_")[:2]))

        for fig in [failures]:
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                margin=go.layout.Margin(
                    l=10, #left margin
                    r=10, #right margin
                    b=50, #bottom margin
                    t=50  #top margin
                )
            )
            fig.update_xaxes(
                color="white", showgrid=False
            )
            fig.update_yaxes(
                color="white", showgrid=False
            )
        return failures

    return app

if __name__ == '__main__':
    app = create_app(PATH)
    app.run_server(debug=True)
