
import numpy
import plotly.graph_objects as go
import matplotlib

from matplotlib import pyplot
from plotly.subplots import make_subplots
from plotly import express

from . import utils

COLOR = "#87cdde"

cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    name="space",
#     colors=["#45d7cd", "#ffc949"]
    colors=["#87cdde", "#ff5554"]
#     colors=["#ffc949", "#ff5554"]
#     colors = ["#1b5bdb", "#ffc949"]
)
matplotlib.cm.register_cmap(cmap=cmap)
matplotlib.cm.register_cmap(cmap=cmap.reversed())

OBJ_PLOT_LIM = {
    "Resolution" : (0, 250),
    "Squirrel" : (0, None),
    "Bleach" : (0, 1),
    "SNR" : (0, 2.0),
    "FFTMetric" : (0, 1.0),
    "Crosstalk" : (0, 1.0),
}

def show_images(config, conf1, sted_image, conf2, idx=0, **kwargs):
    """
    Plots the given data in 1D

    :param X: A `dict` of the selected parameters
    :param y: A `dict` of the objectives

    :returns : A `plotly.Figure`
    """

    conf1, sted_image, conf2 = map(numpy.array, (conf1, sted_image, conf2))

    fig = make_subplots(1, 3, subplot_titles=["conf1", "sted_image", "conf2"])

    fig.add_trace(
        go.Heatmap(z=conf1, zmin=0, zmax=0.7 * conf1.max(), colorscale='hot', showscale=False), row=1, col=1
    )
    if sted_image.ndim == 4:
        sted_image = sted_image[-1]
    fig.add_trace(
        go.Heatmap(z=sted_image[-1], zmin=0, colorscale='hot'), row=1, col=2
    )
    fig.add_trace(
        go.Heatmap(z=conf2, zmin=0, zmax=0.7 * conf1.max(), colorscale='hot', showscale=False), row=1, col=3
    )
    fig.update_yaxes(
        scaleanchor = "x",
        matches = "y",
        scaleratio = 1,
        visible = False,
    )
    fig.update_xaxes(
        scaleanchor = "y",
        matches = "x",
        scaleratio = 1,
        visible = False,
    )

    fig.update_traces(
        colorbar={
            "orientation" : "v",
            "thickness" : 10,
            "len" : 0.4,
            "nticks" : 2,
            "x" : 0.35,
            "y" : 0.725,
        }
    )

    return fig

def plot_failures(config, X, y, idx=0, total=1, show_all=False, **kwargs):
    """
    Plots the given data in 1D

    :param X: A `dict` of the selected parameters
    :param y: A `dict` of the objectives

    :returns : A `plotly.Figure`
    """
    cmap = pyplot.cm.get_cmap("space", total)

    def plot(fig, objs, idx):
        fig.add_trace(
            go.Scatter(x=numpy.arange(len(objs)), y=objs, mode="lines", line={"color" : matplotlib.colors.to_hex(cmap(idx))})
        )

    fig = make_subplots()

    if show_all:
        lim = 0
        for idx, _y in enumerate(y):
            objs = [~utils.isin_bounds(values, obj_name) for obj_name, values in _y.items()]
            objs = numpy.cumsum(numpy.any(objs, axis=0))
            lim = max(lim, kwargs.get("lim", len(objs)))
            plot(fig, objs, idx)
    else:
        objs = [~utils.isin_bounds(values, obj_name) for obj_name, values in y[idx].items()]
        objs = numpy.cumsum(numpy.any(objs, axis=0))
        lim = kwargs.get("lim", len(objs))
        plot(fig, objs, idx)

    fig.add_trace(
        go.Scatter(x=[0, lim], y=[0, lim], line={"dash": "dash", "color" : "silver"})
    )
    fig.update_layout(
        # title="Cummulative failures",
        xaxis_title="Total image (count)",
        yaxis_title="Failures (count)",
        showlegend=False
    )
    fig.update_yaxes(
        range=(0, lim)
    )
    fig.update_xaxes(
        range=(0, lim)
    )

    return fig

def plot_mean_failures(config, X, y, idx=0, total=1, show_all=False, fig=None, **kwargs):
    """
    Plots the given data in 1D

    :param X: A `dict` of the selected parameters
    :param y: A `dict` of the objectives

    :returns : A `plotly.Figure`
    """
    cmap = pyplot.cm.get_cmap("space", total)
    lim = kwargs.get("lim", 200)

    def plot(fig, name, mean, std, idx):
        fig.add_trace(
            go.Scatter(x=numpy.arange(len(std)), y=mean - std, mode="lines", fill="none", line={"color" : matplotlib.colors.to_hex(cmap(idx)), "width" : 0}, hoverinfo='skip', showlegend = False)
        )
        fig.add_trace(
            go.Scatter(x=numpy.arange(len(std)), y=mean + std, mode="lines", fill="tonexty", line={"color" : matplotlib.colors.to_hex(cmap(idx)), "width" : 0}, hoverinfo='skip', showlegend = False)
        )
        fig.add_trace(
            go.Scatter(x=numpy.arange(len(std)), y=mean, mode="lines", line={"color" : matplotlib.colors.to_hex(cmap(idx))}, name=name)
        )

    if isinstance(fig, type(None)):
        fig = make_subplots()

    cumsums = []
    for _y in y:
        objs = [~utils.isin_bounds(values, obj_name) for obj_name, values in _y.items()]
        if (len(y) > 1) and (len(objs[0]) != lim):
            break
        objs = numpy.cumsum(numpy.any(objs, axis=0))
        cumsums.append(objs)

    if cumsums:
        plot(fig, kwargs.get("name", None), numpy.mean(cumsums, axis=0), numpy.std(cumsums, axis=0), idx)

    fig.add_trace(
        go.Scatter(x=[0, lim], y=[0, lim], line={"dash": "dash", "color" : "silver"}, showlegend = False)
    )
    fig.update_layout(
        # title="Cummulative failures",
        xaxis_title="Total image (count)",
        yaxis_title="Failures (count)",
        showlegend=True
    )
    fig.update_yaxes(
        range=(0, lim)
    )
    fig.update_xaxes(
        range=(0, lim)
    )

    return fig

def plot_objectives(config, X, y, _type="line", step=10):
    """
    Plots the given objectives

    :param X: A `dict` of the selected parameters
    :param y: A `dict` of the objectives

    :returns : A `plotly.Figure`
    """
    fig = make_subplots(1, len(config["obj_names"]), subplot_titles=config["obj_names"])
    fig.update_layout(
        # title="Objectives",
        showlegend=False
    )
    for i, obj_name in enumerate(config["obj_names"]):
        if _type == "line":
            fig.add_trace(
                go.Scatter(x=numpy.arange(len(y[obj_name])), y=y[obj_name], mode="lines", line={"color" : COLOR}), row=1, col=i + 1
            )
        elif _type == "boxplot":
            values = y[obj_name]
            for idx in range(0, len(values), step):
                vals = values[idx : idx + step]
                fig.add_trace(go.Box(x=[idx] * len(vals), y=vals, fillcolor=COLOR, line={"color" : "white"}), row=1, col=i + 1)
        else:
            pass
        fig.update_yaxes(
            range=OBJ_PLOT_LIM[obj_name], row=1, col=i+1
        )
    return fig

def plot_parameters(config, X, y, ndims):
    """
    Plots the given parameters

    :param X: A `dict` of the selected parameters
    :param y: A `dict` of the objectives

    :returns : A `plotly.Figure`
    """
    fig = make_subplots(1, sum(ndims), subplot_titles=[config["param_names"][i] for i in range(len(config["param_names"])) for _ in range(ndims[i])])
    for i in range(len(config["param_names"])):
        for j in range(ndims[i]):
            idx = sum(ndims[:i]) + j
            fig.add_trace(
                go.Scatter(
                    x=numpy.arange(len(X[config["param_names"][i]])), y=X[config["param_names"][i]][:, j],
                    mode="lines", line={"color" : COLOR}
                ), row=1, col=idx + 1
            )
            fig.update_yaxes(
                range=(config["x_mins"][i], config["x_maxs"][i]), row=1, col=idx + 1
            )
    fig.update_layout(
        # title="Parameters",
        showlegend=False,
        autosize=True,
    )
    return fig
