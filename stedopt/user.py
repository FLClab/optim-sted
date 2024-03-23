
import numpy
import matplotlib

from matplotlib import pyplot

class PickIndex:
    def __init__(self, idx):
        self.idx = idx

def select(thetas, objectives, with_time, times, figsize=(8, 8), borders=None, *args, **kwargs):
    """Asks the user to select the best option by clicking on the points from the
    :mod:`matplotlib` figure. If several points overlap, select the one that minimizes
    the time (or third objective).

    :param thetas: A 2d-array of options sampled from the algorithms.
    :param objectives: A list of objectives name.
    :param with_time: (bool) Wheter of not to consider *times* as an objective.
    :param times: An array of time for acquiring an image using each configuration in *thetas*.
    :param figsize: The size of figure to display.
    :return: The index of the selected point.
    """
    index = PickIndex(None)

    def onpick(event):
        """Handles the event from the :mod:`matplotlib` to select the points. It
        also handles the situation where several points overlap.

        :param event: A `matplotlib.backend_bases.Event`.
        """
        candidates = event.ind
        index.idx = numpy.random.choice(candidates)

    def update_annotation(ind):
        """
        Updates the annotation text box

        :param ind: An hovered point
        """
        idx = ind["ind"][0]
        pos = sc.get_offsets()[idx]
        annotation.xy = pos

        text = ""
        for i in range(len(objectives)):
            text += "{}: {:0.2f}".format(objectives[i].label, thetas[i][idx].item())
            if i != len(objectives) - 1:
                text += "\n"
        # text = "{}: {:0.2f}\n{}: {:0.2f}\n{}: {:0.2f}".format(
        #     objectives[0].label, thetas[0][idx].item(),
        #     objectives[1].label, thetas[1][idx].item(),
        #     objectives[2].label, thetas[2][idx].item())

        annotation.set_text(text)
        # annotation.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
        annotation.get_bbox_patch().set_facecolor("gray")
        annotation.get_bbox_patch().set_alpha(0.7)

    def hover(event):
        """
        Handles the event from the :mod:`matplotlib` to hover above points

        :param event: A `matplotlib.backend_bases.Event`
        """
        vis = annotation.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annotation(ind)
                annotation.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annotation.set_visible(False)
                    fig.canvas.draw_idle()

    fig, ax = pyplot.subplots(figsize=figsize)
    ax.grid(True)

    if with_time:
        title = ax.set_title(f"Pick the best option by clicking on the point. Tmin={numpy.min(times):0.2e}, Tmax={numpy.max(times):0.2e}")
    else:
        title = ax.set_title(f"Pick the best option by clicking on the point.")

    # 3 points tolerance
    if with_time:
        time_range = times.max() - times.min() + 1e-11

    cmap = pyplot.cm.get_cmap(kwargs.get("cmap", "rainbow"))
    if len(thetas)==3:
        if with_time:
            sc = ax.scatter(thetas[0], thetas[1], s=(times-times.min())/time_range * 60 + 20, c=thetas[2], marker="o", alpha=0.5, picker=3, cmap=cmap)
        else:
            sc = ax.scatter(thetas[0], thetas[1], c=thetas[2], marker="o", alpha=0.5, picker=3, cmap=cmap)#, vmin=0, vmax=1)
        cbar = pyplot.colorbar(sc, ax=ax)
        cbar.set_label(objectives[2].label)
    elif len(thetas)==2:
        if with_time:
            sc = ax.scatter(thetas[0], thetas[1], s=(times-times.min())/time_range * 60 + 20, marker="o", alpha=0.5, picker=3)
        else:
            sc = ax.scatter(thetas[0], thetas[1], marker="o", alpha=0.5, picker=3)
    ax.set(
        xlabel = objectives[0].label, ylabel = objectives[1].label
    )
    if borders is not None:
        borders = [(m , M) for m, M in zip(borders["mins"], borders["maxs"])]
        ax.set(
            xlim = borders[0], ylim = borders[1]
        )
        if len(borders)>=3:
            sc.set_clim(borders[2])

    selected_point = kwargs.get("selected_point", None)
    if isinstance(selected_point, numpy.ndarray):
        ax.scatter(selected_point[0], selected_point[1], c="gray", s=100, marker="x")

    # Creates an annotation text box
    annotation = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annotation.set_visible(False)

    fig.canvas.mpl_connect("motion_notify_event", hover)
    fig.canvas.mpl_connect("pick_event", onpick)
    pyplot.show(block=True)

    if isinstance(index.idx, type(None)):
        print("User did not select any points... Picking at random")
        return None#numpy.random.choice(len(thetas[0]))
    return index.idx
