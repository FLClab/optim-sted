
import numpy
import json
import os

from matplotlib import pyplot
from scipy.spatial import distance

from .user import select
from .prefnet import PrefNet
from .pareto import find_dominating

OBJ_NAMES = {
    "FWHM (nm)" : "Resolution",
    "Signal Ratio" : "SNR",
    "Bleach" : "Bleach",
    "Squirrel" : "Squirrel"
}

class PreferenceArticulator:
    """
    Implements a `PreferenceArticulator` object. The `PreferenceArticulator` is
    responsible to make the tradeoff between the objectives
    """
    def __init__(self, config, mode, model_config=None):
        """
        Instantiates the `PreferenceArticulator`

        :param config: A `dict` of the configuration
        :param mode: A `str` of the mode to select
        :param model_config: A `dict` of the model configuration
        """
        self.config = config
        self.mode = mode
        assert mode in ["random", "optim", "region", "prefnet"], f"The selected mode `{mode}` does not exist"

        # Loads the prefnet model
        if self.mode == "prefnet":
            self.model = PrefNet(**model_config)
            self.model = self.model.loading(**model_config)
            self.model.eval()
            self.config = json.load(open(model_config["config_path"], "r"))
        elif self.mode == "region":
            if "SNR" in config["obj_names"]:
                self.regions = json.load(open(os.path.join(os.path.dirname(__file__), "articulation", "preferences_resolution-snr-bleach.json"), "r"))
            else:
                self.regions = json.load(open(os.path.join(os.path.dirname(__file__), "articulation", "preferences.json"), "r"))

    def __call__(self, thetas, objectives, with_time, times, *args, **kwargs):
        """
        Implements the generic `__call__` method of the `PreferenceArticulator`.
        When called the appropriate preference articulation method is selected

        :param thetas: A `numpy.ndarray` of options sampled from the algorithms.
        :param objectives: A list of objectives name.
        :param with_time: (bool) Wheter of not to consider *times* as an objective.
        :param times: An array of time for acquiring an image using each configuration in *thetas*.

        :returns : An `int` of the selected index
        """
        func = getattr(self, f"_{self.mode}_articulation")
        return func(thetas, objectives, with_time, times, *args, **kwargs)

    def _random_articulation(self, thetas, objectives, with_time, times, *args, **kwargs):
        """
        Implements the `random` mode. 
        
        It selects a random points from the options

        :param thetas: A `numpy.ndarray` of options sampled from the algorithms.
        :param objectives: A list of objectives name.
        :param with_time: (bool) Wheter of not to consider *times* as an objective.
        :param times: An array of time for acquiring an image using each configuration in *thetas*.

        :returns : An `int` of the selected index
        """
        selected_index = numpy.random.choice(len(thetas[0]))
        thetas = numpy.array(thetas).squeeze().T
        updated_index = find_dominating(selected_index, thetas, objectives)
        return updated_index, selected_index

    def _optim_articulation(self, thetas, objectives, with_time, times, *args, **kwargs):
        """
        Implements the `optim` mode.
        It asks the user to select a point from a cloud of points.

        :param thetas: A `numpy.ndarray` of options sampled from the algorithms.
        :param objectives: A list of objectives name.
        :param with_time: (bool) Wheter of not to consider *times* as an objective.
        :param times: An array of time for acquiring an image using each configuration in *thetas*.

        :returns : An `int` of the selected index
        """
        selected_index = None
        while isinstance(selected_index, type(None)):
            selected_index = select(thetas, objectives, with_time, times, *args, **kwargs)
        thetas = numpy.array(thetas).squeeze().T
        updated_index = find_dominating(selected_index, thetas, objectives)
        return updated_index, selected_index

    def _region_articulation(self, thetas, objectives, with_time, times, *args, **kwargs):
        """
        Implements the `region` mode. 

        .. warning::
            This is an experimental mode.
        
        It selects a point based on pre-defined regions by an expert user. 
        The regions are cuboids where the maximum desired objectives is specified (case of minimization).
        The point that is within the region and farthest from the cuboid corner
        is selected.

        :param thetas: A `numpy.ndarray` of options sampled from the algorithms.
        :param objectives: A list of objectives name.
        :param with_time: (bool) Wheter of not to consider *times* as an objective.
        :param times: An array of time for acquiring an image using each configuration in *thetas*.

        :returns : An `int` of the selected index
        """
        selected_point = None
        for region in self.regions:
            is_contained = numpy.ones(len(thetas[0]), dtype=bool)
            obj_values = []
            for obj, theta in zip(objectives, thetas):
                obj_name = OBJ_NAMES[obj.label]

                # We invert in cases where the objective should be maximized
                factor = -1. if obj_name in ["SNR"] else 1.

                contained = factor * theta.squeeze() <= factor * region[obj_name]
                is_contained = numpy.logical_and(is_contained, contained)

                obj_values.append(region[obj_name])

            obj_values = numpy.array(obj_values)[numpy.newaxis]
            if numpy.any(is_contained):
                # print(region)
                indices = numpy.argwhere(is_contained).ravel()

                available_points = (numpy.array([theta[indices].ravel() for theta in thetas]).T - obj_values) / (obj_values + 1e-3)
                distances = distance.cdist((obj_values - obj_values) / (obj_values + 1e-3), available_points)
                selected_index = indices[numpy.argmax(distances)]
                selected_point = numpy.array([theta[selected_index] for theta in thetas])
                break

        # _ = select(thetas, objectives, with_time, times, selected_point=selected_point, *args, **kwargs)
        thetas = numpy.array(thetas).squeeze().T
        updated_index = find_dominating(selected_index, thetas, objectives)
        return updated_index, selected_index

    def _prefnet_articulation(self, thetas, objectives, with_time, times, *args, **kwargs):
        """
        Implements the `prefnet` mode. 
        
        It asks a `PrefNet` model which point should be selected from the cloud of points.

        :param thetas: A `numpy.ndarray` of options sampled from the algorithms.
        :param objectives: A list of objectives name.
        :param with_time: (bool) Wheter of not to consider *times* as an objective.
        :param times: An array of time for acquiring an image using each configuration in *thetas*.

        :returns : An `int` of the selected index
        """
        # Converts to numpy ndarray and resize
        # _thetas = numpy.array(thetas).copy()
        thetas = numpy.array(thetas)[...,0].T

        # Rescales the data appropriately
        thetas = (thetas - self.config["train_mean"]) / self.config["train_std"]

        # Predicts the objectives
        scores = self.model.predict(thetas)

        # index = numpy.argmax(scores)
        # thetas = thetas * self.config["train_std"] + self.config["train_mean"]
        # fig, ax = pyplot.subplots(figsize=(5,5))
        # ax.scatter(thetas[index, 0], thetas[index, 1], c="#cc0000", s=250, marker="*")
        # sc = ax.scatter(thetas[:, 0], thetas[:, 1], c=thetas[:, 2])
        # fig.colorbar(sc, ax=ax)
        # pyplot.show(block=True)

        selected_index = numpy.argmax(scores)

        # selected_point = numpy.array([theta[selected_index] for theta in _thetas])
        # manual_selected_index = select(_thetas, objectives, with_time, times, selected_point=selected_point, *args, **kwargs)
        # if not isinstance(manual_selected_index, (type(None))):
        #     print(f"Manual override: {selected_index} -> {manual_selected_index}")
        #     selected_index = manual_selected_index

        updated_index = find_dominating(selected_index, thetas, objectives)
        # if updated_index != selected_index:
        #     print(f"Selected index dominated : {selected_index} -> {updated_index}")
        return updated_index, selected_index
