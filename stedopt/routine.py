
import os
import json
import random
import glob

from collections import  defaultdict

from banditopt import algorithms

from .tools import ParameterSpaceGenerator
from .configuration import Configurator

class RoutineGenerator:
    """
    Creates a `RoutineGenerator` that allows to modify the configuration of the
    fluorescence or the microscope.

    There are two modes that are available to the user a `random` and `select`.
    - The random mode simply samples from the available modes
    - The select mode allows the user to specify a mode
    """
    def __init__(self, config, mode, routine_id=0, *args, **kwargs):
        """
        Instantiates a `RoutineGenerator`

        :param config: A `dict` of the configuration
        :param mode: A `str` of the selection mode
        :param routine_id: An `int` of the routine id (necessary with `select` mode)
        """
        self.config = config
        self.mode = mode
        self.routine_id = routine_id

        assert isinstance(routine_id, int), "Routine ID is not supported. Must be an `int`"

        if self.mode != "default":
            routine_file = kwargs.get("routine_file", "routines.json")
            self.routines = json.load(open(os.path.join(os.path.dirname(__file__), "routines", routine_file), "r"))
            # Some keys need to be of type int
            for routine in self.routines:
                for key in ["sigma_abs", "sigma_ste"]:
                    tmp = {}
                    for to_convert, value in routine["fluo"][key].items():
                        tmp[int(to_convert)] = value
                    routine["fluo"][key] = tmp

            if self.mode == "random":
                self.routine_id = random.choice(range(len(self.routines)))
            self.routine = self.routines[self.routine_id]
        else:
            self.routine = {}

    def generate(self, *args, **kwargs):
        return self.routine

class RoutineSelector:
    """
    Attempts to select a reliable model from the available models.

    A reliable model is a model that has been trained on a set of data and
    has a good predictive performance on the given task.

    .. reference::
        Saber, H., Saci, L., Maillard, O.-A. & Durand, A. Routine Bandits: Minimizing Regret on Recurring Problems. in Machine Learning and Knowledge Discovery in Databases. Research Track (eds. Oliver, N., Pérez-Cruz, F., Kramer, S., Read, J. & Lozano, J. A.) 3–18 (Springer International Publishing, Cham, 2021). doi:10.1007/978-3-030-86486-6_1.
    """
    def __init__(self, config, regressor, models, use=False, *args, **kwargs):
        """
        Instantiates a `RoutineSelector`

        :param config: A `dict` of the configuration
        :param regressor: A reference to the regressor object
        :param models: A `list` of model path to load from
        """
        self.config = config
        self.regressor = regressor
        self.use = use
        if self.use:
            self.models = self.load_models(models)
        else:
            self.models = {}

    def load_models(self, models):
        """
        Loads multiple models

        :param models: A `list` of models
        :return: A `defaultdict` containing the loaded models
        """
        loaded_models = defaultdict(list)
        for model in models:
            # Load model configuration
            config = Configurator(os.path.join(model, "config.yml"), load_base=False)

            multi_models = glob.glob(os.path.join(model, "**", "*.ckpt"), recursive=True)
            trials = set([model.split(os.path.sep)[-2] for model in multi_models])
            for trial in trials:
                for obj_name in self.config["obj_names"]:
                    args = self.config["regressor_args"]["default"].copy()
                    for key, value in self.config["regressor_args"][obj_name].items():
                        args[key] = value
                    algo = algorithms.TS_sampler(self.regressor(**args))
                    algo.load_ckpt(path=model, prefix=obj_name, trial=trial)

                    loaded_models[obj_name].append({"name" : os.path.join(model, trial), "algo" : algo})
        return loaded_models

    def get_reliable(self, algos):
        """
        Samples a reliable model from the available models

        :param algos: A `list` of default models
        :return: A `list` of reliable models
        """
        if not self.use:
            return algos
        reliable = []
        for i, obj_name in enumerate(self.config["obj_names"]):
            available_algos = self.models[obj_name]
            print(f"[----] {obj_name}: {len(available_algos)} available models")
            if len(available_algos) > 0:
                reliable.append(random.choice(self.models[obj_name])["algo"])
            else:
                reliable.append(algos[i])
        return reliable

    def clear_unreliable(self, X, y):
        """
        Clears the unreliable models from the list of available models. A model
        is considered unreliable if an acquired objective point with a given
        parameterization lies outside of the confidence of a model.

        :param X: A `list` of input parameterizations
        :param y: A `list` of acquired objective points
        """
        if not self.use:
            return
        for obj_name, value in zip(self.config["obj_names"], y):
            remove_indices = []
            for i, algo in enumerate(self.models[obj_name]):
                mean, std = algo["algo"].predict(X)
                if (value <= (mean - 3 * std).item()) or (value >= (mean + 3 * std).item()):
                    print(obj_name, "{:0.4f}".format(mean.item()), "{:0.4f}".format(std.item()), "{:0.4f}".format(value))
                    # Out of distribution we clear the model
                    remove_indices.append(i)
            for i in reversed(remove_indices):
                print(f"[!!!!] {obj_name}: Removing {self.models[obj_name][i]['name']}")
                del self.models[obj_name][i]
