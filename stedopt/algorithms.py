
import torch
import os

from banditopt import algorithms, models

# import sys
# sys.path.insert(0, "..")
from .defaults import regressors_dict

class AlgorithmsConfigurator:
    """
    Implements the configuration of the algorithms based on the given configuration file.
    """
    def __init__(self, config, **kwargs):
        """
        Instantiates the `AlgorithmsConfigurator` object

        :param config: A `dict` of the configuration file
        """
        self.config = config

        self.configurations = {}
        # Objectives use default values unless specified
        for i, name in enumerate(config["obj_names"]):
            kw = config["regressor_args"]["default"].copy()
            kw["datamap_opts"] = config["datamap_opts"]
            kw["idx"] = i
            for key, value in kwargs.items():
                kw[key] = value
            for key, value in config["regressor_args"][name].items():
                kw[key] = value
            self.configurations[name] = kw

        self.algos = self._base_build()
        self.algos = self._update_build(self.algos)

    def _base_build(self):
        """
        Base build of the algorithms from the configurations

        :returns : A `list` of algorithms
        """
        algos = []
        for key, values in self.configurations.items():
            algos.append(algorithms.TS_sampler(regressors_dict[self.config["regressor_name"]](**values)))
        return algos

    def _update_build(self, algos):
        """
        Updates the build of the algorithms

        :param algos: A `list` of algorithms to update

        :returns : An updated `list` of algorithms
        """
        # Update if encoder is shared across models
        if self.config["regressor_args"]["default"].get("share_ctx", False):
            context_encoder = models.ContextEncoder(**self.config["datamap_opts"], **self.config["regressor_args"]["default"])
            if torch.cuda.is_available():
                context_encoder = context_encoder.cuda()
            for algo in algos:
                algo.regressor.model.context_encoder = context_encoder

        # Loads pretrained context encoder if necessary
        pretrained_opts = self.config["regressor_args"]["default"].get("pretrained_opts", {"use" : False})
        teacher_opts = self.config["regressor_args"]["default"].get("teacher_opts", {"use" : False})
        if pretrained_opts["use"]:
            path = pretrained_opts["path"]
            trial = pretrained_opts["trial"]
            for obj_name, algo in zip(self.config["obj_names"], algos):

                algo.regressor.model.load_pretrained(
                    path=os.path.join(path, "models", str(trial), f"{obj_name}_model.ckpt")
                )

                # Load all parameters
                if pretrained_opts["load_all"]:
                    model = torch.load(os.path.join(path, "models", str(trial), f"{obj_name}_model.ckpt"), map_location=lambda storage, loc: storage)
                    algo.regressor.set_U(model.U)

                # Cases where teacher model is used    
                if teacher_opts["use"]:
                    algo.regressor.teacher_model.load_pretrained(
                        path=os.path.join(path, "models", str(trial), f"{obj_name}_model.ckpt")
                    )
        return algos

    def __getitem__(self, item):
        """
        Implements the `__getitem__` method of the class
        """
        return self.algos[item]

    def __len__(self):
        """
        Implements the `__len__` method of the class
        """
        return len(self.algos)
