
import numpy
from functools import partial
from banditopt import algorithms, utils
from tqdm import trange
import time

import pandas as pd
import array
import random
from math import sqrt
from deap import base
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools

def find_dominating(idx, points, objectives, minimize=True):
    """
    Verfies wheter a single point is dominated by any other points, if so the
    dominating point is returned

    :param idx: An `int` of the index of the point
    :param points: A `numpy.ndarray` of points to compare to
    :param objectives: A `list` of objectives
    :param minimize: (bool) Whether minimization should be considered

    :return : A `numpy.ndarray` of a single point
    """
    point = points[idx]
    is_dominated = []
    for i, obj in enumerate(objectives):
        if obj.select_optimal == numpy.argmin:
            is_dominated.append(point[i] > points[:, i])
        else:
            is_dominated.append(point[i] < points[:, i])
    is_dominated = numpy.all(numpy.array(is_dominated).T, axis=1)
    if not numpy.any(is_dominated):
        return idx
    dominating = numpy.argwhere(is_dominated).ravel()
    return numpy.random.choice(dominating)

class ParetoSampler():
    """
    A class to sample the parameter space in a pareto-optimal way.

    There are multiple modes available:
    - `default` : Samples the entire parameter space
    - `nsga` : Samples the parameter space using NSGA-II
    - `grid` : Samples the parameter space using a grid
    - `uniform` : Samples the parameter space using a uniform sampling
    """
    def __init__(self, mode, obj_dict, config, x_mins, x_maxs, n_points, ndims, *args, **kwargs):

        self.mode = mode
        assert mode in ["default", "nsga", "grid", "uniform"], f"The selected mode `{mode}` does not exist"
        self.obj_dict = obj_dict
        self.config = config

        self.ndims = ndims
        self.x_mins = [float(x_mins[i]) for i, ndim in enumerate(self.ndims) for _ in range(ndim)]
        self.x_maxs = [float(x_maxs[i]) for i, ndim in enumerate(self.ndims) for _ in range(ndim)]
        self.n_points = [n_points[i] for i, ndim in enumerate(self.ndims) for _ in range(ndim)]

        if self.mode in ["default", "grid"]:
            grids = numpy.meshgrid(*[numpy.linspace(self.x_mins[i], self.x_maxs[i], self.n_points[i]) for i in range(len(self.x_mins))])
            self.X = numpy.hstack([grid.ravel()[:,numpy.newaxis] for grid in grids])
        else:
            self.X = numpy.random.uniform(self.x_mins, self.x_maxs, size=(10000, len(self.x_maxs)))
            self.pop = None
            self.NSGAII_kwargs = kwargs.get("NSGAII_kwargs", {
                "NGEN" : 50, "MU" : 100, "L" : 10, "percent-replace" : 0.3,
            })

    def __call__(self, iter_idx, algos, with_time, *args, **kwargs):
        """
        Generic `__call__` method of the `ParetoSampler`

        :param iter_idx: An `int` of the iteration index
        :param algos: A `list` of algorithms to optimize
        :param with_time: A `bool` whether time is being optimized

        :returns : A `numpy.ndarray` of the selected parameters
                   A `list` of the objectives evaluated at the selected parameters
                   A `numpy.ndarray` of the evaluated objectives
                   A `numpy.ndarray` of the time per pixel
                   A `numpy.ndarray` of the non-dominated parameters (if required)
        """
        func = getattr(self, f"_{self.mode}_sample")

        # Sets the algorithms in sampling mode
        for algo in algos:
            algo.set_sampling_mode(True)

        X, y_samples, points, timesperpixel, ndf = func(algos, with_time, *args, **kwargs)

        # Sets the algorithms in normal mode
        for algo in algos:
            algo.set_sampling_mode(False)

        # When sampling the first points, we return the default points
        if iter_idx < self.config["knowledge_opts"]["num_random_samples"]:
            return self.X, y_samples, points, timesperpixel, ndf
        return X, y_samples, points, timesperpixel, ndf

    def _default_sample(self, algos, with_time, *args, **kwargs):
        """
        Implements the `default` sampling of the parameter space.

        This corresponds to a `grid` sampling.

        :param algos: A `list` of algorithms to optimize
        :param with_time: A `bool` whether time is being optimized

        :returns : A `numpy.ndarray` of the selected parameters
                   A `list` of the objectives evaluated at the selected parameters
                   A `numpy.ndarray` of the evaluated objectives
                   A `numpy.ndarray` of the time per pixel
                   A `numpy.ndarray` of the non-dominated parameters (if required)
        """
        # Samples the ys
        y_samples = numpy.array([algo.sample(self.X) for algo in algos])

        # Create the timesperpixel
        if "pdt" not in self.config["param_names"]:
            timesperpixel = numpy.ones((self.X.shape[0], 1)) * self.config["default_values_dict"]["pdt"]
            timesperpixel = timesperpixel.flatten()
        else:
            col = self.config["param_names"].index("pdt")
            timesperpixel = self.X[:,col]

        # Create points
        if with_time:
            points = numpy.concatenate([y_samples[i] if self.obj_dict[self.config["obj_names"][i]].select_optimal==numpy.argmax else -y_samples[i] for i in range(len(y_samples))]+[-timesperpixel[:,numpy.newaxis]], axis=1)
        else:
            points = numpy.concatenate([y_samples[i] if self.obj_dict[self.config["obj_names"][i]].select_optimal==numpy.argmax else -y_samples[i] for i in range(len(y_samples))], axis=1)

        return self.X, y_samples, points, timesperpixel, numpy.arange(self.X.shape[0])

    def _grid_sample(self, algos, with_time, *args, **kwargs):
        """
        Implements a `grid` sampling of the parameter space followed by a pareto
        front calculation

        :param algos: A `list` of algorithms to optimize
        :param with_time: A `bool` whether time is being optimized

        :returns : A `numpy.ndarray` of the selected parameters
                   A `list` of the objectives evaluated at the selected parameters
                   A `numpy.ndarray` of the evaluated objectives
                   A `numpy.ndarray` of the time per pixel
                   A `numpy.ndarray` of the non-dominated parameters (if required)
        """
        # Samples the ys
        y_samples = numpy.array([algo.sample(self.X) for algo in algos])

        # Create the timesperpixel
        if "pdt" not in self.config["param_names"]:
            timesperpixel = numpy.ones((self.X.shape[0], 1)) * self.config["default_values_dict"]["pdt"]
            timesperpixel = timesperpixel.flatten()
        else:
            col = self.config["param_names"].index("pdt")
            timesperpixel = self.X[:,col]

        # Create points
        if with_time:
            points = numpy.concatenate([y_samples[i] if self.obj_dict[self.config["obj_names"][i]].select_optimal==numpy.argmax else -y_samples[i] for i in range(len(y_samples))]+[-timesperpixel[:,numpy.newaxis]], axis=1)
        else:
            points = numpy.concatenate([y_samples[i] if self.obj_dict[self.config["obj_names"][i]].select_optimal==numpy.argmax else -y_samples[i] for i in range(len(y_samples))], axis=1)
        weights = [+1 if self.obj_dict[obj_name].select_optimal==numpy.argmax else -1 for obj_name in self.config["obj_names"]]
        if with_time:
            weights += [-1]

        # Computes the pareto front
        ndf = utils.pareto_front(points=points, weights=weights)
        X_sample = self.X[ndf,:]
        y_samples = [y[ndf] for y in y_samples]
        timesperpixel = timesperpixel[ndf]

        return X_sample, y_samples, points, timesperpixel, ndf

    def _uniform_sample(self, algos, with_time, *args, **kwargs):
        """
        Implements a `uniform` sampling of the parameter space

        :param algos: A `list` of algorithms to optimize
        :param with_time: A `bool` whether time is being optimized

        :returns : A `numpy.ndarray` of the selected parameters
                   A `list` of the objectives evaluated at the selected parameters
                   A `numpy.ndarray` of the evaluated objectives
                   A `numpy.ndarray` of the time per pixel
                   A `numpy.ndarray` of the non-dominated parameters (if required)
        """
        X_sample = numpy.random.uniform(self.x_mins, self.x_maxs, size=(10000, len(self.x_maxs)))

        # Sample the ys
        y_samples = [algo.sample(X_sample) for i, algo in enumerate(algos)]

        # Create points
        if with_time:
            points = numpy.concatenate([y_samples[i] if self.obj_dict[self.config["obj_names"][i]].select_optimal==numpy.argmax else -y_samples[i] for i in range(len(y_samples))]+[-timesperpixel[:,numpy.newaxis]], axis=1)
        else:
            points = numpy.concatenate([y_samples[i] if self.obj_dict[self.config["obj_names"][i]].select_optimal==numpy.argmax else -y_samples[i] for i in range(len(y_samples))], axis=1)

        # Create the timesperpixel
        if "pdt" not in self.config["param_names"]:
            timesperpixel = numpy.ones((X_sample.shape[0], 1)) * self.config["default_values_dict"]["pdt"]
            timesperpixel = timesperpixel.flatten()
        else:
            col = self.config["param_names"].index("pdt")
            timesperpixel = X_sample[:,col]

        return X_sample, y_samples, points, timesperpixel, numpy.arange(len(X_sample))

    def _nsga_sample(self, algos, with_time, *args, **kwargs):
        """
        Implements a `NSGA-II` sampling of the parameter space

        :param algos: A `list` of algorithms to optimize
        :param with_time: A `bool` whether time is being optimized

        :returns : A `numpy.ndarray` of the selected parameters
                   A `list` of the objectives evaluated at the selected parameters
                   A `numpy.ndarray` of the evaluated objectives
                   A `numpy.ndarray` of the time per pixel
                   A `numpy.ndarray` of the non-dominated parameters (if required)
        """
        sampler = algorithms.MO_function_sample(algos, with_time, self.config["param_names"], *args, **kwargs)
        nsga_weights = [+1 if self.obj_dict[obj_name].select_optimal==numpy.argmax else -1 for obj_name in self.config["obj_names"]]
        if with_time:
            nsga_weights += [-1]

        # Gets the conditions if any
        conditions = kwargs.get("conditions", [])
        # X_sample, logbook, ngens = utils.NSGAII(sampler.evaluate, self.x_mins, self.x_maxs, nsga_weights, min_std=numpy.sqrt(2e-4 * len(self.config["obj_names"])), conditions=conditions, **self.NSGAII_kwargs)
        X_sample, logbook, ngens, self.pop = utils.NSGAII(
            sampler.evaluate, self.x_mins, self.x_maxs, nsga_weights,
            min_std=numpy.sqrt(2e-4 * len(self.config["obj_names"])),
            conditions=conditions, verbose=False, **self.NSGAII_kwargs,
            pop = self.pop
        )

        # Sample the ys
        y_samples = [algo.sample(X_sample, seed=sampler.seeds[i], *args, **kwargs) for i, algo in enumerate(algos)]

        # Create the timesperpixel
        if "pdt" not in self.config["param_names"]:
            timesperpixel = numpy.ones((X_sample.shape[0], 1)) * self.config["default_values_dict"]["pdt"]
            timesperpixel = timesperpixel.flatten()
        else:
            col = self.config["param_names"].index("pdt")
            timesperpixel = X_sample[:,col]

        # Create points
        if with_time:
            points = numpy.concatenate([y_samples[i] if self.obj_dict[self.config["obj_names"][i]].select_optimal==numpy.argmax else -y_samples[i] for i in range(len(y_samples))]+[-timesperpixel[:,numpy.newaxis]], axis=1)
        else:
            points = numpy.concatenate([y_samples[i] if self.obj_dict[self.config["obj_names"][i]].select_optimal==numpy.argmax else -y_samples[i] for i in range(len(y_samples))], axis=1)

        return X_sample, y_samples, points, timesperpixel, numpy.arange(len(X_sample))
