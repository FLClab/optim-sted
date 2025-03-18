
import numpy
import copy
import random
import json
import os
import datetime
import h5py
import glob
import uuid
import tifffile

from functools import partial
from scipy import optimize, interpolate
from matplotlib import pyplot
from collections import defaultdict
from skimage import filters, transform, measure

from abberior import microscope, user, utils
from banditopt.utils import bigger_than, get_foreground
from tiffwrapper import imwrite

from .experiment import Experiment
from .create import create_microscope, create_dymin_microscope, create_rescue_microscope
from .configuration import Configurator
# from .pareto import bigger_than

PATH = "../data"

def get_resolution(emitter, positions=None, pixelsize=1., delta=8, avg=1):
    """
    Computes the resolution from center point in the image. A resolution of 0 is
    returned if no signal is available.

    :param emitter: A `numpy.ndarray` of emitter
    :param pixelsize: A `float` of the pixelsize
    :param delta: The center crop size

    :returns : A `float` of the resolution of the image
    """
    def gaussian(x,a,x0,sigma):
        return a*numpy.exp(-(x-x0)**2/(2*sigma**2))
    def fit(func, x, y):
        try:
            popt, pcov = optimize.curve_fit(func, x, y, bounds=((-numpy.inf, -numpy.inf, 0), numpy.inf))
            return popt[-1]
        except RuntimeError:
            return 0.

    # Returns the center resolution
    if isinstance(positions, type(None)):
        positions = [(emitter.shape[0] // 2, emitter.shape[1] // 2)]

    # Calculates the resolutions
    resolutions = []
    for ypos, xpos in positions:
        y = emitter[
            max(0, ypos - avg) : min(ypos + avg, emitter.shape[0]),
            max(0, xpos - delta) : min(xpos + delta, emitter.shape[1])
        ].sum(axis=0)
        x = numpy.arange(len(y)) - numpy.argmax(y)

        if y.sum() < 1:
            resolutions.append(0.)
            continue
        sigma = fit(gaussian, x, y)
        resolutions.append(2 * numpy.sqrt(2 * numpy.log(2)) * sigma * pixelsize)
    return numpy.quantile(resolutions, q=0.75)

def get_bleach(before, after):
    """
    Computes the bleach from the number of molecules before and after the imaging.

    :param before: A `numpy.ndarray` of the number of molecules
    :param after: A `numpy.ndarray` of the number of molecules left

    :returns : A `float` of the ratio of molecules left
    """
    where = before != 0
    init_mol = before[where].mean()
    ratio = after[where].mean() / init_mol
    return ratio

def datamap_generator(shape, sources, molecules, shape_sources=(1, 1), random_state=None):
    """
    Function to generate a datamap with randomly located molecules.
    :param shape: A tuple representing the shape of the datamap. If only 1 number is passed, a square datamap will be
                  generated.
    :param sources: Number of molecule sources to be randomly placed on the datamap.
    :param molecules: Average number of molecules contained on each source. The actual number of molecules will be
                      determined by poisson sampling.
    :param shape_sources : A `tuple` of the shape of the sources
    :param random_state: Sets the seed of the random number generator.
    :returns: A datamap containing the randomly placed molecules
    """
    numpy.random.seed(random_state)
    if type(shape) == int:
        shape = (shape, shape)
    datamap = numpy.zeros(shape)
    pos = []
    for i in range(sources):
        row, col = numpy.random.randint(0, shape[0] - shape_sources[0]), numpy.random.randint(0, shape[1] - shape_sources[1])
        datamap[row : row + shape_sources[0], col : col + shape_sources[1]] += numpy.random.poisson(molecules)
        pos.append([row + shape_sources[0] // 2, col + shape_sources[1] // 2])
    return datamap, numpy.array(pos)

class KnowledgeGenerator:
    """
    Creates a `KnowledgeGenerator` for the first acquired images at the begining
    of an optimization run. The knowledge generator handles `expert`, `pareto` or
    `random` knowledge.
    """
    def __init__(self, microscope, config, ndims, mode="expert", use=False, *args, **kwargs):
        """
        Instantiates the `KnowledgeGenerator`

        :param microscope: A `str` or `dict` of the microscope name
        :param config: A `Configurator` of the configuration
        :param ndims: A `list` of the dimensions associated with each parameter
        :param mode: A `str` of the mode of the knowledge generator
        :param use: A `bool` of whether to use the knowledge generator
        """
        self.microscope = microscope
        if isinstance(self.microscope, dict):
            self.microscope = self.microscope["mode"]
        self.config = config
        self.ndims = ndims
        self.use = use
        self.mode = mode

        self.expert_knowledge = kwargs.get("expert_knowledge", "")
        self.pareto_samples = kwargs.get("pareto_samples", [])
        if isinstance(self.pareto_samples, str):
            self.pareto_samples = [self.pareto_samples]

    def __call__(self, *args, **kwargs):
        """
        Implements a generic `__call__` method
        """
        if self.use:
            func = getattr(self, f"_{self.mode}_knowledge")
            return func(*args, **kwargs)
        return self._random_knowledge(*args, **kwargs)

    def _pareto_knowledge(self, *args, **kwargs):
        """
        Randomly samples from a pareto distribution of parameters. This assumes
        that the parameters are the same between Pareto and current optimization.

        :param X: A `numpy.ndarray` of possible parameters

        :returns : A `numpy.ndarray` of the sampled parameters
        """
        assert len(self.pareto_samples) > 0, "Pareto Samples options should be given"
        X = []
        for path in self.pareto_samples:
            with h5py.File(os.path.join(path, "optim.hdf5"), "r") as file:
                trial = str(max(map(int, file["X"].keys())))
                X.extend(file["X"][trial][()])
            config = Configurator(os.path.join(path, "config.yml"), load_base=False)
            assert config.param_names == self.config.param_names, f"Parameters mismatch between Pareto parameters {config.param_names} and current parameters {self.config.param_names}"
        choice = random.choice(X)
        return choice[:, numpy.newaxis]

    def _expert_knowledge(self, *args, **kwargs):
        """
        Randomly samples from an expert distribution.

        Assumes that the expert knowledge is stored in the `knowledge` folder

        The following structure is expected of the knowledge file:
        .. code-block:: python
        {
            "knowledge": [
                {
                    "p_sted": 0,
                    "p_ex": 0.000002,
                    "pdt": 0.00001
                },
                ...
            ]
        }

        :returns : A `numpy.ndarray` of the sampled parameters
        """
        if self.expert_knowledge:
            knowledge = json.load(open(os.path.join(os.path.dirname(__file__), "knowledge", f"{self.expert_knowledge}.json"), "r"))["knowledge"]
        else:
            knowledge = json.load(open(os.path.join(os.path.dirname(__file__), "knowledge", f"{self.microscope}.json"), "r"))["knowledge"]
        knowledge = random.choice(knowledge)
        param = []
        for i, param_name in enumerate(self.config.param_names):
            for _ in range(self.ndims[i]):
                # Ensures parameters are of float type
                param.append(float(knowledge[param_name]))
        return numpy.array(param)[:, numpy.newaxis]

    def _random_knowledge(self, X, *args, **kwargs):
        """
        Randomly samples from a uniform distribution

        :param X: A `numpy.ndarray` of possible parameters

        :returns : A `numpy.ndarray` of the sampled parameters
        """
        return X[numpy.random.randint(X.shape[0])][:, numpy.newaxis]


class DatamapGenerator:
    """
    Creates a `DatamapGenerator` that allows to create a datamap with a variable
    number of fluorescent molecules in each sources
    """
    def __init__(
        self, mode="generate", is_variable=False, shape=(50, 50), sources=10,
        molecules=(10, 100), shape_sources=(3, 3), random_state=None, path=None,
        molecules_scale=0.1, *args, **kwargs
    ):
        """
        Instantiates the `DatamapGenerator`

        :param mode: A `str` of the mode of the datamap generator
        :param is_variable: A `bool` wheter the number of molecules should be sampled
        :param shape: A `tuple` of the shape of the datamap
        :param sources: A `int` of the number of sources
        :param molecules: A `tuple` of the number of molecules to sample
        :param shape_sources: A `tuple` of the shape of the sources
        :param random_state: A `int` of the random state
        :param path: A `str` or a `list` of the path to the datamap
        :param molecules_scale: A `float` of the scale of the molecules
        """
        self.mode = mode
        assert self.mode in ["generate", "generatewithoutoverlap", "real-complete", "real", "real-filtered", "real-filtered-subset"]
        self.is_variable = is_variable
        self.molecules = molecules
        self.molecules_scale = molecules_scale
        self.shape = shape
        self.sources = sources
        self.shape_sources = shape_sources
        self.idx = None

        self.path = path
        if isinstance(self.path, str):
            self.path = [self.path]
        if isinstance(self.path, (list, tuple)) and ("real" in self.mode):
            self.datamaps = []
            for path in self.path:
                exclude = os.path.join(path, "exclude.json")
                files = sorted(glob.glob(os.path.join(path, "*.npy")))
                if os.path.isfile(exclude):
                    exclude_files = json.load(open(exclude, "r"))
                    files = list(filter(lambda file: not any([exclude in file for exclude in exclude_files]), files))
                self.datamaps.extend(files)

            # Removes datamaps with high variations
            statistics = []
            for datamap in self.datamaps:
                datamap = numpy.load(datamap)
                statistics.append({
                    "mean" : numpy.mean(datamap[datamap > 0.01]),
                    "std" : numpy.std(datamap[datamap > 0.01]),
                    "median" : numpy.median(datamap[datamap > 0.01]),
                    "quantiles" : numpy.diff(numpy.quantile(datamap[datamap > 0.01], [0.5, 0.75])),
                    "foreground" : get_foreground(datamap).sum() / datamap.size
                })
            median = numpy.median([stats["median"] for stats in statistics])
            quantiles = numpy.median([stats["quantiles"] for stats in statistics])
            if "real-filtered" in self.mode:
                self.datamaps = [
                    datamap for stats, datamap in zip(statistics, self.datamaps)
                    if (stats["median"] > median - quantiles) \
                    and (stats["median"] < median + quantiles) \
                    and (stats["foreground"] < 0.01) # Removes low-contrast datamaps
                ]
                if "subset" in self.mode:
                    # NOTE: The subset is not relevant anymore with the exclusion
                    self.datamaps = numpy.array(self.datamaps)[[2, 4, 7, 8, 12, 13, 22, 23, 27, 28, 31]]
            elif self.mode == "real":
                self.datamaps = [
                    datamap for stats, datamap in zip(statistics, self.datamaps)
                    if (stats["median"] > median - quantiles) \
                    and (stats["median"] < median + quantiles)
                ]
            else:
                # Complete dataset
                self.datamaps = [
                    datamap for stats, datamap in zip(statistics, self.datamaps)
                ]
            self.mode = "real"

        self.random = numpy.random.RandomState(random_state)

    def __call__(self, **kwargs):
        """
        Implements a generic `__call__` method
        """
        if isinstance(self.molecules, (list, tuple)):
            molecules = self.random.randint(*self.molecules)
            # molecules = self.molecules[0] if self.random.rand() > 0.5 else self.molecules[1]
        else:
            molecules = self.molecules
            scale = self.random.normal(loc=molecules, scale=self.molecules_scale * molecules)
            while (scale < 1) or (scale > molecules):
                scale = self.random.normal(loc=molecules, scale=self.molecules_scale * molecules)
            molecules = int(scale)
        return getattr(self, f"_{self.mode}_datamap_generator")(
            shape = self.shape,
            sources = self.sources,
            molecules = molecules,
            shape_sources = self.shape_sources,
            **kwargs
        )

    def _generate_datamap_generator(self, shape, sources, molecules, shape_sources=(1, 1), *args, **kwargs):
        """
        Function to generate a datamap with randomly located molecules.

        :param shape: A `int` or `tuple` representing the shape of the datamap.
        :param sources: Number of molecule sources to be randomly placed on the datamap.
        :param molecules: Average number of molecules contained on each source. The actual number of molecules will be
                          determined by poisson sampling.
        :param shape_sources : A `tuple` of the shape of the sources
        :param random_state: Sets the seed of the random number generator.
        :return: A `numpy.ndarray` of the datamap containing the randomly placed molecules
        """
        if isinstance(shape, int):
            shape = (shape, shape)
        if isinstance(sources, (list, tuple)):
            sources = self.random.randint(*sources)
        datamap = numpy.zeros(shape)
        pos = []
        for i in range(sources):
            row, col = self.random.randint(0, shape[0] - shape_sources[0]), self.random.randint(0, shape[1] - shape_sources[1])
            datamap[row : row + shape_sources[0], col : col + shape_sources[1]] += self.random.poisson(molecules)
            pos.append([row + shape_sources[0] // 2, col + shape_sources[1] // 2])
        return datamap, numpy.array(pos)

    def _generatewithoutoverlap_datamap_generator(self, shape, sources, molecules, shape_sources=(1, 1), *args, **kwargs):
        """
        Function to generate a datamap with randomly located molecules without
        overlap.

        :param shape: A tuple representing the shape of the datamap. If only 1 number is passed, a square datamap will be
                      generated.
        :param sources: Number of molecule sources to be randomly placed on the datamap.
        :param molecules: Average number of molecules contained on each source. The actual number of molecules will be
                          determined by poisson sampling.
        :param shape_sources : A `tuple` of the shape of the sources
        :param random_state: Sets the seed of the random number generator.

        :returns: A `numpy.ndarray` of the datamap containing the randomly placed molecules
        """
        if isinstance(shape, int):
            shape = (shape, shape)
        if isinstance(sources, (list, tuple)):
            sources = self.random.randint(*sources)

        datamap = numpy.zeros(shape)

        # Creates sampling mask
        mask = numpy.zeros_like(datamap)
        mask[: min(-1, -1 * shape_sources[0] + 1), : min(-1, -1 * shape_sources[1] + 1)] = 1

        pos = []
        for i in range(sources):

            argwhere = numpy.argwhere(mask)
            if len(argwhere) < 1:
                break
            row, col = argwhere[self.random.choice(len(argwhere))]
            mask[max(0, row - shape_sources[0]) : row + shape_sources[0], max(0, col - shape_sources[1]) : col + shape_sources[1]] = 0
            datamap[row : row + shape_sources[0], col : col + shape_sources[1]] += self.random.poisson(molecules)

            pos.append([row + shape_sources[0] // 2, col + shape_sources[1] // 2])
        return datamap, numpy.array(pos)

    def _real_datamap_generator(self, molecules, *args, **kwargs):
        """
        Methods that allows to use the datamaps that were generated from real
        experiments by a U-Net.

        :param molecules: An `int` of the average number of molecules to samples
        """
        idx = kwargs.get("idx", None)
        if isinstance(idx, int):
            datamap = self.datamaps[idx]
        else:
            idx = self.random.randint(len(self.datamaps))
            self.idx = idx
            datamap = self.datamaps[idx]
        datamap = numpy.load(datamap)

        if isinstance(self.shape, int):
            shape = (self.shape, self.shape)
        else:
            shape = self.shape

        # Pads the array in cases where the desired crop is bigger than the datamaps
        pady, padx = max(0, shape[0] - datamap.shape[0]), max(0, shape[1] - datamap.shape[1])
        datamap = numpy.pad(datamap, ((pady, pady), (padx, padx)), mode="symmetric")

        # Random crop of the datamap
        # We filter the datamap with a gaussian kernel to sample where the
        # molecules are
        sy, sx = shape[0] // 2, shape[1] // 2
        prob = filters.gaussian(datamap * (datamap >= 0.5).astype(int), sigma=10)[sy : -sy + 1, sx : -sx + 1] + 1e-9 # adds numerical stability
        prob = prob / prob.sum()
        choice = numpy.random.choice(prob.size, p=prob.ravel())

        # Multiplies by a random number of molecules
        # scale = self.random.normal(loc=molecules, scale=self.molecules_scale * molecules)
        # while (scale < 1) or (scale > molecules):
        #     scale = self.random.normal(loc=molecules, scale=self.molecules_scale * molecules)
        datamap *= int(molecules)

        j, i = numpy.unravel_index(choice, prob.shape)
        j += sy
        i += sx
        # j, i = self.random.randint(0, datamap.shape[0] - shape[0]), self.random.randint(0, datamap.shape[1] - shape[1])
        datamap = datamap[j - sy : j + sy, i - sx : i + sx]

        return datamap, []

class ParameterSpaceGenerator:
    """
    Generate the parameter space of the given microscope.

    The parameter space is generated based on the microscope and the parameters
    and use the following format: ``_{microscope}_parameter_space``

    We use the ``_AbberiorX_parameter_space`` convention when a real microscopy systems
    is used for the optimization.

    :param microscope: A `str` of the microscope name
    """
    def __init__(self, microscope):

        self.microscope = microscope

    def __call__(self, param_names, x_mins, x_maxs, n_divs_default):
        """
        Implements the call function of the microscope specific implementation

        :param param_names: A `list` of parameter names
        :param x_mins: A `list` of minimal values associated with each parameter
        :param x_maxs: A `list` of maximal values associated with each parameter

        :returns : A `list` of ndims
                   A `list` of parameter space bounds
                   A `list` of n points
                   A `list` of conditions
        """
        return getattr(self, f"_{self.microscope}_parameter_space")(param_names, x_mins, x_maxs, n_divs_default)

    def _DyMIN_parameter_space(self, param_names, x_mins, x_maxs, n_divs_default):
        """
        Implements the DyMIN parameter space creation

        :param param_names: A `list` of parameter names
        :param x_mins: A `list` of minimal values associated with each parameter
        :param x_maxs: A `list` of maximal values associated with each parameter

        :returns : A `list` of ndims
                   A `list` of parameter space bounds
                   A `list` of n points
                   A `list` of conditions
        """
        # Defines necessary variables
        ndims, param_space_bounds, conditions = [], [], []
        for i, param_name in enumerate(param_names):
            if param_name in ["decision_time", "threshold_count"]:
                ndims.append(2)
            else:
                ndims.append(1)
            param_space_bounds.extend([[x_mins[i], x_maxs[i]]] * ndims[-1])
        n_points = [n_divs_default]*sum(ndims)
        return ndims, param_space_bounds, n_points, conditions

    def _STED_parameter_space(self, param_names, x_mins, x_maxs, n_divs_default):
        """
        Implements the RESCue parameter space creation

        :param param_names: A `list` of parameter names
        :param x_mins: A `list` of minimal values associated with each parameter
        :param x_maxs: A `list` of maximal values associated with each parameter

        :returns : A `list` of ndims
                   A `list` of parameter space bounds
                   A `list` of n points
                   A `list` of conditions
        """
        # Defines necessary variables
        ndims, param_space_bounds, conditions = [], [], []
        for i, param_name in enumerate(param_names):
            ndims.append(1)
            param_space_bounds.extend([[x_mins[i], x_maxs[i]]] * ndims[-1])
        n_points = [n_divs_default]*sum(ndims)
        return ndims, param_space_bounds, n_points, conditions

    def _RESCue_parameter_space(self, param_names, x_mins, x_maxs, n_divs_default):
        """
        Implements the RESCue parameter space creation

        :param param_names: A `list` of parameter names
        :param x_mins: A `list` of minimal values associated with each parameter
        :param x_maxs: A `list` of maximal values associated with each parameter

        :returns : A `list` of ndims
                   A `list` of parameter space bounds
                   A `list` of n points
                   A `list` of conditions
        """
        ndims, param_space_bounds, n_points, _ = self._STED_parameter_space(param_names, x_mins, x_maxs, n_divs_default)
        conditions = []
        if ("lower_threshold" in param_names) and ("upper_threshold" in param_names):
            index_lTh = param_names.index("lower_threshold")
            index_uTh = param_names.index("upper_threshold")
            conditions.append(partial(
                bigger_than,
                low = sum(ndims[:index_lTh]),
                high = sum(ndims[:index_uTh]),
                bound_low = [bound[0] for bound in param_space_bounds],
                bound_up = [bound[1] for bound in param_space_bounds]
            ))
        return ndims, param_space_bounds, n_points, conditions

    def _AbberiorSTED_parameter_space(self, param_names, x_mins, x_maxs, n_divs_default):
        """
        Implements the RESCue parameter space creation

        :param param_names: A `list` of parameter names
        :param x_mins: A `list` of minimal values associated with each parameter
        :param x_maxs: A `list` of maximal values associated with each parameter

        :returns : A `list` of ndims
                   A `list` of parameter space bounds
                   A `list` of n points
                   A `list` of conditions
        """
        # Defines necessary variables
        ndims, param_space_bounds, conditions = [], [], []
        for i, param_name in enumerate(param_names):
            ndims.append(1)
            param_space_bounds.extend([[x_mins[i], x_maxs[i]]] * ndims[-1])
        n_points = [n_divs_default]*sum(ndims)
        return ndims, param_space_bounds, n_points, conditions

    def _AbberiorRESCue_parameter_space(self, param_names, x_mins, x_maxs, n_divs_default):
        """
        Implements the Abberior parameter space creation

        :param param_names: A `list` of parameter names
        :param x_mins: A `list` of minimal values associated with each parameter
        :param x_maxs: A `list` of maximal values associated with each parameter

        :returns : A `list` of ndims
                   A `list` of parameter space bounds
                   A `list` of n points
                   A `list` of conditions
        """
        # Defines necessary variables
        ndims, param_space_bounds, conditions = [], [], []
        for i, param_name in enumerate(param_names):
            if param_name in ["ch2_LTh_thresholds", "ch2_LTh_times"]:
                for j in reversed(range(1, 4)):
                    conditions.append(partial(
                        bigger_than,
                        low = j - 1 + sum(ndims),
                        high = j + sum(ndims),
                        bound_low=[0.] * sum(ndims) + [x_mins[i]] * 4,
                        bound_up=[0.] * sum(ndims) + [x_maxs[i]] * 4
                    ))
                ndims.append(4)
            else:
                ndims.append(1)
            param_space_bounds.extend([[x_mins[i], x_maxs[i]]] * ndims[-1])

        if ("ch1_p_sted" in param_names) and ("ch2_p_sted" in param_names):
            index_ch2 = param_names.index("ch1_p_sted")
            index_ch3 = param_names.index("ch2_p_sted")
            conditions.append(partial(
                bigger_than,
                low = sum(ndims[:index_ch2]),
                high = sum(ndims[:index_ch3]),
                bound_low = [bound[0] for bound in param_space_bounds],
                bound_up = [bound[1] for bound in param_space_bounds]
            ))

        n_points = [n_divs_default]*sum(ndims)
        return ndims, param_space_bounds, n_points, conditions

    def _AbberiorDyMIN_parameter_space(self, param_names, x_mins, x_maxs, n_divs_default):
        """
        Implements the Abberior parameter space creation

        :param param_names: A `list` of parameter names
        :param x_mins: A `list` of minimal values associated with each parameter
        :param x_maxs: A `list` of maximal values associated with each parameter

        :returns : A `list` of ndims
                   A `list` of parameter space bounds
                   A `list` of n points
                   A `list` of conditions
        """
        # Defines necessary variables
        ndims, param_space_bounds, conditions = [], [], []
        for i, param_name in enumerate(param_names):
            if param_name in ["ch2_LTh_thresholds", "ch2_LTh_times"]:
                for j in reversed(range(1, 4)):
                    conditions.append(partial(
                        bigger_than,
                        low = j - 1 + sum(ndims),
                        high = j + sum(ndims),
                        bound_low=[0.] * sum(ndims) + [x_mins[i]] * 4,
                        bound_up=[0.] * sum(ndims) + [x_maxs[i]] * 4
                    ))
                ndims.append(4)
            else:
                ndims.append(1)
            param_space_bounds.extend([[x_mins[i], x_maxs[i]]] * ndims[-1])

        if ("ch2_p_sted" in param_names) and ("ch3_p_sted" in param_names):
            index_ch2 = param_names.index("ch2_p_sted")
            index_ch3 = param_names.index("ch3_p_sted")
            conditions.append(partial(
                bigger_than,
                low = sum(ndims[:index_ch2]),
                high = sum(ndims[:index_ch3]),
                bound_low = [bound[0] for bound in param_space_bounds],
                bound_up = [bound[1] for bound in param_space_bounds]
            ))
        n_points = [n_divs_default]*sum(ndims)
        return ndims, param_space_bounds, n_points, conditions

class ExperimentGenerator:
    """
    Experiment generator that creates a microscope specific experiment from which
    the user can simply acquire. 
    
    This is for simulation experiments only.

    :param microscope: A `str` of the microscope
    :param conf_params: A `dict` of the confocal parameters
    :param default_values_dict: A `dict` of default values
    :param ndims: A `list` of the dimensions associated with each parameter
    :param param_names: A `list` of the parameter name
    """
    def __init__(self, microscope, conf_params, default_values_dict, ndims, param_names):

        self.microscope = microscope
        self.conf_params = conf_params
        self.default_values_dict = default_values_dict
        self.ndims = ndims
        self.param_names = param_names

    def __call__(self, x_selected, molecules_disposition, *args, **kwargs):
        """
        Creates an `Experiment` that contains the STED and confocal microscope

        :param x_selected: A `numpy.ndarray` of the selected parameters
        :param molecules_disposition: A `numpy.ndarray` of the molecules disposition

        :returns : An `Experiment`
        """
        return getattr(self, f"_{self.microscope}_experiment")(x_selected, molecules_disposition, *args, **kwargs)

    def confocal_experiment(self, molecules_disposition, *args, **kwargs):
        """
        Creates an `Experiment` that contains a confocal microscope

        :param molecules_disposition: A `numpy.ndarray` of the molecules disposition

        :returns : An `Experiment`
        """
        confocal_microscope, datamap, conf_params = create_microscope(
            datamap = {
                "whole_datamap" : molecules_disposition,
                "datamap_pixelsize" : self.default_values_dict["pixelsize"]
            },
            imaging = self.conf_params,
            **kwargs
        )

        # Creates the experiment
        experiment = Experiment()
        experiment.add("conf", confocal_microscope, datamap, conf_params)

        return experiment

    def _STED_experiment(self, x_selected, molecules_disposition, *args, **kwargs):
        """
        Creates an `Experiment` that contains the STED and confocal microscope

        :param x_selected: A `numpy.ndarray` of the selected parameters
        :param molecules_disposition: A `numpy.ndarray` of the molecules disposition

        :returns : An `Experiment`
        """
        # Updates the x selected dict for future use
        x_selected_dict = {}
        for i, (ndim, pname) in enumerate(zip(self.ndims, self.param_names)):
            x_selected_dict[pname] = x_selected[sum(self.ndims[:i]) : sum(self.ndims[:i]) + ndim]
            if pname == "threshold_count":
                tmp = numpy.zeros(3)
                tmp[:2] = x_selected_dict[pname].ravel()
                x_selected_dict[pname] = tmp.astype(int)
            if pname == "decision_time":
                tmp = -1 * numpy.ones(3)
                tmp[:2] = x_selected_dict[pname].ravel()
                x_selected_dict[pname] = tmp

        # Acquire conf1, sted_image and conf2
        STED_microscope, datamap, sted_params = create_microscope(
            datamap = {
                "whole_datamap" : molecules_disposition,
                "datamap_pixelsize" : self.default_values_dict["pixelsize"]
            },
            imaging={
                pname : x_selected_dict[pname].item() if pname in self.param_names
                else self.default_values_dict[pname]
                for pname in ["pdt", "p_ex", "p_sted"]
            },
            **kwargs
        )
        confocal_microscope, _, conf_params = create_microscope(
            imaging = self.conf_params,
            **kwargs
        )

        # By passing the same datamap to DyMIN and conf we ensure that they
        # both have access to the same datamap
        experiment = Experiment()
        experiment.add("STED", STED_microscope, datamap, sted_params)
        experiment.add("conf", confocal_microscope, datamap, conf_params)

        return x_selected_dict, experiment

    def _DyMIN_experiment(self, x_selected, molecules_disposition, *args, **kwargs):
        """
        Creates an `Experiment` that contains the STED and confocal microscope

        :param x_selected: A `numpy.ndarray` of the selected parameters
        :param molecules_disposition: A `numpy.ndarray` of the molecules disposition

        :returns : An `Experiment`
        """

        # Updates the x selected dict for future use
        x_selected_dict = {}
        for i, (ndim, pname) in enumerate(zip(self.ndims, self.param_names)):
            x_selected_dict[pname] = x_selected[sum(self.ndims[:i]) : sum(self.ndims[:i]) + ndim]
            if pname == "threshold_count":
                tmp = numpy.zeros(3)
                tmp[:2] = x_selected_dict[pname].ravel()
                x_selected_dict[pname] = tmp.astype(int)
            if pname == "decision_time":
                tmp = -1 * numpy.ones(3)
                tmp[:2] = x_selected_dict[pname].ravel()
                x_selected_dict[pname] = tmp

        DyMIN_microscope, datamap, dymin_params = create_dymin_microscope(
            datamap = {
                "whole_datamap" : molecules_disposition,
                "datamap_pixelsize" : self.default_values_dict["pixelsize"]
            },
            microscope = {
                pname : x_selected_dict[pname] if pname in self.param_names
                else self.default_values_dict[pname]
                for pname in ["scale_power", "decision_time", "threshold_count"]
            },
            imaging={
                pname : x_selected_dict[pname].item() if pname in self.param_names
                else self.default_values_dict[pname]
                for pname in ["pdt", "p_ex", "p_sted"]
            },
            **kwargs
        )
        confocal_microscope, _, conf_params = create_microscope(
            imaging = self.conf_params,
            **kwargs
        )

        # By passing the same datamap to sted and conf we ensure that they
        # both have access to the same datamap
        experiment = Experiment()
        experiment.add("STED", DyMIN_microscope, datamap, dymin_params)
        experiment.add("conf", confocal_microscope, datamap, conf_params)

        return x_selected_dict, experiment

    def _RESCue_experiment(self, x_selected, molecules_disposition, *args, **kwargs):
        """
        Creates an `Experiment` that contains the STED and confocal microscope

        :param x_selected: A `numpy.ndarray` of the selected parameters
        :param molecules_disposition: A `numpy.ndarray` of the molecules disposition

        :returns : An `Experiment`
        """
        # Updates the x selected dict for future use
        x_selected_dict = {}
        for i, (ndim, pname) in enumerate(zip(self.ndims, self.param_names)):
            x_selected_dict[pname] = x_selected[sum(self.ndims[:i]) : sum(self.ndims[:i]) + ndim]
            if pname in ["lower_threshold", "upper_threshold"]:
                tmp = -1 * numpy.ones(2)
                tmp[:1] = x_selected_dict[pname].ravel()
                x_selected_dict[pname] = tmp.astype(int)
            if pname == "decision_time":
                tmp = -1 * numpy.ones(2)
                tmp[:1] = x_selected_dict[pname].ravel()
                x_selected_dict[pname] = tmp

        RESCue_microscope, datamap, rescue_params = create_rescue_microscope(
            datamap = {
                "whole_datamap" : molecules_disposition,
                "datamap_pixelsize" : self.default_values_dict["pixelsize"]
            },
            microscope = {
                pname : x_selected_dict[pname] if pname in self.param_names
                else self.default_values_dict[pname]
                for pname in ["lower_threshold", "upper_threshold", "decision_time"]
            },
            imaging = {
                pname : x_selected_dict[pname].item() if pname in self.param_names
                else self.default_values_dict[pname]
                for pname in ["pdt", "p_ex", "p_sted"]
            },
            **kwargs
        )
        confocal_microscope, _, conf_params = create_microscope(
            imaging = self.conf_params
        )

        # By passing the same datamap to sted and conf we ensure that they
        # both have access to the same datamap
        experiment = Experiment()
        experiment.add("STED", RESCue_microscope, datamap, rescue_params)
        experiment.add("conf", confocal_microscope, datamap, conf_params)

        return x_selected_dict, experiment


class MicroscopeConfigurator:
    """
    Configuration of a real microscopy system.

    :param defaults: A `module` containing the default setter methods
    :param microscope: A `dict` of the microscope
    :param conf_params: A `dict` of the confocal parameters
    :param default_values_dict: A `dict` of default values
    :param ndims: A `list` of the dimensions associated with each parameter
    :param param_names: A `list` of the parameter name
    :param config_conf: A confocal measurement configuration
    :param config_sted: A STED measurement configuration
    :param other_configs: A `list` of other configurations
    """
    def __init__(self, defaults, microscope_conf, params_conf, default_values_dict, ndims, param_names,
                    config_conf, config_sted, other_configs=None):
        self.defaults = getattr(defaults, microscope_conf["mode"])
        self.microscope_conf = microscope_conf
        self.conf_params = params_conf
        self.default_values_dict = default_values_dict
        self.ndims = ndims
        self.param_names = param_names
        self.config_conf, self.config_sted = config_conf, config_sted
        self.other_configs = other_configs

        # Sets default confocal configuration
        exc_id = getattr(self.defaults, "EXC_ID")
        if isinstance(exc_id, (tuple, list)):
            for i, e_id in enumerate(exc_id):
                microscope.set_power(self.config_conf, self.conf_params[f"ch{i + 1}_p_ex"], laser_id=e_id, channel_id=i)
        else:
            microscope.set_power(self.config_conf, self.conf_params["p_ex"], laser_id=exc_id)
        microscope.set_dwelltime(self.config_conf, self.conf_params["pdt"])
        microscope.set_imagesize(self.config_conf, *self.default_values_dict["imagesize"])
        microscope.set_pixelsize(self.config_conf, *self.default_values_dict["pixelsize"])

        # for conf in [self.config_conf, self.config_sted, *other_configs]:
        #     print("SPECTRUM")
        #     print(microscope.get_spectral_min(conf))
        #     print(microscope.get_spectral_max(conf))

        if other_configs:
            for conf in other_configs:
                microscope.set_imagesize(conf, *self.default_values_dict["imagesize"])
                microscope.set_pixelsize(conf, *self.default_values_dict["pixelsize"])

        # Sets default STED configuration
        getattr(self, "_{}_defaults".format(self.microscope_conf["mode"]))()

        # If the microscope is in advanced mode, we need to activate some advanced
        # parameters
        # if self.microscope_conf["acquisition-mode"]:
        #     getattr(self, "_{}_advanced_settings".format(self.microscope_conf["mode"]))()
        # else:
        getattr(self, "_{}_{}_settings".format(self.microscope_conf["mode"], self.microscope_conf["acquisition-mode"]))()

    def __call__(self, x_selected):
        return getattr(self, "_{}_configuration".format(self.microscope_conf["mode"]))(x_selected)

    def _AbberiorSTED_configuration(self, x_selected):
        """
        Set the given configuration on the microscope

        :returns : A `dict` of the selected parameters
        """
        # Updates the x selected dict for future use
        x_selected_dict = {}
        for i, (ndim, pname) in enumerate(zip(self.ndims, self.param_names)):
            x_selected_dict[pname] = x_selected[sum(self.ndims[:i]) : sum(self.ndims[:i]) + ndim]

        # Sets the parameter using the implemented partial methods
        for key, value in x_selected_dict.items():
            if len(value) == 1:
                value = value.item()
            else:
                value = value.ravel().tolist()
            getattr(self.defaults, key)(self.config_sted, value)

        return x_selected_dict

    def _AbberiorSTED_defaults(self):
        """
        Sets the defaults values of the miroscope
        """
        microscope.set_imagesize(self.config_sted, *self.default_values_dict["imagesize"])
        microscope.set_pixelsize(self.config_sted, *self.default_values_dict["pixelsize"])
        microscope.set_dwelltime(self.config_sted, self.default_values_dict["pdt"])

        # When we have multiple channels we need to set the power for each channel
        exc_id = getattr(self.defaults, "EXC_ID")
        if isinstance(exc_id, (tuple, list)):
            for i, e_id in enumerate(exc_id):
                microscope.set_power(self.config_sted, self.default_values_dict["p_ex"], laser_id=e_id, channel_id=i)
                microscope.set_power(self.config_sted, self.default_values_dict["p_sted"],
                                     laser_id=getattr(self.defaults, "STED_ID"), channel_id=i)
        else:
            microscope.set_power(self.config_sted, self.default_values_dict["p_ex"], laser_id=getattr(self.defaults, "EXC_ID"), channel_id=0)
            microscope.set_power(self.config_sted, self.default_values_dict["p_sted"], laser_id=getattr(self.defaults, "STED_ID"), channel_id=0)

    def _AbberiorSTED_normal_settings(self):
        """
        Sets microscope to the advanced settings
        """
        pass
        # microscope.activate_linestep(self.config_sted, True)
        # microscope.activate_pixelstep(self.config_sted, False)

    def _AbberiorDyMIN_configuration(self, x_selected):
        """
        Set the given configuration on the microscope

        :param x_selected: A `numpy.ndarray` of the selected parameters

        :returns : A `dict` of the selected parameters
        """
        # Updates the x selected dict for future use
        x_selected_dict = {}
        for i, (ndim, pname) in enumerate(zip(self.ndims, self.param_names)):
            x_selected_dict[pname] = x_selected[sum(self.ndims[:i]) : sum(self.ndims[:i]) + ndim]

        # Sets the parameter using the implemented partial methods
        for key, value in x_selected_dict.items():
            if len(value) == 1:
                value = value.item()
            else:
                value = value.ravel().tolist()
            getattr(self.defaults, key)(self.config_sted, value)

        return x_selected_dict

    def _AbberiorDyMIN_defaults(self):
        """
        Sets the defaults values of the miroscope in DyMIN mode
        """
        microscope.set_imagesize(self.config_sted, *self.default_values_dict["imagesize"])
        microscope.set_pixelsize(self.config_sted, *self.default_values_dict["pixelsize"])
        microscope.set_dwelltime(self.config_sted, self.default_values_dict["pdt"])
        for channel_id in range(microscope.get_num_channels(self.config_sted)):
            microscope.set_power(self.config_sted, self.default_values_dict["p_ex"], laser_id=getattr(self.defaults, "EXC_ID"), channel_id=channel_id)
        for channel_id, scale in enumerate([0, 0.1, 1.]):
            microscope.set_power(self.config_sted, scale * self.default_values_dict["p_sted"], laser_id=getattr(self.defaults, "STED_ID"), channel_id=channel_id)

    def _AbberiorDyMIN_advanced_settings(self):
        """
        Sets microscope to the advanced settings in DyMIN mode
        """
        microscope.set_rescue_mode(self.config_sted, mode=1, channel_id=1)
        microscope.set_LTh_num_times(self.config_sted, num_times=4, channel_id=1)
        microscope.set_LTh_auto(self.config_sted, value=False, channel_id=1)
        microscope.set_UTh_auto(self.config_sted, value=False, channel_id=1)

    def _AbberiorDyMIN_normal_settings(self):
        """
        Sets microscope to the normal settings in DyMIN mode
        """
        microscope.set_rescue_mode(self.config_sted, mode=5, channel_id=1)
        microscope.set_LTh_auto(self.config_sted, value=True, channel_id=1)
        microscope.set_UTh_auto(self.config_sted, value=True, channel_id=1)

    def _AbberiorDyMIN_naive_settings(self):
        """
        Sets microscope to the naive settings
        """
        microscope.set_rescue_mode(self.config_sted, mode=3, channel_id=0)
        microscope.set_LTh_auto(self.config_sted, value=False, channel_id=0)
        microscope.set_LTh_num_times(self.config_sted, num_times=1, channel_id=0)
        microscope.set_UTh_auto(self.config_sted, value=True, channel_id=0)

        microscope.set_rescue_mode(self.config_sted, mode=3, channel_id=1)
        microscope.set_LTh_auto(self.config_sted, value=False, channel_id=1)
        microscope.set_LTh_num_times(self.config_sted, num_times=1, channel_id=1)
        microscope.set_UTh_auto(self.config_sted, value=True, channel_id=1)

    def _AbberiorRESCue_configuration(self, x_selected):
        """
        Set the given configuration on the microscope in RESCue mode

        :param x_selected: A `numpy.ndarray` of the selected parameters

        :returns : A `dict` of the selected parameters
        """
        # Updates the x selected dict for future use
        x_selected_dict = {}
        for i, (ndim, pname) in enumerate(zip(self.ndims, self.param_names)):
            x_selected_dict[pname] = x_selected[sum(self.ndims[:i]) : sum(self.ndims[:i]) + ndim]

        # Sets the parameter using the implemented partial methods
        for key, value in x_selected_dict.items():
            if len(value) == 1:
                value = value.item()
            else:
                value = value.ravel().tolist()
            getattr(self.defaults, key)(self.config_sted, value)

        return x_selected_dict

    def _AbberiorRESCue_defaults(self):
        """
        Sets the defaults values of the miroscope in RESCue mode
        """
        microscope.set_imagesize(self.config_sted, *self.default_values_dict["imagesize"])
        microscope.set_dwelltime(self.config_sted, self.default_values_dict["pdt"])
        microscope.set_pixelsize(self.config_sted, *self.default_values_dict["pixelsize"])
        for channel_id in range(microscope.get_num_channels(self.config_sted)):
            microscope.set_power(self.config_sted, self.default_values_dict["p_ex"], laser_id=getattr(self.defaults, "EXC_ID"), channel_id=channel_id)
        for channel_id, scale in enumerate([0, 1.]):
            microscope.set_power(self.config_sted, scale * self.default_values_dict["p_sted"], laser_id=getattr(self.defaults, "STED_ID"), channel_id=channel_id)

    def _AbberiorRESCue_advanced_settings(self):
        """
        Sets microscope to the advanced settings in RESCue mode
        """
        microscope.set_rescue_mode(self.config_sted, mode=6, channel_id=0)
        microscope.set_rescue_mode(self.config_sted, mode=1, channel_id=1)
        microscope.set_LTh_num_times(self.config_sted, num_times=4, channel_id=1)
        microscope.set_LTh_auto(self.config_sted, value=False, channel_id=1)
        microscope.set_UTh_auto(self.config_sted, value=False, channel_id=1)

    def _AbberiorRESCue_normal_settings(self):
        """
        Sets microscope to the normal settings in RESCue mode
        """
        microscope.set_rescue_mode(self.config_sted, mode=6, channel_id=0)
        microscope.set_rescue_mode(self.config_sted, mode=1, channel_id=1)
        microscope.set_LTh_auto(self.config_sted, value=True, channel_id=1)
        microscope.set_UTh_auto(self.config_sted, value=True, channel_id=1)

def get_user_input(question, expected_answers):
    if not question.endswith(" "):
        question += " "
    answer = input(question)
    while answer not in expected_answers:
        answer = input(question)
    return answer

def wait_for_user_to_annotate_image(image):
    print("Annotate the image in the napari plugin...")
    print("The image is located at: ", image)
    get_user_input("Once you are done, press 'q' to continue.", ["q"])

def wait_for_drag_and_drop_annotation_path():
    print("Drag and drop the image you want to annotate here.")
    path = input()
    print(path)
    return path

def load_annotations_and_ask_which_ids_to_keep(path):
    annotation = tifffile.imread(path)
    uniques = numpy.unique(annotation)
    uniques = [str(u) for u in uniques]
    print("The unique values in the annotation are: ", uniques)
    keep = int(get_user_input("Which labels do you want to keep? ", uniques))
    mask = annotation == keep
    return mask, annotation, keep

def convert_mask_to_rectangles(mask, image=None):
    label = measure.label(mask)
    rprops = measure.regionprops(label, intensity_image=image)
    rectangles = []
    for rprop in rprops:
        minr, minc, maxr, maxc = rprop.bbox
        rectangles.append((
            (minc, minr),
            (maxc, maxr)
        ))
    return rectangles

def convert_rectangles_to_regions(config, rectangles):
    regions = utils.rect2regions(rectangles, microscope.get_pixelsize(config)) # New window size
    # points = utils.get_rect_center(rectangles, microscope.get_pixelsize(config))
    points = utils.get_rect_center(rectangles, microscope.get_pixelsize(config), microscope.get_resolution(config))
    x_offset, y_offset = microscope.get_offsets(config)
    rect_region_offset = [(x + x_offset, y + y_offset) for (x, y) in points]
    return rect_region_offset, regions, rectangles # returns the offset and the regions dimensions

class RegionSelector:
    """
    Creates a `RegionSelector` to store possible regions.

    Regions are stored into a buffer and are queried by an optimization routine.
    When the buffer is empty new regions are asked from the user.
    """
    def __init__(self, config_overview, config):
        """
        Instanciates a `RegionSelector`

        :param config_overview: An overview configuration
        :param overview: A `str` of the name of the overview
        """
        self.config_overview = config_overview
        self.mode = config["region_opts"]["mode"]
        assert self.mode in ["manual", "auto"], "The mode {} is not implemented".format(self.mode)
        self.overview = config["region_opts"]["overview"]
        self.config = config
        self.buffer = []
        self.t = 0

    def __iter__(self):
        """
        Implements a `__iter__` of the `RegionSelector`

        :returns : The iterator
        """
        return self

    def __next__(self):
        """
        Implements a `__next__` of the `RegionSelector`

        returns : The next element into the buffer
        """
        if len(self.buffer) < 1:
            func = getattr(self, f"_fill_{self.mode}_buffer")
            func()
        item = self.buffer.pop(0)
        return item

    def _fill_manual_buffer(self):
        """
        Asks the user to fill the buffer with some regions
        """
        print("[!!!!] Now would be a good time to move the overwiew...")
        input("[----] Once done hit enter!")
        regions_offset = user.get_regions(self.overview, self.config_overview)
        for offset in regions_offset:
            self.buffer.append(offset)
    
    def _fill_auto_buffer(self):
        """
        Automatically fills the buffer with some regions
        """
        print("[!!!!] Now would be a good time to move the overwiew...")
        input("[----] Once done hit enter!")

        overview_image = microscope.get_overview(self.config_overview, name=self.overview)

        # Saves the overview image
        imwrite(os.path.join(self.config["save_folder"], f"overview-{self.t}.tif"), overview_image.astype(numpy.uint16))
        wait_for_user_to_annotate_image(os.path.join(self.config["save_folder"], f"overview-{self.t}.tif"))

        # Loads the annotations
        image_path = wait_for_drag_and_drop_annotation_path()
        mask, raw, keep_idx = load_annotations_and_ask_which_ids_to_keep(image_path)
        rectangles = convert_mask_to_rectangles(mask, overview_image)

        rect_region_offset, regions, rectangles = convert_rectangles_to_regions(self.config_overview, rectangles)

        for offset in rect_region_offset:
            self.buffer.append(offset)

        self.t += 1

class PointSelector:
    """
    Creates a small interface for the user to manually input the next point
    """
    def __init__(self, param_names, param_space_bounds, obj_names, mode="terminal"):
        self.param_names = param_names
        self.param_space_bounds = param_space_bounds
        self.obj_names = obj_names

        self.mode = mode
        self.select_mode = getattr(self, f"select_{self.mode}")

    def select(self, *args, **kwargs):
        return self.select_mode(*args, **kwargs)

    def select_terminal(self, X, y, show=True):
        """
        Implements a selection of the parameter from the terminal
        """
        if show:
            fig, axes = pyplot.subplots(y.shape[-1], X.shape[-1], figsize=(15, 5), tight_layout=True)
            for i in range(X.shape[-1]):
                for j in range(y.shape[-1]):
                    axes[j, i].scatter(X[:, i], y[:, j], c=numpy.arange(len(X)))
                    axes[j, i].set(
                        xlabel = self.param_names[i] if j == y.shape[-1]-1 else None, ylabel=self.obj_names[j] if i == 0 else None,
                        xlim = self.param_space_bounds[i]
                    )

        x_selected = []
        for param_name, bound in zip(self.param_names, self.param_space_bounds):
            if param_name == "pdt":
                param_name = param_name + " (us)"
                bound = bound.copy()
                bound[0], bound[1] = bound[0] * 1e+6, bound[1] * 1e+6
            answer = None
            while not isinstance(answer, (int, float)):
                answer = input("{} [{} - {}] : ".format(param_name, bound[0], bound[1]))
                try:
                    answer = float(answer)
                except:
                    pass
            if param_name == "pdt (us)":
                answer = answer / 1e+6
            x_selected.append(answer)

        if show:
            pyplot.show(block=True)
        return numpy.array(x_selected)[:, numpy.newaxis]

    def select_gui(self):
        """
        Implements a selection of the parameter using a small GUI
        """
        def callback(var):
            pass

        import tkinter
        from tkinter import Tk, ttk, StringVar
        root = Tk()
        root.title("Point selection")
        frm = ttk.Frame(root, padding=10)
        frm.grid()
        string_variables = []
        for i, (param_name, bound) in enumerate(zip(self.param_names, self.param_space_bounds)):
            if param_name == "pdt":
                param_name = param_name + " (us)"
                bound = bound.copy()
                bound[0], bound[1] = bound[0] * 1e+6, bound[1] * 1e+6
            ttk.Label(frm, text="{} [{} - {}] : ".format(param_name, *bound), justify=tkinter.LEFT).grid(column=0, row=i)
            var = StringVar()
            var.trace("w", lambda name, index, mode, var=var: callback(var))
            e = ttk.Entry(frm, textvariable=var).grid(column=1, row=i)
            string_variables.append(var)
        root.mainloop()

        x_selected = []
        for answer in string_variables:
            answer = float(answer.get())
            if param_name == "pdt (us)":
                answer = answer / 1e+6
            x_selected.append(answer)
        return numpy.array(x_selected)[:, numpy.newaxis]

def plot_scores(scores, reps, experiment, X, Y, filter_keys=None,
                score_keys=["resolution", "bleach"], interp=False):
    """
    Plots the scores
    """
    # Reorder the dict to take into account the repetitions
    keys = list(set(["_".join(key.split("_")[:-1]) for key in scores.keys()]))
    data = defaultdict(list)
    for key in keys:
        for rep in range(reps):
            data[key].append(scores["{}_{}".format(key, rep)])

    figaxes = {score_key : pyplot.subplots(figsize=(3,3), tight_layout=True) for score_key in score_keys}
    minmax = {score_key : [numpy.inf, -numpy.inf] for score_key in score_keys}
    xyz = {score_key : [] for score_key in score_keys}
    for key, values in data.items():

        x, y = eval(X), eval(Y)
        # Filters the good labels
        if (isinstance(filter_keys, list)):
            flags = []
            for filter_key in filter_keys:
                flags.append(eval(filter_key))
            if not all(flags):
                continue

        for score_key in score_keys:
            fig, ax = figaxes[score_key]
            mean = numpy.mean([value[score_key] for value in values])
            xyz[score_key].append([x, y, mean])
            ax.scatter(x, y, s=200, c=mean)
            if mean < minmax[score_key][0]:
                minmax[score_key][0] = mean
            if mean > minmax[score_key][1]:
                minmax[score_key][1] = mean

    if interp:
        for score_key in score_keys:
            x, y, z = numpy.array(xyz[score_key]).T
            func = interpolate.interp2d(x, y, z, kind="linear")
            xx, yy = numpy.linspace(x.min(), x.max(), 50), numpy.linspace(y.min(), y.max(), 50)
            xstep, ystep = numpy.diff(xx).mean(), numpy.diff(yy).mean()

            print(xx.min(), xx.max(), xstep)
            print(yy.min(), yy.max(), ystep)
            fig, ax = figaxes[score_key]
            ax.imshow(
                func(xx, yy),
                # extent=(xx.min() - xstep / 2, xx.max() + xstep / 2, yy.min() - ystep / 2, yy.max() + ystep / 2),
                origin="lower"
            )

    for key, (fig, ax) in figaxes.items():
        ax.set(title=key)
        for s in ax.collections:
            s.set_clim(*minmax[key])
        fig.colorbar(ax.collections[0], ax=ax)
    return figaxes

def preference_articulation(points, objs, increase_size=0.01):
    """
    Defines an automatic preference articulation

    :param points: A `numpy.ndarray` of points
    :param objs: A `list` of objective names
    :param increase_size: A `float` of the increase size

    :returns : An `int` of the index
    """
    bounds = {
        "Resolution" : {"min" : 40, "max" : 80},
        "Bleach" : {"min" : 0, "max" : 0.1},
        "Squirrel" : {"min" : 0, "max" : 12}
    }

    def isin(points, objs, bounds):
        conditions = []
        for obj_name, values in zip(objs, points):
            conditions.append((values <= bounds[obj_name]["max"]) * (values >= bounds[obj_name]["min"]))
        conditions = numpy.prod(conditions, axis=0)
        return conditions

    conditions = isin(points, objs, bounds)

    # If the conditions are all met then we return a random point within bounds
    if numpy.any(conditions):
        return random.choice(numpy.argwhere(conditions).ravel())

    # When no points are within the bounds then we iteratively increase the size of the available bounds
    # until a point is found
    def increase_bounds(i, increase_size, objs, bounds):
        if isinstance(increase_size, type(float)):
            increase_size = [increase_size] * len(objs)
        _bounds = copy.deepcopy(bounds)
        for (obj, value) in zip(objs, increase_size):
            _bounds[obj]["max"] = bounds[obj]["max"] + i * value * (bounds[obj]["max"] - bounds[obj]["min"])
        return _bounds

    i = 1
    while not numpy.any(conditions):
        _bounds = increase_bounds(i, increase_size, objs, bounds)
        conditions = isin(points, objs, _bounds)
        i += 1
    return random.choice(numpy.argwhere(conditions).ravel())

def create_dataset(group, key, **kwargs):
    """
    Creates a `h5py.Dataset`. If the key to the dataset already exists we delete it
    and replace it with the new data

    :param group: A `h5py.Group`
    :param data: A `str` of the name of the dataset
    :param value: A `numpy.ndarray` of the new dataset

    :returns : A `h5py.Dataset`
    """
    if key in group:
        del group[key]

    pixelsize = kwargs.pop("pixelsize", None)

    dataset = group.create_dataset(key, compression="gzip", compression_opts=4, **kwargs)
    if pixelsize:
        dataset.attrs["element_size_um"] = [1, pixelsize * 1e+6, pixelsize * 1e+6]
    return dataset

def create_group(group, key, exist_ok=False):
    """
    Creates a `hyp5.Group`. If the key is already existing and `exist_ok` is `True`
    than we simply return the group. Otherwise, we simply proceed with normal
    `h5py` implementation

    :param group: A `h5py.Group` or `h5py.File`
    :param key: A `str` of the name of the group to create
    :param exist_ok: (optional) A `bool` wheter the key may exist

    :returns : A `h5py.Group`
    """
    if exist_ok:
        if key in group:
            return group[key]
        else:
            return group.create_group(key)
    return group.create_group(key)

def create_savename(**kwargs):
    """
    Creates path name of the save folder

    :returns : A `str` of the savepath
    """
    dry_run = kwargs.get("dry_run")
    save_folder = kwargs.get("save_folder", PATH)
    if dry_run:
        return os.path.join(save_folder, "debug")

    dtime = kwargs.get("dtime", datetime.datetime.today().strftime("%Y%m%d-%H%M%S"))
    unique_id = str(uuid.uuid4())[:8]
    microscope = kwargs.get("microscope", None)
    optim = kwargs.get("optim", None)
    ndims = kwargs.get("ndims", None)
    degree = kwargs.get("degree", None)
    objectives = kwargs.get("objectives", None)
    regressor_name = kwargs.get("regressor_name", None)

    folder_name = "{dtime}_{unique_id}_{microscope}_{optim}_{regressor_name}".format(
        dtime=dtime, unique_id=unique_id, microscope=microscope, optim=optim, regressor_name=regressor_name
    )
    return os.path.join(save_folder, folder_name)

def print_(x_selected, y_result, obj_names, param_names):
    """
    Implements a print helper function

    :param hide: A `bool` whether the print is occuring
    """
    print("========================================================")
    print("[----] Selected")
    for key, value in x_selected.items():
        print("[----]", key, value)
    print("[----] Objectives")
    for name, _y in zip(obj_names, y_result):
        print("[----]", name, "{:0.4f}".format(_y))
    print("========================================================")

def get_trials(save_folder, nbre_trials, optim_length):
    """
    Verifies the trial at which the restoration should take place

    :param save_folder: A `str` of the save folder
    :param nbre_trials: An `int` of the number of trials
    :param optim_length: An `int` for the length of the optimization

    :returns : An `int` of the start trial
    """
    # We assume that the file is missing and iterate until a file is found
    missing_trials = []
    if os.path.isfile(os.path.join(save_folder, "optim.hdf5")):
        with h5py.File(os.path.join(save_folder, "optim.hdf5"), "r") as file:
            for i in range(nbre_trials):
                if (str(i) not in file["X"]) or (len(file["X"][str(i)]) != optim_length):
                    missing_trials.append(i)
    else:
        for i in range(nbre_trials):
            if not os.path.isfile(os.path.join(save_folder, f"X_{i}.csv")):
                missing_trials.append(i)

    return missing_trials

    # if (i + 1 == nbre_trials) and (len(data) == optim_length):
    #     # Everything is done
    #     return nbre_trials
    # elif len(data) == optim_length:
    #     # The last experiment was completed
    #     return i + 1
    # else:
    #     # The last experiment was not completed
    #     return i

def mrange(num_processes, trials):
    """
    Creates a generator of multiple range objects.

    :param num_processes: An `int` of the number of processes
    :param start: An `int` of the starting range point
    :param stop: An `int` of the stoping range point

    :yields : A `range` iterator object
    """
    for s in range(0, len(trials), num_processes):
        yield [trials[i] for i in range(s, min(len(trials), s + num_processes))]
