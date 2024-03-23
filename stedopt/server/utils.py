
import numpy
import os
import yaml
import copy
import sklearn
import h5py
import shutil
import time

from matplotlib import pyplot
from scipy.spatial import distance
from collections import defaultdict
from tqdm.auto import tqdm, trange

BOUNDS = {
    "Resolution" : {
        "min" : -numpy.inf,
        "max" : 140.,
    },
    "SNR" : {
        "min" : 0.1,
        "max" : numpy.inf,
    },
    "Bleach" : {
        "min" : -numpy.inf,
        "max" : 0.5,
    },
    "Squirrel" : {
        "min" : -numpy.inf,
        "max" : 0.4,
    },
    "FFTMetric": {
        "min": -numpy.inf,
        "max": 0.35,
    },
    "Crosstalk": {
        "min": -numpy.inf,
        "max": 0.25,
    }
}

def isin_bounds(y, obj_name):
    """
    Verifies wheter the objectives are within the bounds

    :param y: A `numpy.ndarray` of the objective values
    :param obj_name: A `str` of the objective name

    :return : A `numpy.ndarray` of the parameters
    """
    return numpy.logical_and(y >= BOUNDS[obj_name]["min"], y <= BOUNDS[obj_name]["max"])

def rescale_obj(config, y, obj_name):
    """
    Rescales the objectives using the scaling function

    :param y: A `numpy.ndarray` of the objective value
    :param obj_name: A `str` of the objective name

    :return : A `numpy.ndarray` of the rescaled array
    """
    return (y) * (CONFIG["obj_normalization"][obj_name]["max"] - CONFIG["obj_normalization"][obj_name]["min"]) + CONFIG["obj_normalization"][obj_name]["min"]

def scale_obj(config, y, obj_name):
    """
    Scales the objectives using the scaling min max function

    :param y: A `numpy.ndarray` of the objective value
    :param obj_name: A `str` of the objective name

    :return : A `numpy.ndarray` of the rescaled array
    """
    return (y - CONFIG["obj_normalization"][obj_name]["min"]) / (CONFIG["obj_normalization"][obj_name]["max"] - CONFIG["obj_normalization"][obj_name]["min"])

def scale_param(config, X, param_name):
    """
    Scales the parameters using the scaling function

    :param X: A `numpy.ndarray` of the parameters
    :param param_name: A `str` of the parameter

    :return : A `numpy.ndarray` of the parameters
    """
    m = CONFIG["x_mins"][CONFIG["param_names"].index(param_name)]
    M = CONFIG["x_maxs"][CONFIG["param_names"].index(param_name)]
    return (X - m) / (M - m)

def load_images(config, path, trial=0, idx=0, **kwargs):
    """
    Loads the data from the given path

    :param path: A `str` of the path
    :param trial: An `int` of the number of path

    :returns : A `dict` of the X
               A `dict` of the y
    """
    conf1, sted_image, conf2 = None, None, None
    if os.path.isfile(os.path.join(path, "optim.hdf5")):

        # shutil.copy(os.path.join(path, "optim.hdf5"), os.path.join(path, "tmp.hdf5"))
        with h5py.File(os.path.join(path, "optim.hdf5"), "r+") as file:
            conf1 = file["conf1"][f"{trial}"][idx][()]
            sted_image = file["sted"][f"{trial}"][idx][()]
            conf2 = file["conf2"][f"{trial}"][idx][()]

        # if os.path.isfile(os.path.join(path, "tmp.hdf5")):
        #     os.remove(os.path.join(path, "tmp.hdf5"))

        if conf1.ndim > 2:
            conf1 = conf1[0]
        if conf2.ndim > 2:
            conf2 = conf2[-1]

    return conf1, sted_image, conf2

def load_data(config, path, trial=0, scale=False, gridsearch=False, slc=slice(None, None), **kwargs):
    """
    Loads the data from the given path

    :param path: A `str` of the path
    :param trial: An `int` of the number of path

    :returns : A `dict` of the X
               A `dict` of the y
    """
    ndims = kwargs.get("ndims")

    if os.path.isfile(os.path.join(path, "optim.hdf5")):

        # shutil.copy(os.path.join(path, "optim.hdf5"), os.path.join(path, "tmp.hdf5"))
        with h5py.File(os.path.join(path, "optim.hdf5"), "r+") as file:
            if trial == "all":
                X, y = [], []
                other_X, other_y = None, None
                for key in sorted(file["X"].keys(), key=lambda x : int(x)):
                    X.append(file["X"][key][slc])
                    y.append(file["y"][key][slc])
                X = numpy.concatenate(X, axis=0)
                y = numpy.concatenate(y, axis=0)
            else:
                X = file["X"][f"{trial}"][slc]
                y = file["y"][f"{trial}"][slc]

                other_X, other_y = [], []
                for key in sorted(file["X"].keys(), key=lambda x : int(x)):
                    other_X.append(file["X"][key][slc])
                    other_y.append(file["y"][key][slc])


        # if os.path.isfile(os.path.join(path, "tmp.hdf5")):
        #     os.remove(os.path.join(path, "tmp.hdf5"))
    else:
        X = numpy.loadtxt(os.path.join(path, f"X_{trial}.csv"), delimiter=",")
        y = numpy.loadtxt(os.path.join(path, f"y_{trial}.csv"), delimiter=",")
    if X.ndim == 1:
        X = X[:, numpy.newaxis]
    X = {
        param_name : scale_param(config, X[:, sum(ndims[:i]) : sum(ndims[:i]) + ndims[i]], param_name)
                     if scale else X[:, sum(ndims[:i]) : sum(ndims[:i]) + ndims[i]]
        for i, param_name in enumerate(config["param_names"])
    }
    other_X = [
        {
            param_name : scale_param(config, o_X[:, sum(ndims[:i]) : sum(ndims[:i]) + ndims[i]], param_name)
                         if scale else o_X[:, sum(ndims[:i]) : sum(ndims[:i]) + ndims[i]]
            for i, param_name in enumerate(config["param_names"])
        }
        for o_X in other_X
    ]
    y = {
        obj_name : y[:, i]
        for i, obj_name in enumerate(config["obj_names"])
    }
    other_y = [
        {
            obj_name : o_y[:, i]
            for i, obj_name in enumerate(config["obj_names"])
        }
        for o_y in other_y
    ]
    return X, y, other_X, other_y

def get_data(path, trial):

    config = yaml.load(open(os.path.join(path, "config.yml"), "r"), Loader=yaml.Loader)
    ndims = []
    for param_name in config["param_names"]:
        if (param_name in ["decision_time", "threshold_count"]) and (config["microscope"] == "DyMIN"):
            ndims.append(2)
        else:
            ndims.append(1)
    N_POINTS = [config["n_divs_default"]]*sum(ndims)
    slc = slice(None, 200)

    try:
        # Case where data can be read
        X, y, all_X, all_y = load_data(config, path, trial=str(trial), slc=slc, ndims=ndims)

        data = {
            "config" : config,
            "X" : {key : value.tolist() for key, value in X.items()},
            "y" : {key : value.tolist() for key, value in y.items()},
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
        data = {
            "config" : config
        }
    return data


def load_history(path, trial=0, scale=False, gridsearch=False):
    """
    Loads the data from the given path

    :param path: A `str` of the path
    :param trial: An `int` of the number of path

    :returns : A `dict` of the X
               A `dict` of the y
    """
    if os.path.isfile(os.path.join(path, "optim.hdf5")):
        with h5py.File(os.path.join(path, "optim.hdf5"), "r") as file:
            history = file["history"][f"{trial}"]
            X, y, ctx = [], [], []
            for i in range(CONFIG["optim_length"]):
                if str(i) not in history:
                    break
                X.append(history[str(i)]["X"][()])
                y.append(history[str(i)]["y"][()])
                ctx.append(history[str(i)]["ctx"][()])
    X, y, ctx = numpy.array(X).squeeze(), numpy.array(y), numpy.array(ctx)
    X = {
        param_name : X[:, :, sum(NDIMS[:i]) : sum(NDIMS[:i]) + NDIMS[i]]
        for i, param_name in enumerate(CONFIG["param_names"])
    }
    y = {
        obj_name : y[:, :, i]
        for i, obj_name in enumerate(CONFIG["obj_names"])
    }
    return X, y, ctx.squeeze()

def get_grid(**kwargs):
    """
    Creates a grid from the given parameters

    :returns : A `numpy.ndarray` of the grid
    """
    grid = [numpy.linspace(0, 350e-3, 50)]
    for i, param_name in enumerate(CONFIG["param_names"]):
        if param_name == "p_sted":
            continue
        for j in range(NDIMS[i]):
            if NDIMS[i] > 1:
                param = kwargs.get("_".join((param_name, str(j))))
            else:
                param = kwargs.get(param_name)

            if param_name in ["p_ex", "decision_time", "pdt"]:
                grid.append(numpy.tile(param * 1e-6, (50, )))
            else:
                grid.append(numpy.tile(param, (50, )))
    return numpy.stack(grid, axis=-1)
