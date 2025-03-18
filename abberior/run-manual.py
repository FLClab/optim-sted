
import numpy
import os
import time
import yaml
import platform
import skimage.io as skio
import copy
import datetime
import matplotlib
import h5py
matplotlib.use("tkagg")

from matplotlib import pyplot
from shutil import copyfile

import abberior

from banditopt import algorithms, objectives, user, utils
from banditopt.algorithms import TS_sampler

from stedopt.tools import (datamap_generator, create_dataset, create_group,
                        ParameterSpaceGenerator, MicroscopeConfigurator,
                        RegionSelector, PointSelector)
from stedopt.articulation import PreferenceArticulator
from stedopt.pareto import ParetoSampler

import defaults

PATH = os.path.join(
    "C:", os.sep, "Users", "abberior", "Desktop", "DATA", "abilodeau",
    "20211029_User-Study"
)

# PATH = "../data"
SAVE = True

# Define the objectives and regressors here
obj_dict = {
    "SNR" : objectives.Signal_Ratio(75),
    "Bleach" : objectives.Bleach(),
    "Resolution" : objectives.Resolution(pixelsize=15e-9),
    "Squirrel" : objectives.Squirrel()
}
regressors_dict = {
    "sklearn_BayesRidge":algorithms.sklearn_BayesRidge,
    "sklearn_GP":algorithms.sklearn_GP,
}

# Sets the configuration
config_overview = abberior.microscope.get_config("Setting overview configuration.")
config_conf = abberior.microscope.get_config("Setting confocal configuration.")
config_sted = abberior.microscope.get_config("Setting STED configuration.")

def create_savename(**kwargs):
    """
    Creates path name of the save folder

    :returns : A `str` of the savepath
    """
    dry_run = kwargs.get("dry_run")
    save_folder = kwargs.get("save_folder", PATH)
    if dry_run:
        return os.path.join(PATH, "debug")

    dtime = kwargs.get("dtime", datetime.datetime.today().strftime("%Y%m%d-%H%M%S"))
    microscope = kwargs.get("microscope", None)
    optim = kwargs.get("optim", None)
    ndims = kwargs.get("ndims", None)
    degree = kwargs.get("degree", None)
    objectives = kwargs.get("objectives", None)
    folder_name = "{dtime}_{microscope}_{optim}_{ndims}params_{objectives}objectives_degree{degree}".format(
        dtime=dtime, microscope=microscope, optim=optim, ndims=ndims, degree=degree, objectives=objectives
    )
    return os.path.join(save_folder, folder_name)

def print_(x_selected, y_result, obj_names, param_names):
    """
    Implements a print helper function
    """
    print("============================")
    print("[----] Selected")
    for key, value in x_selected.items():
        print("[----]", key, value)
    print("[----] Objectives")
    for name, _y in zip(obj_names, y_result):
        print("[----]", name, "{:0.4f}".format(_y))
    print("============================")

def get_start_trial(save_folder, nbre_trials, optim_length):
    """
    Verifies the trial at which the restoration should take place

    :param save_folder: A `str` of the save folder
    :param nbre_trials: An `int` of the number of trials
    :param optim_length: An `int` for the length of the optimization

    :returns : An `int` of the start trial
    """
    # We assume that the file is missing and iterate until a file is found
    if os.path.isfile(os.path.join(save_folder, "optim.hdf5")):
        with h5py.File(os.path.join(save_folder, "optim.hdf5"), "r") as file:
            flag = False
            for i in reversed(range(nbre_trials)):
                flag = f"{i}" in file["X"]
                if flag:
                    break
            data = file["X"][f"{i}"][()]
    else:
        flag = False
        for i in reversed(range(nbre_trials)):
            flag = os.path.isfile(os.path.join(save_folder, f"X_{i}.csv"))
            if flag:
                break
        # We verify that the latest file is complete
        data = numpy.loadtxt(os.path.join(save_folder, f"X_{i}.csv"), delimiter=",")

    if (i + 1 == nbre_trials) and (len(data) == optim_length):
        # Everything is done
        return nbre_trials
    elif len(data) == optim_length:
        # The last experiment was completed
        return i + 1
    else:
        # The last experiment was not completed
        return i

def run_TS(config, save_folder=PATH, regressor_name="sklearn_BayesRidge",
            regressor_args= {"default":{}, "SNR":{}, "bleach":{}}, n_divs_default = 25,
            param_names = ["p_ex", "p_sted"], with_time=True, default_values_dict={"dwelltime":20e-6},
            params_conf = { 'p_ex':100e-6, 'p_sted':0, 'dwelltime':10.0e-6,}, x_mins=[400e-6*2**-3, 900e-6*2**-3, ],
            x_maxs=[400e-6*2**4, 900e-6*2**4], obj_names=["SNR", "bleach"], optim_length = 30,
            nbre_trials = 2, borders=None, dry_run=False, obj_normalization=None,
            pareto_opts={"mode" : "default"}, prefart=None, model_config=None, restore_folder=None,
            microscope_conf={"mode" : "AbberiorRESCue", "advanced" : False}, **kwargs):
    """This function does multi-objective Thompson sampling optimization of parameters of simulated STED images.

    :param config: Dictionary of all the function parameters to be saved as a yaml file
    :save_folder: Directory that will be created to store the optimization data
    :regressor_name: Name (str key of a dictionary) of the regressor class
    :regressor_args: Dictionary of the arguments of the regressor class arguments
    :n_divs_default: Number (int) of discretizations of the paramter space along each axis
    :param_names: Parameters (str name of arguments for the STED simulator) to optimize
    :with_time: True if the imaging time is optimized
    :default_values_dict: Default argument values dictionary for the STED simulator (STED)
    :params_conf: Default argument values dictionary for the STED simulator (confocals 1 and 2)
    :x_mins: List of the smallest parameter values of the parameter space (same order as param_names)
    :x_maxs: List of the largest parameter values of the parameter space (same order as param_names)
    :obj_names: Name (str key of a dictionary) of objectives
    :optim_length: Number of iterations of an optimization
    :nbre_trials: Number of trials
    :borders: None, or List of tuples (minval, maxval) to cap the objective values in the visualization
              for tradeoff selection
    """
    # Defines necessary variables
    ndims, param_space_bounds, n_points, conditions = ParameterSpaceGenerator(microscope_conf["mode"])(
        param_names, x_mins, x_maxs, n_divs_default
    )

    # Gets the information from the computer
    config["computer"] = platform.uname()._asdict()

    im_dir_names = (
        "conf1", "sted", "conf2", "X", "y",
        "dts_sampling", "dts_update", "s_lb", "s_ub", "dts",
        "fluomap", "y_samples", "pareto_indexes", "selected_indexes"
    )
    if isinstance(restore_folder, str):
        save_folder = restore_folder
        start_trial = get_start_trial(save_folder, nbre_trials, optim_length)

        # Creates the file logging
        with h5py.File(os.path.join(save_folder, "optim.hdf5"), "a") as logging:
            for dir_name in im_dir_names:
                create_group(logging, dir_name, exist_ok=True)
    else:
        # Creates the savename folder
        save_folder = create_savename(
            dry_run = dry_run,
            save_folder = save_folder,
            optim = prefart,
            ndims = sum(ndims),
            degree = regressor_args["default"]["degree"],
            objectives = len(obj_names),
            microscope = microscope_conf["mode"]
        )
        print("========================================================")
        print("[----] Savefolder")
        print("[----]", save_folder)
        print("========================================================\n")

        # Creates the output directory
        if not os.path.isfile(save_folder):
            os.makedirs(save_folder, exist_ok=dry_run)
        root = os.path.dirname(os.path.abspath(__file__))
        copyfile(os.path.abspath(__file__), os.path.join(save_folder,"run.py"))
        copyfile(algorithms.__file__, os.path.join(save_folder,"algorithms.py"))
        with open(os.path.join(save_folder, "config.yml"), 'w') as f:
            yaml.dump(config, f)

        # Save template
        abberior.microscope.measurement.save_as(os.path.join(save_folder, "template.msr"))

        # We initialize start trial to 0
        start_trial = 0

        # Creates the file logging
        with h5py.File(os.path.join(save_folder, "optim.hdf5"), "w") as logging:
            for dir_name in im_dir_names:
                group = logging.create_group(dir_name)

    # Creates the configuration
    microscope_configurator = MicroscopeConfigurator(
        defaults, microscope_conf, params_conf, default_values_dict, ndims, param_names,
        config_conf, config_sted
    )
    region_selector = RegionSelector(
        config_overview, config=config.region_opts
    )
    point_selector = PointSelector(param_names, param_space_bounds, obj_names)

    # Iterates across the selected number of trials
    for no_trial in range(start_trial, nbre_trials):
        print(f"[----] Trial {no_trial}...")

        with h5py.File(os.path.join(save_folder, "optim.hdf5"), "a") as logging:
            # Creates the image dataset
            for dir_name in ["conf1", "conf2"]:
                create_dataset(
                    logging[dir_name], str(no_trial),
                    shape=(optim_length, *[int(default_values_dict["imagesize"][i] / default_values_dict["pixelsize"][i]) for i in range(2)]),
                    dtype=numpy.uint16
                )
            for dir_name in ["sted"]:
                num_channels = abberior.microscope.get_num_channels(config_sted)
                create_dataset(
                    logging[dir_name], str(no_trial),
                    shape=(optim_length, num_channels, *[int(default_values_dict["imagesize"][i] / default_values_dict["pixelsize"][i]) for i in range(2)]),
                    dtype=numpy.uint16
                )
            # Creates the groups for y_samples, pareto_indexes and selected_indexes
            for dir_name in im_dir_names[-3:]:
                create_group(logging[dir_name], str(no_trial), exist_ok=True)

        #Define the algos
        algos = []
        for name in obj_names:
            args = regressor_args["default"].copy()
            args["param_space_bounds"] = param_space_bounds
            for key, value in regressor_args[name].items():
                args[key] = value
            algos.append(TS_sampler(regressors_dict[regressor_name](**args)))

        s_lb, s_ub, dts, dts_sampling, dts_update = [], [], [], [], []
        for iter_idx in range(optim_length):
            print(f"[----] Iteration {iter_idx}...")

            # Sample objective values over the parameter space
            t0 = time.time()

            # Points selection
            if iter_idx > 0:
                _X = algos[0].X
                _y = numpy.hstack([algos[i].y[:,numpy.newaxis] for i in range(len(obj_names))])
                print(_X.shape, _y.shape)
            else:
                _X = numpy.zeros((1, len(param_names)))
                _y = numpy.zeros((1, len(obj_names)))
            x_selected = point_selector.select(_X, _y, show=True)

            # Configures the microscope accordingly
            xoff, yoff = next(region_selector)
            abberior.microscope.set_offsets(config_conf, xoff, yoff)
            abberior.microscope.set_offsets(config_sted, xoff, yoff)
            x_selected_dict = microscope_configurator(x_selected)

            # Acquire conf1, sted, conf2
            conf1, _ = abberior.microscope.acquire(config_conf)
            sted_image, _ = abberior.microscope.acquire(config_sted)
            conf2, _ = abberior.microscope.acquire(config_conf)

            conf1 = numpy.array(conf1)[0][0]
            sted_image = numpy.array(sted_image).squeeze(axis=1)
            conf2 = numpy.array(conf2)[0][0]

            # Save acquired image
            with h5py.File(os.path.join(save_folder, "optim.hdf5"), "a") as logging:
                logging["conf1"][str(no_trial)][iter_idx] = conf1
                logging["sted"][str(no_trial)][iter_idx] = sted_image
                logging["conf2"][str(no_trial)][iter_idx] = conf2

            # foreground on confocal image
            fg_c = utils.get_foreground(conf1)
            # foreground on sted image
            if numpy.any(sted_image):
                fg_s = utils.get_foreground(sted_image[-1])
            else:
                fg_s = numpy.ones_like(fg_c)
            # remove STED foreground points not in confocal foreground, if any
            fg_s *= fg_c

            # Evaluate the objective results
            # obj_dict["Resolution"] = Resolution(pixelsize=default_values_dict["pixelsize"], positions=positions) #Just in case the pixelsize have changed
            obj_dict["Resolution"] = objectives.Resolution(pixelsize=default_values_dict["pixelsize"][0]) #Just in case the pixelsize have changed
            y_result = numpy.array([obj_dict[name].evaluate([sted_image[-1]], conf1, conf2, fg_s, fg_c) for name in obj_names])

            print_(x_selected_dict, y_result, obj_names, param_names)

            # Normalize the objectives accordingly
            # y_result = numpy.array([
            #     (_y - obj_normalization[obj_name]["min"]) / (obj_normalization[obj_name]["max"] - obj_normalization[obj_name]["min"])
            #     for obj_name, _y in zip(obj_names, y_result)
            # ])

#            y_result = obj_func.sample(x_selected.T)
            print(f"[----] Took {time.time() - t0}")
            t0 = time.time()
            [algos[i].update(x_selected.T, y_result[i].flatten()) for i in range(len(obj_names))]
            dt_update = time.time()-t0
            #save s_lb and s_ub, and calculation time
            # dts_sampling.append(dt_sampling)
            dts_update.append(dt_update)
#            s_lb.append(algo.s_lb)
#            s_ub.append(algo.s_ub)

            # Save data to logging
            with h5py.File(os.path.join(save_folder, "optim.hdf5"), "a") as logging:
                create_dataset(logging["X"], str(no_trial), data=algos[0].X)
                y_array = numpy.hstack([algos[i].y[:,numpy.newaxis] for i in range(len(obj_names))])
                create_dataset(logging["y"], str(no_trial), data=y_array)
                create_dataset(logging["dts_sampling"], str(no_trial), data=dts_sampling)
                create_dataset(logging["dts_update"], str(no_trial), data=dts_update)

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/config_abberior.yml",
                        help="Path a config file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Wheter a dry-run is run")
    parser.add_argument("--prefart", type=str, default="random",
                        help="Selects the preference articulation method. Select from: `random`, `optim`, `prefnet`")
    parser.add_argument("--degree", type=int, default=None,
                        help="Selects the degree of the polynomial used for regression")
    parser.add_argument("--restore-folder", type=str, default=None,
                        help="Selects the model from which to restore training")
    args = parser.parse_args()

    # Loads config from given file
    if isinstance(args.restore_folder, str):
        config = yaml.load(open(os.path.join(args.restore_folder, "config.yml"), "r"), Loader=yaml.Loader)
    else:
        config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

        # Updates the degree of the polynomial if given in parser
        if isinstance(args.degree, int):
            config["regressor_args"]["default"]["degree"] = args.degree

    # Runs TS
    run_TS(config=config, **config, dry_run=args.dry_run, prefart=args.prefart, restore_folder=args.restore_folder)
