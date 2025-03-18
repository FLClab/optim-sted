
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
import random
matplotlib.use("tkagg")

from copy import deepcopy
from matplotlib import pyplot
from shutil import copyfile
from collections import defaultdict

import abberior

from banditopt import algorithms, objectives, user, utils, models
from banditopt.algorithms import TS_sampler

from stedopt.tools import create_dataset, create_group, ParameterSpaceGenerator, \
                        KnowledgeGenerator, create_savename, print_, get_trials, \
                        mrange, MicroscopeConfigurator, RegionSelector
from stedopt.articulation import PreferenceArticulator
from stedopt.pareto import ParetoSampler
from stedopt.experiment import Experiment
from stedopt.configuration import Configurator
from stedopt.routine import RoutineGenerator
from stedopt.context import ContextHandler
from stedopt.algorithms import AlgorithmsConfigurator
from stedopt.defaults import obj_dict, regressors_dict

import defaults

# PATH = os.path.join(
#     "C:", os.sep, "Users", "abberior", "Desktop", "DATA", "abilodeau",
#     "20230409_STED-optim"
# )
PATH = os.path.join(os.getcwd(), "data", "20230409_STED-optim")

# PATH = "../data/abberior"
SAVE = True

# Sets the configuration
config_overview = abberior.microscope.get_config("Setting overview configuration.")
config_conf = abberior.microscope.get_config("Setting confocal configuration.")
config_sted = abberior.microscope.get_config("Setting STED configuration.")
config_focus = abberior.microscope.get_config("Setting FOCUS configuration.")

def run_TS(config, prefart="random", restore_folder=None, dry_run=False, verbose=True):
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
    # Sets random state
    numpy.random.seed(config.random_state)
    random.seed(config.random_state)

    printf = print if verbose else lambda x : x

    # Defines necessary variables
    ndims, param_space_bounds, n_points, conditions = ParameterSpaceGenerator(config.microscope_conf["mode"])(
        config.param_names, config.x_mins, config.x_maxs, config.n_divs_default
    )

    # Stores in configuration
    config.config["ndims"] = ndims
    config.config["param_space_bounds"] = param_space_bounds
    config.config["n_points"] = n_points
    config.config["conditions"] = conditions

    # This is necessary for algorithms configuration
    config.config["datamap_opts"] = {
        "shape" : [int(config.default_values_dict["imagesize"][i] / config.default_values_dict["pixelsize"][i]) for i in range(2)]
    }

    # Gets the information from the computer
    config.config["computer"] = platform.uname()._asdict()

    config.config["im_dir_names"] = (
        "conf1", "sted", "conf2", "X", "y",
        "dts_sampling", "dts_update", "s_lb", "s_ub", "dts",
        "fluomap", "y_samples", "pareto_indexes", "selected_indexes", "history"
    )
    if isinstance(restore_folder, str):
        save_folder = restore_folder
        config.config["save_folder"] = save_folder
        print("========================================================")
        print("[----] Restoring from")
        print("[----]", save_folder)
        print("========================================================\n")
        trials = get_trials(save_folder, config.nbre_trials, config.optim_length)

        # Creates the file logging
        with h5py.File(os.path.join(save_folder, "optim.hdf5"), "a") as logging:
            for dir_name in config.im_dir_names:
                create_group(logging, dir_name, exist_ok=True)
    else:
        # Creates the savename folder
        save_folder = create_savename(
            dry_run = dry_run,
            save_folder = PATH,
            optim = prefart,
            ndims = sum(ndims),
            objectives = len(config.obj_names),
            microscope = config.microscope_conf["mode"],
            regressor_name = config.regressor_name
        )
        config.config["save_folder"] = save_folder
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
        copyfile(models.__file__, os.path.join(config.save_folder, "models.py"))
        copyfile(defaults.__file__, os.path.join(config.save_folder, "defaults.py"))
        with open(os.path.join(save_folder, "config.yml"), 'w') as f:
            yaml.dump(config.config, f)

        # Save template
        abberior.microscope.measurement.save_as(os.path.join(save_folder, "template.msr"))

        # We initialize start trial to 0
        trials = list(range(config.nbre_trials))

        # Creates the file logging
        with h5py.File(os.path.join(save_folder, "optim.hdf5"), "w") as logging:
            for dir_name in config.im_dir_names:
                group = logging.create_group(dir_name)

    # Creates the preference articulator
    pref_articulator = PreferenceArticulator(
        config=config.config, mode=prefart, model_config=config.model_config
    )
    pareto_sampler = ParetoSampler(
        x_mins=config.x_mins, x_maxs=config.x_maxs, n_points=n_points, ndims=ndims, obj_dict=obj_dict,
        config=config.config, **config.pareto_opts
    )
    microscope_configurator = MicroscopeConfigurator(
        defaults, config.microscope_conf, config.params_conf, config.default_values_dict, ndims, config.param_names,
        config_conf, config_sted
    )
    region_selector = RegionSelector(
        config_overview, config=config.region_opts
    )
    knowledge_generator = KnowledgeGenerator(
        config.microscope_conf, config=config, ndims=ndims, **config.knowledge_opts
    )
    context_handler = ContextHandler(
        config.ctx_opts
    )

    # Iterates across the selected number of trials
    for no_trial in trials:
        print(f"[----] Trial {no_trial}...")
        trial_id = "{:4d}".format(no_trial).replace(" ", "-")

        with h5py.File(os.path.join(save_folder, "optim.hdf5"), "a") as logging:
            # Creates the image dataset
            for dir_name in ["conf1", "conf2"]:
                create_dataset(
                    logging[dir_name], str(no_trial),
                    shape=(config.optim_length, config.default_values_dict["num_acquisition"], *[int(config.default_values_dict["imagesize"][i] / config.default_values_dict["pixelsize"][i]) for i in range(2)]),
                    dtype=numpy.uint16
                )
            for dir_name in ["sted"]:
                num_channels = abberior.microscope.get_num_channels(config_sted)
                create_dataset(
                    logging[dir_name], str(no_trial),
                    shape=(config.optim_length, config.default_values_dict["num_acquisition"], num_channels, *[int(config.default_values_dict["imagesize"][i] / config.default_values_dict["pixelsize"][i]) for i in range(2)]),
                    dtype=numpy.uint16
                )
            # Creates the groups for y_samples, pareto_indexes and selected_indexes
            for dir_name in config.im_dir_names[-4:]:
                create_group(logging[dir_name], str(no_trial), exist_ok=True)

        #Define the algos
        algos = AlgorithmsConfigurator(config.config, param_space_bounds=config.param_space_bounds)

        s_lb, s_ub, dts, dts_sampling, dts_update = [], [], [], [], []
        for iter_idx in range(config.optim_length):
            print(f"[{trial_id}] Iteration {iter_idx}...")

            # Sample objective values over the parameter space
            t0 = time.time()

            # Sets the next regions to images
            xoff, yoff = next(region_selector)
            abberior.microscope.set_offsets(config_conf, xoff, yoff)
            abberior.microscope.set_offsets(config_sted, xoff, yoff)
            abberior.microscope.set_offsets(config_focus, xoff, yoff)

            input("Now is a good time to move focus... (press enter when done)")

            # Acquire a confocal image that will be used in the history
            conf2, _ = abberior.microscope.acquire(config_conf)
            conf2 = numpy.array(conf2)[0][0]
            conf_init = conf2.copy()
            fg_init = utils.get_foreground(conf2)

            history = defaultdict(list)
            for n in range(config.default_values_dict["num_acquisition"]):

                # Extract context from the last acquired confocal image
                fg_c = utils.get_foreground(conf2)
                context = context_handler(conf2, fg_c)
                if context_handler.use_ctx:
                    history["ctx"].append(numpy.array([context]))

                if n == 0:
                    X, y_samples, points_arr2d, timesperpixel, ndf = pareto_sampler(
                        iter_idx, algos, config.with_time, conditions=config.conditions,
                        history=history
                    )

                    if SAVE:
                        with h5py.File(os.path.join(save_folder, "optim.hdf5"), "a") as logging:
                            create_dataset(logging["y_samples"][str(no_trial)], str(iter_idx), data=y_samples)
                            create_dataset(logging["pareto_indexes"][str(no_trial)], str(iter_idx), data=ndf)

                    # Points selection
                    if iter_idx < config.knowledge_opts["num_random_samples"]:
                        x_selected = knowledge_generator(X)
                        for condition in conditions:
                            x_selected = condition(x_selected)
                    else:
                        # Selects the optimal point
                        index, selected_index = pref_articulator(
                            y_samples, [obj_dict[name] for name in config.obj_names],
                            config.with_time, timesperpixel, **config.articulation_opts
                        )
                        x_selected = X[index, :][:,numpy.newaxis]

                        # Saves pareto indices and samples
                        if SAVE:
                            with h5py.File(os.path.join(save_folder, "optim.hdf5"), "a") as logging:
                                create_dataset(logging["selected_indexes"][str(no_trial)], str(iter_idx), data=[index, selected_index])

                # Configures the microscope accordingly
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
                    logging["conf1"][str(no_trial)][iter_idx, n] = conf1
                    logging["sted"][str(no_trial)][iter_idx, n] = sted_image
                    logging["conf2"][str(no_trial)][iter_idx, n] = conf2

                # foreground on confocal image
                fg_c = utils.get_foreground(conf1)
                # foreground on sted image
                if numpy.any(sted_image):
                    fg_s = utils.get_foreground(sted_image[-1])
                else:
                    fg_s = numpy.ones_like(fg_c)
                # remove STED foreground points not in confocal foreground, if any
                fg_s *= fg_init

                # Evaluate the objective results
                # obj_dict["Resolution"] = Resolution(pixelsize=default_values_dict["pixelsize"], positions=positions) #Just in case the pixelsize have changed
                # obj_dict["Resolution"] = objectives.Resolution(pixelsize=default_values_dict["pixelsize"][0]) #Just in case the pixelsize have changed
                # y_result = numpy.array([obj_dict[name].evaluate([sted_image[-1]], conf1, conf2, fg_s, fg_c) for name in config.obj_names])
                # Evaluate the objective results
                obj_dict["Resolution"] = objectives.Resolution(pixelsize=config.default_values_dict["pixelsize"]) #Just in case the pixelsize have changed
                y_result = []
                for name in config.obj_names:
                    if name == "Resolution":
                        # Valid for all microscopes
                        acquisition = sted_image[-1]
                        value = obj_dict[name].evaluate([acquisition], conf_init, conf2, fg_s, fg_init)
                    elif name == "Squirrel":
                        threshold = numpy.quantile(conf_init, 0.95)
                        fg = conf_init > threshold
                        value = obj_dict[name].evaluate([sted_image[-1]], conf_init, conf2, fg_s, fg_init)
                    else:
                        value = obj_dict[name].evaluate([sted_image[-1]], conf_init, conf2, fg_s, fg_init)
                    y_result.append(value)
                y_result = numpy.array(y_result)

                # Keep track of the history
                history["X"].append(x_selected)
                history["y"].append(y_result)

            # We evaluate the resolution on the first image only
            y_result[0] = history["y"][0][0]

            if not config.hide_acquisition:
                print_(x_selected_dict, y_result, config.obj_names, config.param_names)

            printf(f"[{trial_id}] Took {time.time() - t0}")
            t0 = time.time()
            weights = [1.0] * len(config.obj_names)
            if config.knowledge_opts["update_posterior"]:
                update_posterior = True
            else:
                update_posterior = iter_idx >= config.knowledge_opts["num_random_samples"] - 1
            shuffled_range = list(range(len(config.obj_names)))
            random.shuffle(shuffled_range)
            [
                algos[i].update(
                    x_selected.T, y_result[i].flatten(), history=history, weights=[weights[i]],
                    update_posterior=update_posterior
                ) for i in shuffled_range
            ]
            dt_update = time.time()-t0
            #save s_lb and s_ub, and calculation time
            # dts_sampling.append(dt_sampling)
            dts_update.append(dt_update)
    #            s_lb.append(algo.s_lb)
    #            s_ub.append(algo.s_ub)

            # Save data to logging
            with h5py.File(os.path.join(save_folder, "optim.hdf5"), "a") as logging:
                create_dataset(logging["X"], str(no_trial), data=algos[0].X)
                y_array = numpy.hstack([algos[i].y[:,numpy.newaxis] for i in range(len(config.obj_names))])
                create_dataset(logging["y"], str(no_trial), data=y_array)
                create_dataset(logging["dts_sampling"], str(no_trial), data=dts_sampling)
                create_dataset(logging["dts_update"], str(no_trial), data=dts_update)

                create_group(logging["history"][str(no_trial)], str(iter_idx))
                for key, value in history.items():
                    create_dataset(logging["history"][str(no_trial)][str(iter_idx)], key, data=numpy.array(value))

            # Saves checkpoint of model
            for i, obj_name in enumerate(config.obj_names):
                algos[i].save_ckpt(config.save_folder, prefix=obj_name, trial=str(no_trial))

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/config_sted.yml",
                        help="Path a config file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Wheter a dry-run is run")
    parser.add_argument("--prefart", type=str, default="random",
                        help="Selects the preference articulation method. Select from: `random`, `optim`, `prefnet`, `region`")
    parser.add_argument("--restore-folder", type=str, default=None,
                        help="Selects the model from which to restore training")
    parser.add_argument("--options", type=str, default=None, nargs="+",
                        help="Modifies the configuration file. Nested dictionary keys can be accessed with the '/'")
    args = parser.parse_args()

    if args.options:
        assert len(args.options) % 2 == 0, "The number of options should be a multiple of 2 (key : value)"

    # Loads config from given file
    if isinstance(args.restore_folder, str):
        config = yaml.load(open(os.path.join(args.restore_folder, "config.yml"), "r"), Loader=yaml.Loader)
        configurator = Configurator(config)
    else:
        configurator = Configurator(args.config, options=args.options)

    # Runs TS
    # print(configurator)
    run_TS(config=configurator, dry_run=args.dry_run, prefart=args.prefart, restore_folder=args.restore_folder)
