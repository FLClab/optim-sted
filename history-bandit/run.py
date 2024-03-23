
import numpy
import os
import time
import yaml
import platform
import copy
import datetime
import matplotlib
import h5py
import random
import multiprocessing

from copy import deepcopy
from matplotlib import pyplot
from shutil import copyfile
from collections import defaultdict
from functools import partial

from pysted import base
from banditopt import algorithms, objectives, user, utils, models
from banditopt.algorithms import TS_sampler

from stedopt.tools import create_dataset, create_group, ParameterSpaceGenerator, ExperimentGenerator, \
                        DatamapGenerator, KnowledgeGenerator, create_savename, print_, get_trials, \
                        mrange
from stedopt.articulation import PreferenceArticulator
from stedopt.pareto import ParetoSampler
from stedopt.experiment import Experiment
from stedopt.configuration import Configurator
from stedopt.routine import RoutineGenerator
from stedopt.context import ContextHandler
from stedopt.algorithms import AlgorithmsConfigurator
from stedopt.defaults import obj_dict, regressors_dict

SAVE = True

def run_trial(no_trial, config, pref_articulator, pareto_sampler, experiment_generator, knowledge_generator, routine_generator, context_handler, verbose=False):
    """
    Runs a single trial of optimization run

    :param no_trial: An `int` of the trial number
    :param config: A `Configurator` object
    """
    # Sets random state in each trial
    numpy.random.seed(config.random_state)
    random.seed(config.random_state)

    printf = print if verbose else lambda x : x

    # Creates a datamap generator and update random state of generator
    opts = deepcopy(config.datamap_opts)
    if opts.get("sequence_per_trial", True):
        opts["random_state"] = config.datamap_opts["random_state"] + no_trial
    datamap_generator = DatamapGenerator(**opts)

    print(f"[----] Trial {no_trial}...")
    trial_id = "{:4d}".format(no_trial).replace(" ", "-")

    saved = False
    while not saved:
        try:
            with h5py.File(os.path.join(config.save_folder, "optim.hdf5"), "a") as logging:
                # Creates the image dataset
                for dir_name in ["conf1", "conf2"]:
                    create_dataset(logging[dir_name], str(no_trial), shape=(config.optim_length, config.default_values_dict["num_acquisition"], *datamap_generator.shape), dtype=numpy.uint16, pixelsize=config.default_values_dict["pixelsize"])
                create_dataset(logging["sted"], str(no_trial), shape=(config.optim_length, config.default_values_dict["num_acquisition"], *datamap_generator.shape), dtype=numpy.uint16, pixelsize=config.default_values_dict["pixelsize"])
                create_dataset(logging["other"], str(no_trial), shape=(config.optim_length, config.default_values_dict["num_acquisition"], *datamap_generator.shape), dtype=numpy.float32, pixelsize=config.default_values_dict["pixelsize"])
                # Creates the groups for y_samples, pareto_indexes and selected_indexes
                for dir_name in config.im_dir_names[-4:]:
                    create_group(logging[dir_name], str(no_trial), exist_ok=True)
            saved = True
        except (BlockingIOError, OSError) as err:
            # print(err)
            time.sleep(random.random() * 0.2)

    #Define the algos
    algos = AlgorithmsConfigurator(config.config, param_space_bounds=config.param_space_bounds)

    s_lb, s_ub, dts, dts_sampling, dts_update = [], [], [], [], []
    for iter_idx in range(config.optim_length):
        printf(f"[{trial_id}] Iteration {iter_idx}...")

        # Sample objective values over the parameter space
        t0 = time.time()

        # Create a datamap and acquires a confocal image
        routine = routine_generator.generate()
        molecules_disposition, positions = datamap_generator()
        experiment = experiment_generator.confocal_experiment(molecules_disposition, **routine)
        name, acquired = experiment.acquire("conf", 1, bleach=False)
        conf2 = acquired["acquisition"][-1]
        conf_init = conf2.copy()
        fg_init = utils.get_foreground(conf2)

        history = defaultdict(list)
        for n in range(config.default_values_dict["num_acquisition"]):

            # Extract context from the last acquired confocal image
            fg_c = utils.get_foreground(conf2)
            context = context_handler(conf2, fg_c)
            history["ctx"].append(numpy.array([context]))

            if (n == 0) or (config.regressor_args["default"]["every-step-decision"]):
                X, y_samples, points_arr2d, timesperpixel, ndf = pareto_sampler(
                    iter_idx, algos, config.with_time, conditions=config.conditions,
                    history=history
                )

                if SAVE:
                    saved = False
                    while not saved:
                        try:
                            with h5py.File(os.path.join(config.save_folder, "optim.hdf5"), "a") as logging:
                                create_dataset(logging["y_samples"][str(no_trial)], str(iter_idx), data=y_samples)
                                create_dataset(logging["pareto_indexes"][str(no_trial)], str(iter_idx), data=ndf)
                            saved = True
                        except (BlockingIOError, OSError) as err:
                            # print(err)
                            time.sleep(random.random() * 0.2)

                # Points selection
                if iter_idx < config.knowledge_opts["num_random_samples"]:#(config.regressor_args["default"]["degree"] + 1) ** (numpy.log10(sum(ndims) + 1)) + 1:
                    x_selected = knowledge_generator(X)
                    for condition in config.conditions:
                        x_selected = condition(x_selected)
                else:
                    if config.rescale_opts["use"]:
                        mins, maxs = config.rescale_opts["mins"], config.rescale_opts["maxs"]
                        y_samples = [
                            (y - y.min()) / (y.max() - y.min()) * (maxs[i] - mins[i]) + mins[i] for i, y in enumerate(y_samples)
                        ]

                    # Selects the optimal point
                    index, selected_index = pref_articulator(
                        y_samples, [obj_dict[name] for name in config.obj_names],
                        config.with_time, timesperpixel, **config.articulation_opts
                    )
                    x_selected = X[index, :][:,numpy.newaxis]

                    # Saves pareto indices and samples
                    if SAVE:
                        saved = False
                        while not saved:
                            try:
                                with h5py.File(os.path.join(config.save_folder, "optim.hdf5"), "a") as logging:
                                    create_dataset(logging["selected_indexes"][str(no_trial)], str(iter_idx), data=[index, selected_index])
                                saved = True
                            except (BlockingIOError, OSError) as err:
                                # print(err)
                                time.sleep(random.random() * 0.2)

            # Generate experiment
            x_selected_dict, experiment = experiment_generator(x_selected, molecules_disposition, **routine)

            # Acquire confocal image
            name, acquired = experiment.acquire("conf", 1, bleach=False)
            conf1 = acquired["acquisition"][-1]

            # Acquire STED image
            name, sted_acquired = experiment.acquire("STED", 1, bleach=True, verbose=False)
            sted_image = sted_acquired["acquisition"][-1]

            # Acquire confocal image
            name, acquired = experiment.acquire("conf", 1, bleach=False)
            conf2 = acquired["acquisition"][-1]

            # Update molecules_disposition for the next step
            molecules_disposition = acquired["bleached"][-1]

            saved = False
            while not saved:
                try:
                    # Save acquired image
                    with h5py.File(os.path.join(config.save_folder, "optim.hdf5"), "a") as logging:
                        logging["conf1"][str(no_trial)][iter_idx, n] = conf1
                        logging["sted"][str(no_trial)][iter_idx, n] = sted_image
                        logging["conf2"][str(no_trial)][iter_idx, n] = conf2
                        logging["other"][str(no_trial)][iter_idx, n] = sted_acquired["other"][-1]
                    saved = True
                except (BlockingIOError, OSError) as err:
                    # print(err)
                    time.sleep(random.random() * 0.2)

            # foreground on confocal image
            fg_c = utils.get_foreground(conf1)
            # foreground on sted image
            if numpy.any(sted_image):
                fg_s = utils.get_foreground(sted_image)
            else:
                fg_s = numpy.ones_like(fg_c)
            # remove STED foreground points not in confocal foreground, if any
            fg_s *= fg_init

            # Evaluate the objective results
            obj_dict["Resolution"] = objectives.Resolution(pixelsize=config.default_values_dict["pixelsize"]) #Just in case the pixelsize have changed
            y_result = []
            for name in config.obj_names:
                if name == "Resolution":
                    # Valid for all microscopes
                    valid = sted_acquired["other"][-1] >= 1
                    acquisition = sted_image.copy()
                    acquisition = sted_acquired["acquisition"][0] # Evaluates on first image
                    value = obj_dict[name].evaluate([acquisition], conf_init, conf2, fg_s, fg_init, positions=positions)
                    if value == obj_dict[name].res_cap:
                        if valid.any():
                            min_value = acquisition[valid].min()
                            acquisition[numpy.logical_not(valid)] = min_value
                        # Method did not converge... We ask another metric to compute the resolution
                        value = objectives.FWHMResolution(20e-9).evaluate([acquisition], conf_init, conf2, fg_s, fg_init, positions=positions)
                elif name == "Squirrel":
                    threshold = numpy.quantile(conf_init, 0.95)
                    fg = conf_init > threshold
                    value = obj_dict[name].evaluate([sted_image], conf_init, conf2, fg_s, fg_init)
                else:
                    value = obj_dict[name].evaluate([sted_image], conf_init, conf2, fg_s, fg_init)
                    value = value
                y_result.append(value)
            y_result = numpy.array(y_result)
            # y_result = numpy.array([obj_dict[name].evaluate([sted_image], conf1, conf2, fg_s, fg_c) for name in obj_names])

            # Keep track of the history
            history["X"].append(x_selected)
            history["y"].append(y_result)

        if not config.hide_acquisition:
            if pref_articulator.mode in ["optim", "region"]:
                fig, axes = pyplot.subplots(1, 3, figsize=(10,3))
                for ax, im in zip(axes.ravel(), (conf1, sted_image, conf2)):
                    im = ax.imshow(im, cmap="hot", vmin=0)
                    pyplot.colorbar(im, ax=ax)
                pyplot.show(block=True)
            print_(x_selected_dict, y_result, config.obj_names, config.param_names)

        printf(f"[{trial_id}] Took {time.time() - t0}")
        t0 = time.time()
        # weights = [1.0 if y_result[0].item() < 250 else 0.1, 1.0, 1.0]
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

        saved = False
        while not saved:
            try:
                # Save data to logging
                with h5py.File(os.path.join(config.save_folder, "optim.hdf5"), "a") as logging:
                    create_dataset(logging["X"], str(no_trial), data=algos[0].X)
                    y_array = numpy.hstack([algos[i].y[:,numpy.newaxis] for i in range(len(config.obj_names))])
                    create_dataset(logging["y"], str(no_trial), data=y_array)
                    create_dataset(logging["dts_sampling"], str(no_trial), data=dts_sampling)
                    create_dataset(logging["dts_update"], str(no_trial), data=dts_update)

                    create_group(logging["history"][str(no_trial)], str(iter_idx))
                    for key, value in history.items():
                        create_dataset(logging["history"][str(no_trial)][str(iter_idx)], key, data=numpy.array(value))
                saved = True
            except (BlockingIOError, OSError) as err:
                # print(err)
                time.sleep(random.random() * 0.2)


        # Saves checkpoint of model
        for i, obj_name in enumerate(config.obj_names):
            algos[i].save_ckpt(config.save_folder, prefix=obj_name, trial=str(no_trial))


def run_TS(config, prefart="random", restore_folder=None, dry_run=False):
    """This function does multi-objective Thompson sampling optimization of parameters of simulated STED images.

    :param config: Dictionary of all the function parameters to be saved as a yaml file
    :param prefart: A `str` of the type of preference articulation
    :param restore_folder: A `str` of a previous experiment to restore from
    :param dry_run: (optional) Whether the optimization is should be saved
    """
    # Sets random state
    numpy.random.seed(config.random_state)
    random.seed(config.random_state)

    # Defines necessary variables
    ndims, param_space_bounds, n_points, conditions = ParameterSpaceGenerator(config.microscope)(
        config.param_names, config.x_mins, config.x_maxs, config.n_divs_default
    )

    # Stores in configuration
    config.config["ndims"] = ndims
    config.config["param_space_bounds"] = param_space_bounds
    config.config["n_points"] = n_points
    config.config["conditions"] = conditions

    # Gets the information from the computer
    config.config["computer"] = platform.uname()._asdict()

    config.config["im_dir_names"] = (
        "conf1", "sted", "conf2", "other", "X", "y",
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
        with h5py.File(os.path.join(config.save_folder, "optim.hdf5"), "a") as logging:
            for dir_name in config.im_dir_names:
                create_group(logging, dir_name, exist_ok=True)
    else:
        # Creates the savename folder
        save_folder = create_savename(
            dry_run = dry_run,
            save_folder = config.save_folder,
            optim = prefart,
            ndims = sum(ndims),
            objectives = len(config.obj_names),
            microscope = config.microscope,
            regressor_name = config.regressor_name
        )
        config.config["save_folder"] = save_folder
        print("========================================================")
        print("[----] Savefolder")
        print("[----]", save_folder)
        print("========================================================\n")

        # Creates the output directory
        if not os.path.isfile(config.save_folder):
            os.makedirs(config.save_folder, exist_ok=dry_run)
        root = os.path.dirname(os.path.abspath(__file__))
        copyfile(os.path.abspath(__file__), os.path.join(config.save_folder, "run.py"))
        copyfile(algorithms.__file__, os.path.join(config.save_folder, "algorithms.py"))
        copyfile(models.__file__, os.path.join(config.save_folder, "models.py"))
        with open(os.path.join(config.save_folder, "config.yml"), 'w') as f:
            yaml.dump(config.config, f)

        # We initialize start trial to 0
        trials = list(range(config.nbre_trials))

        # Creates the file logging
        with h5py.File(os.path.join(config.save_folder, "optim.hdf5"), "w") as logging:
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
    experiment_generator = ExperimentGenerator(
        config.microscope, config.params_conf, config.default_values_dict, ndims, config.param_names
    )
    knowledge_generator = KnowledgeGenerator(
        config.microscope, config=config, ndims=ndims, **config.knowledge_opts
    )
    routine_generator = RoutineGenerator(
        config=config.config, **config.routine_opts
    )
    context_handler = ContextHandler(
        config.ctx_opts
    )

    func = partial(run_trial,
        config=config, pref_articulator=pref_articulator, pareto_sampler=pareto_sampler,
        experiment_generator=experiment_generator, knowledge_generator=knowledge_generator,
        routine_generator=routine_generator, context_handler=context_handler,
        verbose=config.multiprocess_opts["verbose"]
    )
    if config.multiprocess_opts["num_processes"] > 1:
        for gen in mrange(config.multiprocess_opts["num_processes"], trials):
            processes = [
                multiprocessing.Process(target=func, args=(i,)) for i in gen
            ]
            for process in processes:
                process.start()
            for process in processes:
                process.join()
    else:
        for i in trials:
            func(i)

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
    run_TS(config=configurator, dry_run=args.dry_run, prefart=args.prefart, restore_folder=args.restore_folder)
