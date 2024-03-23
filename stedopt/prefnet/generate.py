
import os
import yaml
import h5py
import numpy
import random
import matplotlib

from tqdm.auto import tqdm, trange

from stedopt.configuration import Configurator
from stedopt.articulation import PreferenceArticulator
from stedopt.defaults import obj_dict, regressors_dict
from stedopt.tools import create_dataset, create_group

cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    name="nice-prism",
    colors=["#5F4690","#1D6996","#38A6A5","#0F8554","#73AF48","#EDAD08","#E17C05","#CC503E","#94346E"]
)
matplotlib.cm.register_cmap(cmap=cmap)
matplotlib.cm.register_cmap(cmap=cmap.reversed())

DATAPATH = "./data"

def load_pareto_samples(path, trial=0, step=0):
    """
    Loads the Pareto samples from the given path

    :param path: A `str` of the path location of the model
    :param trial: An `int` of the trial index
    :param step: An `int` of the step index

    :returns : A `numpy.ndarray` of the pareto samples
               A `numpy.ndarray` of the selected indices
    """
    with h5py.File(os.path.join(path, "optim.hdf5"), "r") as file:
        y_samples = file["y_samples"][f"{trial}"][f"{step}"][()]
        selected_indexes = file["selected_indexes"][f"{trial}"][f"{step}"][()]

    return y_samples, selected_indexes

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--restart", action="store_true",
                        help="Wheter a new file should be created")
    args = parser.parse_args()

    models = [
        "20221128-132311_0afe1492_DyMIN_prefnet_LinTSDiag",
        "20221128-132311_0582c962_DyMIN_prefnet_LinTSDiag",
        "20221205-153858_51b0486f_DyMIN_prefnet_LinTSDiag",
        "contextual-image/20221125-102843_7afe6f0f_DyMIN_prefnet_ContextualImageLinTSDiag",
        "contextual-image/20221125-102843_993cb345_DyMIN_prefnet_ContextualImageLinTSDiag",
    ]

    # Creates the file logging
    with h5py.File(os.path.join("./optim.hdf5"), "w" if args.restart else "a") as logging:
        for dir_name in ["y_samples", "selected_indexes"]:
            group = create_group(logging, dir_name, exist_ok=True)
            create_group(group, str(0), exist_ok=True)

    random.seed(42)

    current_idx = 0
    for model in tqdm(models, desc="Models"):

        config = yaml.load(open(os.path.join(DATAPATH, model, "config.yml"), "r"), Loader=yaml.Loader)
        config = Configurator(config, load_base=False)

        # Creates the preference articulator
        pref_articulator = PreferenceArticulator(
            config=config.config, mode="optim", model_config=config.model_config
        )

        trial = random.randrange(0, config.nbre_trials)
        for step in trange(5, config.optim_length, leave=False, desc="Steps"):

            # Skips if index is already done...
            with h5py.File("./optim.hdf5", "a") as logging:
                isdone = str(current_idx) in logging["y_samples"][str(0)].keys()
            if isdone:
                current_idx += 1
                continue

            y_samples, selected_index = load_pareto_samples(os.path.join(DATAPATH, model), trial, step)
            timesperpixel = numpy.ones((y_samples.shape[1], 1)) * config.config["default_values_dict"]["pdt"]

            # Selects the optimal point
            config.articulation_opts["cmap"] = "nice-prism"
            index, selected_index = pref_articulator(
                y_samples, [obj_dict[name] for name in config.obj_names],
                config.with_time, timesperpixel, **config.articulation_opts
            )

            with h5py.File("./optim.hdf5", "a") as logging:
                create_dataset(logging["y_samples"][str(0)], str(current_idx), data=y_samples)
                create_dataset(logging["selected_indexes"][str(0)], str(current_idx), data=[index, selected_index])

            current_idx += 1
