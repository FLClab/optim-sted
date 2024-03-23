
import sys
import numpy
import random
import yaml
import os
import torch
import argparse
import time
import json

from tqdm.auto import tqdm, trange
from collections import defaultdict
from torch import nn, optim

from stedopt.tools import ParameterSpaceGenerator, create_savename
from stedopt.algorithms import AlgorithmsConfigurator
from stedopt.configuration import Configurator

from loader import get_loader

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path a config file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Wheter a dry-run is run")
    args = parser.parse_args()

    configpath = os.path.dirname(args.config)
    config = Configurator(args.config, configpath=configpath)
    config.regressor_args["default"]["pretrained_opts"]["use"] = False

    trainer_params = {
        "epochs" : 100,
        "dataloader_opts" : {
            "shuffle" : True,
            "batch_size" : 16
        }
    }

    model_names = [
        "../data/contextual-image/20221214-154044_587c662e_DyMIN_prefnet_ContextualImageLinTSDiag",
        "../data/contextual-image/20221214-154044_cc675f82_DyMIN_prefnet_ContextualImageLinTSDiag",
        "../data/contextual-image/20221214-154044_fd765cda_DyMIN_prefnet_ContextualImageLinTSDiag"
    ]

    # Sets random state
    numpy.random.seed(config.random_state)
    random.seed(config.random_state)
    torch.manual_seed(config.random_state)
    torch.cuda.manual_seed(config.random_state)

    # Defines necessary variables
    ndims, param_space_bounds, n_points, conditions = ParameterSpaceGenerator(config.microscope)(
        config.param_names, config.x_mins, config.x_maxs, config.n_divs_default
    )

    save_folder = create_savename(
        dry_run = args.dry_run,
        save_folder = "../data/pretrained",
        ndims = sum(ndims),
        objectives = len(config.obj_names),
        microscope = config.microscope,
        regressor_name = config.regressor_name
    )
    # Creates the output directory
    if not os.path.isfile(save_folder):
        os.makedirs(save_folder, exist_ok=args.dry_run)
    json.dump(model_names, open(os.path.join(save_folder, "training-data.json"), "w"), sort_keys=True, indent=4)

    #Define the algos
    algos = AlgorithmsConfigurator(config.config, param_space_bounds=param_space_bounds)
    optimizers = [
        optim.Adam(algo.regressor.model.parameters(), lr=algo.regressor.learning_rate)
        for algo in algos
    ]

    train_loader, valid_loader = get_loader(model_names, split_train_valid=0.7)

    stats = defaultdict(list)
    min_valid_loss = numpy.inf

    for epoch in range(trainer_params["epochs"]):

        start = time.time()

        train_losses = defaultdict(list)
        valid_losses = defaultdict(list)

        # Resets U
        for algo in algos:
            algo.regressor.set_U(algo.regressor._lambda * torch.ones((algo.regressor.total_param,)))

        for batch in tqdm(train_loader, desc="Training loader", leave=False):
            X, y, history = batch

            indices = list(range(len(algos)))
            random.shuffle(indices)
            for idx in indices:
                loss = algos[idx].train_batch(X, y, history=history, optimizer=optimizers[idx])
                train_losses[config.obj_names[idx]].append(loss)

        # Updates the U matrix with the training loader
        for batch in tqdm(train_loader, desc="Gradient loader", leave=False):
            X, y, history = batch

            indices = list(range(len(algos)))
            random.shuffle(indices)
            for idx in indices:
                algos[idx].add_gradient_batch(X, y, history=history)

        for batch in tqdm(valid_loader, desc="Validation loader", leave=False):
            X, y, history = batch

            indices = list(range(len(algos)))
            random.shuffle(indices)
            for idx in indices:
                loss = algos[idx].predict_batch(X, y, history=history)
                valid_losses[config.obj_names[idx]].append(loss)

        for dataset, values in zip(["train", "valid"], [train_losses, valid_losses]):
            for name, func in zip(["mean", "std", "median", "min", "max"],
                                  [numpy.mean, numpy.std, numpy.median, numpy.min, numpy.max]):
                for obj_name in config.obj_names:
                    stats[f"{obj_name}-{dataset}-{name}"].append(func(values[obj_name]))
                stats[f"{dataset}-{name}"].append(sum([stats[f"{obj_name}-{dataset}-{name}"][-1] for obj_name in config.obj_names]))

        if stats["valid-mean"][-1] < min_valid_loss:
            print("[!!!!] New best model ({:0.4f}. Loss is better than the previous {:0.4f})".format(stats["valid-mean"][-1], min_valid_loss))
            min_valid_loss = stats["valid-mean"][-1]

            # Save model
            for i, obj_name in enumerate(config.obj_names):
                algos[i].save_ckpt(save_folder, prefix=obj_name, trial=str(0))

        json.dump(stats, open(os.path.join(save_folder, "models", "0", "stats.json"), "w"), sort_keys=True, indent=4)

        print("[----] Epoch {} done!".format(epoch + 1))
        print("[----]     Avg loss train/validation : {:0.4f} / {:0.4f}".format(stats["train-mean"][-1], stats["valid-mean"][-1]))
        print("[----]     Current best model : {:0.4f}".format(min_valid_loss))
        print("[----]     Took {} seconds".format(time.time() - start))
