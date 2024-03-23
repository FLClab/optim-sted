# Optim-STED

Welcome to Optim-STED! This project aims to optimize the performance of stimulated emission depletion (STED) microscopy.

## Installation

See the steps provided in the [installation](installation.md) file.

## Usage

There are three ways of using the ``optim-sted``. 

1. [Synthetic optimization of microscopy task](bandit/README.md)
1. [Synthetic optimization of microscopy task using contextual information](history-bandit/README.md)
1. [Optimization of microscopy task on a real microscope](abberior/README.md)

We herein provide the general methodology to launch an experiment. For specific details, please refer to the corresponding folders.

1. **Configuration**
    
    Update the configuration files or use the provided configuration files in each folder.

1. **Launch optimization**

    Each folder contains a ``run.py`` file that is used to launch an optimization session.

    ```bash
    python run.py --prefart optim --config ./configs/config_sted_3params_LinTSDiag.yml
    ```

1. **Inspect results**

    A folder will be created in ``config.save_folder`` with the output from the optimization saved in as an hdf5 file.
