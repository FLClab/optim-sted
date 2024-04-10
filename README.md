# Optim-STED

<a target="_blank" href="https://colab.research.google.com/drive/128pOe4KwnZ7MH6HFcd-mG1HjwDWvohnq?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Welcome to Optim-STED! This project aims to optimize the performance of stimulated emission depletion (STED) microscopy.

## Installation

See the steps provided in the [installation](installation.md) file.

## Usage

We provide a working example of the library in a [Google Colab](https://colab.research.google.com/drive/1ckVkFQnTTZpQIrUTbbz_NsQdrApsEd_1?usp=sharing) jupyter notebook.

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

## Citation

If you use this repo please cite the following
```bibtex
@article{bilodeau2024development,
  title={Development of AI-assisted microscopy frameworks through realistic simulation in pySTED},
  author={Bilodeau, Anthony and Michaud-Gagnon, Albert and Chabbert, Julia and Turcotte, Benoit and Heine, J{\"o}rn and Durand, Audrey and Lavoie-Cardinal, Flavie},
  journal={bioRxiv},
  pages={2024--03},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```
